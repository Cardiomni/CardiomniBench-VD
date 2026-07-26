"""Pure-PyTorch ``Mamba2`` compatible with ``mamba_ssm.Mamba2`` checkpoints.

Why this exists
---------------
``att_mamba2_unet.pth`` was trained with ``mamba_ssm.Mamba2`` inside every
``AttMamba2Block``. Installing ``mamba-ssm`` requires compiling CUDA extensions,
and this host has no ``nvcc`` (``which nvcc`` is empty; ``pip download
mamba-ssm`` fails during metadata generation because its ``setup.py`` imports
``torch.utils.cpp_extension`` and probes for CUDA). So the checkpoint would be
permanently unusable without a kernel-free path.

This module re-derives the same computation in native PyTorch ops. It is a
*numerical* reimplementation, not a stub: the parameter set, their shapes, and
the state-space recurrence all follow ``mamba_ssm``'s ``Mamba2`` so the
``state_dict`` loads strictly, with identical key names.

Hyperparameters recovered from the checkpoint
---------------------------------------------
Every ``*.att_mamba.mamba.*`` group in ``att_mamba2_unet.pth`` satisfies::

    d_in_proj == 2 * d_inner + 2 * ngroups * d_state + nheads
    conv_dim  == d_inner + 2 * ngroups * d_state

with ``expand=2``, ``d_state=64``, ``ngroups=1``, ``d_conv=4`` and
``headdim=64`` at all three widths (d_model = 64, 128, 256 → nheads = 2, 4, 8).
For example at d_model=256: d_inner=512, conv_dim=512+128=640 (matches
``conv1d.weight`` (640, 1, 4)), d_in_proj=1024+128+8=1160 (matches
``in_proj.weight`` (1160, 256)), ``A_log``/``D``/``dt_bias`` are (8,) = nheads,
and ``norm.weight`` is (512,) = d_inner. These are the ``mamba_ssm`` defaults,
which is what makes an exact reimplementation possible.

Fidelity and cost
-----------------
The scan here is a sequential loop over the sequence length, chunked only over
heads, where ``mamba_ssm`` uses a fused chunked-parallel Triton/CUDA kernel.
Results agree up to floating-point reassociation; throughput does not. For a 96³
patch the longest sequence is 24³ = 13824 tokens (stage 2, windowed) and the
global-attention stages see 12³ = 1728 and 6³ = 216, so a full sliding-window
pass over one CCA volume is minutes, not seconds. Correct and slow beats fast
and wrong for a benchmark number.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormGated(nn.Module):
    """Gated RMSNorm, matching ``mamba_ssm.ops.triton.layernorm_gated``.

    ``mamba_ssm`` normalizes ``x`` and multiplies by ``silu(z)`` in one fused
    kernel; the gate is applied *before* the norm's scale, which is why this
    cannot be replaced by a plain ``RMSNorm`` followed by a multiply.
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        x = x * F.silu(z)
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (x * self.weight.float()).to(z.dtype)


class Mamba2(nn.Module):
    """Mamba-2 layer with a ``mamba_ssm``-identical parameterization.

    Only the arguments used by ``att_mamba2_net.AttMamba2Block`` are honoured
    (``d_model``, ``d_state``, ``d_conv``, ``expand``); the rest exist so the
    constructor signature stays call-compatible.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 128,
        d_conv: int = 4,
        expand: int = 2,
        headdim: int = 64,
        ngroups: int = 1,
        chunk_size: int = 256,
        conv_init=None,
        bias: bool = False,
        conv_bias: bool = True,
        **_ignored,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = expand * d_model
        self.headdim = headdim
        self.ngroups = ngroups
        self.chunk_size = chunk_size

        if self.d_inner % headdim != 0:
            raise ValueError(
                f"d_inner ({self.d_inner}) must be divisible by headdim ({headdim})"
            )
        self.nheads = self.d_inner // headdim

        # in_proj emits, in order: z | x | B | C | dt
        d_in_proj = 2 * self.d_inner + 2 * ngroups * d_state + self.nheads
        self.in_proj = nn.Linear(d_model, d_in_proj, bias=bias)

        # Depthwise causal conv over the x | B | C slice.
        conv_dim = self.d_inner + 2 * ngroups * d_state
        self.conv1d = nn.Conv1d(
            conv_dim,
            conv_dim,
            kernel_size=d_conv,
            groups=conv_dim,
            padding=d_conv - 1,
            bias=conv_bias,
        )

        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.A_log = nn.Parameter(torch.zeros(self.nheads))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.norm = RMSNormGated(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def forward(self, u: torch.Tensor, **_ignored) -> torch.Tensor:
        """Args: ``u`` of shape (batch, seqlen, d_model). Returns the same shape."""
        batch, seqlen, _ = u.shape

        zxbcdt = self.in_proj(u)
        d_bc = self.ngroups * self.d_state
        z, xBC, dt = torch.split(
            zxbcdt, [self.d_inner, self.d_inner + 2 * d_bc, self.nheads], dim=-1
        )

        # Causal depthwise conv + SiLU, dropping the right padding.
        xBC = self.conv1d(xBC.transpose(1, 2))[..., :seqlen].transpose(1, 2)
        xBC = F.silu(xBC)

        x, B, C = torch.split(xBC, [self.d_inner, d_bc, d_bc], dim=-1)

        dt = F.softplus(dt + self.dt_bias)          # (batch, seqlen, nheads)
        A = -torch.exp(self.A_log.float())          # (nheads,)

        y = self._selective_scan(x, dt, A, B, C)
        y = self.norm(y, z)
        return self.out_proj(y)

    def _selective_scan(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        reference: bool = False,
    ) -> torch.Tensor:
        """Scalar-``A`` SSD recurrence, in fp32.

        Mamba-2 restricts ``A`` to one scalar per head, so the state update is an
        elementwise decay rather than a matrix product::

            h_t = exp(A_h * dt_t) * h_{t-1} + dt_t * x_t B_t^T
            y_t = h_t C_t + D_h * x_t

        ``reference=True`` runs that literally, one timestep per iteration. It is
        the readable definition and the correctness oracle, but it costs one
        Python iteration per token and stage 2 sees 24^3 = 13824 tokens per
        window, which measured out at over 15 minutes for a single volume.
        The default path is :meth:`_scan_chunked`, which is algebraically the
        same sum evaluated blockwise; ``tests/test_mamba2_torch.py`` asserts the
        two agree.
        """
        batch, seqlen, _ = x.shape
        nheads, headdim, d_state = self.nheads, self.headdim, self.d_state

        x = x.reshape(batch, seqlen, nheads, headdim).float()
        dt = dt.float()
        # ngroups=1 in this checkpoint: one B/C pair shared by all heads.
        B = B.reshape(batch, seqlen, self.ngroups, d_state).float()
        C = C.reshape(batch, seqlen, self.ngroups, d_state).float()
        if self.ngroups == 1:
            B = B.expand(batch, seqlen, nheads, d_state)
            C = C.expand(batch, seqlen, nheads, d_state)
        else:
            repeat = nheads // self.ngroups
            B = B.repeat_interleave(repeat, dim=2)
            C = C.repeat_interleave(repeat, dim=2)

        if reference:
            y = self._scan_sequential(x, dt, A, B, C)
        else:
            y = self._scan_chunked(x, dt, A, B, C)

        y = y + self.D.view(1, 1, nheads, 1) * x
        return y.reshape(batch, seqlen, self.d_inner).to(self.out_proj.weight.dtype)

    @staticmethod
    def _scan_sequential(
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Literal timestep loop. Correctness oracle for :meth:`_scan_chunked`."""
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        decay = torch.exp(dt * A)

        state = x.new_zeros(batch, nheads, headdim, d_state)
        outputs = []
        for t in range(seqlen):
            state = state * decay[:, t].unsqueeze(-1).unsqueeze(-1)
            state = state + (
                (dt[:, t].unsqueeze(-1) * x[:, t]).unsqueeze(-1) * B[:, t].unsqueeze(-2)
            )
            outputs.append(torch.einsum("bhpn,bhn->bhp", state, C[:, t]))
        return torch.stack(outputs, dim=1)

    def _scan_chunked(
        self,
        x: torch.Tensor,
        dt: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
    ) -> torch.Tensor:
        """Blockwise evaluation of the same sum, one Python step per chunk.

        Unrolling the recurrence gives, for ``cs_t = cumsum(dt)_t``::

            y_t = sum_{s <= t} exp(A (cs_t - cs_s)) dt_s (C_t . B_s) x_s

        Splitting at chunk boundaries makes the ``s`` in the same chunk as ``t``
        a masked Q x Q matrix (computed densely), and every earlier chunk enter
        only through the state carried across the boundary. So the Python loop
        runs once per chunk instead of once per token: 13824 tokens at
        ``chunk_size=128`` is 108 iterations rather than 13824.

        Because ``A < 0`` and ``dt > 0``, every exponent here is <= 0, so the
        exponentials cannot overflow.
        """
        batch, seqlen, nheads, headdim = x.shape
        d_state = B.shape[-1]
        chunk = max(1, int(self.chunk_size))

        pad = (-seqlen) % chunk
        if pad:
            # Zero dt makes a padded step the identity: decay 1, no input.
            x = F.pad(x, (0, 0, 0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            B = F.pad(B, (0, 0, 0, 0, 0, pad))
            C = F.pad(C, (0, 0, 0, 0, 0, pad))
        nchunks = (seqlen + pad) // chunk

        # (b, k, q, h, ...) with k = chunk index, q = position within chunk
        xc = x.view(batch, nchunks, chunk, nheads, headdim)
        dtc = dt.view(batch, nchunks, chunk, nheads)
        Bc = B.view(batch, nchunks, chunk, nheads, d_state)
        Cc = C.view(batch, nchunks, chunk, nheads, d_state)

        # Cumulative decay exponent within each chunk, inclusive of the step.
        cs = torch.cumsum(dtc, dim=2)                       # (b, k, q, h)
        a_in = torch.exp(A * cs)                            # decay from chunk start
        # exponent for the (t, s) pair, t >= s, within a chunk
        rel = cs.unsqueeze(3) - cs.unsqueeze(2)             # (b, k, t, s, h)
        causal = torch.tril(
            torch.ones(chunk, chunk, dtype=torch.bool, device=x.device)
        ).view(1, 1, chunk, chunk, 1)
        decay_ts = torch.where(causal, torch.exp(A * rel), rel.new_zeros(()))

        # Intra-chunk: y[t] = sum_{s<=t} decay_ts * dt_s * (C_t . B_s) * x_s
        CB = torch.einsum("bktha,bksha->bktsh", Cc, Bc)
        weights = decay_ts * CB * dtc.unsqueeze(2)          # (b, k, t, s, h)
        y = torch.einsum("bktsh,bkshp->bkthp", weights, xc)

        # Per-chunk summary state, and the decay across a whole chunk.
        total = cs[:, :, -1, :]                             # (b, k, h)
        a_out = torch.exp(A * (total.unsqueeze(2) - cs))    # start-of-next from s
        chunk_state = torch.einsum(
            "bksh,bkshp,bksha->bkhpa", a_out * dtc, xc, Bc
        )
        chunk_decay = torch.exp(A * total)                  # (b, k, h)

        # Carry states across chunk boundaries: one iteration per chunk.
        states = []
        state = x.new_zeros(batch, nheads, headdim, d_state)
        for k in range(nchunks):
            states.append(state)
            state = state * chunk_decay[:, k].unsqueeze(-1).unsqueeze(-1) + chunk_state[:, k]
        prev = torch.stack(states, dim=1)                   # (b, k, h, p, a)

        # Inter-chunk contribution of the incoming state.
        y = y + torch.einsum("bktha,bkhpa,bkth->bkthp", Cc, prev, a_in)

        y = y.reshape(batch, nchunks * chunk, nheads, headdim)
        return y[:, :seqlen]


__all__ = ["Mamba2", "RMSNormGated"]
