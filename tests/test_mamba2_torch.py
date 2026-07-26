"""The kernel-free Mamba2 must reproduce the recurrence it replaces.

``att_mamba2_unet.pth`` was trained with ``mamba_ssm.Mamba2``, whose CUDA
kernels cannot be built on this host (no ``nvcc``). The checkpoint therefore
runs through :mod:`algorithms.specialist_models.att_mamba2.mamba2_torch`, a
reimplementation in plain PyTorch ops.

That substitution is only legitimate if the arithmetic matches, and a wrong
scan would not announce itself: the state dict still loads, inference still
returns a mask, and the Dice number would simply be wrong in a way that looks
like a domain-shift result. These tests pin the fast path against the literal
timestep loop, which is short enough to read as the definition of the
recurrence, and pin the parameter shapes against the released checkpoint.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from algorithms.specialist_models.att_mamba2.mamba2_torch import Mamba2, RMSNormGated

# (d_model, expected nheads) triples covering every width in the checkpoint:
# stage 2, stage 3, and the bottleneck.
CHECKPOINT_WIDTHS = [(64, 2), (128, 4), (256, 8)]


def _scan_inputs(module: Mamba2, batch: int, seqlen: int, seed: int = 0):
    """Inputs shaped as ``_selective_scan`` hands them to the scan.

    ``dt`` is positive (it comes from a softplus) and ``A`` is negative (it is
    ``-exp(A_log)``); the chunked path relies on both to keep its exponentials
    bounded, so the test must respect the same signs.
    """
    gen = torch.Generator().manual_seed(seed)
    h, p, n = module.nheads, module.headdim, module.d_state
    return {
        "x": torch.randn(batch, seqlen, h, p, generator=gen),
        "dt": torch.rand(batch, seqlen, h, generator=gen) * 0.5 + 0.01,
        "A": -torch.exp(torch.randn(h, generator=gen)),
        "B": torch.randn(batch, seqlen, h, n, generator=gen),
        "C": torch.randn(batch, seqlen, h, n, generator=gen),
    }


@pytest.mark.parametrize("d_model,expected_nheads", CHECKPOINT_WIDTHS)
def test_hyperparameters_match_checkpoint_layout(d_model: int, expected_nheads: int) -> None:
    """Derived dimensions must line up with the released tensor shapes.

    At d_model=256 the checkpoint has ``in_proj.weight`` (1160, 256),
    ``conv1d.weight`` (640, 1, 4) and ``A_log`` (8,); those follow from
    expand=2, d_state=64, ngroups=1, headdim=64, so this test fails if a default
    drifts away from the trained configuration.
    """
    m = Mamba2(d_model=d_model, d_state=64, expand=2, headdim=64)

    assert m.nheads == expected_nheads
    assert m.d_inner == 2 * d_model
    assert m.in_proj.weight.shape == (2 * m.d_inner + 2 * 64 + m.nheads, d_model)
    assert m.conv1d.weight.shape == (m.d_inner + 2 * 64, 1, 4)
    assert m.A_log.shape == (m.nheads,)
    assert m.D.shape == (m.nheads,)
    assert m.dt_bias.shape == (m.nheads,)
    assert m.norm.weight.shape == (m.d_inner,)
    assert m.out_proj.weight.shape == (d_model, m.d_inner)


@pytest.mark.parametrize("d_model,expected_nheads", CHECKPOINT_WIDTHS)
def test_chunked_scan_matches_sequential_reference(d_model: int, expected_nheads: int) -> None:
    """The blockwise sum must equal the timestep-by-timestep recurrence."""
    m = Mamba2(d_model=d_model, d_state=64, expand=2, headdim=64, chunk_size=64).eval()
    assert m.nheads == expected_nheads
    args = _scan_inputs(m, batch=2, seqlen=192, seed=d_model)

    with torch.no_grad():
        reference = m._scan_sequential(**args)
        chunked = m._scan_chunked(**args)

    scale = reference.abs().max()
    relative_error = (reference - chunked).abs().max() / scale
    assert relative_error < 1e-4, (
        f"chunked scan drifted from the reference recurrence "
        f"(relative error {relative_error:.2e}); inference would return a "
        "plausible but numerically wrong mask"
    )


@pytest.mark.parametrize("seqlen", [1, 7, 64, 65, 130])
def test_scan_exact_when_length_is_not_a_multiple_of_chunk(seqlen: int) -> None:
    """Padded tail steps must not leak into the result.

    Padding sets ``dt = 0`` so a padded step decays by ``exp(0) = 1`` and adds
    no input. If that ever stops holding, short sequences break first, which is
    why the lengths here straddle the chunk boundary.
    """
    m = Mamba2(d_model=64, d_state=64, expand=2, headdim=64, chunk_size=64).eval()
    args = _scan_inputs(m, batch=1, seqlen=seqlen, seed=seqlen)

    with torch.no_grad():
        reference = m._scan_sequential(**args)
        chunked = m._scan_chunked(**args)

    relative_error = (reference - chunked).abs().max() / reference.abs().max()
    assert relative_error < 1e-4


def test_forward_is_finite_and_shape_preserving() -> None:
    """A full layer call must return its input shape without NaNs."""
    m = Mamba2(d_model=64, d_state=64, expand=2, headdim=64, chunk_size=64).eval()
    u = torch.randn(2, 100, 64)

    with torch.no_grad():
        y = m(u)

    assert y.shape == u.shape
    assert torch.isfinite(y).all()


def test_gated_norm_applies_gate_before_scaling() -> None:
    """RMSNorm-with-gate is fused upstream, so the order of operations matters.

    ``mamba_ssm`` computes ``rms(x * silu(z)) * weight``. Normalizing first and
    gating afterwards gives a different result whenever the gate is not
    constant, so this pins the sequence rather than just the output shape.
    """
    norm = RMSNormGated(8)
    with torch.no_grad():
        norm.weight.copy_(torch.linspace(0.5, 2.0, 8))
    x = torch.randn(2, 5, 8)
    z = torch.randn(2, 5, 8)

    with torch.no_grad():
        got = norm(x, z)

        gated = x * torch.nn.functional.silu(z)
        expected = gated * torch.rsqrt(gated.pow(2).mean(-1, keepdim=True) + norm.eps)
        expected = expected * norm.weight

    assert torch.allclose(got, expected, atol=1e-6)
