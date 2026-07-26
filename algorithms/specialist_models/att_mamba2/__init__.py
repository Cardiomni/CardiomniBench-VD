"""Attention-Mamba2 U-Net, vendored for the ``coronary_att_mamba2`` checkpoint.

``att_mamba2_net.py`` is copied verbatim from upstream
https://github.com/noahschuetz/coronary-artery-segmentation
(``src/models/segmentation/att_mamba2_net.py``), with one edit: the
``mamba_ssm`` import now falls back to :mod:`mamba2_torch`, a pure-PyTorch
``Mamba2`` with the same parameters, because ``mamba-ssm`` needs ``nvcc`` and
this host has none. See ``mamba2_torch`` for the shape evidence.

Vendoring rather than importing from a clone keeps evaluation reproducible: a
checkout under ``/tmp`` would disappear between runs.
"""

from .att_mamba2_net import MAMBA2_IMPL, AttMamba2UNet, get_att_mamba2_unet

__all__ = ["AttMamba2UNet", "get_att_mamba2_unet", "MAMBA2_IMPL"]
