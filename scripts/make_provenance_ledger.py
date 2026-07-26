"""Emit the provenance ledger for every method the benchmark can run.

Why this is generated rather than hand-written
----------------------------------------------
The facts a reader needs in order to trust a number -- which checkpoint produced
it, which commit of that checkpoint, what the checkpoint was trained on, and what
preprocessing was applied -- already live in three places that the runners
actually import:

    benchmark/vlms.py     VLM registry: repo@revision, trained_on, limitations
    benchmark/methods.py  specialist registry: source, author-reported number
    methods/*.toml        per-method paper link and preprocessing parameters

A hand-maintained Markdown table duplicating those fields would drift the first
time a checkpoint is repinned, and a stale provenance table is worse than none,
because it misattributes numbers with full confidence. So this script reads the
same modules the runners read and renders them. If the ledger and the code
disagree, the ledger is regenerated, never edited.

Preprocessing is reported per method because it is not a shared constant: the
volumetric specialists resample to their own training spacing and window HU to
their own range, and applying one method's window to another's checkpoint is a
silent accuracy loss rather than an error.

Usage
-----
    python -m scripts.make_provenance_ledger --out docs/PROVENANCE.md
    python -m scripts.make_provenance_ledger            # writes to stdout
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

METHODS_DIR = REPO_ROOT / "methods"

# Preprocessing keys worth surfacing, in report order. Anything else in the TOML
# is runner plumbing (channels, strides, paths) and does not change what a number
# means.
#
# The vocabulary is deliberately per-family, because these methods do not share
# one preprocessing model and forcing them into common key names would misreport
# them: the MONAI volumetric nets resample with `pixdim` and window with
# `window_a_*`, nnU-Net carries its own fingerprint (`target_spacing_zyx` plus
# foreground statistics and percentile clipping), CardioSYNTAX is a video model
# (`frames_per_clip`, `input_size`), and CM-UNet is 2D (`pad_to`, `model_input`).
PREPROC_KEYS: tuple[str, ...] = (
    # geometry / resampling
    "pixdim",
    "target_spacing",
    "target_spacing_zyx",
    "spacing",
    "orientation",
    # intensity
    "window_a_min",
    "window_a_max",
    "hu_window",
    "hu_min",
    "hu_max",
    "clip_hu",
    "clip_percentile_00_5",
    "clip_percentile_99_5",
    "foreground_mean_hu",
    "foreground_std_hu",
    "normalization_scheme",
    "normalize_mode",
    "normalize",
    "normalize_mean",
    "normalize_std",
    "rescale_uint16",
    # cropping / input shape
    "body_crop",
    "body_threshold_hu",
    "patch_size",
    "roi_size",
    "pad_to",
    "model_input",
    "input_size",
    "resize",
    "resize_mode",
    # temporal (video models)
    "frames_per_clip",
    "frame_sampling",
    "channel_replication",
    # 2D enhancement
    "unsharp_radius",
    "unsharp_amount",
)

# Rendered as one range entry instead of two, since a window is one decision.
_WINDOW_PAIRS = (
    ("window_a_min", "window_a_max"),
    ("hu_min", "hu_max"),
    ("clip_percentile_00_5", "clip_percentile_99_5"),
)


def _load_tomls() -> dict[str, dict[str, Any]]:
    """Read every methods/*.toml, keyed by the file stem."""
    out: dict[str, dict[str, Any]] = {}
    if not METHODS_DIR.is_dir():
        return out
    for path in sorted(METHODS_DIR.glob("*.toml")):
        try:
            with path.open("rb") as fh:
                out[path.stem] = tomllib.load(fh)
        except (OSError, tomllib.TOMLDecodeError) as exc:  # pragma: no cover
            out[path.stem] = {"_error": str(exc)}
    return out


def _fmt(value: Any) -> str:
    """Render a TOML scalar or short list compactly."""
    if isinstance(value, list):
        if len(value) > 1 and len(set(map(str, value))) == 1:
            return f"{_fmt(value[0])}(iso)"
        return "/".join(_fmt(v) for v in value)
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def _walk(node: Any, keys: tuple[str, ...], into: dict[str, Any], depth: int = 0) -> None:
    """Collect the requested keys from anywhere in a nested TOML table.

    The documents nest to method.provenance and to preprocess.*, so this walks the
    whole tree instead of assuming one layout.
    """
    if depth > 6 or not isinstance(node, dict):
        return
    for key, value in node.items():
        if isinstance(value, dict):
            _walk(value, keys, into, depth + 1)
        elif key in keys:
            into.setdefault(key, value)


def _preproc_for(stem: str, tomls: dict[str, dict[str, Any]]) -> str:
    doc = tomls.get(stem)
    if doc is None:
        return "not declared"
    found: dict[str, Any] = {}
    _walk(doc, PREPROC_KEYS, found)
    if not found:
        return "checkpoint defaults"

    parts: list[str] = []
    consumed: set[str] = set()
    for lo, hi in _WINDOW_PAIRS:
        if lo in found and hi in found:
            label = "clip pct" if "percentile" in lo else "HU"
            parts.append(f"{label} {_fmt(found[lo])} to {_fmt(found[hi])}")
            consumed |= {lo, hi}
    for key in PREPROC_KEYS:
        if key in found and key not in consumed:
            parts.append(f"{key}={_fmt(found[key])}")
    return "; ".join(parts)


_PAPER_KEYS = ("doi", "arxiv", "paper", "upstream", "url")


def _paper_for(stem: str, tomls: dict[str, dict[str, Any]]) -> str:
    """Find the paper link, which lives at method.provenance.paper."""
    doc = tomls.get(stem)
    if not doc:
        return "-"
    found: dict[str, Any] = {}
    _walk(doc, _PAPER_KEYS, found)
    for key in _PAPER_KEYS:
        val = found.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return "-"


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def _truncate(text: str, width: int) -> str:
    text = _md_escape(text)
    return text if len(text) <= width else text[: width - 1].rstrip() + "…"


def build() -> str:
    from benchmark.methods import ALL_METHODS
    from benchmark.vlms import ALL_VLMS

    tomls = _load_tomls()
    lines: list[str] = []

    lines.append("# Provenance ledger")
    lines.append("")
    lines.append(
        "Generated by `python -m scripts.make_provenance_ledger`. Do not edit by "
        "hand: the fields are read from `benchmark/vlms.py`, `benchmark/methods.py` "
        "and `methods/*.toml`, which are what the runners import. Regenerate after "
        "repinning a checkpoint."
    )
    lines.append("")

    # ---------------------------------------------------------------- VLMs
    lines.append("## Vision-language models")
    lines.append("")
    lines.append(
        "Every VLM is pinned to an upstream commit. A Hugging Face repo name alone "
        "does not identify weights, because the branch can be updated in place; the "
        "`@sha` suffix is what makes a row reproducible. All of these are "
        "`not_trained` on coronary angiography, so their scores are zero-shot "
        "transfer and are a floor for the task, not a competitive baseline."
    )
    lines.append("")
    lines.append("| Method | Checkpoint @ commit | Trained on | Local status |")
    lines.append("|---|---|---|---|")
    for m in ALL_VLMS:
        p = m.provenance
        try:
            ok, msg = m.check_available()
        except Exception as exc:  # pragma: no cover - defensive
            ok, msg = False, f"check failed: {exc}"
        status = ("available — " if ok else "UNAVAILABLE — ") + _truncate(msg, 60)
        lines.append(
            f"| `{m.name}` | `{_md_escape(p.source)}` | "
            f"{_truncate(p.trained_on or '-', 46)} | {status} |"
        )
    lines.append("")

    lines.append("### Stated limitations")
    lines.append("")
    for m in ALL_VLMS:
        lim = (m.provenance.limitations or "").strip()
        if lim:
            lines.append(f"- **`{m.name}`** — {_md_escape(lim)}")
    lines.append("")

    # ---------------------------------------------------------- specialists
    lines.append("## Specialist models")
    lines.append("")
    lines.append(
        "These are task-specific supervised models. `in-domain` means the "
        "checkpoint was trained on the same dataset it is scored on here, so its "
        "number is an upper reference rather than a comparison. `cross-domain` "
        "means the checkpoint is applied to a dataset it never saw, which is the "
        "usual reason a published number does not reproduce here."
    )
    lines.append("")
    lines.append("| Method | Weights source | Domain | Author-reported | Paper |")
    lines.append("|---|---|---|---|---|")
    for m in ALL_METHODS:
        domain = "cross-domain" if m.is_zero_shot_transfer else "in-domain"
        lines.append(
            f"| `{m.name}` | {_truncate(m.source or '-', 44)} | {domain} | "
            f"{_truncate(m.reported or 'not stated', 44)} | {_paper_for(m.name, tomls)} |"
        )
    lines.append("")

    # -------------------------------------------------------- preprocessing
    lines.append("## Preprocessing per method")
    lines.append("")
    lines.append(
        "Read from `methods/*.toml`. These are not interchangeable: each "
        "checkpoint expects the spacing and HU window it was trained with, and "
        "substituting another method's values degrades accuracy silently instead "
        "of raising an error. `checkpoint defaults` means the TOML declares no "
        "override; `not declared` means no TOML exists for that method."
    )
    lines.append("")
    lines.append("| Method config | Preprocessing |")
    lines.append("|---|---|")
    for stem in sorted(tomls):
        lines.append(f"| `{stem}` | {_md_escape(_preproc_for(stem, tomls))} |")
    lines.append("")

    named = {m.name for m in ALL_METHODS}
    orphans = sorted(set(tomls) - named)
    if orphans:
        lines.append(
            "Configs without a registry entry (variants reachable only by explicit "
            "config, not by method name): "
            + ", ".join(f"`{s}`" for s in orphans)
        )
        lines.append("")

    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write here instead of stdout (e.g. docs/PROVENANCE.md).",
    )
    args = ap.parse_args(argv)

    text = build()
    if args.out is None:
        print(text)
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
