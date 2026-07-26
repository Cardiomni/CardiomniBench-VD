#!/usr/bin/env python
"""CCA 3D coronary segmentation agent — thin shell over monai_unet_runner.

⚠️  CORRECTED VERSION — replaces the broken preprocessing that scored Dice 0.048
────────────────────────────────────────────────────────────────────────────────
The original implementation (see git history or the warning block that was here)
imported 7 MONAI transforms but never called them. It scored Dice 0.048 on
case_cca_0001_0 where the correct preprocessing scores 0.536 on the same
checkpoint — an 11× gap caused by missing resampling, RAS orientation, body crop,
wrong HU window, and whole-volume z-score.

This version is a ~40-line shell that delegates to the verified
``benchmark/runners/monai_unet_runner.py`` pipeline. One source of truth for
preprocessing (``methods/coronary_unet.toml``), one code path.

Usage is unchanged:
    python cca_unet_agent.py --case-dir data/tasks/cca_segmentation/cases/case_cca_0001_0 \\
                             --output-dir /tmp/cca_out \\
                             --device cuda:1
────────────────────────────────────────────────────────────────────────────────
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument(
        "--method",
        default="coronary_unet",
        help="Which methods/<name>.toml to use (default: coronary_unet)",
    )
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError:
        print(
            "[cca_unet] PyTorch / MONAI required; use e.g. /opt/anaconda3/envs/gkp-gsa/bin/python",
            file=sys.stderr,
        )
        return 1

    from benchmark.io_spec import load_case_input
    from benchmark.method_config import load_method_config
    from benchmark.runners import monai_unet_runner

    case = load_case_input(args.case_dir)
    method_config = load_method_config(args.method)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # The runner's predict() is typed for the SpecialistMethod protocol, but
    # MethodConfig is a duck-type match: it has .name and .weights_path, which
    # is all the runner reads from the method argument (the preprocessing /
    # architecture config comes from load_method_config inside predict).
    prediction = monai_unet_runner.predict(
        method=method_config,  # type: ignore[arg-type]
        case=case,
        output_dir=args.output_dir,
        device=args.device,
    )

    # Write prediction.json matching the old schema so existing callers don't break.
    pred_json = args.output_dir / "prediction.json"
    with pred_json.open("w") as f:
        json.dump(prediction.to_dict(), f, indent=2, ensure_ascii=False)

    print(
        f"[cca_unet] {case.case_id}: wrote {prediction.mask_path.name} + prediction.json",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
