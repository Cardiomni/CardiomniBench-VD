"""The provenance ledger must stay derivable from the registries it documents.

The ledger's value is that a reader can trust it to say which weights produced a
number. That only holds if it is regenerated from the live registries rather than
edited, so these tests check the generator's contract instead of the prose:

  * every registered method appears, so a method cannot be evaluated while going
    undocumented;
  * every VLM row carries a pinned ``repo@sha``, since a bare repo name does not
    identify weights;
  * preprocessing is reported per method and is not silently collapsed to
    "checkpoint defaults" when the TOML does declare values -- that failure mode
    is invisible in the rendered table, which is exactly why it is tested;
  * the committed docs/PROVENANCE.md is not stale with respect to the code.

These run offline: the generator reads TOMLs and registry dataclasses, and the
only filesystem-dependent field (local weight status) is not asserted on.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.make_provenance_ledger import (  # noqa: E402
    _load_tomls,
    _paper_for,
    _preproc_for,
    build,
)


@pytest.fixture(scope="module")
def ledger() -> str:
    return build()


@pytest.fixture(scope="module")
def tomls() -> dict:
    return _load_tomls()


def test_every_vlm_is_listed_with_a_pinned_revision(ledger: str) -> None:
    from benchmark.vlms import ALL_VLMS

    assert ALL_VLMS, "VLM registry is empty; the ledger would be vacuous"
    for m in ALL_VLMS:
        assert f"`{m.name}`" in ledger, f"{m.name} missing from ledger"
        src = m.provenance.source
        assert "@" in src, f"{m.name} is not pinned: {src!r}"
        repo, _, sha = src.partition("@")
        assert re.fullmatch(r"[0-9a-f]{40}", sha), (
            f"{m.name} revision is not a full commit sha: {sha!r}"
        )
        assert src in ledger, f"{m.name} pinned source not rendered"


def test_every_specialist_is_listed(ledger: str) -> None:
    from benchmark.methods import ALL_METHODS

    assert ALL_METHODS, "specialist registry is empty"
    for m in ALL_METHODS:
        assert f"`{m.name}`" in ledger, f"{m.name} missing from ledger"


def test_no_vlm_leaked_into_the_specialist_registry() -> None:
    """The duplicate VLM list in methods.py was removed; keep it removed.

    Two registries for the same checkpoints let a run resolve unpinned weights
    under a colliding name, which is unattributable after the fact.
    """
    from benchmark.methods import ALL_METHODS
    from benchmark.vlms import BY_NAME as VLM_BY_NAME

    for m in ALL_METHODS:
        assert m.name not in VLM_BY_NAME, (
            f"{m.name} is registered in both methods.py and vlms.py"
        )
        assert str(m.family) != "vlm", (
            f"{m.name} is a VLM in methods.py; VLMs belong in vlms.py where "
            "revisions are pinned"
        )


def test_declared_preprocessing_is_not_reported_as_defaults(tomls: dict) -> None:
    """A TOML that declares preprocessing must not render as 'checkpoint defaults'.

    The generator matches a fixed key vocabulary. When a method uses key names
    outside it (nnU-Net's target_spacing_zyx, CardioSYNTAX's frames_per_clip),
    the row silently degrades to 'checkpoint defaults' while the real values sit
    in the file. This asserts the vocabulary keeps up with the configs.
    """
    assert tomls, "no methods/*.toml found"
    for stem, doc in tomls.items():
        section = doc.get("preprocess") or doc.get("preprocessing") or {}
        substantive = {k: v for k, v in section.items() if not k.startswith("_")}
        if not substantive:
            continue
        rendered = _preproc_for(stem, tomls)
        assert rendered not in ("checkpoint defaults", "not declared"), (
            f"{stem} declares {sorted(substantive)[:4]} but the ledger reports "
            f"{rendered!r}; add the key(s) to PREPROC_KEYS"
        )


def test_paper_link_is_found_when_declared(tomls: dict) -> None:
    for stem, doc in tomls.items():
        prov = (doc.get("method") or {}).get("provenance") or {}
        if not prov.get("paper"):
            continue
        assert _paper_for(stem, tomls) != "-", f"{stem} declares a paper but none rendered"


def test_committed_ledger_is_current(ledger: str) -> None:
    """docs/PROVENANCE.md must match what the generator produces now.

    Compared without the local-weight-status column, which legitimately differs
    by machine and by what has finished downloading.
    """
    path = REPO_ROOT / "docs" / "PROVENANCE.md"
    if not path.is_file():
        pytest.skip("docs/PROVENANCE.md not generated yet")

    def normalize(text: str) -> list[str]:
        out = []
        for line in text.splitlines():
            if "available" in line or "UNAVAILABLE" in line:
                line = re.sub(r"\|[^|]*(?:available|UNAVAILABLE)[^|]*\|", "| |", line)
            out.append(line.rstrip())
        return [ln for ln in out if ln]

    assert normalize(path.read_text(encoding="utf-8")) == normalize(ledger), (
        "docs/PROVENANCE.md is stale; regenerate with "
        "`python -m scripts.make_provenance_ledger --out docs/PROVENANCE.md`"
    )
