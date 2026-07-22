"""Judge backends — grade subjective rubric criteria (the '换 rubric CLI/LLM' axis).

One ``Judge`` protocol, three interchangeable implementations selected by
``judge.backend``:

    mock — deterministic canned grade (offline; gray-box path, no API key)
    llm  — call an LLM chat API (Anthropic-style; needs api_key_env in env)
    cli  — shell out to an external judge program that prints grade JSON

Every backend takes a fully-rendered prompt string and returns the same dict:
    {"grade": "A"|"B"|"C"|..., "reasoning": str, "evidence_quotes": [str], ...}

Points are NOT decided here — the backend only emits a categorical grade, exactly
like BiomniBench's judge. The rubric's grade->points table is applied in scoring.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from typing import Any, Dict, List, Optional

from .config import JudgeConfig

logger = logging.getLogger(__name__)


def make_judge(cfg: JudgeConfig) -> "Judge":
    """Factory: build the judge backend named by cfg.backend."""
    if cfg.backend == "mock":
        return MockJudge(cfg)
    if cfg.backend == "llm":
        return LLMJudgeBackend(cfg)
    if cfg.backend == "cli":
        return CLIJudge(cfg)
    raise ValueError(f"unknown judge.backend {cfg.backend!r}")


def parse_grade_json(text: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
    """Extract a grade dict from judge output (tolerates markdown fences / prose)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        candidate = brace.group(0) if brace else text
    try:
        result = json.loads(candidate)
    except json.JSONDecodeError as e:
        return {"grade": None, "reasoning": f"parse error: {e}",
                "evidence_quotes": [], "parse_error": True, "raw": text[:500]}

    if "grade" not in result:
        result["grade"] = None
    result.setdefault("reasoning", "")
    result.setdefault("evidence_quotes", [])
    if valid_grades and result["grade"] not in valid_grades:
        result["grade_out_of_scale"] = True
    return result


class Judge:
    """Base judge protocol."""

    def __init__(self, cfg: JudgeConfig):
        self.cfg = cfg

    def grade(self, prompt: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
        raise NotImplementedError


class MockJudge(Judge):
    """Return a fixed grade without any external call. Enables offline runs."""

    def grade(self, prompt: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
        grade = self.cfg.mock_grade
        if valid_grades and grade not in valid_grades:
            grade = valid_grades[0]
        return {
            "grade": grade,
            "reasoning": "[MOCK JUDGE] deterministic grade for pipeline gray-box testing.",
            "evidence_quotes": [],
            "confidence": "n/a",
            "backend": "mock",
        }


class CLIJudge(Judge):
    """Delegate grading to an external CLI program.

    The command template may reference ``{prompt_file}`` (a temp file holding the
    rendered prompt) and ``{model}``. The program must print grade JSON to stdout.
    This is the '换 rubric CLI' path — plug in any scorer that speaks JSON.
    """

    def grade(self, prompt: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
        import tempfile

        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tf:
            tf.write(prompt)
            prompt_file = tf.name
        try:
            cmd = self.cfg.command.format(prompt_file=prompt_file, model=self.cfg.model)
            proc = subprocess.run(
                cmd, shell=True, capture_output=True, text=True, timeout=300
            )
            if proc.returncode != 0:
                return {"grade": None, "reasoning": f"cli judge failed: {proc.stderr[:400]}",
                        "evidence_quotes": [], "error": True}
            return parse_grade_json(proc.stdout, valid_grades)
        except (subprocess.TimeoutExpired, OSError) as e:
            return {"grade": None, "reasoning": f"cli judge error: {e}",
                    "evidence_quotes": [], "error": True}
        finally:
            try:
                os.unlink(prompt_file)
            except OSError:
                pass


class LLMJudgeBackend(Judge):
    """Grade via an LLM chat API (Anthropic messages format by default).

    The client import is lazy so the pipeline runs with mock/cli backends even
    when no LLM SDK or API key is present. Selecting backend=llm without the SDK
    or key raises a clear, actionable error.
    """

    def grade(self, prompt: str, valid_grades: Optional[List[str]] = None) -> Dict[str, Any]:
        text = self._call(prompt)
        return parse_grade_json(text, valid_grades)

    def _call(self, prompt: str) -> str:
        api_key = os.environ.get(self.cfg.api_key_env)
        if not api_key:
            raise RuntimeError(
                f"judge.backend=llm needs {self.cfg.api_key_env} set in the environment"
            )
        try:
            from anthropic import Anthropic
        except ImportError as e:
            raise RuntimeError(
                "judge.backend=llm requires the 'anthropic' package (pip install anthropic)"
            ) from e

        client = Anthropic(api_key=api_key)
        resp = client.messages.create(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text
