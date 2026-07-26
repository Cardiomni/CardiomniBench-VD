#!/usr/bin/env python3
"""
Claude 3 Opus Baseline Agent for CardiomniBench-VD

BLOCKED: REQUIRES EXTERNAL API, NO KEYS AVAILABLE
────────────────────────────────────────────────────────────────────────────────
This agent calls the Anthropic API (``anthropic.Anthropic``), which requires
``ANTHROPIC_API_KEY`` in the environment. The H20 server has no API key
configured, and per project policy external API calls that transmit case data
are prohibited without explicit approval.

If you need Claude results, use the benchmark's VLM runner with local checkpoints
(``benchmark/vlms.py`` + ``hf_cache/``) rather than the Anthropic API.

Schema note: this reads ``input.images[]`` and writes ``prediction_data`` with
fusion-era fields (``dominance`` / ``segments`` / ``syntax_score``). The current
four public tasks use different schemas per data/tasks/AGENT_SPEC.md.
────────────────────────────────────────────────────────────────────────────────

Pure VLM reasoning without tools (对应EchoAgent的Baseline)
"""

import json
import argparse
import base64
from pathlib import Path
from typing import Dict, Any
import os

try:
    from anthropic import Anthropic
except ImportError:
    print("Warning: anthropic package not installed. Run: pip install anthropic")
    Anthropic = None


class ClaudeAgent:
    """Claude 3 Opus baseline agent - pure VLM reasoning"""

    def __init__(self, api_key: str = None, model: str = "claude-3-opus-20240229"):
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        self.model = model
        self.client = None

        if Anthropic and self.api_key:
            self.client = Anthropic(api_key=self.api_key)

    def encode_image(self, image_path: Path) -> tuple:
        """Encode image to base64 and detect media type"""
        with open(image_path, "rb") as f:
            image_data = base64.standard_b64encode(f.read()).decode("utf-8")

        # Detect media type from extension
        ext = image_path.suffix.lower()
        media_type_map = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp"
        }
        media_type = media_type_map.get(ext, "image/png")

        return image_data, media_type

    def build_prompt(self, task_spec: Dict) -> str:
        """Build prompt for coronary angiography analysis"""

        case_id = task_spec.get("case_id", "unknown")
        task_type = task_spec.get("case_metadata", {}).get("task_type", "unknown")

        prompt = f"""You are an expert interventional cardiologist analyzing coronary angiography images.

Task: {task_type}
Case ID: {case_id}

Please analyze the provided coronary angiography image(s) systematically:

## Analysis Steps:

1. **Dominance Assessment**
   - Identify the dominant coronary circulation pattern
   - Options: right-dominant, left-dominant, or co-dominant

2. **Vessel Identification** (SYNTAX nomenclature)
   - Left Main (LM): segment 5
   - LAD: segments 6 (proximal), 7 (mid), 8 (distal)
   - Diagonal branches: 9 (D1), 10 (D2)
   - LCX: segments 11 (proximal), 13 (mid), 15 (distal)
   - Obtuse marginal: 12 (OM1), 14 (OM2)
   - RCA: segments 1 (proximal), 2 (mid), 3 (distal)
   - PDA: segment 16, PLV: segment 16

3. **Stenosis Quantification**
   - For each visible segment, estimate stenosis percentage (0-100%)
   - Classify severity: <50% (mild), 50-69% (moderate), 70-99% (severe), 100% (occlusion)

4. **SYNTAX Score Calculation**
   - Identify all lesions with ≥50% stenosis
   - Consider lesion complexity factors
   - Calculate total SYNTAX score

## Output Format:

Provide your analysis as a JSON object:

{{
  "dominance": "right|left|co-dominant",
  "segments": [
    {{
      "segment_id": "LAD_6",
      "vessel": "LAD",
      "position": "proximal",
      "stenosis_percent": 60,
      "severity": "moderate",
      "confidence": "high"
    }}
  ],
  "syntax_score": {{
    "total": 15.5,
    "left": 10.0,
    "right": 5.5,
    "risk_tier": "low|intermediate|high"
  }},
  "reasoning": "Detailed clinical reasoning for your assessment..."
}}

Please provide ONLY the JSON object, no additional text.
"""
        return prompt

    def run_task(self, task_spec_path: Path, output_path: Path):
        """Execute task using Claude"""

        # Load task spec
        with open(task_spec_path) as f:
            task_spec = json.load(f)

        case_id = task_spec.get("case_id", "unknown")
        print(f"[Claude] Analyzing {case_id}...")

        if not self.client:
            print("[Claude] ✗ Anthropic client not initialized (missing API key)")
            prediction = self._create_placeholder_prediction(case_id)
        else:
            # Get input images
            input_data = task_spec.get("input", {})
            images = input_data.get("images", [])

            if not images:
                print("[Claude] ✗ No images in task spec")
                prediction = self._create_placeholder_prediction(case_id)
            else:
                # Build prompt
                prompt = self.build_prompt(task_spec)

                # Prepare image for API
                image_file = images[0].get("file_path", "")
                image_path = task_spec_path.parent / image_file

                if not image_path.exists():
                    print(f"[Claude] ✗ Image not found: {image_path}")
                    prediction = self._create_placeholder_prediction(case_id)
                else:
                    # Call Claude API
                    try:
                        image_data, media_type = self.encode_image(image_path)

                        response = self.client.messages.create(
                            model=self.model,
                            max_tokens=2000,
                            messages=[
                                {
                                    "role": "user",
                                    "content": [
                                        {
                                            "type": "image",
                                            "source": {
                                                "type": "base64",
                                                "media_type": media_type,
                                                "data": image_data,
                                            },
                                        },
                                        {
                                            "type": "text",
                                            "text": prompt
                                        }
                                    ],
                                }
                            ],
                        )

                        # Parse response
                        content = response.content[0].text
                        prediction = self._parse_response(content, case_id)

                    except Exception as e:
                        print(f"[Claude] ✗ API call failed: {e}")
                        prediction = self._create_placeholder_prediction(case_id)

        # Write prediction.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)

        print(f"[Claude] ✓ Written {output_path}")

    def _parse_response(self, content: str, case_id: str) -> Dict:
        """Parse Claude response to prediction format"""

        # Try to extract JSON from response
        try:
            # Look for JSON block
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0]
                parsed = json.loads(json_str)
            elif "{" in content and "}" in content:
                # Try to extract raw JSON
                start = content.index("{")
                end = content.rindex("}") + 1
                json_str = content[start:end]
                parsed = json.loads(json_str)
            else:
                parsed = {}
        except:
            parsed = {}

        # Build prediction
        prediction = {
            "case_id": case_id,
            "agent_name": "claude_baseline",
            "model_version": self.model,
            "prediction_data": {
                "dominance": parsed.get("dominance", "unknown"),
                "segments": parsed.get("segments", []),
                "syntax_score": parsed.get("syntax_score", {}),
            },
            "reasoning_trace": parsed.get("reasoning", content[:500]),
        }

        return prediction

    def _create_placeholder_prediction(self, case_id: str) -> Dict:
        """Create placeholder prediction when API fails"""
        return {
            "case_id": case_id,
            "agent_name": "claude_baseline",
            "model_version": self.model,
            "prediction_data": {
                "dominance": "unknown",
                "segments": [],
                "syntax_score": {
                    "total": 0.0,
                    "left": 0.0,
                    "right": 0.0,
                    "risk_tier": "unknown"
                },
            },
            "reasoning_trace": "Placeholder prediction (API not available)",
        }


def main():
    parser = argparse.ArgumentParser(description="Claude 3 Opus Baseline Agent")
    parser.add_argument("--task-spec", type=Path, required=True,
                        help="Path to task_spec.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write prediction.json")
    parser.add_argument("--model", type=str, default="claude-3-opus-20240229",
                        help="Anthropic model name")
    parser.add_argument("--api-key", type=str, default=None,
                        help="Anthropic API key (or use ANTHROPIC_API_KEY env var)")

    args = parser.parse_args()

    agent = ClaudeAgent(api_key=args.api_key, model=args.model)
    agent.run_task(args.task_spec, args.output)


if __name__ == "__main__":
    main()
