#!/usr/bin/env python3
"""
GPT-4V Baseline Agent for CardiomniBench-VD

BLOCKED: REQUIRES EXTERNAL API, NO KEYS AVAILABLE
────────────────────────────────────────────────────────────────────────────────
This agent calls the OpenAI API (``openai.OpenAI``), which requires
``OPENAI_API_KEY`` in the environment. The H20 server has no API key configured,
and per project policy external API calls that transmit case data are prohibited
without explicit approval.

If you need GPT-4V results, use the benchmark's VLM runner with local checkpoints
(``benchmark/vlms.py`` + ``hf_cache/``) rather than the OpenAI API.

Schema note: this reads ``input.images[]`` and writes ``prediction_data`` with
fusion-era fields (``dominance`` / ``segments`` / ``syntax_score``). The current
four public tasks use different schemas per data/tasks/AGENT_SPEC.md.
────────────────────────────────────────────────────────────────────────────────

Pure VLM reasoning without tools (对应EchoAgent的Baseline)
"""

import json
import yaml
import argparse
import base64
from pathlib import Path
from typing import Dict, Any
import os

try:
    from openai import OpenAI
except ImportError:
    print("Warning: openai package not installed. Run: pip install openai")
    OpenAI = None


class GPT4VAgent:
    """GPT-4V baseline agent - pure VLM reasoning"""

    def __init__(self, api_key: str = None, model: str = "gpt-4o"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.client = None

        if OpenAI and self.api_key:
            self.client = OpenAI(api_key=self.api_key)

    def encode_image(self, image_path: Path) -> str:
        """Encode image to base64"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode('utf-8')

    def build_prompt(self, task_spec: Dict) -> str:
        """Build prompt for coronary angiography analysis"""

        case_id = task_spec.get("case_id", "unknown")
        task_type = task_spec.get("case_metadata", {}).get("task_type", "unknown")

        # Base prompt
        prompt = f"""You are an expert cardiologist analyzing coronary angiography images.

Task: {task_type}
Case ID: {case_id}

Please analyze the provided coronary angiography image(s) and provide:

1. **Dominance type**: Determine if the coronary circulation is right-dominant, left-dominant, or co-dominant.

2. **Vessel segments**: Identify all visible coronary artery segments using SYNTAX nomenclature:
   - Left Main (LM)
   - LAD segments (proximal, mid, distal)
   - Diagonal branches (D1, D2)
   - LCX segments (proximal, mid, distal)
   - Obtuse marginal branches (OM1, OM2)
   - RCA segments (proximal, mid, distal)
   - PDA, PLV

3. **Stenosis assessment**: For each visible segment, estimate stenosis percentage (0-100%).

4. **SYNTAX score**: If applicable, calculate the SYNTAX score based on lesion locations and characteristics.

Provide your analysis in a structured JSON format:
{{
  "dominance": "right|left|co-dominant",
  "segments": [
    {{
      "segment_id": "LAD_6",
      "vessel": "LAD",
      "position": "proximal",
      "stenosis_percent": 60,
      "confidence": "high|medium|low"
    }}
  ],
  "syntax_score": {{
    "total": 0.0,
    "left": 0.0,
    "right": 0.0,
    "risk_tier": "low|intermediate|high"
  }},
  "reasoning": "Your clinical reasoning here..."
}}
"""
        return prompt

    def run_task(self, task_spec_path: Path, output_path: Path):
        """Execute task using GPT-4V"""

        # Load task spec
        with open(task_spec_path) as f:
            task_spec = json.load(f)

        case_id = task_spec.get("case_id", "unknown")
        print(f"[GPT-4V] Analyzing {case_id}...")

        if not self.client:
            print("[GPT-4V] ✗ OpenAI client not initialized (missing API key)")
            # Create placeholder prediction
            prediction = self._create_placeholder_prediction(case_id)
        else:
            # Get input images
            input_data = task_spec.get("input", {})
            images = input_data.get("images", [])

            if not images:
                print("[GPT-4V] ✗ No images in task spec")
                prediction = self._create_placeholder_prediction(case_id)
            else:
                # Build prompt
                prompt = self.build_prompt(task_spec)

                # Prepare image for API
                image_file = images[0].get("file_path", "")
                image_path = task_spec_path.parent / image_file

                if not image_path.exists():
                    print(f"[GPT-4V] ✗ Image not found: {image_path}")
                    prediction = self._create_placeholder_prediction(case_id)
                else:
                    # Call GPT-4V API
                    try:
                        response = self.client.chat.completions.create(
                            model=self.model,
                            messages=[
                                {
                                    "role": "system",
                                    "content": "You are an expert cardiologist analyzing coronary angiography."
                                },
                                {
                                    "role": "user",
                                    "content": [
                                        {"type": "text", "text": prompt},
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": f"data:image/png;base64,{self.encode_image(image_path)}"
                                            }
                                        }
                                    ]
                                }
                            ],
                            max_tokens=2000
                        )

                        # Parse response
                        content = response.choices[0].message.content
                        prediction = self._parse_response(content, case_id)

                    except Exception as e:
                        print(f"[GPT-4V] ✗ API call failed: {e}")
                        prediction = self._create_placeholder_prediction(case_id)

        # Write prediction.json
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(prediction, f, indent=2)

        print(f"[GPT-4V] ✓ Written {output_path}")

    def _parse_response(self, content: str, case_id: str) -> Dict:
        """Parse GPT-4V response to prediction format"""

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
            "agent_name": "gpt4v_baseline",
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
            "agent_name": "gpt4v_baseline",
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
    parser = argparse.ArgumentParser(description="GPT-4V Baseline Agent")
    parser.add_argument("--task-spec", type=Path, required=True,
                        help="Path to task_spec.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Path to write prediction.json")
    parser.add_argument("--model", type=str, default="gpt-4o",
                        help="OpenAI model name")
    parser.add_argument("--api-key", type=str, default=None,
                        help="OpenAI API key (or use OPENAI_API_KEY env var)")

    args = parser.parse_args()

    agent = GPT4VAgent(api_key=args.api_key, model=args.model)
    agent.run_task(args.task_spec, args.output)


if __name__ == "__main__":
    main()
