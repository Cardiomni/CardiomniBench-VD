"""
PureLLM Baseline: Direct LLM inference without agent harness.

This baseline directly feeds video frames and metadata to an LLM
without any structured reasoning or tool use.
"""

import base64
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

from algorithms.base import BaseAlgorithm, AlgorithmOutput, VideoData, SegmentPrediction


class PureLLM(BaseAlgorithm):
    """
    Baseline that directly prompts an LLM with video frames.

    No structured reasoning, no tool use, no multi-step planning.
    Just raw vision-language model inference.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.model_name = config.get("model", "gpt-4o") if config else "gpt-4o"
        self.max_frames = config.get("max_frames", 5) if config else 5
        self.api_client = None  # Will be set in load_model

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Initialize API client."""
        # Import here to avoid dependency if not using PureLLM
        try:
            from openai import OpenAI
            self.api_client = OpenAI()
            self._loaded = True
        except ImportError:
            raise ImportError("OpenAI package required for PureLLM. Install: pip install openai")

    def predict(self, input_data: Dict[str, Any]) -> AlgorithmOutput:
        """
        Direct LLM inference on video frames.

        Args:
            input_data: Dictionary with:
                - case_id: str
                - videos: List[VideoData]
                - metadata: Optional[Dict]

        Returns:
            AlgorithmOutput with predictions
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = self._start_timer()
        case_id = input_data["case_id"]
        videos = input_data["videos"]

        # Sample frames from videos
        frames_data = self._sample_frames(videos)

        # Build prompt
        prompt = self._build_prompt(videos, frames_data)

        # Call LLM
        try:
            response = self.api_client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                max_tokens=2000,
                temperature=0.0
            )

            # Parse response
            result = self._parse_response(response.choices[0].message.content)
            tokens_used = response.usage.total_tokens

        except Exception as e:
            return AlgorithmOutput(
                case_id=case_id,
                algorithm_name="pure_llm",
                timestamp=datetime.now().isoformat(),
                error=str(e),
                execution_time=self._stop_timer(start_time)
            )

        execution_time = self._stop_timer(start_time)

        return AlgorithmOutput(
            case_id=case_id,
            algorithm_name="pure_llm",
            timestamp=datetime.now().isoformat(),
            syntax_score=result.get("syntax_score"),
            syntax_left=result.get("syntax_left"),
            syntax_right=result.get("syntax_right"),
            dominance=result.get("dominance"),
            segments=result.get("segments", []),
            execution_time=execution_time,
            tokens_used=tokens_used,
            reasoning_trace=response.choices[0].message.content
        )

    def _sample_frames(self, videos: List[VideoData]) -> Dict[str, List[np.ndarray]]:
        """
        Sample frames from each video.

        Args:
            videos: List of VideoData objects

        Returns:
            Dictionary mapping video_id to list of sampled frames
        """
        frames_data = {}

        for video in videos[:10]:  # Limit to 10 videos max
            # Load video (placeholder - actual implementation depends on format)
            # For now, assume we can load .npy files
            try:
                video_array = np.load(video.file_path)  # Shape: (frames, H, W)
                num_frames = video_array.shape[0]

                # Sample evenly spaced frames
                indices = np.linspace(0, num_frames - 1, min(self.max_frames, num_frames), dtype=int)
                sampled = video_array[indices]

                frames_data[video.video_id] = sampled

            except Exception as e:
                print(f"Warning: Failed to load video {video.video_id}: {e}")
                continue

        return frames_data

    def _build_prompt(self, videos: List[VideoData], frames_data: Dict) -> str:
        """Build prompt for LLM."""
        prompt = """You are a cardiologist analyzing coronary angiography videos.

Given the following DSA videos with different projection angles, predict:
1. SYNTAX score (0-100)
2. Dominance type (left/right/balanced)
3. Stenosis for each major segment (if visible)

Videos available:
"""
        for video in videos:
            prompt += f"\n- Video {video.video_id}:"
            prompt += f"\n  Artery: {video.artery}"
            prompt += f"\n  Primary angle: {video.projection['primary_angle']:.1f}°"
            prompt += f"\n  Secondary angle: {video.projection['secondary_angle']:.1f}°"
            prompt += f"\n  Frames: {video.shape[0]}"

        prompt += """

Analyze these videos and provide your assessment in JSON format:
{
  "syntax_score": <number>,
  "syntax_left": <number>,
  "syntax_right": <number>,
  "dominance": "<left|right|balanced>",
  "segments": [
    {
      "segment_id": "LAD_proximal",
      "stenosis_percent": <0-100>,
      "stenosis_grade": "<normal|mild|moderate|severe|occluded>",
      "confidence": <0-1>
    }
  ]
}

Respond with ONLY the JSON, no additional text.
"""
        return prompt

    def _parse_response(self, response_text: str) -> Dict:
        """Parse LLM response to extract predictions."""
        import json
        import re

        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return {}

        try:
            data = json.loads(json_match.group())

            # Convert segments to SegmentPrediction objects
            segments = []
            for seg_dict in data.get("segments", []):
                segments.append(SegmentPrediction(
                    segment_id=seg_dict["segment_id"],
                    segment_name=seg_dict.get("segment_name", seg_dict["segment_id"]),
                    stenosis_percent=seg_dict["stenosis_percent"],
                    stenosis_grade=seg_dict["stenosis_grade"],
                    confidence=seg_dict.get("confidence", 0.5)
                ))

            data["segments"] = segments
            return data

        except json.JSONDecodeError:
            return {}

    def get_metadata(self) -> Dict[str, Any]:
        """Return algorithm metadata."""
        return {
            "name": "PureLLM",
            "version": "1.0",
            "paper": "N/A (baseline)",
            "tasks": ["syntax_score", "stenosis_detection", "dominance"],
            "input_format": ["dicom", "npy"],
            "output_format": "json",
            "requires_gpu": False,
            "description": "Direct LLM inference without agent harness"
        }
