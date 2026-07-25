"""
Base classes and interfaces for CardiomniBench-VD algorithms.

All algorithms (specialist models, baselines, tools) should implement
the BaseAlgorithm interface for standardized evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time


@dataclass
class VideoData:
    """Standard video data structure."""
    video_id: str
    file_path: str
    modality: str  # "XA", "CT", etc.
    artery: str  # "LCA", "RCA", "Unknown"
    projection: Dict[str, float]  # {"primary_angle": ..., "secondary_angle": ...}
    shape: List[int]  # [frames, height, width]
    frame_rate: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SegmentPrediction:
    """Prediction for a coronary segment."""
    segment_id: str  # "LAD_proximal", "D1", "RCA_mid", etc.
    segment_name: str
    stenosis_percent: float  # 0-100
    stenosis_grade: str  # "normal" | "mild" | "moderate" | "severe" | "occluded"
    confidence: float  # 0-1
    evidence: Optional[Dict[str, Any]] = None  # key_frames, views used, etc.


@dataclass
class AlgorithmOutput:
    """Standard algorithm output structure."""
    case_id: str
    algorithm_name: str
    timestamp: str

    # Main predictions
    syntax_score: Optional[float] = None
    syntax_left: Optional[float] = None
    syntax_right: Optional[float] = None
    dominance: Optional[str] = None  # "left" | "right" | "balanced"
    segments: List[SegmentPrediction] = field(default_factory=list)

    # Execution metrics
    execution_time: float = 0.0  # seconds
    tokens_used: int = 0  # for LLM-based methods
    gpu_memory_mb: float = 0.0

    # Provenance
    reasoning_trace: Optional[str] = None
    view_selection_log: Optional[List[Dict]] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "case_id": self.case_id,
            "algorithm_name": self.algorithm_name,
            "timestamp": self.timestamp,
            "syntax_score": self.syntax_score,
            "syntax_left": self.syntax_left,
            "syntax_right": self.syntax_right,
            "dominance": self.dominance,
            "segments": [
                {
                    "segment_id": s.segment_id,
                    "segment_name": s.segment_name,
                    "stenosis_percent": s.stenosis_percent,
                    "stenosis_grade": s.stenosis_grade,
                    "confidence": s.confidence,
                    "evidence": s.evidence
                }
                for s in self.segments
            ],
            "execution_time": self.execution_time,
            "tokens_used": self.tokens_used,
            "gpu_memory_mb": self.gpu_memory_mb,
            "reasoning_trace": self.reasoning_trace,
            "view_selection_log": self.view_selection_log,
            "error": self.error
        }


class BaseAlgorithm(ABC):
    """
    Base class for all algorithms in CardiomniBench-VD.

    Subclasses must implement:
    - load_model()
    - predict()
    - get_metadata()
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize algorithm.

        Args:
            config: Algorithm-specific configuration
        """
        self.config = config or {}
        self.model = None
        self._loaded = False

    @abstractmethod
    def load_model(self, checkpoint_path: Optional[str] = None):
        """
        Load model weights and prepare for inference.

        Args:
            checkpoint_path: Path to model checkpoint (None for default)
        """
        pass

    @abstractmethod
    def predict(self, input_data: Dict[str, Any]) -> AlgorithmOutput:
        """
        Run inference on a single case.

        Args:
            input_data: Dictionary containing:
                - case_id: str
                - videos: List[VideoData]
                - metadata: Optional[Dict]

        Returns:
            AlgorithmOutput with predictions and metrics
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:
        """
        Return algorithm metadata.

        Returns:
            Dictionary with keys:
                - name: str
                - version: str
                - paper: str (URL or citation)
                - tasks: List[str] (e.g., ["syntax_score", "stenosis_detection"])
                - input_format: List[str] (e.g., ["dicom", "npy"])
                - output_format: str
                - requires_gpu: bool
        """
        pass

    def batch_predict(
        self,
        cases: List[Dict[str, Any]],
        num_workers: int = 1
    ) -> List[AlgorithmOutput]:
        """
        Run inference on multiple cases.

        Args:
            cases: List of input_data dictionaries
            num_workers: Number of parallel workers (if supported)

        Returns:
            List of AlgorithmOutput objects
        """
        results = []
        for case in cases:
            try:
                result = self.predict(case)
                results.append(result)
            except Exception as e:
                # Record error but continue
                error_result = AlgorithmOutput(
                    case_id=case.get("case_id", "unknown"),
                    algorithm_name=self.get_metadata()["name"],
                    timestamp=datetime.now().isoformat(),
                    error=str(e)
                )
                results.append(error_result)

        return results

    def _start_timer(self):
        """Helper to start execution timer."""
        return time.time()

    def _stop_timer(self, start_time: float) -> float:
        """Helper to stop execution timer and return elapsed seconds."""
        return time.time() - start_time


class ToolAlgorithm(BaseAlgorithm):
    """
    Base class for lightweight tools (e.g., projection angle calculator).

    Tools are fast, deterministic functions that don't require GPU.
    """

    def load_model(self, checkpoint_path: Optional[str] = None):
        """Tools don't need model loading."""
        self._loaded = True

    def get_metadata(self) -> Dict[str, Any]:
        """Default metadata for tools."""
        return {
            "name": self.__class__.__name__,
            "version": "1.0",
            "paper": "N/A",
            "tasks": ["utility"],
            "input_format": ["any"],
            "output_format": "dict",
            "requires_gpu": False
        }


class LLMBasedAlgorithm(BaseAlgorithm):
    """
    Base class for LLM-based methods (PureLLM, Agent harnesses).

    Tracks token usage and LLM-specific metrics.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.llm_client = None
        self.total_tokens = 0

    def _count_tokens(self, text: str, model: str = "gpt-4") -> int:
        """
        Estimate token count (simplified).

        Args:
            text: Input text
            model: Model name for tokenizer

        Returns:
            Estimated token count
        """
        # Rough estimate: ~4 chars per token
        return len(text) // 4

    def _call_llm(
        self,
        prompt: str,
        model: str = "gpt-4o",
        max_tokens: int = 4096
    ) -> tuple[str, int]:
        """
        Call LLM and track token usage.

        Args:
            prompt: Input prompt
            model: Model identifier
            max_tokens: Maximum response tokens

        Returns:
            Tuple of (response_text, tokens_used)
        """
        raise NotImplementedError("Subclass must implement LLM calling")


def list_available_algorithms() -> List[str]:
    """
    List all registered algorithms.

    Returns:
        List of algorithm names
    """
    # TODO: Implement registry pattern
    return [
        "deepcoro_clip",
        "cardiosyntax",
        "pure_llm",
        "qca_tool",
        "view_classifier"
    ]


def load_algorithm(
    name: str,
    config: Optional[Dict] = None,
    checkpoint_path: Optional[str] = None
) -> BaseAlgorithm:
    """
    Factory function to load an algorithm by name.

    Args:
        name: Algorithm identifier
        config: Algorithm-specific config
        checkpoint_path: Path to model weights

    Returns:
        Initialized algorithm instance
    """
    # TODO: Implement registry pattern
    algorithm_map = {
        # "deepcoro_clip": DeepCOROCLIP,
        # "cardiosyntax": CardioSyntax,
        # "pure_llm": PureLLM,
    }

    if name not in algorithm_map:
        raise ValueError(f"Unknown algorithm: {name}")

    algo = algorithm_map[name](config)
    algo.load_model(checkpoint_path)
    return algo


def get_algorithm_doc(name: str) -> str:
    """
    Get documentation for an algorithm.

    Args:
        name: Algorithm identifier

    Returns:
        Documentation string (markdown format)
    """
    # TODO: Load from markdown files
    return f"Documentation for {name} not yet available."
