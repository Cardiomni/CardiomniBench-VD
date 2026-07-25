"""
Algorithm registry and discovery system for CardiomniBench-VD.

Provides utilities to list, load, and query available algorithms.
"""

from typing import Dict, List, Type, Optional
from pathlib import Path
import importlib
import json

from .base import BaseAlgorithm


class AlgorithmRegistry:
    """Central registry for all available algorithms."""

    _algorithms: Dict[str, Type[BaseAlgorithm]] = {}
    _metadata_cache: Dict[str, Dict] = {}

    @classmethod
    def register(cls, name: str, algorithm_class: Type[BaseAlgorithm]):
        """
        Register an algorithm.

        Args:
            name: Unique algorithm identifier
            algorithm_class: Algorithm class (subclass of BaseAlgorithm)
        """
        cls._algorithms[name] = algorithm_class

    @classmethod
    def get(cls, name: str) -> Type[BaseAlgorithm]:
        """
        Get algorithm class by name.

        Args:
            name: Algorithm identifier

        Returns:
            Algorithm class

        Raises:
            KeyError: If algorithm not found
        """
        if name not in cls._algorithms:
            raise KeyError(f"Algorithm '{name}' not found. Available: {list(cls._algorithms.keys())}")
        return cls._algorithms[name]

    @classmethod
    def list_all(cls) -> List[str]:
        """List all registered algorithm names."""
        return list(cls._algorithms.keys())

    @classmethod
    def get_metadata(cls, name: str) -> Dict:
        """
        Get cached metadata for an algorithm.

        Args:
            name: Algorithm identifier

        Returns:
            Metadata dictionary
        """
        if name not in cls._metadata_cache:
            # Instantiate to get metadata (lightweight)
            algo_class = cls.get(name)
            algo_instance = algo_class()
            cls._metadata_cache[name] = algo_instance.get_metadata()

        return cls._metadata_cache[name]

    @classmethod
    def list_by_task(cls, task: str) -> List[str]:
        """
        List algorithms that support a specific task.

        Args:
            task: Task name (e.g., "syntax_score", "stenosis_detection")

        Returns:
            List of algorithm names
        """
        result = []
        for name in cls.list_all():
            metadata = cls.get_metadata(name)
            if task in metadata.get("tasks", []):
                result.append(name)
        return result


def load_algorithm(name: str, config: Optional[Dict] = None, checkpoint: Optional[str] = None) -> BaseAlgorithm:
    """
    Load and initialize an algorithm.

    Args:
        name: Algorithm identifier
        config: Algorithm-specific configuration
        checkpoint: Path to model checkpoint

    Returns:
        Initialized algorithm instance

    Example:
        >>> model = load_algorithm('deepcoro_clip', config={'device': 'cuda'})
        >>> result = model.predict(case_data)
    """
    algo_class = AlgorithmRegistry.get(name)
    instance = algo_class(config=config)

    if checkpoint or hasattr(instance, 'default_checkpoint'):
        checkpoint_path = checkpoint or getattr(instance, 'default_checkpoint', None)
        if checkpoint_path:
            instance.load_model(checkpoint_path)

    return instance


def list_available_algorithms() -> List[str]:
    """
    List all available algorithm names.

    Returns:
        List of algorithm identifiers

    Example:
        >>> algos = list_available_algorithms()
        >>> print(algos)
        ['deepcoro_clip', 'cardiosyntax', 'pure_llm', 'qca_tool']
    """
    return AlgorithmRegistry.list_all()


def get_algorithm_doc(name: str) -> str:
    """
    Get documentation for an algorithm.

    Args:
        name: Algorithm identifier

    Returns:
        Formatted documentation string

    Example:
        >>> doc = get_algorithm_doc('deepcoro_clip')
        >>> print(doc)
    """
    metadata = AlgorithmRegistry.get_metadata(name)

    doc = f"""
Algorithm: {metadata['name']}
Version: {metadata.get('version', 'N/A')}
Paper: {metadata.get('paper', 'N/A')}

Tasks: {', '.join(metadata.get('tasks', []))}
Input formats: {', '.join(metadata.get('input_format', []))}
Requires GPU: {metadata.get('requires_gpu', False)}

Usage:
    from algorithms import load_algorithm

    model = load_algorithm('{name}')
    result = model.predict(case_data)
"""
    return doc.strip()


def discover_algorithms():
    """
    Auto-discover and register algorithms from subdirectories.

    Looks for `register.py` in each subdirectory and imports it.
    """
    algorithms_dir = Path(__file__).parent

    for subdir in algorithms_dir.iterdir():
        if not subdir.is_dir():
            continue
        if subdir.name.startswith('_') or subdir.name in ['__pycache__']:
            continue

        register_file = subdir / 'register.py'
        if register_file.exists():
            # Import the register module
            module_path = f"algorithms.{subdir.name}.register"
            try:
                importlib.import_module(module_path)
            except Exception as e:
                print(f"Warning: Failed to register algorithms from {subdir.name}: {e}")


# Auto-discover on import
discover_algorithms()
