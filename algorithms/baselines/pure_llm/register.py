"""Register PureLLM baseline."""

from algorithms import AlgorithmRegistry
from .pure_llm import PureLLM

AlgorithmRegistry.register("pure_llm", PureLLM)
