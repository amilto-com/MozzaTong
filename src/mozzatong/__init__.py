"""
MozzaTong: State-of-the-art LLM for converting natural language to TONG programming language.

This package provides tools for:
- Fine-tuning language models for natural language to TONG conversion
- Generating training datasets
- Inference and code generation
- CLI interface for easy usage
"""

__version__ = "0.1.0"
__author__ = "AMILTO SARL"
__email__ = "contact@amilto.com"

from .inference import TongGenerator
from .trainer import TongTrainer
from .dataset import TongDatasetGenerator

__all__ = ["TongGenerator", "TongTrainer", "TongDatasetGenerator"]