"""
Configuration package for triplets extraction.

Provides a clean, modular configuration system with no if/else chains.
"""

from typing import Union

from .base import BaseExtractorConfig, ExtractionType
from .extractors import ExtractorRegistry, RefineConfig, ReviewConfig, SwarmConfig
from .factory import ExtractorFactory
from .pipeline import PipelineConfig

# Type union for convenience
ExtractorConfig = Union[RefineConfig, ReviewConfig, SwarmConfig]

# Public API - clean modular interfaces
__all__ = [
    # Main interfaces
    "PipelineConfig",
    "ExtractorFactory",
    # Base classes and enums
    "ExtractionType",
    "BaseExtractorConfig",
    # Individual extractor configs
    "RefineConfig",
    "ReviewConfig",
    "SwarmConfig",
    "ExtractorConfig",
    # Registry (for advanced usage)
    "ExtractorRegistry",
]
