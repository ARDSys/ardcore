"""
Extractor registry system that eliminates if/else chains.

Each extractor config automatically registers itself, providing a clean
way to dispatch to the appropriate extractor without hard-coded conditionals.
"""

from typing import Dict, Type

from ..base import BaseExtractorConfig
from .refine import RefineConfig
from .review import ReviewConfig
from .swarm import SwarmConfig


class ExtractorRegistry:
    """Registry for extractor configurations - no if/else chains needed!"""

    _configs: Dict[str, Type[BaseExtractorConfig]] = {}

    @classmethod
    def register(
        cls, extraction_type: str, config_class: Type[BaseExtractorConfig]
    ) -> None:
        """Register an extractor config class."""
        cls._configs[extraction_type] = config_class

    @classmethod
    def get_config_class(cls, extraction_type: str) -> Type[BaseExtractorConfig]:
        """Get the config class for an extraction type."""
        config_class = cls._configs.get(extraction_type)
        if not config_class:
            available_types = list(cls._configs.keys())
            raise ValueError(
                f"Unknown extraction_type: {extraction_type}. "
                f"Available types: {available_types}"
            )
        return config_class

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of all available extraction types."""
        return list(cls._configs.keys())


# Auto-register all extractor configs (no manual if/else needed!)
ExtractorRegistry.register(RefineConfig.get_extraction_type(), RefineConfig)
ExtractorRegistry.register(ReviewConfig.get_extraction_type(), ReviewConfig)
ExtractorRegistry.register(SwarmConfig.get_extraction_type(), SwarmConfig)

# Export everything for easy imports
__all__ = [
    "ExtractorRegistry",
    "RefineConfig",
    "ReviewConfig",
    "SwarmConfig",
]
