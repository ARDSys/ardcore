"""
Clean extractor factory using registry pattern - no if/else chains!
"""

from typing import Any, Callable, Dict

from loguru import logger

from .extractors import ExtractorRegistry


class ExtractorFactory:
    """
    Factory for creating extraction generators based on configuration.

    Uses registry pattern to eliminate if/else chains completely.
    """

    @classmethod
    def create_generator(cls, config_dict: Dict[str, Any]) -> Callable:
        """
        Create generator function from configuration dictionary.

        Args:
            config_dict: Complete configuration dictionary containing extraction_type
                        and extractor-specific parameters

        Returns:
            Callable: Generator function that takes text and returns triplets

        Raises:
            ValueError: If extraction_type is unknown or configuration is invalid
        """
        # Get extraction type
        extraction_type = config_dict.get("extraction_type")
        if not extraction_type:
            raise ValueError("extraction_type is required in config")

        logger.info(f"Using extraction type: {extraction_type}")

        # Use registry to get the right config class (no if/else needed!)
        config_class = ExtractorRegistry.get_config_class(extraction_type)

        # Create and validate the config
        extractor_config = config_class.from_dict(config_dict)

        # Return the generator (each config knows how to create its own generator)
        return extractor_config.get_generator()

    @classmethod
    def get_available_types(cls) -> list[str]:
        """Get list of all available extraction types."""
        return ExtractorRegistry.get_available_types()
