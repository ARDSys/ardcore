"""
Base configuration classes and enums for triplets extraction.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict


class ExtractionType(Enum):
    """Enum for extraction types to avoid string-based conditionals."""

    REFINE = "refine"
    REVIEW = "review"
    SWARM = "swarm"


class BaseExtractorConfig(ABC):
    """
    Abstract base class for extractor configurations.

    Provides a common interface for all extractor configs and eliminates
    the need for if/else chains in the factory.
    """

    @abstractmethod
    def validate(self) -> None:
        """Validate configuration parameters."""
        pass

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        pass

    @classmethod
    @abstractmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "BaseExtractorConfig":
        """Create configuration from dictionary."""
        pass

    @abstractmethod
    def get_generator(self) -> Callable:
        """Get the generator function for this extractor type."""
        pass

    @classmethod
    @abstractmethod
    def get_extraction_type(cls) -> str:
        """Get the extraction type string for this config."""
        pass
