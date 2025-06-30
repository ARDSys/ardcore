"""
Configuration for refine extraction method.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..base import BaseExtractorConfig


@dataclass
class RefineConfig(BaseExtractorConfig):
    """Configuration for refine extraction method."""

    extractor_model_name: str = "gpt-4o"
    refiner_model_name: str = "gpt-4o"
    max_iterations: int = 3
    refiner_model_sleep: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate refine-specific parameters."""
        if not self.extractor_model_name or not isinstance(
            self.extractor_model_name, str
        ):
            raise ValueError(
                f"extractor_model_name must be a non-empty string, got {self.extractor_model_name}"
            )

        if not self.refiner_model_name or not isinstance(self.refiner_model_name, str):
            raise ValueError(
                f"refiner_model_name must be a non-empty string, got {self.refiner_model_name}"
            )

        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )

        if self.refiner_model_sleep < 0:
            raise ValueError(
                f"refiner_model_sleep must be non-negative, got {self.refiner_model_sleep}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "RefineConfig":
        """Create refine config from dictionary, filtering relevant parameters."""
        refine_fields = {f.name for f in cls.__dataclass_fields__.values()}
        refine_params = {k: v for k, v in config_dict.items() if k in refine_fields}
        return cls(**refine_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert refine config to dictionary."""
        return {
            "extractor_model_name": self.extractor_model_name,
            "refiner_model_name": self.refiner_model_name,
            "max_iterations": self.max_iterations,
            "refiner_model_sleep": self.refiner_model_sleep,
        }

    def get_generator(self) -> Callable:
        """Get the generator function for refine extraction."""
        from ard.data.triplets_extractor.extract_refine import extract_refine_generator

        return lambda text: extract_refine_generator(text, self.to_dict())

    @classmethod
    def get_extraction_type(cls) -> str:
        """Get the extraction type string for this config."""
        return "refine"
