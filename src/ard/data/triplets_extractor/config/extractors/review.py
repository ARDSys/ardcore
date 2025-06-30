"""
Configuration for review extraction method.
"""

from dataclasses import dataclass
from typing import Any, Callable, Dict

from ..base import BaseExtractorConfig


@dataclass
class ReviewConfig(BaseExtractorConfig):
    """Configuration for review extraction method."""

    extractor_model_name: str = "gpt-4o"
    reviewer_model_name: str = "gpt-4o"
    max_iterations: int = 3
    reviewer_model_sleep: int = 0

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate review-specific parameters."""
        if not self.extractor_model_name or not isinstance(
            self.extractor_model_name, str
        ):
            raise ValueError(
                f"extractor_model_name must be a non-empty string, got {self.extractor_model_name}"
            )

        if not self.reviewer_model_name or not isinstance(
            self.reviewer_model_name, str
        ):
            raise ValueError(
                f"reviewer_model_name must be a non-empty string, got {self.reviewer_model_name}"
            )

        if self.max_iterations <= 0:
            raise ValueError(
                f"max_iterations must be positive, got {self.max_iterations}"
            )

        if self.reviewer_model_sleep < 0:
            raise ValueError(
                f"reviewer_model_sleep must be non-negative, got {self.reviewer_model_sleep}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "ReviewConfig":
        """Create review config from dictionary, filtering relevant parameters."""
        review_fields = {f.name for f in cls.__dataclass_fields__.values()}
        review_params = {k: v for k, v in config_dict.items() if k in review_fields}
        return cls(**review_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert review config to dictionary."""
        return {
            "extractor_model_name": self.extractor_model_name,
            "reviewer_model_name": self.reviewer_model_name,
            "max_iterations": self.max_iterations,
            "reviewer_model_sleep": self.reviewer_model_sleep,
        }

    def get_generator(self) -> Callable:
        """Get the generator function for review extraction."""
        from ard.data.triplets_extractor.extract_review import extract_review_generator

        return lambda text: extract_review_generator(text, self.to_dict())

    @classmethod
    def get_extraction_type(cls) -> str:
        """Get the extraction type string for this config."""
        return "review"
