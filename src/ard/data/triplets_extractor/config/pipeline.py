"""
Pipeline configuration for triplets generation (common parameters).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional


@dataclass
class PipelineConfig:
    """
    Configuration for the triplets generation pipeline (common parameters).

    Handles text processing, distillation, and other pipeline-level concerns.
    """

    # === TEXT PROCESSING CONFIGURATION ===
    chunk_method: str = "fixed"
    chunk_size: int = 1000
    chunk_overlap: int = 100

    # === DISTILLATION CONFIGURATION ===
    skip_distillation: bool = False
    distillation_model: str = "gpt-4o"

    # === CORE GENERATION CONFIGURATION ===
    context_model: str = "gpt-4o"
    scientific_domain: str = "bioscience"

    # === PARALLEL PROCESSING CONFIGURATION ===
    max_workers: int = 8

    # === METADATA (AUTO-GENERATED) ===
    timestamp: Optional[str] = field(
        default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S")
    )

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate pipeline configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")

        if self.chunk_overlap < 0:
            raise ValueError(
                f"chunk_overlap must be non-negative, got {self.chunk_overlap}"
            )

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than chunk_size ({self.chunk_size})"
            )

        if self.max_workers <= 0:
            raise ValueError(f"max_workers must be positive, got {self.max_workers}")

        valid_chunk_methods = ["fixed"]
        if self.chunk_method not in valid_chunk_methods:
            raise ValueError(
                f"chunk_method must be one of {valid_chunk_methods}, got {self.chunk_method}"
            )

        if not self.distillation_model or not isinstance(self.distillation_model, str):
            raise ValueError(
                f"distillation_model must be a non-empty string, got {self.distillation_model}"
            )

        if not self.context_model or not isinstance(self.context_model, str):
            raise ValueError(
                f"context_model must be a non-empty string, got {self.context_model}"
            )

        if not self.scientific_domain or not isinstance(self.scientific_domain, str):
            raise ValueError(
                f"scientific_domain must be a non-empty string, got {self.scientific_domain}"
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "PipelineConfig":
        """Create pipeline config from dictionary, filtering relevant parameters."""
        pipeline_fields = {f.name for f in cls.__dataclass_fields__.values()}
        pipeline_params = {k: v for k, v in config_dict.items() if k in pipeline_fields}
        return cls(**pipeline_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline config to dictionary."""
        return {
            "chunk_method": self.chunk_method,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "skip_distillation": self.skip_distillation,
            "distillation_model": self.distillation_model,
            "context_model": self.context_model,
            "scientific_domain": self.scientific_domain,
            "max_workers": self.max_workers,
            "timestamp": self.timestamp,
        }

    def merge_with(
        self, overrides: Optional[Dict[str, Any]] = None
    ) -> "PipelineConfig":
        """Create new config with overrides applied."""
        if overrides is None:
            overrides = {}
        current_dict = self.to_dict()
        current_dict.update(overrides)
        return self.from_dict(current_dict)
