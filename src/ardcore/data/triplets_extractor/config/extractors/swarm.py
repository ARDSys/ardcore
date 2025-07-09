"""
Configuration for swarm extraction method.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from ..base import BaseExtractorConfig


# TODO: use approach from `SwarmKGConfig`
@dataclass
class SwarmConfig(BaseExtractorConfig):
    """Configuration for swarm extraction method."""

    prompt_repository: str
    scientific_domain: str = "bioscience"

    # Homogeneous swarm config (single model replicated)
    extractor_model_name: Optional[str] = None
    swarm_size: int = 1
    extractor_model_params: Dict[str, Any] = field(default_factory=dict)

    # Heterogeneous swarm config (multiple different models)
    extractor_model_names: List[str] = field(default_factory=list)

    # Merging configuration for swarm
    merging_model_name: Optional[str] = None
    merging_model_params: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate configuration after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate swarm-specific parameters."""
        if not self.prompt_repository or not isinstance(self.prompt_repository, str):
            raise ValueError(
                f"prompt_repository must be a non-empty string, got {self.prompt_repository}"
            )

        if not self.scientific_domain or not isinstance(self.scientific_domain, str):
            raise ValueError(
                f"scientific_domain must be a non-empty string, got {self.scientific_domain}"
            )

        # Check that either homogeneous or heterogeneous config is provided
        has_homogeneous = self.extractor_model_name and self.swarm_size > 0
        has_heterogeneous = (
            self.extractor_model_names and len(self.extractor_model_names) > 0
        )

        if not has_homogeneous and not has_heterogeneous:
            raise ValueError(
                "Swarm extraction requires either homogeneous config (extractor_model_name + swarm_size > 0) "
                "or heterogeneous config (non-empty extractor_model_names)"
            )

        if has_homogeneous and has_heterogeneous:
            raise ValueError(
                "Swarm extraction cannot have both homogeneous and heterogeneous config. "
                "Use either extractor_model_name OR extractor_model_names, not both."
            )

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "SwarmConfig":
        """Create swarm config from dictionary, filtering relevant parameters."""
        swarm_fields = {f.name for f in cls.__dataclass_fields__.values()}
        swarm_params = {k: v for k, v in config_dict.items() if k in swarm_fields}
        return cls(**swarm_params)

    def to_dict(self) -> Dict[str, Any]:
        """Convert swarm config to dictionary."""
        return {
            "prompt_repository": self.prompt_repository,
            "scientific_domain": self.scientific_domain,
            "extractor_model_name": self.extractor_model_name,
            "swarm_size": self.swarm_size,
            "extractor_model_params": self.extractor_model_params,
            "extractor_model_names": self.extractor_model_names,
            "merging_model_name": self.merging_model_name,
            "merging_model_params": self.merging_model_params,
        }

    def get_generator(self) -> Callable:
        """Get the generator function for swarm extraction."""
        from ardcore.data.triplets_extractor.extract_swarm import (
            extract_swarm_generator,
        )

        return lambda text: extract_swarm_generator(text, self.to_dict())

    @classmethod
    def get_extraction_type(cls) -> str:
        """Get the extraction type string for this config."""
        return "swarm"
