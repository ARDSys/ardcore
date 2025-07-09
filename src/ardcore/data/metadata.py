import hashlib
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, ForwardRef, List, Optional

# Forward reference for type hints with self-referencing types
MetadataRef = ForwardRef("Metadata")


class MetadataType(Enum):
    """Enum for different types of metadata."""

    PAPER = "paper"


@dataclass
class Metadata:
    """
    Dataclass for storing metadata about a scientific paper or document.

    Attributes:
        doi (str, optional): Digital Object Identifier
        pm_id (int, optional): PubMed ID
        pmc_id (str, optional): PubMed Central ID
        gs_id (str, optional): Google Scholar ID
        extra_id (Optional[str]): Extra identifier
        category (str, optional): Category of the document
        title (str, optional): Title of the document
        abstract (str, optional): Abstract of the document
        gs_meta (str, optional): Google Scholar metadata as a string (usually JSON)
        scraped_date (str, optional): Date when the document was scraped
        id (str, optional): Unique identifier for the document
        authors (Optional[List[str]]): List of authors
        citations (Optional[List[Metadata]]): List of papers this document cites
        cited_by (Optional[List[Metadata]]): List of papers that cite this document
        type (MetadataType): Type of the item (paper, image, etc.)
        additional_metadata (Dict[str, Any], optional): Any additional metadata
    """

    doi: Optional[str] = None
    pm_id: Optional[int] = None
    pmc_id: Optional[str] = None
    gs_id: Optional[str] = None
    extra_id: Optional[str] = None
    category: Optional[str] = None
    title: Optional[str] = None
    abstract: Optional[str] = None
    gs_meta: Optional[str] = None
    scraped_date: Optional[str] = None
    id: Optional[str] = None
    authors: Optional[List[str]] = None
    citations: Optional[List[MetadataRef]] = field(default_factory=list)  # type: ignore
    cited_by: Optional[List[MetadataRef]] = field(default_factory=list)  # type: ignore
    type: MetadataType = MetadataType.PAPER
    additional_metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """
        Validate that at least one identifier is present and generate ID if not provided.
        """

        # Ensure at least one identifier is present
        if not any(
            [self.doi, self.pm_id, self.pmc_id, self.gs_id, self.extra_id, self.id]
        ):
            raise ValueError(
                "At least one identifier (doi, pm_id, pmc_id, gs_id, extra_id, or id) must be provided"
            )

        # Generate ID if not provided
        if self.id is None:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """
        Generate a unique hash-based identifier from all available identifiers.

        Returns:
            str: A hexadecimal hash string that uniquely identifies this metadata
        """
        # Collect all available identifiers
        identifiers = []

        for identifier_name in ["doi", "pm_id", "pmc_id", "gs_id", "extra_id"]:
            identifier_value = getattr(self, identifier_name, None)
            if identifier_value:
                identifiers.append(f"{identifier_name}:{str(identifier_value)}")

        # If we have any identifiers, generate a hash from all of them
        if identifiers:
            # Join all identifiers with a separator and create a hash
            combined_identifier = "|".join(identifiers)
            return hashlib.sha256(combined_identifier.encode()).hexdigest()

        # This should never happen due to validation in __post_init__
        raise ValueError("Cannot generate ID: no identifiers available")

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert metadata to a dictionary for serialization.

        Returns:
            Dict[str, Any]: Dictionary representation of the metadata
        """
        result = {
            "doi": self.doi,
            "pm_id": self.pm_id,
            "pmc_id": self.pmc_id,
            "gs_id": self.gs_id,
            "extra_id": self.extra_id,
            "category": self.category,
            "title": self.title,
            "abstract": self.abstract,
            "gs_meta": self.gs_meta,
            "scraped_date": self.scraped_date,
            "id": self.id,
            "authors": self.authors,
            "_internal": {
                "type": self.type.value,
            },
        }

        # Add citations and cited_by if they exist
        if self.citations:
            result["citations"] = [citation.to_dict() for citation in self.citations]

        if self.cited_by:
            result["cited_by"] = [citing.to_dict() for citing in self.cited_by]

        # Remove None values
        result = {k: v for k, v in result.items() if v is not None}

        # Add additional_metadata to the result if it's not empty
        if self.additional_metadata:
            result["_internal"]["additional_metadata"] = self.additional_metadata

        return result

    def to_json(self) -> str:
        """
        Convert metadata to a JSON string.

        Returns:
            str: JSON representation of the metadata
        """
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Metadata":
        """
        Create a Metadata object from a dictionary.

        Args:
            data: Dictionary containing metadata fields

        Returns:
            Metadata: A new Metadata object
        """
        # Convert all keys to lowercase
        data_copy = {k.lower(): v for k, v in data.items()}

        # Make a copy of the data to avoid modifying the original
        metadata_dict = data_copy.copy()

        # Extract internal fields if present
        internal_data = metadata_dict.pop("_internal", {})

        # Handle the type field - default to PAPER if not present
        metadata_type = None
        if "type" in internal_data:
            metadata_type = MetadataType(internal_data["type"])

        # Handle additional metadata
        additional_metadata = internal_data.get("additional_metadata", {})

        # Extract citations and cited_by for separate processing
        citations_data = metadata_dict.pop("citations", [])
        cited_by_data = metadata_dict.pop("cited_by", [])

        # Remove ID field if it's None (will be generated in post_init)
        if "id" in metadata_dict and metadata_dict["id"] is None:
            del metadata_dict["id"]

        # Handle special case mappings
        key_mappings = {
            "pm_id": ["pmid"],
            "pmc_id": ["pmcid"],
            "gs_id": ["gsid"],
            "gs_meta": ["gsmeta", "gs_metadata"],
            "scraped_date": ["scrapeddate", "scraped_at", "date_scraped"],
        }

        # Apply mappings for special cases
        for target_key, source_keys in key_mappings.items():
            for source_key in source_keys:
                if source_key in metadata_dict and target_key not in metadata_dict:
                    metadata_dict[target_key] = metadata_dict.pop(source_key)

        # Create the metadata object with type and additional_metadata
        # Only pass type if it's not the default to avoid redundancy
        kwargs = {**metadata_dict, "additional_metadata": additional_metadata}
        if metadata_type is not None:
            kwargs["type"] = metadata_type

        # Create the metadata object
        metadata_obj = cls(**kwargs)

        # Process citations and cited_by after the object is created
        if citations_data:
            metadata_obj.citations = [
                cls.from_dict(citation) for citation in citations_data
            ]

        if cited_by_data:
            metadata_obj.cited_by = [cls.from_dict(citing) for citing in cited_by_data]

        return metadata_obj

    @classmethod
    def from_json(cls, json_str: str) -> "Metadata":
        """
        Create a Metadata object from a JSON string.

        Args:
            json_str: JSON string containing metadata

        Returns:
            Metadata: A new Metadata object
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    def __eq__(self, other):
        """
        Check if two metadata objects are equal based on their identifiers.

        Raises:
            ValueError: If identifiers are inconsistent (some match, some don't)
        """
        if not isinstance(other, Metadata):
            return False

        # Track which identifiers match and which don't
        matching_ids = []
        non_matching_ids = []

        # Check all identifiers
        for attr in ["doi", "pm_id", "pmc_id", "gs_id", "extra_id", "id"]:
            self_val = getattr(self, attr, None)
            other_val = getattr(other, attr, None)

            # Only compare if both objects have this identifier
            if self_val is not None and other_val is not None:
                if self_val == other_val:
                    matching_ids.append(attr)
                else:
                    non_matching_ids.append(attr)

        # If we have both matching and non-matching identifiers, that's inconsistent
        if matching_ids and non_matching_ids:
            raise ValueError(
                f"Inconsistent identifiers: {', '.join(matching_ids)} match but "
                f"{', '.join(non_matching_ids)} don't match"
            )

        # Return True if any identifiers match
        return len(matching_ids) > 0


# Resolve the forward reference
Metadata.__annotations__["citations"] = Optional[List[Metadata]]
Metadata.__annotations__["cited_by"] = Optional[List[Metadata]]
