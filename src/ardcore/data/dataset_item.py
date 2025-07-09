import datetime
import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Set, Type, TypeVar, Union

from ardcore.data.metadata import Metadata
from ardcore.data.triplets import Triplets
from ardcore.data.types import KGGenerator
from ardcore.storage.file import (
    LocalStorageBackend,
    S3StorageBackend,
    StorageBackend,
    StorageManager,
)

T = TypeVar("T", bound="DatasetItem")


class DataCategory(Enum):
    """Standard categories for dataset item files."""

    RAW = "raw"
    PROCESSED = "processed"
    KG = "kg"

    @classmethod
    def values(cls) -> Set[str]:
        """Get all category values as strings."""
        return {category.value for category in cls}


class DatasetItem(ABC):
    """
    Abstract base class for any item in our dataset.
    Extend this class for specific dataset items (e.g., ResearchPaper).
    """

    def __init__(self, metadata: Metadata, storage_backend: Optional[str] = None):
        self._metadata = metadata
        self._storage_manager = StorageManager()
        self._storage_backend_name = storage_backend

        # Initialize the directory structure
        self._initialize_directory_structure()

        # Save metadata to storage
        self._save_metadata()

    def _initialize_directory_structure(self):
        """Initialize the standard directory structure for this item."""
        backend = self._get_storage_backend()

        # Create standard directories
        for category in DataCategory:
            # We don't need to store anything here, just ensure the directory exists
            # The _get_item_dir method in LocalStorageBackend will create the directory
            backend._get_item_dir(self.id, category.value)

        # For KG category, we don't create subdirectories yet as they will be created as needed

    def get_metadata(self) -> Metadata:
        """Returns the structured metadata for this item."""
        return self._metadata

    def update_metadata(self, metadata: Metadata) -> None:
        """
        Update the metadata for this item and save it to storage.

        Args:
            metadata: New metadata for this item
        """
        self._metadata = metadata
        self._save_metadata()

    @classmethod
    def from_metadata(
        cls: Type[T], metadata: Metadata, storage_backend: Optional[str] = None
    ) -> T:
        """Factory method to create a DatasetItem from a Metadata object."""
        return cls(metadata, storage_backend)

    @classmethod
    def from_doi(cls: Type[T], doi: str) -> T:
        """Factory method to create a DatasetItem from a DOI."""
        metadata = Metadata(doi=doi)
        return cls(metadata)

    @classmethod
    def from_pm_id(cls: Type[T], pm_id: str) -> T:
        """Factory method to create a DatasetItem from a PMID."""
        metadata = Metadata(pm_id=pm_id)
        return cls(metadata)

    @classmethod
    def from_local(
        cls: Type[T], item_id: str, storage_backend: Optional[str] = "local"
    ) -> T:
        """
        Factory method to create a DatasetItem from a local storage.

        Args:
            item_id: The unique identifier for the item
            storage_backend: Optional name of the storage backend to use

        Returns:
            T: A new instance of the class

        Raises:
            FileNotFoundError: If the metadata file doesn't exist
            ValueError: If the metadata is invalid
        """
        # Get the storage manager and backend
        storage_manager = StorageManager()
        backend = storage_manager.get_backend(storage_backend)

        try:
            # Load metadata from storage
            metadata_json = backend.get_file(item_id, "metadata.json")
            metadata = Metadata.from_json(metadata_json.decode("utf-8"))

            # if metadata.type == MetadataType.PAPER:
            #     return ResearchPaper.from_metadata(metadata, storage_backend)
            # else:
            #     raise ValueError(f"Unsupported metadata type: {metadata.type}")

            # Create and return the item
            return cls(metadata, storage_backend)
        except FileNotFoundError:
            raise FileNotFoundError(f"No metadata found for item with ID: {item_id}")
        except Exception as e:
            raise ValueError(
                f"Error loading metadata for item with ID {item_id}: {str(e)}"
            )

    @classmethod
    def from_s3(cls: Type[T], item_id: str, storage_backend: Optional[str] = "s3") -> T:
        """
        Factory method to create a DatasetItem from a s3 storage.

        Args:
            item_id: The unique identifier for the item
            storage_backend: Optional name of the storage backend to use

        Returns:
            T: A new instance of the class

        Raises:
            FileNotFoundError: If the metadata file doesn't exist
            ValueError: If the metadata is invalid
        """
        # Get the storage manager and backend
        storage_manager = StorageManager()
        backend = storage_manager.get_backend(storage_backend)

        try:
            # Load metadata from storage
            metadata_json = backend.get_file(item_id, "metadata.json")
            metadata = Metadata.from_json(metadata_json.decode("utf-8"))

            # if metadata.type == MetadataType.PAPER:
            #     return ResearchPaper.from_metadata(metadata, storage_backend)
            # else:
            #     raise ValueError(f"Unsupported metadata type: {metadata.type}")

            # Create and return the item
            return cls(metadata, storage_backend)
        except FileNotFoundError:
            raise FileNotFoundError(f"No metadata found for item with ID: {item_id}")
        except Exception as e:
            raise ValueError(
                f"Error loading metadata for item with ID {item_id}: {str(e)}"
            )

    @property
    def id(self) -> str:
        """
        Get the unique identifier for this dataset item.
        Uses the metadata's id property.

        Returns:
            str: Unique identifier for this item
        """
        return self._metadata.id

    def _save_metadata(self) -> None:
        """Save metadata to storage."""
        metadata_json = self._metadata.to_json()
        self.save_file(
            "metadata.json",
            metadata_json.encode("utf-8"),
        )

    def _get_storage_backend(self) -> StorageBackend:
        """Get the storage backend for this item."""
        return self._storage_manager.get_backend(self._storage_backend_name)

    def save_file(
        self,
        file_path: Union[str, Path],
        data: Union[bytes, BinaryIO],
        category: Optional[str] = None,
    ) -> str:
        """
        Save a file associated with this dataset item.

        Args:
            file_path: Path where the file should be stored (relative to the item's directory)
            data: File content as bytes or file-like object
            category: Optional category of the file (e.g., 'raw', 'processed', 'kg')
                     If None, the file will be stored at the top level of the item directory

        Returns:
            str: The full path where the file was saved
        """
        # Validate category if provided
        if category is not None and category not in DataCategory.values():
            if not category.startswith(f"{DataCategory.KG.value}/"):
                raise ValueError(
                    f"Invalid category: {category}. Must be one of {DataCategory.values()} "
                    f"or a subdirectory of {DataCategory.KG.value}/"
                )

        backend = self._get_storage_backend()
        return backend.save_file(self.id, file_path, data, category)

    def get_file(
        self, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bytes:
        """
        Retrieve a file associated with this dataset item.

        Args:
            file_path: Path of the file to retrieve (relative to the item's directory)
            category: Optional category of the file (e.g., 'raw', 'processed', 'kg')
                     If None, the file will be retrieved from the top level of the item directory

        Returns:
            bytes: The file content
        """
        backend = self._get_storage_backend()
        return backend.get_file(self.id, file_path, category)

    def list_files(self, category: Optional[str] = None) -> List[str]:
        """
        List all files associated with this dataset item.

        Args:
            category: Optional category to filter by
                     If None, all files will be listed (including top-level files)

        Returns:
            List[str]: List of file paths
        """
        backend = self._get_storage_backend()
        return backend.list_files(self.id, category)

    def delete_file(
        self, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bool:
        """
        Delete a file associated with this dataset item.

        Args:
            file_path: Path of the file to delete (relative to the item's directory)
            category: Optional category of the file (e.g., 'raw', 'processed', 'kg')
                     If None, the file will be deleted from the top level of the item directory

        Returns:
            bool: True if the file was deleted, False otherwise
        """
        backend = self._get_storage_backend()
        return backend.delete_file(self.id, file_path, category)

    def create_kg_version(self, version_name: str) -> str:
        """
        Create a new knowledge graph version directory.

        Args:
            version_name: Name of the KG version (e.g., 'baseline_1')

        Returns:
            str: The full path to the created directory

        Raises:
            ValueError: If the version name is invalid
        """
        if not version_name or "/" in version_name or "\\" in version_name:
            raise ValueError(f"Invalid KG version name: {version_name}")

        kg_subdir = f"{DataCategory.KG.value}/{version_name}"

        # Create an empty config file to ensure the directory is created
        config = {"version": version_name, "created_at": str(datetime.datetime.now())}
        self.save_file(
            "config.json",
            json.dumps(config, indent=2).encode("utf-8"),
            category=kg_subdir,
        )

        return kg_subdir

    def list_kg_versions(self) -> List[str]:
        """
        List all knowledge graph versions available for this item.

        Returns:
            List[str]: List of KG version names
        """
        backend = self._get_storage_backend()
        versions = []

        if isinstance(backend, LocalStorageBackend):
            kg_dir = backend._get_item_dir(self.id, DataCategory.KG.value)

            if kg_dir.exists():
                for path in kg_dir.iterdir():
                    if path.is_dir():
                        versions.append(path.name)

        elif isinstance(backend, S3StorageBackend):
            for dir in backend.list_directory(f"{self.id}/{DataCategory.KG.value}/")[
                "directories"
            ]:
                versions.append(dir)

        return versions

    def save_kg_file(
        self,
        version_name: str,
        file_path: Union[str, Path],
        data: Union[bytes, BinaryIO],
    ) -> str:
        """
        Save a file in a specific knowledge graph version.

        Args:
            version_name: Name of the KG version (e.g., 'baseline_1')
            file_path: Path where the file should be stored (relative to the version directory)
            data: File content as bytes or file-like object

        Returns:
            str: The full path where the file was saved
        """
        if not version_name or "/" in version_name or "\\" in version_name:
            raise ValueError(f"Invalid KG version name: {version_name}")

        kg_subdir = f"{DataCategory.KG.value}/{version_name}"
        return self.save_file(file_path, data, category=kg_subdir)

    def get_kg_file(self, version_name: str, file_path: Union[str, Path]) -> bytes:
        """
        Retrieve a file from a specific knowledge graph version.

        Args:
            version_name: Name of the KG version (e.g., 'baseline_1')
            file_path: Path of the file to retrieve (relative to the version directory)

        Returns:
            bytes: The file content
        """
        if not version_name or "/" in version_name or "\\" in version_name:
            raise ValueError(f"Invalid KG version name: {version_name}")

        kg_subdir = f"{DataCategory.KG.value}/{version_name}"
        return self.get_file(file_path, category=kg_subdir)

    def list_kg_files(self, version_name: str) -> List[str]:
        """
        List all files in a specific knowledge graph version.

        Args:
            version_name: Name of the KG version (e.g., 'baseline_1')

        Returns:
            List[str]: List of file paths
        """
        if not version_name or "/" in version_name or "\\" in version_name:
            raise ValueError(f"Invalid KG version name: {version_name}")

        kg_subdir = f"{DataCategory.KG.value}/{version_name}"
        return self.list_files(category=kg_subdir)

    def get_triplets(
        self, kg_version: Optional[str] = None, build_graph: bool = False
    ) -> "Triplets":
        """
        Get the triplets from a knowledge graph version.

        Args:
            kg_version: Optional name of the KG version (e.g., 'baseline_1').
                        If not provided, the latest version in alphabetical order will be used.
            build_graph: Whether to build the graph during initialization.
                        If False, the graph will be built on-demand when needed.

        Returns:
            Triplets: A Triplets object containing the triplets from the specified KG version

        Raises:
            ValueError: If the specified kg_version doesn't exist
            FileNotFoundError: If there are no KG versions available or if the triplets file doesn't exist
        """
        # Get available KG versions
        versions = self.list_kg_versions()

        if not versions:
            raise FileNotFoundError(
                f"No knowledge graph versions found for item {self.id}"
            )

        # If kg_version is not specified, use the latest version in alphabetical order
        if kg_version is None:
            kg_version = sorted(versions)[-1]
        elif kg_version not in versions:
            raise ValueError(
                f"Knowledge graph version '{kg_version}' not found. Available versions: {versions}"
            )

        # Load triplets from the specified KG version
        return Triplets.from_dataset_item(self, kg_version, build_graph=build_graph)

    def __repr__(self) -> str:
        """Get the string representation for this item."""
        return f"DatasetItem(id={self.id})"

    # @abstractmethod
    # def pull_metadata(self) -> str:
    #     """Get the full metadata for this item."""
    #     pass

    # @abstractmethod
    # def pull_raw_data(self) -> None:
    #     """Get the raw data for this item."""
    #     pass

    # @abstractmethod
    # def process_data(self) -> None:
    #     """Process the data for this item."""
    #     pass

    @abstractmethod
    def generate_kg(
        self,
        config: Optional[Dict],
        kg_generator: Optional[KGGenerator] = None,
        kg_version: Optional[str] = None,
        overwrite: bool = False,
    ) -> "Triplets":
        """
        Generate triplets from the dataset item content.

        This method should handle dataset-specific concerns (caching, text retrieval,
        persistence, versioning) while delegating the core triplet generation logic
        to TripletsGenerationPipeline.

        Args:
            config: Optional configuration dictionary for triplet generation
            kg_generator: Optional triplet generator function (KGGenerator type)
            kg_version: Optional KG version name for storage (e.g., 'baseline_1')
            overwrite: Whether to overwrite existing KG version

        Returns:
            Triplets: Generated triplets with metadata and configuration
        """
        pass

    @abstractmethod
    def add_context_to_triplets(self, kg_version: str) -> "Triplets":
        """Add context to the triplets."""
        pass
