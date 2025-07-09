import os
import random
from typing import TYPE_CHECKING, Dict, List, Optional, Type

from loguru import logger

from ardcore.data.dataset_item import DatasetItem
from ardcore.storage.file import S3StorageBackend, StorageManager

if TYPE_CHECKING:
    from ardcore.data.triplets import Triplets


class Dataset:
    """
    A collection of DatasetItem objects representing a dataset.

    The Dataset class provides a container for managing and accessing a collection
    of DatasetItem objects. It supports initialization with a list of items,
    loading from a local directory, and standard container operations.

    Attributes:
        items (List[DatasetItem]): The list of DatasetItem objects in the dataset.
    """

    def __init__(self, items: List[DatasetItem]) -> None:
        """
        Initialize a Dataset with a list of DatasetItem objects.

        Args:
            items (List[DatasetItem]): A list of DatasetItem objects to include in the dataset.
        """
        self.items = items

    @classmethod
    def from_local(
        cls,
        path: str,
        daset_item_type: Type[DatasetItem],
        storage_manager: StorageManager,
        kg_version: str,
        overwrite: bool = False,
    ) -> "Dataset":
        """
        Create a Dataset by loading DatasetItem objects from a local directory.

        This method scans the specified directory and attempts to create a DatasetItem
        for each item found. Items that cannot be loaded (e.g., due to missing metadata)
        are skipped with a warning.

        Args:
            path (str): The path to the directory containing dataset items.
            daset_item_type (DatasetItem): The type of `DatasetItem` to create for each item in the directory.
            storage_manager (StorageManager): The storage manager to use for accessing item data.
            kg_version (str): The knowledge graph version to check for. This is used to determine
                whether an item should be skipped if `overwrite` is `False`.
            overwrite (bool): If `True`, all items in the directory are loaded. If `False`,
                items that already have the specified `kg_version` are skipped.
                Defaults to `False`.

        Returns:
            Dataset: A new Dataset instance containing the successfully loaded items.
        """

        dir_names_with_items = cls._get_items_dir_names(
            path=path,
            storage_manager=storage_manager,
            kg_version=kg_version,
            overwrite=overwrite,
        )
        items = cls._load_items_from_dir_names(
            path=path,
            daset_item_type=daset_item_type,
            dir_names_with_items=dir_names_with_items,
        )

        # minimize the risk of processing the same item multiple times in case of issues
        random.shuffle(items)

        return cls(items)

    @classmethod
    def _get_items_dir_names(
        cls,
        path: str,
        storage_manager: StorageManager,
        kg_version: str,
        overwrite: bool,
    ) -> List[str]:
        existing_directories = os.listdir(path)
        new_directories = []

        if overwrite:
            return existing_directories

        # skip items if they have `kg_version` dir
        for directory in existing_directories:
            kg_versions = storage_manager.get_backend("local").list_directory(
                directory + "/kg"
            )["directories"]
            if kg_version and kg_version in kg_versions:
                logger.info(f"Item {directory} already exists, skipping")
            else:
                new_directories.append(directory)
                logger.info(f"Item {directory} does not exist, passing")

        return new_directories

    @classmethod
    def _load_items_from_dir_names(
        cls,
        path: str,
        daset_item_type: Type[DatasetItem],
        dir_names_with_items: List[str],
    ):
        items = []
        for item_dir_name in dir_names_with_items:
            # skip files
            if os.path.isfile(os.path.join(path, item_dir_name)):
                continue
            try:
                items.append(daset_item_type.from_local(item_dir_name))
            except FileNotFoundError:
                logger.warning(f"Item {item_dir_name} not found, skipping")
        return items

    @classmethod
    def from_s3(cls, path: str) -> "Dataset":
        """
        Create a Dataset by loading DatasetItem objects from a s3 directory.

        This method scans the specified directory and attempts to create a DatasetItem
        for each item found. Items that cannot be loaded (e.g., due to missing metadata)
        are skipped with a warning.

        Args:
            path (str): The path to the directory containing dataset items.

        Returns:
            Dataset: A new Dataset instance containing the successfully loaded items.
        """
        storage_manager = StorageManager()
        storage_manager.register_backend(name="s3", backend=S3StorageBackend(path))
        backend = storage_manager.get_backend(name="s3")

        items = []

        for item in backend.list_directory()["directories"]:
            # skip files
            try:
                new_item = DatasetItem.from_local(item, "s3")
                items.append(new_item)

            except FileNotFoundError:
                logger.warning(f"Item {item} not found, skipping")

        return cls(items)

    def get_triplets(
        self,
        kg_version: Optional[str] = None,
        build_graph: bool = False,
        skip_errors: bool = True,
    ) -> Dict[str, "Triplets"]:
        """
        Get triplets from all items in the dataset.

        This method retrieves triplets from each DatasetItem in the dataset.
        If an item doesn't have the specified KG version (or any KG version if none is specified),
        it will be skipped if skip_errors is True, otherwise an error will be raised.

        Args:
            kg_version: Optional name of the KG version (e.g., 'baseline_1').
                        If not provided, the latest version for each item will be used.
            build_graph: Whether to build the graph during initialization.
                        If False, the graph will be built on-demand when needed.
            skip_errors: Whether to skip items that don't have the specified KG version
                        or any KG version if none is specified.

        Returns:
            Dict[str, Triplets]: A dictionary mapping item IDs to their Triplets objects

        Raises:
            FileNotFoundError: If skip_errors is False and an item doesn't have the specified KG version
                              or any KG version if none is specified.
        """

        result = {}

        for item in self.items:
            try:
                triplets = item.get_triplets(
                    kg_version=kg_version, build_graph=build_graph
                )
                result[item.id] = triplets
            except (FileNotFoundError, ValueError) as e:
                if skip_errors:
                    logger.warning(
                        f"Could not get triplets for item {item.id}: {str(e)}"
                    )
                else:
                    raise

        return result

    def __repr__(self) -> str:
        """
        Return a string representation of the Dataset for debugging.

        Returns:
            str: A string representation of the Dataset, including its items.
        """
        return f"Dataset(items={self.items})"

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Dataset.

        Returns:
            str: A string representation of the Dataset, including its items.
        """
        return f"Dataset(items={self.items})"

    def __len__(self) -> int:
        """
        Return the number of items in the Dataset.

        Returns:
            int: The number of DatasetItem objects in the dataset.
        """
        return len(self.items)
