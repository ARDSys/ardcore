from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union


class StorageBackend(ABC):
    """
    Abstract base class for storage backends.
    Implementations should handle storing and retrieving files for DatasetItems.
    """

    @abstractmethod
    def save_file(
        self,
        item_id: str,
        file_path: Union[str, Path],
        data: Union[bytes, BinaryIO],
        category: Optional[str] = None,
    ) -> str:
        """
        Save a file for a dataset item.

        Args:
            item_id: Unique identifier for the dataset item
            file_path: Path where the file should be stored (relative to the item's directory)
            data: File content as bytes or file-like object
            category: Optional category of the file (e.g., 'raw', 'processed', 'extracted')
                     If None, the file will be stored at the top level of the item directory

        Returns:
            str: The full path where the file was saved
        """
        pass

    @abstractmethod
    def get_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bytes:
        """
        Retrieve a file for a dataset item.

        Args:
            item_id: Unique identifier for the dataset item
            file_path: Path of the file to retrieve (relative to the item's directory)
            category: Optional category of the file (e.g., 'raw', 'processed', 'extracted')
                     If None, the file will be retrieved from the top level of the item directory

        Returns:
            bytes: The file content
        """
        pass

    @abstractmethod
    def list_files(self, item_id: str, category: Optional[str] = None) -> List[str]:
        """
        List all files for a dataset item.

        Args:
            item_id: Unique identifier for the dataset item
            category: Optional category to filter by
                     If None, all files will be listed (including top-level files)

        Returns:
            List[str]: List of file paths
        """
        pass

    @abstractmethod
    def delete_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bool:
        """
        Delete a file for a dataset item.

        Args:
            item_id: Unique identifier for the dataset item
            file_path: Path of the file to delete (relative to the item's directory)
            category: Optional category of the file (e.g., 'raw', 'processed', 'extracted')
                     If None, the file will be deleted from the top level of the item directory

        Returns:
            bool: True if the file was deleted, False otherwise
        """
        pass

    @abstractmethod
    def list_directory(self, prefix: str = "") -> Dict[str, List[str]]:
        """
        List immediate contents under a specific prefix/directory.

        Args:
            prefix: Path prefix to list contents from (default: root)

        Returns:
            Dict with 'files' and 'directories' lists
        """
        pass
