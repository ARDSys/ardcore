import shutil
from pathlib import Path
from typing import BinaryIO, Dict, List, Optional, Union

from ardcore.storage.file import StorageBackend


class LocalStorageBackend(StorageBackend):
    """
    Local filesystem storage backend for DatasetItems.
    """

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the local storage backend.

        Args:
            base_dir: Base directory for storing all dataset items
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _get_item_dir(self, item_id: str, category: Optional[str] = None) -> Path:
        """
        Get the directory for a specific item and category.

        Args:
            item_id: Unique identifier for the dataset item
            category: Optional category of the file
                     If None, returns the top-level item directory

        Returns:
            Path: The directory path
        """
        if category is None:
            # Return the top-level item directory
            item_dir = self.base_dir / item_id
        else:
            # Return the category subdirectory
            item_dir = self.base_dir / item_id / category

        item_dir.mkdir(parents=True, exist_ok=True)
        return item_dir

    def save_file(
        self,
        item_id: str,
        file_path: Union[str, Path],
        data: Union[bytes, BinaryIO],
        category: Optional[str] = None,
    ) -> str:
        """Save a file to local storage."""
        item_dir = self._get_item_dir(item_id, category)
        file_path = Path(file_path)

        # Create parent directories if needed
        full_path = item_dir / file_path
        full_path.parent.mkdir(parents=True, exist_ok=True)

        # Write the file
        if isinstance(data, bytes):
            with open(full_path, "wb") as f:
                f.write(data)
        else:
            with open(full_path, "wb") as f:
                shutil.copyfileobj(data, f)

        return str(full_path)

    def get_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bytes:
        """Retrieve a file from local storage."""
        item_dir = self._get_item_dir(item_id, category)
        full_path = item_dir / Path(file_path)

        if not full_path.exists():
            raise FileNotFoundError(f"File not found: {full_path}")

        with open(full_path, "rb") as f:
            return f.read()

    def list_files(self, item_id: str, category: Optional[str] = None) -> List[str]:
        """List all files for a dataset item in local storage."""
        item_base_dir = self.base_dir / item_id

        if not item_base_dir.exists():
            return []

        result = []

        if category:
            # List files in a specific category
            category_dir = item_base_dir / category
            if category_dir.exists():
                for path in category_dir.glob("**/*"):
                    if path.is_file():
                        # Use Path.as_posix() to ensure consistent forward slashes
                        result.append(path.relative_to(category_dir).as_posix())
        else:
            # List files in all categories and at the top level

            # First, list top-level files
            for path in item_base_dir.glob("*"):
                if path.is_file():
                    result.append(path.name)

            # Then, list files in all category subdirectories
            for category_dir in item_base_dir.glob("*"):
                if category_dir.is_dir():
                    category_name = category_dir.name
                    for path in category_dir.glob("**/*"):
                        if path.is_file():
                            # Use Path.as_posix() to ensure consistent forward slashes
                            result.append(
                                f"{category_name}/{path.relative_to(category_dir).as_posix()}"
                            )

        return result

    def delete_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bool:
        """Delete a file from local storage."""
        item_dir = self._get_item_dir(item_id, category)
        full_path = item_dir / Path(file_path)

        if not full_path.exists():
            return False

        full_path.unlink()
        return True

    def list_directory(self, prefix: str = "") -> Dict[str, List[str]]:
        """List immediate contents in local storage directory."""
        base_path = self.base_dir / prefix

        if not base_path.exists():
            return {"files": [], "directories": []}

        files = []
        directories = []

        for path in base_path.iterdir():
            if path.is_file():
                files.append(path.name)
            elif path.is_dir():
                directories.append(path.name)

        return {"files": sorted(files), "directories": sorted(directories)}
