import os
import warnings
from pathlib import Path
from typing import Dict, Optional

from ard.storage.file import LocalStorageBackend, S3StorageBackend, StorageBackend


class StorageManager:
    """
    Manages storage backends for the application.
    Provides a unified interface for accessing different storage backends.
    """

    _instance = None
    _backends: Dict[str, StorageBackend] = {}
    _default_backend_name = "local"

    def __new__(
        cls,
        storage_type: Optional[str] = None,
        storage_path: Optional[Path] = None,
        storage_name: Optional[str] = None,
    ):
        """Singleton pattern to ensure only one instance of StorageManager exists."""
        if cls._instance is None:
            cls._instance = super(StorageManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(
        self,
        storage_type: Optional[str] = None,
        storage_path: Optional[Path] = None,
        storage_name: Optional[str] = None,
    ):
        """Initialize the storage manager if not already initialized.

        Args:
            storage_type: Optional type of storage ('local' or 's3')
            storage_path: Optional path for storage
        """
        if not getattr(self, "_initialized", False):
            self._initialized = True

            if storage_type and storage_path:
                self.add_backend(storage_type, storage_path, storage_name)

            else:
                # Set up default backends if no specific backend is provided
                self._setup_default_backends()
        elif storage_type and storage_path and storage_name:
            warnings.warn(
                "StorageManager already initialized",
                RuntimeWarning,
            )
            self.add_backend(storage_type, storage_path, storage_name)

    def add_backend(
        self, storage_type: str, storage_path: Path, storage_name: Optional[str] = None
    ):
        """Add a storage backend to the manager."""
        if storage_name is None:
            storage_name = storage_type

        if storage_type == "s3":
            self.register_backend(storage_name, S3StorageBackend(storage_path))
        elif storage_type == "local":
            self.register_backend(storage_name, LocalStorageBackend(storage_path))

    def _setup_default_backends(self):
        """Set up the default storage backends."""
        # Set up local storage backend
        data_dir = os.environ.get("ARD_DATA_DIR", None)
        if data_dir is None:
            # Default to a directory in the user's home directory
            data_dir = Path.home() / ".ard" / "data"

        self.register_backend("local", LocalStorageBackend(data_dir))

    def register_backend(self, name: str, backend: StorageBackend):
        """
        Register a storage backend with the manager.

        Args:
            name: Name to identify the backend
            backend: The storage backend instance
        """
        self._backends[name] = backend

    def get_backend(self, name: Optional[str] = None) -> StorageBackend:
        """
        Get a storage backend by name.

        Args:
            name: Name of the backend to get. If None, returns the default backend.

        Returns:
            StorageBackend: The requested storage backend

        Raises:
            ValueError: If the requested backend is not registered
        """
        if name is None:
            name = self._default_backend_name

        if name not in self._backends:
            raise ValueError(f"Storage backend '{name}' is not registered")

        return self._backends[name]

    def set_default_backend(self, name: str):
        """
        Set the default storage backend.

        Args:
            name: Name of the backend to set as default

        Raises:
            ValueError: If the requested backend is not registered
        """
        if name not in self._backends:
            raise ValueError(f"Storage backend '{name}' is not registered")

        self._default_backend_name = name
