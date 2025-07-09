import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from ardcore.storage.file import LocalStorageBackend, StorageBackend, StorageManager


class MockStorageBackend(StorageBackend):
    """Mock storage backend for testing."""

    def __init__(self):
        self.saved_files = {}

    def save_file(self, item_id, file_path, data, category="raw"):
        key = (item_id, str(file_path), category)
        if isinstance(data, bytes):
            self.saved_files[key] = data
        else:
            # Read file-like object
            self.saved_files[key] = data.read()
        return f"mock://{item_id}/{category}/{file_path}"

    def get_file(self, item_id, file_path, category="raw"):
        key = (item_id, str(file_path), category)
        if key not in self.saved_files:
            raise FileNotFoundError(f"File not found: {key}")
        return self.saved_files[key]

    def list_files(self, item_id, category=None):
        files = []
        for i, path, cat in self.saved_files.keys():
            if i == item_id and (category is None or cat == category):
                files.append(path)
        return files

    def delete_file(self, item_id, file_path, category="raw"):
        key = (item_id, str(file_path), category)
        if key not in self.saved_files:
            return False
        del self.saved_files[key]
        return True

    def list_directory(self, item_id, category=None):
        files = []
        for i, path, cat in self.saved_files.keys():
            if i == item_id and (category is None or cat == category):
                files.append(path)
        return files


class TestStorageManager:
    """Tests for the StorageManager class."""

    def test_singleton_pattern(self, reset_storage_manager):
        """Test that StorageManager follows the singleton pattern."""
        manager1 = StorageManager()
        manager2 = StorageManager()

        assert manager1 is manager2

    def test_default_backend_setup(self, reset_storage_manager):
        """Test that the default backend is set up correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Set environment variable for data directory
            os.environ["ARD_DATA_DIR"] = temp_dir

            manager = StorageManager()
            backend = manager.get_backend()

            assert isinstance(backend, LocalStorageBackend)
            assert backend.base_dir == Path(temp_dir)

    def test_register_and_get_backend(self, reset_storage_manager):
        """Test registering and retrieving a backend."""
        manager = StorageManager()
        mock_backend = MockStorageBackend()

        # Register the mock backend
        manager.register_backend("mock", mock_backend)

        # Get the mock backend
        retrieved_backend = manager.get_backend("mock")
        assert retrieved_backend is mock_backend

    def test_get_nonexistent_backend(self, reset_storage_manager):
        """Test that getting a non-existent backend raises an error."""
        manager = StorageManager()

        with pytest.raises(ValueError):
            manager.get_backend("nonexistent")

    def test_set_default_backend(self, reset_storage_manager):
        """Test setting the default backend."""
        manager = StorageManager()
        mock_backend = MockStorageBackend()

        # Register the mock backend
        manager.register_backend("mock", mock_backend)

        # Set it as the default
        manager.set_default_backend("mock")

        # Get the default backend
        default_backend = manager.get_backend()
        assert default_backend is mock_backend

    def test_set_nonexistent_default_backend(self, reset_storage_manager):
        """Test that setting a non-existent default backend raises an error."""
        manager = StorageManager()

        with pytest.raises(ValueError):
            manager.set_default_backend("nonexistent")

    @patch.object(StorageManager, "_setup_default_backends")
    def test_initialization_once(self, mock_setup, reset_storage_manager):
        """Test that the initialization happens only once."""
        StorageManager()
        assert mock_setup.call_count == 1

        StorageManager()
        assert mock_setup.call_count == 1  # Still 1, not called again
