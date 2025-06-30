import json
import tempfile
from io import BytesIO
from pathlib import Path

import pytest

from ard.data.metadata import Metadata, MetadataType
from ard.data.research_paper import ResearchPaper
from ard.storage.file import LocalStorageBackend, StorageManager


class ConcreteDatasetItem(ResearchPaper):
    """Concrete implementation of DatasetItem for testing."""

    pass


class TestDatasetItemStorage:
    """Tests for the storage capabilities of DatasetItem."""

    @pytest.fixture
    def reset_storage_manager(self):
        """Reset the StorageManager singleton between tests."""
        # Reset the singleton instance
        StorageManager._instance = None
        StorageManager._backends = {}
        StorageManager._default_backend_name = "local"

        yield

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def dataset_item(self, storage_manager_with_local_backend, temp_dir):
        """Create a DatasetItem instance with a local storage backend."""
        manager, backend_name = storage_manager_with_local_backend

        # Create a dataset item that uses this backend
        metadata = Metadata(doi="10.1234/test.item")
        return ConcreteDatasetItem(metadata, storage_backend=backend_name)

    def test_id_property(self, dataset_item):
        """Test that the id property returns the metadata's id."""
        assert dataset_item.id == dataset_item.get_metadata().id

    def test_save_and_get_file(self, dataset_item):
        """Test saving and retrieving a file."""
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        saved_path = dataset_item.save_file(file_path, content, category)
        assert Path(saved_path).exists()

        # Get the file
        retrieved_content = dataset_item.get_file(file_path, category)
        assert retrieved_content == content

    def test_save_file_with_file_object(self, dataset_item):
        """Test saving a file using a file-like object."""
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Create a file-like object
        file_obj = BytesIO(content)

        # Save the file
        saved_path = dataset_item.save_file(file_path, file_obj, category)
        assert Path(saved_path).exists()

        # Get the file
        retrieved_content = dataset_item.get_file(file_path, category)
        assert retrieved_content == content

    def test_list_files(self, dataset_item):
        """Test listing files."""
        # Save some files
        dataset_item.save_file("file1.txt", b"Content 1", "raw")
        dataset_item.save_file("file2.txt", b"Content 2", "raw")
        dataset_item.save_file("report.pdf", b"PDF content", "processed")

        # List all files
        all_files = dataset_item.list_files()
        # 4 files: metadata.json, file1.txt, file2.txt, report.pdf
        assert len(all_files) == 4

        # List files by category
        raw_files = dataset_item.list_files("raw")
        assert len(raw_files) == 2
        assert "file1.txt" in raw_files
        assert "file2.txt" in raw_files

        processed_files = dataset_item.list_files("processed")
        assert len(processed_files) == 1
        assert "report.pdf" in processed_files

    def test_delete_file(self, dataset_item):
        """Test deleting a file."""
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        dataset_item.save_file(file_path, content, category)

        # Verify it exists
        assert dataset_item.get_file(file_path, category) == content

        # Delete the file
        result = dataset_item.delete_file(file_path, category)
        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            dataset_item.get_file(file_path, category)

    def test_default_storage_backend(self, reset_storage_manager, temp_dir):
        """Test that the default storage backend is used when none is specified."""
        # Set up the default storage backend
        backend = LocalStorageBackend(temp_dir)
        manager = StorageManager()
        manager.register_backend("local", backend)

        # Create a dataset item without specifying a backend
        metadata = Metadata(doi="10.1234/test.item")
        item = ConcreteDatasetItem(metadata)

        # Test that it uses the default backend
        file_path = "test.txt"
        content = b"Hello, world!"

        saved_path = item.save_file(file_path, content)
        assert Path(saved_path).exists()

        retrieved_content = item.get_file(file_path)
        assert retrieved_content == content

    def test_metadata_saved_on_init(
        self, dataset_item, storage_manager_with_local_backend
    ):
        """Test that metadata is saved to storage when a DatasetItem is created."""
        manager, backend_name = storage_manager_with_local_backend
        backend = manager.get_backend(backend_name)

        # Check that metadata.json exists at the top level
        metadata_json = backend.get_file(
            dataset_item.id, "metadata.json", category=None
        )
        assert metadata_json is not None

        # Parse the JSON and check the content
        metadata_dict = json.loads(metadata_json.decode("utf-8"))
        assert metadata_dict["doi"] == "10.1234/test.item"

    def test_update_metadata(self, dataset_item, storage_manager_with_local_backend):
        """Test updating metadata and verifying it's saved to storage."""
        manager, backend_name = storage_manager_with_local_backend
        backend = manager.get_backend(backend_name)

        # Update the metadata
        new_metadata = Metadata(
            doi="10.1234/test.item",
            title="Updated Title",
            authors=["New Author"],
            type=MetadataType.PAPER,
        )
        dataset_item.update_metadata(new_metadata)

        # Check that the updated metadata is saved at the top level
        metadata_json = backend.get_file(
            dataset_item.id, "metadata.json", category=None
        )
        metadata_dict = json.loads(metadata_json.decode("utf-8"))
        print(metadata_dict)

        assert metadata_dict["title"] == "Updated Title"
        assert metadata_dict["authors"] == ["New Author"]
        assert metadata_dict["_internal"]["type"] == MetadataType.PAPER.value

    def test_from_local(self, storage_manager_with_local_backend, temp_dir):
        """Test creating a DatasetItem from local storage."""
        manager, backend_name = storage_manager_with_local_backend

        # First create and save an item
        metadata = Metadata(
            doi="10.1234/test.item",
            title="Test Item",
            authors=["Test Author"],
            type=MetadataType.PAPER,
        )
        original_item = ConcreteDatasetItem(metadata, storage_backend=backend_name)
        item_id = original_item.id

        # Save a file to verify it can be accessed later
        original_item.save_file("test.txt", b"Test content", "raw")

        # Now load it using from_local
        loaded_item = ConcreteDatasetItem.from_local(
            item_id, storage_backend=backend_name
        )

        # Verify the metadata was loaded correctly
        loaded_metadata = loaded_item.get_metadata()
        assert loaded_metadata.doi == "10.1234/test.item"
        assert loaded_metadata.title == "Test Item"
        assert loaded_metadata.authors == ["Test Author"]
        assert loaded_metadata.type == MetadataType.PAPER

        # Verify we can access the file
        content = loaded_item.get_file("test.txt", "raw")
        assert content == b"Test content"
