import os
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from ard.storage.file import LocalStorageBackend, S3StorageBackend


class TestLocalStorageBackend:
    """Tests for the LocalStorageBackend class."""

    def test_init(self, temp_dir):
        """Test that the backend initializes correctly."""
        backend = LocalStorageBackend(temp_dir)
        assert backend.base_dir == Path(temp_dir)
        assert backend.base_dir.exists()

    def test_save_and_get_file(self, local_storage_backend):
        """Test saving and retrieving a file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        saved_path = local_storage_backend.save_file(
            item_id, file_path, content, category
        )
        assert Path(saved_path).exists()
        assert Path(saved_path).name == file_path

        # Get the file
        retrieved_content = local_storage_backend.get_file(item_id, file_path, category)
        assert retrieved_content == content

    def test_save_and_get_file_with_subdirectories(self, local_storage_backend):
        """Test saving and retrieving a file in a subdirectory."""
        item_id = "test-item-123"
        file_path = os.path.join("subdir", "nested", "test.txt")
        content = b"Hello, world!"
        category = "processed"

        # Save the file
        saved_path = local_storage_backend.save_file(
            item_id, file_path, content, category
        )
        assert Path(saved_path).exists()
        assert str(Path(saved_path)).endswith(file_path)

        # Get the file
        retrieved_content = local_storage_backend.get_file(item_id, file_path, category)
        assert retrieved_content == content

    def test_list_files(self, local_storage_backend):
        """Test listing files."""
        item_id = "test-item-123"

        # Save some files
        local_storage_backend.save_file(item_id, "file1.txt", b"Content 1", "raw")
        local_storage_backend.save_file(item_id, "file2.txt", b"Content 2", "raw")
        subdir_file = os.path.join("subdir", "file3.txt")
        local_storage_backend.save_file(item_id, subdir_file, b"Content 3", "raw")
        local_storage_backend.save_file(
            item_id, "report.pdf", b"PDF content", "processed"
        )

        # List all files
        all_files = local_storage_backend.list_files(item_id)
        assert len(all_files) == 4

        # List files by category
        raw_files = local_storage_backend.list_files(item_id, "raw")
        assert len(raw_files) == 3
        assert "file1.txt" in raw_files
        assert "file2.txt" in raw_files
        assert any(
            os.path.normpath(f) == os.path.normpath(subdir_file) for f in raw_files
        )

        processed_files = local_storage_backend.list_files(item_id, "processed")
        assert len(processed_files) == 1
        assert "report.pdf" in processed_files

    def test_delete_file(self, local_storage_backend):
        """Test deleting a file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        local_storage_backend.save_file(item_id, file_path, content, category)

        # Verify it exists
        assert local_storage_backend.get_file(item_id, file_path, category) == content

        # Delete the file
        result = local_storage_backend.delete_file(item_id, file_path, category)
        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            local_storage_backend.get_file(item_id, file_path, category)

        # Try to delete a non-existent file
        result = local_storage_backend.delete_file(item_id, "nonexistent.txt", category)
        assert result is False

    def test_file_overwrite(self, local_storage_backend):
        """Test that saving a file with the same path overwrites the existing file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content1 = b"Original content"
        content2 = b"Updated content"
        category = "raw"

        # Save the file with original content
        local_storage_backend.save_file(item_id, file_path, content1, category)
        assert local_storage_backend.get_file(item_id, file_path, category) == content1

        # Save the file with updated content
        local_storage_backend.save_file(item_id, file_path, content2, category)
        assert local_storage_backend.get_file(item_id, file_path, category) == content2

    def test_top_level_files(self, local_storage_backend):
        """Test saving and retrieving files at the top level (no category)."""
        item_id = "test-item-123"
        file_path = "metadata.json"
        content = b'{"key": "value"}'

        # Save the file at the top level
        saved_path = local_storage_backend.save_file(
            item_id, file_path, content, category=None
        )
        assert Path(saved_path).exists()

        # Get the file from the top level
        retrieved_content = local_storage_backend.get_file(
            item_id, file_path, category=None
        )
        assert retrieved_content == content

        # List all files including top level
        all_files = local_storage_backend.list_files(item_id)
        assert file_path in all_files

        # Delete the file from the top level
        result = local_storage_backend.delete_file(item_id, file_path, category=None)
        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            local_storage_backend.get_file(item_id, file_path, category=None)


@pytest.fixture
def s3_storage_backend():
    """Fixture that provides a mocked S3 storage backend for testing."""
    with mock_aws():
        # Create a mock S3 bucket
        conn = boto3.resource("s3", region_name="us-east-1")
        conn.create_bucket(Bucket="test-bucket")

        # Create and return the S3 backend
        backend = S3StorageBackend("s3://test-bucket")
        yield backend


class TestS3StorageBackend:
    """Tests for the S3StorageBackend class."""

    def test_init(self):
        """Test that the backend initializes correctly."""
        with mock_aws():
            # Create a mock S3 bucket
            conn = boto3.resource("s3", region_name="us-east-1")
            conn.create_bucket(Bucket="test-bucket")

            # Test initialization
            backend = S3StorageBackend("s3://test-bucket")
            assert backend.bucket == "test-bucket"
            assert backend.base_dir == ""

            # Test with prefix
            backend = S3StorageBackend("s3://test-bucket/prefix")
            assert backend.bucket == "test-bucket"
            assert backend.base_dir == "prefix"

    def test_save_and_get_file(self, s3_storage_backend):
        """Test saving and retrieving a file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        saved_path = s3_storage_backend.save_file(item_id, file_path, content, category)
        assert saved_path.startswith("s3://")

        # Get the file
        retrieved_content = s3_storage_backend.get_file(item_id, file_path, category)
        assert retrieved_content == content

    def test_save_and_get_file_with_subdirectories(self, s3_storage_backend):
        """Test saving and retrieving a file in a subdirectory."""
        item_id = "test-item-123"
        file_path = os.path.join("subdir", "nested", "test.txt")
        content = b"Hello, world!"
        category = "processed"

        # Save the file
        saved_path = s3_storage_backend.save_file(item_id, file_path, content, category)
        assert saved_path.startswith("s3://")

        # Get the file
        retrieved_content = s3_storage_backend.get_file(item_id, file_path, category)
        assert retrieved_content == content

    def test_list_files(self, s3_storage_backend):
        """Test listing files."""
        item_id = "test-item-123"

        # Save some files
        s3_storage_backend.save_file(item_id, "file1.txt", b"Content 1", "raw")
        s3_storage_backend.save_file(item_id, "file2.txt", b"Content 2", "raw")
        subdir_file = os.path.join("subdir", "file3.txt")
        s3_storage_backend.save_file(item_id, subdir_file, b"Content 3", "raw")
        s3_storage_backend.save_file(item_id, "report.pdf", b"PDF content", "processed")

        # List all files
        all_files = s3_storage_backend.list_files(item_id)
        assert len(all_files) == 4

        # List files by category
        raw_files = s3_storage_backend.list_files(item_id, "raw")
        assert len(raw_files) == 3
        assert "file1.txt" in raw_files
        assert "file2.txt" in raw_files
        assert any(
            os.path.normpath(f) == os.path.normpath(subdir_file) for f in raw_files
        )

        processed_files = s3_storage_backend.list_files(item_id, "processed")
        assert len(processed_files) == 1
        assert "report.pdf" in processed_files

    def test_delete_file(self, s3_storage_backend):
        """Test deleting a file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content = b"Hello, world!"
        category = "raw"

        # Save the file
        s3_storage_backend.save_file(item_id, file_path, content, category)

        # Verify it exists
        assert s3_storage_backend.get_file(item_id, file_path, category) == content

        # Delete the file
        result = s3_storage_backend.delete_file(item_id, file_path, category)
        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            s3_storage_backend.get_file(item_id, file_path, category)

        # Try to delete a non-existent file
        result = s3_storage_backend.delete_file(item_id, "nonexistent.txt", category)
        assert result is False

    def test_file_overwrite(self, s3_storage_backend):
        """Test that saving a file with the same path overwrites the existing file."""
        item_id = "test-item-123"
        file_path = "test.txt"
        content1 = b"Original content"
        content2 = b"Updated content"
        category = "raw"

        # Save the file with original content
        s3_storage_backend.save_file(item_id, file_path, content1, category)
        assert s3_storage_backend.get_file(item_id, file_path, category) == content1

        # Save the file with updated content
        s3_storage_backend.save_file(item_id, file_path, content2, category)
        assert s3_storage_backend.get_file(item_id, file_path, category) == content2

    def test_top_level_files(self, s3_storage_backend):
        """Test saving and retrieving files at the top level (no category)."""
        item_id = "test-item-123"
        file_path = "metadata.json"
        content = b'{"key": "value"}'

        # Save the file at the top level
        saved_path = s3_storage_backend.save_file(
            item_id, file_path, content, category=None
        )
        assert saved_path.startswith("s3://")

        # Get the file from the top level
        retrieved_content = s3_storage_backend.get_file(
            item_id, file_path, category=None
        )
        assert retrieved_content == content

        # List all files including top level
        all_files = s3_storage_backend.list_files(item_id)
        assert file_path in all_files

        # Delete the file from the top level
        result = s3_storage_backend.delete_file(item_id, file_path, category=None)
        assert result is True

        # Verify it's gone
        with pytest.raises(FileNotFoundError):
            s3_storage_backend.get_file(item_id, file_path, category=None)

    def test_list_directory(self, s3_storage_backend):
        """Test listing files in a directory."""
        item_id = "test-item-123"

        # Save files in different directories
        s3_storage_backend.save_file(item_id, "file1.txt", b"Content 1", "raw")
        s3_storage_backend.save_file(item_id, "file2.txt", b"Content 2", "raw")
        s3_storage_backend.save_file(item_id, "subdir1/file3.txt", b"Content 3", "raw")
        s3_storage_backend.save_file(item_id, "subdir1/file4.txt", b"Content 4", "raw")
        s3_storage_backend.save_file(
            item_id, "subdir1/nested/file5.txt", b"Content 5", "raw"
        )
        s3_storage_backend.save_file(item_id, "subdir2/file6.txt", b"Content 6", "raw")

        # List root directory for this item/category
        root_prefix = f"{item_id}/raw"
        root_listing = s3_storage_backend.list_directory(root_prefix)
        assert "file1.txt" in root_listing["files"]
        assert "file2.txt" in root_listing["files"]
        assert "subdir1" in root_listing["directories"]
        assert "subdir2" in root_listing["directories"]

        # List subdirectory
        subdir_prefix = f"{item_id}/raw/subdir1"
        subdir_listing = s3_storage_backend.list_directory(subdir_prefix)
        assert "file3.txt" in subdir_listing["files"]
        assert "file4.txt" in subdir_listing["files"]
        assert "nested" in subdir_listing["directories"]

        # List nested subdirectory
        nested_prefix = f"{item_id}/raw/subdir1/nested"
        nested_listing = s3_storage_backend.list_directory(nested_prefix)
        assert "file5.txt" in nested_listing["files"]
