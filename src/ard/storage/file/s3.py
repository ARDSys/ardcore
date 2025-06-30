from pathlib import Path, PurePosixPath
from typing import BinaryIO, Dict, List, Optional, Union

import boto3

from ard.storage.file import StorageBackend


class S3StorageBackend(StorageBackend):
    """
    Amazon S3 storage backend for DatasetItems.
    """

    def __init__(self, base_dir: Union[str, Path]):
        """
        Initialize the S3 storage backend.

        Args:
            base_dir: S3 URI in the format 's3://bucket-name/optional/prefix'

        Set Fields Example:
            self.bucket = "beeard-papers"
            self.base_dir = "rheumatology_test/subfolder1/subfolder2"
            self.s3_base_dir = "s3://beeard-papers/rheumatology_test/subfolder1/subfolder2"

        """
        self.s3 = boto3.client("s3")

        # Parse S3 URI
        if str(base_dir).startswith("s3:"):
            if not str(base_dir).startswith("s3://"):
                p = PurePosixPath(base_dir)
                parts = p.parts
                base_dir = "s3://" + "/".join(parts[1:])
            else:
                pass
        else:
            raise ValueError("base_dir must be an S3 URI (s3://bucket/path)")

        parts = str(base_dir)[5:].split("/", 1)
        self.bucket = parts[0]
        # self.base_dir = base_dir
        self.base_dir = parts[1] if len(parts) > 1 else ""
        self.base_dir = self.base_dir.rstrip("/")
        self.s3_base_dir = f"s3://{self.bucket}/{self.base_dir}"

    def _get_item_dir(self, item_id: str, category: Optional[str] = None) -> str:
        """
        Get the directory for a specific item and category.

        Args:
            item_id: Unique identifier for the dataset item
            category: Optional category of the file
                     If None, returns the top-level item directory

        Returns:
            str: The S3 key prefix for the directory
        """
        if category is None:
            # Return the top-level item directory
            parts = [self.base_dir, str(item_id)] if self.base_dir else [str(item_id)]
        else:
            # Return the category subdirectory
            parts = (
                [self.base_dir, str(item_id), str(category)]
                if self.base_dir
                else [str(item_id), str(category)]
            )

        return "/".join(filter(None, parts))

    def _normalize_path(self, path: Union[str, Path]) -> str:
        """
        Normalize path separators to forward slashes for S3.

        Args:
            path: Path to normalize

        Returns:
            str: Normalized path with forward slashes
        """
        return str(Path(path)).replace("\\", "/")

    def save_file(
        self,
        item_id: str,
        file_path: Union[str, Path],
        data: Union[bytes, BinaryIO],
        category: Optional[str] = None,
    ) -> str:
        """Save a file to S3 storage."""
        item_dir = self._get_item_dir(item_id, category)
        key = f"{item_dir}/{self._normalize_path(file_path)}"

        if isinstance(data, bytes):
            from io import BytesIO

            data = BytesIO(data)

        try:
            self.s3.upload_fileobj(data, self.bucket, key)
        except Exception as e:
            raise RuntimeError(
                f"Failed to upload to s3://{self.bucket}/{key}: {str(e)}"
            )

        return f"s3://{self.bucket}/{key}"

    def get_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bytes:
        """Retrieve a file from S3 storage."""
        item_dir = self._get_item_dir(item_id, category)
        key = f"{item_dir}/{self._normalize_path(file_path)}"

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=key)
            content = response["Body"].read()

            # Handle case where moto or other mock returns raw HTTP response with chunked encoding
            if b"\r\n" in content:
                # Check for common chunked encoding patterns
                if (
                    content.startswith(b"d\r\n")
                    or content.startswith(b"10\r\n")
                    or content.startswith(b"f\r\n")
                ):
                    # Extract content between first \r\n and next \r\n
                    parts = content.split(b"\r\n", 2)
                    if len(parts) >= 2:
                        return parts[1]

            return content
        except self.s3.exceptions.NoSuchKey:
            raise FileNotFoundError(f"File not found: s3://{self.bucket}/{key}")

    def list_files(self, item_id: str, category: Optional[str] = None) -> List[str]:
        """List all files for a dataset item in S3 storage."""
        item_base_dir = self._get_item_dir(item_id)
        result = []
        paginator = self.s3.get_paginator("list_objects_v2")

        if category:
            # List files in a specific category
            prefix = f"{item_base_dir}/{category}/"
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        key = obj["Key"][len(prefix) :]
                        if key:  # Skip empty keys
                            result.append(key)
        else:
            # List files in all categories and at the top level
            prefix = f"{item_base_dir}/"
            for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
                if "Contents" in page:
                    for obj in page["Contents"]:
                        rel_path = obj["Key"][len(prefix) :]
                        if rel_path:  # Skip empty keys
                            result.append(rel_path)

        return result

    def delete_file(
        self, item_id: str, file_path: Union[str, Path], category: Optional[str] = None
    ) -> bool:
        """Delete a file from S3 storage."""
        item_dir = self._get_item_dir(item_id, category)
        key = f"{item_dir}/{self._normalize_path(file_path)}"

        try:
            # Check if object exists first
            self.s3.head_object(Bucket=self.bucket, Key=key)
            # If we get here, the object exists, so delete it
            self.s3.delete_object(Bucket=self.bucket, Key=key)
            return True
        except self.s3.exceptions.ClientError as e:
            error_code = e.response["Error"]["Code"]
            if error_code == "404":
                # File doesn't exist
                return False
            # Re-raise other errors
            raise

    def list_directory(self, in_prefix: str = "") -> Dict[str, List[str]]:
        """List immediate contents under S3 prefix."""
        files = set()
        directories = set()

        # Ensure prefix ends with / if not empty
        if in_prefix and not in_prefix.endswith("/"):
            in_prefix = in_prefix + "/"

        # Construct the full prefix properly
        if self.base_dir:
            if in_prefix:
                prefix = f"{self.base_dir}/{in_prefix}"
            else:
                prefix = f"{self.base_dir}/"
        else:
            prefix = in_prefix  # Could be empty string or a path

        paginator = self.s3.get_paginator("list_objects_v2")

        # Use delimiter to get directory-like listing
        for page in paginator.paginate(
            Bucket=self.bucket, Prefix=prefix, Delimiter="/"
        ):
            # Handle files (objects at this level)
            if "Contents" in page:
                for obj in page["Contents"]:
                    key = obj["Key"]
                    # Remove prefix to get just the filename
                    name = key[len(prefix) :]
                    if name:  # Skip the directory marker itself
                        files.add(name)

            # Handle directories (common prefixes)
            if "CommonPrefixes" in page:
                for prefix_obj in page["CommonPrefixes"]:
                    prefix_name = prefix_obj["Prefix"]
                    # Remove base prefix and trailing slash
                    name = prefix_name[len(prefix) :].rstrip("/")
                    if name:
                        directories.add(name)

        return {"files": sorted(list(files)), "directories": sorted(list(directories))}
