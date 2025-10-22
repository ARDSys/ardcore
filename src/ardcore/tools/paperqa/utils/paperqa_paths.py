"""Shared PaperQA path utilities."""

import os
from pathlib import Path


class PaperQAPaths:
    """Manages paths for PaperQA operations.

    This class centralizes all path logic for PaperQA operations, including
    S3 paths for papers and indexes, and local temporary directories.

    Args:
        corpus_name: Optional corpus name. If not provided, reads from PAPERQA_CORPUS_NAME env variable.

    Raises:
        ValueError: If corpus_name is not provided and PAPERQA_CORPUS_NAME is not set in environment.

    Example:
        >>> # Using environment variable
        >>> paths = PaperQAPaths()
        >>> paths.papers_s3
        's3://your-papers-bucket/my-corpus/'

        >>> # Using explicit corpus name
        >>> paths = PaperQAPaths(corpus_name="custom-corpus")
        >>> paths.papers_s3
        's3://your-papers-bucket/custom-corpus/'
    """

    def __init__(self, corpus_name: str | None = None):
        """Initialize PaperQAPaths with corpus name from parameter or environment.

        Args:
            corpus_name: Optional corpus name. If not provided, reads from PAPERQA_CORPUS_NAME env variable.

        Raises:
            ValueError: If corpus_name is not provided and PAPERQA_CORPUS_NAME is not set.
        """
        if corpus_name is not None:
            self.corpus_name = corpus_name
        else:
            corpus_name_env = os.getenv("PAPERQA_CORPUS_NAME")
            if not corpus_name_env:
                raise ValueError(
                    "corpus_name parameter not provided and PAPERQA_CORPUS_NAME not set in .env"
                )
            self.corpus_name = corpus_name_env

    @property
    def papers_bucket(self) -> str:
        """Get S3 bucket for raw papers.

        Returns:
            S3 bucket (e.g., 's3://your-papers-bucket')

        Raises:
            ValueError: If PAPERQA_PAPERS_BUCKET is not set or invalid
        """
        papers_bucket = os.getenv("PAPERQA_PAPERS_BUCKET")

        if not papers_bucket:
            raise ValueError(
                "PAPERQA_PAPERS_BUCKET environment variable is required. "
                "Set it in .env (e.g., PAPERQA_PAPERS_BUCKET=s3://your-papers-bucket)"
            )

        if not papers_bucket.startswith("s3://"):
            raise ValueError(
                f"PAPERQA_PAPERS_BUCKET must start with 's3://', got: {papers_bucket}"
            )

        return papers_bucket.rstrip("/")

    @property
    def index_bucket(self) -> str:
        """Get S3 bucket for index storage.

        Note: PaperQA creates subdirectories automatically with index name.

        Returns:
            S3 bucket (e.g., 's3://your-index-bucket')

        Raises:
            ValueError: If PAPERQA_INDEX_BUCKET is not set or invalid
        """
        index_bucket = os.getenv("PAPERQA_INDEX_BUCKET")

        if not index_bucket:
            raise ValueError(
                "PAPERQA_INDEX_BUCKET environment variable is required. "
                "Set it in .env (e.g., PAPERQA_INDEX_BUCKET=s3://your-index-bucket)"
            )

        if not index_bucket.startswith("s3://"):
            raise ValueError(
                f"PAPERQA_INDEX_BUCKET must start with 's3://', got: {index_bucket}"
            )

        return index_bucket.rstrip("/")

    @property
    def papers_s3(self) -> str:
        """Get S3 path for raw papers.

        Returns:
            S3 path: {PAPERQA_PAPERS_BUCKET}/{corpus_name}/
        """
        return f"{self.papers_bucket}/{self.corpus_name}/"

    @property
    def index_s3(self) -> str:
        """Get full S3 path for this corpus's index.

        Returns:
            S3 path: {PAPERQA_INDEX_BUCKET}/{corpus_name}/
        """
        return f"{self.index_bucket}/{self.corpus_name}/"

    @property
    def local_index_base(self) -> str:
        """Get local base directory for index storage.

        Note: PaperQA creates subdirectories automatically with index name.

        Returns:
            Cross-platform base path: ~/.paperqa
        """
        return str(Path.home() / ".paperqa")

    @property
    def local_index(self) -> Path:
        """Get local index directory for this corpus.

        Returns:
            Path to local index: ~/.paperqa/{corpus_name}
        """
        return Path(self.local_index_base) / self.corpus_name

    @property
    def build_temp_base(self) -> str:
        """Get local temp base directory for build operations.

        Note: PaperQA will create subdirectories (files.zip, docs/, index/) inside this path.
        Papers should be placed in {base}/papers/.

        Returns:
            Temp base path: /tmp/paperqa/{corpus_name}
        """
        return f"/tmp/paperqa/{self.corpus_name}"

    @property
    def build_papers_dir(self) -> str:
        """Get local temp directory for papers during build.

        Returns:
            Papers temp path: /tmp/paperqa/{corpus_name}/papers
        """
        return f"{self.build_temp_base}/papers"

    @property
    def build_index_dir(self) -> str:
        """Get local temp directory for index during build.

        Returns:
            Index temp path: /tmp/paperqa/{corpus_name}/index
        """
        return f"{self.build_temp_base}/index"


# Backward compatibility: keep old function for code that might still use it
def get_corpus_name() -> str:
    """Get corpus name from environment.

    .. deprecated::
        Use PaperQAPaths class instead.

    Returns:
        Corpus name (e.g., 'my-corpus-name')

    Raises:
        ValueError: If PAPERQA_CORPUS_NAME is not set
    """
    corpus_name = os.getenv("PAPERQA_CORPUS_NAME")
    if not corpus_name:
        raise ValueError("PAPERQA_CORPUS_NAME must be set in .env")
    return corpus_name
