"""S3 sync utilities for PaperQA."""

import subprocess
from pathlib import Path

from loguru import logger

from .paperqa_paths import PaperQAPaths


def sync_from_s3(s3_path: str, local_path: str, description: str) -> int:
    """Sync data from S3 to local path using aws s3 sync.

    Args:
        s3_path: S3 path to sync from
        local_path: Local path to sync to
        description: Description for logging

    Returns:
        Number of files found in the local directory after sync
    """
    logger.info(f"⬇️  Syncing {description} from {s3_path} to {local_path}")
    local_path_obj = Path(local_path)
    local_path_obj.mkdir(parents=True, exist_ok=True)

    # Check if S3 path has any files
    try:
        ls_result = subprocess.run(
            ["aws", "s3", "ls", s3_path, "--recursive"],
            check=False,  # Don't fail if path doesn't exist
            capture_output=True,
            text=True,
        )
        s3_has_files = bool(ls_result.stdout.strip())
    except Exception:
        s3_has_files = False

    try:
        result = subprocess.run(
            ["aws", "s3", "sync", s3_path, local_path],
            check=True,
            capture_output=True,
            text=True,
        )

        # Count files downloaded by parsing output
        downloads = result.stdout.count("download:")

        if result.stdout:
            logger.debug(f"Sync output: {result.stdout}")

        # Count actual files in the directory after sync
        file_count = sum(1 for f in local_path_obj.rglob("*") if f.is_file())

        if downloads > 0:
            logger.info(
                f"✅ {description} synced from S3: {downloads} file(s) downloaded (total {file_count} files available)"
            )
        elif file_count > 0:
            if s3_has_files:
                # S3 has files and local has files but nothing downloaded - already in sync
                logger.info(
                    f"📦 {description}: already up-to-date locally ({file_count} files, nothing new from S3)"
                )
            else:
                # S3 is empty but local has files from previous run
                logger.info(
                    f"🆕 No {description} in S3 (will create new), but {file_count} files present locally from previous run"
                )
        else:
            # No files locally and nothing downloaded - S3 is empty
            logger.info(f"🆕 No {description} found in S3; will build from scratch")

        return file_count
    except subprocess.CalledProcessError as e:
        # Check if path doesn't exist (acceptable for new index)
        if "does not exist" in e.stderr or "NoSuchBucket" in e.stderr:
            # Count what we have locally even if S3 doesn't exist
            file_count = sum(1 for f in local_path_obj.rglob("*") if f.is_file())
            if file_count > 0:
                logger.info(
                    f"ℹ️  No {description} in S3 (will create new), but {file_count} files present locally from previous run"
                )
            else:
                logger.info(
                    f"ℹ️  No existing {description} found in S3; will create new"
                )
            return file_count
        else:
            logger.error(f"❌ Failed to sync {description}: {e.stderr}")
            raise
    except FileNotFoundError as e:
        raise RuntimeError(
            "AWS CLI not found. Please install it: "
            "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
        ) from e


def sync_to_s3(local_path: str, s3_path: str, description: str) -> None:
    """Sync data from local path to S3 using aws s3 sync.

    Args:
        local_path: Local path to sync from
        s3_path: S3 path to sync to
        description: Description for logging
    """
    logger.info(f"⬆️  Syncing {description} from {local_path} to {s3_path}")

    try:
        result = subprocess.run(
            ["aws", "s3", "sync", local_path, s3_path, "--delete"],
            check=True,
            capture_output=True,
            text=True,
        )
        logger.info(f"✅ {description} uploaded successfully")
        if result.stdout:
            logger.debug(f"Upload output: {result.stdout}")
    except subprocess.CalledProcessError as e:
        logger.error(f"❌ Failed to upload {description}: {e.stderr}")
        raise


def ensure_paperqa_index_available(corpus_name: str | None = None) -> Path:
    """Ensure the PaperQA index for the configured corpus is synced from S3.

    Always syncs from S3 to ensure the latest index is available locally.
    The sync is efficient - only new or modified files are downloaded.

    Args:
        corpus_name: Optional corpus name. If not provided, reads from PAPERQA_CORPUS_NAME env variable.

    Returns
    -------
    Path
        The local directory containing the PaperQA index.
    """
    paths = PaperQAPaths(corpus_name=corpus_name)
    index_path = paths.local_index

    # Always sync from S3 to ensure we have the latest index
    logger.info(
        "📥 [PAPERQA] Syncing %s index from %s to ensure latest version",
        paths.corpus_name,
        paths.index_s3,
    )
    files_synced = sync_from_s3(paths.index_s3, paths.local_index_base, "PaperQA index")

    if files_synced == 0:
        logger.warning(f"⚠️  No index files found for corpus '{paths.corpus_name}'")

    logger.info("✅ [PAPERQA] Index ready at %s", index_path)
    return index_path
