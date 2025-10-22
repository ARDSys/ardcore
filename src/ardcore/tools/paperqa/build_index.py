"""Build or update PaperQA index from S3-stored PDFs.

This script:
1. Syncs PDFs from S3 to a local directory
2. Syncs existing index (if any) from S3
3. Builds/updates the PaperQA index incrementally with error handling
4. Syncs the updated index back to S3

The script uses robust error handling to skip problematic PDFs (corrupted, malformed, etc.)
instead of crashing on the first error. This is essential for large corpora (1000+ documents)
where some files may be problematic.

Configuration via environment variables (loaded from .env):
- PAPERQA_CORPUS_NAME: Corpus name (required, e.g., my-corpus-name)
- PAPERQA_PAPERS_BUCKET: S3 bucket for papers (required, e.g., s3://your-papers-bucket)
- PAPERQA_INDEX_BUCKET: S3 bucket for indexes (required, e.g., s3://your-index-bucket)

Usage:
    # Use environment variable:
    uv run python -m src.langgraph.tools.paperqa.build_index

    # Pass corpus name as command-line argument:
    uv run python -m src.langgraph.tools.paperqa.build_index --corpus-name my-corpus

    # Or programmatically:
    from src.langgraph.tools.paperqa.build_index import main
    main(corpus_name="my-corpus")
"""

import asyncio
from collections import Counter
from pathlib import Path

import anyio
import click
from dotenv import load_dotenv
from loguru import logger

from paperqa import Settings
from paperqa.agents.search import SearchIndex, process_file
from paperqa.settings import AgentSettings, IndexSettings

from .utils import s3_sync
from .utils.paperqa_paths import PaperQAPaths

# Load AWS credentials and config from .env
load_dotenv()


async def index_papers_with_error_handling(
    paper_dir: Path,
    index_dir: Path,
    corpus_name: str,
    settings: Settings,
    checkpoint_interval: int = 100,
    index_s3_path: str | None = None,
) -> tuple[int, int, int, list[tuple[str, str]]]:
    """Index papers with error handling to skip problematic PDFs.

    Uses paper-qa's SearchIndex for proper persistence.

    Args:
        paper_dir: Directory containing PDF files
        index_dir: Directory where index should be stored
        corpus_name: Name of the corpus
        settings: PaperQA settings
        checkpoint_interval: Save index every N successfully indexed files (default: 100)
        index_s3_path: Optional S3 path to upload checkpoints to (e.g., s3://bucket/corpus/)

    Returns:
        Tuple of (successful_count, failed_count, skipped_count, failed_files)
        where failed_files is a list of (filename, error_message) tuples
    """
    pdf_files = sorted(list(paper_dir.glob("*.pdf")))
    successful = 0
    failed = 0
    skipped = 0
    failed_files = []

    logger.info(f"📚 Found {len(pdf_files)} PDF files to index")

    # Create SearchIndex - paper-qa's way of persisting index
    search_index = SearchIndex(
        fields=[*SearchIndex.REQUIRED_FIELDS, "title", "year"],
        index_name=corpus_name,
        index_directory=str(index_dir),
    )

    # Get list of already-indexed files
    existing_files = await search_index.index_files
    existing_file_paths = set(existing_files.keys())

    if existing_file_paths:
        logger.info(f"ℹ️  Found {len(existing_file_paths)} already-indexed documents")

    total_files = len(pdf_files)
    semaphore = anyio.Semaphore(settings.agent.index.concurrency)

    # Track files processed since last checkpoint save
    files_since_last_save = 0
    logger.info(f"⚙️  Checkpoint interval: saving every {checkpoint_interval} files")

    for idx, pdf_file in enumerate(pdf_files, start=1):
        # Get relative path (paper-qa uses relative paths)
        rel_file_path = pdf_file.relative_to(paper_dir)
        rel_file_str = str(rel_file_path)

        # Skip if already indexed
        if rel_file_str in existing_file_paths:
            skipped += 1
            if skipped <= 10:  # Log first 10 skipped files
                logger.info(
                    f"⏭️  [{idx}/{total_files}] Skipping already-indexed: {pdf_file.name}"
                )
            elif skipped == 11:
                logger.info("⏭️  (suppressing further skip messages...)")
            continue

        try:
            logger.info(f"📄 [{idx}/{total_files}] Processing: {pdf_file.name}")

            # Use paper-qa's process_file function with error handling
            # This is what get_directory_index uses internally
            async with semaphore:
                await process_file(
                    rel_file_path,
                    search_index,
                    manifest={},  # Empty manifest dict
                    semaphore=semaphore,
                    settings=settings,
                    processed_counter=Counter(),  # Empty Counter object
                    progress_bar_update=None,  # No progress bar
                )

            successful += 1
            files_since_last_save += 1
            logger.info(
                f"✅ [{idx}/{total_files}] Successfully indexed: {pdf_file.name}"
            )

            # Periodically save progress to avoid losing work on crash
            if files_since_last_save >= checkpoint_interval:
                logger.info(
                    f"💾 Checkpoint: saving progress ({successful} files indexed so far)..."
                )
                await search_index.save_index()
                files_since_last_save = 0
                logger.info("✅ Checkpoint saved locally")

                # Upload checkpoint to S3 if configured
                if index_s3_path:
                    logger.info("☁️  Uploading checkpoint to S3...")
                    s3_sync.sync_to_s3(str(index_dir), index_s3_path, "checkpoint")
                    logger.info("✅ Checkpoint uploaded to S3")

        except Exception as e:
            failed += 1
            error_msg = f"{type(e).__name__}: {str(e)[:200]}"
            failed_files.append((pdf_file.name, error_msg))
            logger.warning(
                f"⚠️  [{idx}/{total_files}] Skipping {pdf_file.name} due to error: {error_msg}"
            )

    # Final save at the end (if there were changes since last checkpoint)
    logger.info(f"💾 Saving final index to {index_dir}")
    try:
        await search_index.save_index()
        logger.info(f"✅ Index saved successfully to {index_dir}/{corpus_name}/")
    except Exception as e:
        logger.error(f"❌ Failed to save index: {e}")
        raise

    return successful, failed, skipped, failed_files


@click.command()
@click.option(
    "--corpus-name",
    type=str,
    default=None,
    help="Corpus name (if not provided, reads from PAPERQA_CORPUS_NAME env variable)",
)
@click.option(
    "--checkpoint-interval",
    type=int,
    default=100,
    help="Save index checkpoint every N successfully indexed files (default: 100)",
)
def main(corpus_name: str | None = None, checkpoint_interval: int = 100) -> None:
    """Build or update PaperQA index and sync to S3.

    Args:
        corpus_name: Optional corpus name. If not provided, reads from PAPERQA_CORPUS_NAME env variable.
        checkpoint_interval: Save index every N successfully indexed files (default: 100)
    """
    # Initialize paths with corpus name (from parameter or environment)
    paths = PaperQAPaths(corpus_name=corpus_name)

    logger.info("🚀 Starting PaperQA index build/update")
    logger.info(f"📋 Corpus: {paths.corpus_name}")
    logger.info(f"📁 Papers S3: {paths.papers_s3}")
    logger.info(f"📁 Index S3: {paths.index_s3}")
    logger.info(f"📁 Local base: {paths.build_temp_base}")

    # Helper for counting files
    def _count_files(directory: str) -> int:
        """Count files recursively in a directory.

        Args:
            directory: Directory path to scan.

        Returns:
            Total number of files under the directory (0 if path doesn't exist).
        """
        base = Path(directory)
        if not base.exists():
            return 0
        return sum(1 for f in base.rglob("*") if f.is_file())

    # Capture pre-sync state
    papers_before = _count_files(paths.build_papers_dir)
    index_before = _count_files(paths.build_index_dir)

    # 1) Sync PDFs from S3
    papers_after = s3_sync.sync_from_s3(paths.papers_s3, paths.build_papers_dir, "PDFs")
    papers_delta = max(papers_after - papers_before, 0)
    if papers_after == 0:
        # Fail if no papers found
        error_msg = (
            f"❌ No papers found for corpus '{paths.corpus_name}' at {paths.papers_s3}\n"
            f"   Cannot build index without papers. Please check:\n"
            f"   1. The corpus name is correct: '{paths.corpus_name}'\n"
            f"   2. Papers have been uploaded to S3: {paths.papers_s3}\n"
            f"   3. You have the correct AWS credentials configured"
        )
        logger.error(error_msg)
        raise ValueError(f"No papers found for corpus '{paths.corpus_name}'")
    else:
        if papers_delta > 0 and papers_before == 0:
            logger.info(f"📥 Papers sync: fresh download ({papers_after} files).")
        elif papers_delta > 0:
            logger.info(
                f"📥 Papers sync: {papers_delta} new/updated files (total {papers_after})."
            )
        else:
            logger.info(
                f"📦 Papers sync: up-to-date (no changes, {papers_after} files)."
            )

    # 2) Sync existing index from S3 (if it exists)
    index_after = s3_sync.sync_from_s3(paths.index_s3, paths.build_index_dir, "index")
    index_delta = max(index_after - index_before, 0)
    if index_after == 0:
        logger.info("🆕 No existing index found in S3; will build from scratch.")
    else:
        if index_before == 0:
            logger.info(f"⬇️  Pulled existing index from S3 ({index_after} files).")
        elif index_delta > 0:
            logger.info(
                f"🔄 Index sync: {index_delta} new/updated files (total {index_after})."
            )
        else:
            logger.info(f"📦 Index sync: up-to-date (no changes, {index_after} files).")

    # 3) Build or update the index with robust error handling
    logger.info("🔨 Building/updating PaperQA index (with error tolerance)...")

    # Configure settings for paper-qa
    settings = Settings(
        agent=AgentSettings(
            index=IndexSettings(
                name=paths.corpus_name,
                paper_directory=paths.build_papers_dir,
                index_directory=paths.build_index_dir,
                sync_with_paper_directory=False,  # We handle this manually for error tolerance
                concurrency=5,
            )
        )
    )

    try:
        # Determine build mode for logging
        build_mode = "from-scratch" if index_after == 0 else "incremental-update"
        logger.info(f"🧠 Index build mode: {build_mode}")

        # Index papers with error handling to skip problematic PDFs
        successful, failed, skipped, failed_files = asyncio.run(
            index_papers_with_error_handling(
                Path(paths.build_papers_dir),
                Path(paths.build_index_dir),
                paths.corpus_name,
                settings,
                checkpoint_interval=checkpoint_interval,
                index_s3_path=paths.index_s3,  # Upload checkpoints to S3
            )
        )

        # Log summary
        logger.info("📊 Indexing complete:")
        logger.info(f"   ✅ Successfully indexed: {successful} files")
        logger.info(f"   ⏭️  Skipped (already indexed): {skipped} files")
        logger.info(f"   ⚠️  Failed to index: {failed} files")

        if failed_files:
            logger.warning("⚠️  Failed files (showing first 20):")
            for filename, error in failed_files[:20]:
                logger.warning(f"   - {filename}: {error}")
            if len(failed_files) > 20:
                logger.warning(f"   ... and {len(failed_files) - 20} more")

        # Fail the build if no files were successfully indexed AND there were files to process
        # (Don't fail if all files were skipped because they're already indexed)
        if successful == 0 and skipped == 0 and failed > 0:
            raise ValueError(
                "No files were successfully indexed. All files failed. Check the errors above."
            )

        logger.info("✅ Index build/update completed successfully")

    except Exception as e:
        logger.error(f"❌ Index build failed: {type(e).__name__}: {e}")
        raise

    # 4) Sync updated index back to S3
    s3_sync.sync_to_s3(paths.build_index_dir, paths.index_s3, "index")

    logger.info("🎉 PaperQA index build complete and uploaded to S3")


if __name__ == "__main__":
    main()
