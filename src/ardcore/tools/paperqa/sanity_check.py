"""Sanity check for PaperQA search tool.

This script:
1. Initializes PaperQA (syncs from S3 if needed, loads index)
2. Runs two searches to verify the tool works correctly
3. Validates that the index is reused across queries

Usage:
    cd ard/workflows/generate_hypothesis
    uv run python -m src.langgraph.tools.paperqa.sanity_check

Or run directly (e.g., from PyCharm):
    cd ard/workflows/generate_hypothesis
    python src/langgraph/tools/paperqa/sanity_check.py
"""

import logging

from dotenv import load_dotenv
from loguru import logger

from ardcore.utils.logging_config import setup_workflow_logging

from .paperqa_manager import paperqa_search

# Configure both standard logging and loguru centrally
setup_workflow_logging(level=logging.INFO)  # Change to logging.DEBUG for verbose output

# Load AWS credentials and config from .env
load_dotenv()


def _print_results(results: str) -> None:
    if len(results) <= 1600:
        logger.info(f"Results:\n{results}")

    logger.info(f"Results:\n{results[:800]}...\n...\n...{results[-1600:]}")


logger.info("=" * 80)
logger.info("PaperQA Search Tool Test")
logger.info("=" * 80)
logger.info("(Index loads when module is imported)")

# Test 1: First search
logger.info("\n🧪 TEST 1: First search")
logger.info("-" * 80)
result1 = paperqa_search.invoke("What is psylocybin?")
_print_results(result1)

# Test 2: Second search (should reuse loaded index)
logger.info("\n🧪 TEST 2: Second search (should reuse loaded index)")
logger.info("-" * 80)
result2 = paperqa_search.invoke("What are adverese events when you take psylocybin?")
_print_results(result2)

logger.info("\n" + "=" * 80)
logger.info("✅ Test complete! Verify above:")
logger.info("  - Initialization should show: 'Syncing index...' (if not cached)")
logger.info("  - Both searches should complete without re-syncing")
logger.info("=" * 80)
