import asyncio
import time

from langchain_core.tools import tool
from loguru import logger

from paperqa import Settings
from paperqa.agents import ask

from .utils import s3_sync
from .utils.logging import supress_excessive_paperqa_logs
from .utils.paperqa_paths import PaperQAPaths

# Initialize paths once at module level
_PATHS = PaperQAPaths()

# Patch PaperQA logging before any searches run
supress_excessive_paperqa_logs()


def _sync_paperqa_index():
    """Ensure the PaperQA index for the current corpus is available locally.

    This function performs side effects (network and filesystem) by syncing the
    index from remote storage when necessary.
    """
    logger.info(
        f"📚 [PAPERQA] Ensuring index is available for corpus '{_PATHS.corpus_name}'"
    )
    s3_sync.ensure_paperqa_index_available()


def _get_paperqa_settings():
    """Build and return PaperQA Settings configured for the corpus.

    Returns:
        Settings: Settings object configured to use the local index
    """
    logger.info(f"📖 [PAPERQA] Configuring settings for index at {_PATHS.local_index}")
    settings = Settings()
    settings.agent.index.name = _PATHS.corpus_name
    settings.agent.index.index_directory = _PATHS.local_index_base
    settings.agent.index.sync_with_paper_directory = (
        False  # use `build_index.py` instead
    )

    # Cannot use gpt-5 due to: `Unsupported value: 'temperature'...`
    # settings.llm = "gpt-5-mini-2025-08-07"
    # settings.summary_llm = "gpt-5-mini-2025-08-07"

    # below params changed due to errors while generating hypothesis
    settings.answer.max_concurrent_requests = 1
    settings.agent.search_count = 5  # Reduced from default 8
    settings.agent.max_timesteps = 50  # Add upper limit on environment steps
    settings.verbosity = 0  # Reduced from 3

    # just an arbitrary decision ;)
    settings.answer.evidence_k = 10
    settings.answer.answer_max_sources = 5

    logger.info("✅ [PAPERQA] Settings configured and ready")
    return settings


# Load settings once at module initialization
_sync_paperqa_index()
_SETTINGS = _get_paperqa_settings()


@tool
def paperqa_search(query: str) -> str:
    """Search a curated database of scientific papers for evidence and citations.

    This tool searches through a specialized corpus of peer-reviewed research papers
    and returns evidence-based answers with inline citations. Use this tool to find
    empirical evidence, experimental results, and published findings that support or
    refute claims in the hypothesis.

    Args:
        query: A specific research question or claim to find evidence for

    Returns:
        Evidence-based answer with citations from indexed scientific papers
    """
    query_preview = query[:100] + ("..." if len(query) > 100 else "")
    logger.info(f"🔍 [PAPERQA] Starting search for query: '{query_preview}'")

    start_time = time.time()

    try:
        logger.debug("🌐 [PAPERQA] Running PaperQA query...")

        # Use the ask() function from paperqa.agents which properly queries and generates answers
        result = ask(query, settings=_SETTINGS)

        # If ask() returned a coroutine or Task (happens in async contexts), await it
        if asyncio.iscoroutine(result) or isinstance(result, asyncio.Task):
            response = asyncio.run(result)
        else:
            response = result

        elapsed = time.time() - start_time

        # Extract the formatted answer from the AnswerResponse
        answer = response.session.formatted_answer
        answer_len = len(answer)

        # Count citations (rough heuristic)
        num_citations = answer.count("(") if "(" in answer else 0

        logger.info(
            f"✅ [PAPERQA] Search completed in {elapsed:.2f}s "
            f"(length: {answer_len} chars, citations: ~{num_citations})"
        )

        return answer

    except Exception as e:
        elapsed = time.time() - start_time
        error_type = type(e).__name__
        error_msg = str(e)

        logger.error(
            f"❌ [PAPERQA] Search failed after {elapsed:.2f}s: {error_type}: {error_msg}"
        )

        # Contextual error hints
        emsg = error_msg.lower()
        if "index" in emsg or "not found" in emsg:
            logger.warning(
                "📂 [PAPERQA] Index not found or corrupted. "
                "Ensure PAPERQA_INDEX_DIR points to a valid index."
            )
        elif "timeout" in emsg or "timed out" in emsg:
            logger.warning("⏰ [PAPERQA] Query timed out; try a simpler question")
        elif "no answer" in emsg or "insufficient" in emsg:
            logger.warning("⚠️ [PAPERQA] No sufficient evidence found in indexed papers")
        else:
            logger.warning(f"❓ [PAPERQA] Unknown error type: {error_type}")

        return (
            f"PaperQA search failed for query '{query_preview}': {error_type}: {error_msg}. "
            f"The indexed corpus may not contain relevant papers for this query."
        )
