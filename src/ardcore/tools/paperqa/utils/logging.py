"""Helpers for aligning PaperQA logging with ARD workflows."""

from __future__ import annotations

import logging
from typing import Iterable

from loguru import logger

_PAPERQA_LOGGER_NAMES: tuple[str, ...] = (
    "paperqa",
    "paperqa.agents",
    "paperqa.agents.tools",
    "paperqa.agents.search",
    "paperqa.agents.main",
    "paperqa.agents.main.agent_callers",
    "LiteLLM",
)

_PATCH_APPLIED = False


def supress_excessive_paperqa_logs() -> None:
    """Force PaperQA verbosity presets to keep internal logs at ERROR."""

    global _PATCH_APPLIED

    if _PATCH_APPLIED:
        return

    try:
        import paperqa.agents as paperqa_agents  # type: ignore import-not-found
    except ImportError as exc:  # pragma: no cover - defensive
        logger.warning("⚠️ [PAPERQA] Unable to adjust PaperQA verbosity: %s", exc)
        return

    _force_presets_to_error(paperqa_agents.LOG_VERBOSITY_MAP.values())

    _PATCH_APPLIED = True


def _force_presets_to_error(verbosity_maps: Iterable[dict[str, int]]) -> None:
    """Ensure PaperQA presets never promote internal loggers above ERROR."""

    for preset in verbosity_maps:
        for name in _PAPERQA_LOGGER_NAMES:
            preset[name] = logging.ERROR
