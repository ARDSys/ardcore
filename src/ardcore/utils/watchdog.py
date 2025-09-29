"""
Reusable watchdog utility for ARD workflows.
"""

import os
import sys
import time
import threading
from typing import Callable, Optional

from loguru import logger


class Watchdog:
    """Terminate the process if the timeout elapses.

    - Starts a background timer on enter; cancels on exit.
    - On timeout: logs, runs optional on_timeout, flushes, then exits.
    - Uses os._exit to guarantee termination even if the main thread is blocked.
    """

    def __init__(
        self,
        timeout_seconds: int,
        task_name: str,
        exit_code: int = 2,
        on_timeout: Optional[Callable[[], None]] = None,
        flush_seconds: float = 2.0,
    ):
        self.timeout_seconds = timeout_seconds
        self.task_name = task_name
        self.exit_code = exit_code
        self.on_timeout = on_timeout
        self.flush_seconds = flush_seconds
        self._timer: Optional[threading.Timer] = None

    def __enter__(self):
        def _trigger():
            try:
                logger.error(
                    f"⏰ Watchdog timeout exceeded ({self.timeout_seconds}s) for '{self.task_name}'. "
                    "Requesting shutdown."
                )
                if self.on_timeout:
                    try:
                        self.on_timeout()
                    except Exception as e:
                        logger.warning(f"Watchdog on_timeout raised: {type(e).__name__}: {e}")
                try:
                    sys.stdout.flush()
                    sys.stderr.flush()
                except Exception:
                    pass
                time.sleep(self.flush_seconds)
            finally:
                os._exit(self.exit_code)

        self._timer = threading.Timer(self.timeout_seconds, _trigger)
        self._timer.daemon = True
        self._timer.start()
        logger.info(
            f"🐶 Watchdog armed for '{self.task_name}' with timeout={self.timeout_seconds}s "
            f"(exit_code={self.exit_code})"
        )
        return self

    def cancel(self):
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None

    def __exit__(self, exc_type, exc, tb):
        self.cancel()


def watchdog_timeout(
    timeout_seconds: int,
    task_name: str,
    exit_code: int = 2,
    on_timeout: Optional[Callable[[], None]] = None,
    flush_seconds: float = 2.0,
) -> Watchdog:
    """Create a Watchdog context for the given task."""
    return Watchdog(
        timeout_seconds=timeout_seconds,
        task_name=task_name,
        exit_code=exit_code,
        on_timeout=on_timeout,
        flush_seconds=flush_seconds,
    )


