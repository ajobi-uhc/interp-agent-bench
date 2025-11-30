"""Shared test utilities."""

import signal
import sys
from contextlib import contextmanager


@contextmanager
def auto_cleanup(sandbox):
    """Auto-cleanup sandbox on exit or interrupt."""
    def cleanup(signum=None, frame=None):
        try:
            sandbox.terminate()
        except:
            pass
        if signum:
            sys.exit(0)

    # Register signal handler
    signal.signal(signal.SIGINT, cleanup)

    try:
        yield sandbox
    finally:
        cleanup()
