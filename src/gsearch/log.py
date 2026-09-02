import logging
from pathlib import Path


def configure_logging(
    level: int = logging.INFO, logfile: str | Path | None = None
) -> logging.Logger:
    """Attach a handler to the ``gsearch`` package logger.

    Application-side helper: library modules only call ``logging.getLogger(__name__)``;
    the notebook or CLI calls this to decide the level and destination.

    Args:
        level: Logging level for the ``gsearch`` logger (e.g. ``logging.DEBUG``).
        logfile: If given, write logs to this file; otherwise write to stderr.

    Returns:
        The configured ``gsearch`` package logger.
    """
    handler = logging.FileHandler(logfile) if logfile else logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(name)s %(levelname)s | %(message)s")
    )

    pkg = logging.getLogger("gsearch")
    pkg.handlers.clear()  # idempotent: safe to call repeatedly in a notebook
    pkg.setLevel(level)
    pkg.addHandler(handler)
    pkg.propagate = False  # gsearch handler is the final stop; don't double-log via root

    return pkg
