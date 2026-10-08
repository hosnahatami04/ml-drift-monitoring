"""Logging setup for the driftmon package."""

import logging

_HANDLER_NAME = "driftmon"

_FORMAT = "%(asctime)s level=%(levelname)s logger=%(name)s %(message)s"
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S%z"


def setup_logging(level: str = "INFO") -> None:
    """Attach a single stream handler to the root logger.

    Safe to call more than once: the level is updated, the handler is not
    duplicated.
    """
    root = logging.getLogger()
    root.setLevel(level.upper())

    if any(handler.get_name() == _HANDLER_NAME for handler in root.handlers):
        return

    handler = logging.StreamHandler()
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT))
    root.addHandler(handler)
