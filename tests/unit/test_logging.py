import logging

import pytest

from driftmon.logging import setup_logging


@pytest.fixture(autouse=True)
def clean_root_logger() -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        if handler.get_name() == "driftmon":
            root.removeHandler(handler)


def _driftmon_handlers() -> list[logging.Handler]:
    return [handler for handler in logging.getLogger().handlers if handler.get_name() == "driftmon"]


def test_setup_attaches_single_handler() -> None:
    setup_logging()

    assert len(_driftmon_handlers()) == 1


def test_repeated_setup_does_not_duplicate_handlers() -> None:
    setup_logging()
    setup_logging()
    setup_logging("DEBUG")

    assert len(_driftmon_handlers()) == 1


def test_setup_applies_level() -> None:
    setup_logging("DEBUG")

    assert logging.getLogger().level == logging.DEBUG
