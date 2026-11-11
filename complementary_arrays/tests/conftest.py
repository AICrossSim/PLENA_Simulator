"""Fixtures of the reproduction tests: the data root (COMPLEMENTARY_ARRAYS_DATA) and the reference results in the
output layout (COMPLEMENTARY_ARRAYS_REFERENCE); tests that need them are skipped when they are not set."""

import os
from pathlib import Path

import pytest


def _root(name: str) -> Path:
    value = os.environ.get(name)
    if not value or not Path(value).is_dir():
        pytest.skip(f"{name} is not set")
    return Path(value)


@pytest.fixture(scope="session")
def data_root() -> Path:
    return _root("COMPLEMENTARY_ARRAYS_DATA")


@pytest.fixture(scope="session")
def reference() -> Path:
    return _root("COMPLEMENTARY_ARRAYS_REFERENCE")
