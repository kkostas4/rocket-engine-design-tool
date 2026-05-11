"""conftest.py — project-root pytest configuration.

Adds the repo root to sys.path so that ``import physics`` resolves
without installing the package.
"""
import sys
import os

import pytest

sys.path.insert(0, os.path.dirname(__file__))


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with -m 'not slow')"
    )
