"""Pytest configuration and shared fixtures; adds the project root to sys.path
so `src.*` imports resolve from within tests/."""

import sys
from pathlib import Path

import pytest

# Add project root to sys.path so `from src.xxx import ...` works
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def root_dir():
    """Return the project root directory as a Path object."""
    return ROOT
