#!/usr/bin/env python3
"""Alias for reproduce.py.

The manuscript's Data Availability statement refers to `run_pipeline.py`; the
canonical entry point in this repository is `reproduce.py`. This thin wrapper
forwards all arguments so either name regenerates every figure and table.
"""
import runpy
import sys
from pathlib import Path

if __name__ == "__main__":
    sys.argv[0] = str(Path(__file__).resolve().parent / "reproduce.py")
    runpy.run_path(sys.argv[0], run_name="__main__")
