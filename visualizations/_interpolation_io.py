"""Shared input/output path helpers for the figure generators.

Centralizes the Monte Carlo demand CSV paths and manuscript figure
directory so the generators agree on where things live, and exposes a
pchip-only ``--interpolation`` CLI shim for provenance.
"""

from __future__ import annotations

import argparse
from pathlib import Path

# Repository root (one directory up from visualizations/).
BASE_DIR = Path(__file__).resolve().parent.parent

# The shipped configuration.
INTERPOLATION_METHODS = ("pchip",)


def demand_csv_path(method: str = "pchip") -> Path:
    """Path to the per-scenario Monte Carlo demand CSV.

    The shipped results are PCHIP-annualized and live at the canonical
    location; the ``method`` argument is accepted for call-site
    compatibility but always resolves there.
    """
    return BASE_DIR / "outputs" / "data" / "material_demand_by_scenario.csv"


def demand_summary_csv_path(method: str = "pchip") -> Path:
    """Path to the cross-scenario Monte Carlo demand-summary CSV."""
    return BASE_DIR / "outputs" / "data" / "material_demand_summary.csv"


def figures_output_dir(method: str = "pchip", subdir: str = "manuscript") -> Path:
    """Figure output directory, ``outputs/figures/<subdir>/``."""
    return BASE_DIR / "outputs" / "figures" / subdir


def setup_interpolation_env() -> str:
    """Compatibility shim retained from the multi-cadence era.

    Originally set the ``MATERIALS_INTERPOLATION`` environment variable so
    that ``config`` would reroute its paths. The condensed release uses a
    single configuration, so this is now a no-op that simply reports the
    shipped method name.
    """
    return "pchip"


def add_interpolation_arg(parser: argparse.ArgumentParser) -> None:
    """Attach the (pchip-only) ``--interpolation`` flag to ``parser``."""
    parser.add_argument(
        "--interpolation",
        choices=list(INTERPOLATION_METHODS),
        default="pchip",
        help=(
            "Monte Carlo output cadence to read. This condensed release "
            "ships only 'pchip' (PCHIP-annualized, the manuscript default)."
        ),
    )
