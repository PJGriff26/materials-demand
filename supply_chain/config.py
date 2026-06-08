"""Shared configuration for the supply-chain package: file paths, plot
defaults, and the two canonical material taxonomies (DEMAND_TO_RISK and
MATERIAL_GROUPS). Single source of truth for where the package reads and
writes, imported by every supply-chain module and figure generator.
"""

from pathlib import Path
import sys

# ── Reproducibility ──────────────────────────────────────────────────────────
# Re-export the canonical pipeline seed so supply-chain code uses the same RNG
# source as the Monte Carlo simulation. We bootstrap the repository root onto
# sys.path first because this module is imported both as `supply_chain.config`
# and (when a figure script is run directly) as a bare `config`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from src.random_config import RANDOM_SEED as _PIPELINE_SEED  # noqa: E402

RANDOM_SEED = _PIPELINE_SEED

# ── Paths ────────────────────────────────────────────────────────────────────
# Everything is resolved relative to the repository root (the parent of the
# supply_chain/ directory), so the package is fully relocatable: clone it
# anywhere and the paths still resolve. There are no absolute paths anywhere
# in this codebase.
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"

# Input data.
# DEMAND_FILE is the Monte Carlo demand output, annualized with a shape-
# preserving PCHIP interpolation of the NREL 3-year capacity cadence
# (Fritsch & Carlson 1980). It carries both the raw 3-year-bucket columns
# (mean, p2, p97, ...) and the companion per-year columns (mean_annual,
# p2_annual, p97_annual, ...). The figures use the *_annual columns.
DEMAND_FILE = OUTPUTS_DIR / "data" / "material_demand_by_scenario.csv"
NREL_SCENARIOS_FILE = DATA_DIR / "StdScen24_annual_national.csv"
MATERIAL_INTENSITY_FILE = DATA_DIR / "intensity_data.csv"

# Supply-chain / risk data: all machine-readable, all from citable sources:
#   USGS MCS 2025 Data Release (DOI 10.5066/P13XCP3R): production, reserves,
#     per-commodity trade, and net import reliance.
#   US Census Bureau International Trade API: bilateral import-partner shares.
#   OECD Country Risk Classifications (January 2026): country risk tiers.
MCS2025_DIR = DATA_DIR / "usgs_mcs_2025"
OECD_CRC_DIR = DATA_DIR / "oecd_crc"
CENSUS_TRADE_DIR = DATA_DIR / "census_trade"

# Output directories.
RESULTS_DIR = OUTPUTS_DIR / "data"
FIGURES_MANUSCRIPT_DIR = OUTPUTS_DIR / "figures" / "manuscript"

# ── Plot defaults ────────────────────────────────────────────────────────────
FIGURE_DPI = 300          # publication raster resolution
FIGURE_FORMAT = ["png"]   # formats supply_chain_analysis writes per figure

# ── Demand material → USGS MCS 2025 risk-material mapping ─────────────────────
# 18 direct matches plus the six thin-film byproducts; every rare-earth element
# maps to the aggregate "Rare Earths" entry because that is the only resolution
# USGS publishes. ("Gadium" is a typo for Gadolinium in the source
# intensity_data.csv; it is preserved verbatim so the demand series keys still
# join; renaming it would silently break the merge.)
DEMAND_TO_RISK = {
    "Aluminum": "Aluminum", "Boron": "Boron", "Cement": "Cement",
    "Chromium": "Chromium", "Copper": "Copper", "Lead": "Lead",
    "Magnesium": "Magnesium", "Manganese": "Manganese",
    "Molybdenum": "Molybdenum", "Nickel": "Nickel", "Niobium": "Niobium",
    "Silicon": "Silicon", "Silver": "Silver", "Steel": "Steel",
    "Tin": "Tin", "Vanadium": "Vanadium", "Yttrium": "Yttrium",
    "Zinc": "Zinc",
    # Thin-film byproducts. USGS MCS 2025 coverage varies by commodity
    # (some have per-country reserves, some only production); materials
    # without reserves are skipped cleanly in the reserves panels.
    "Cadmium": "Cadmium",
    "Gallium": "Gallium",
    "Germanium": "Germanium",
    "Indium": "Indium",
    "Selenium": "Selenium",
    "Tellurium": "Tellurium",
    # Rare earth elements → aggregate "Rare Earths".
    "Dysprosium": "Rare Earths", "Neodymium": "Rare Earths",
    "Praseodymium": "Rare Earths", "Terbium": "Rare Earths",
    "Gadium": "Rare Earths",
}

# ── Material classes ─────────────────────────────────────────────────────────
# The four classes used to group and color materials throughout the figures
# (Figs 3, 4, and 7). This is the canonical copy; the standalone Fig 3 and
# Fig 4 generators keep a local copy of the same grouping so they remain
# runnable on their own, but the membership is identical to this dict.
MATERIAL_GROUPS = {
    "Bulk commodities":    ["Cement", "Steel", "Aluminum", "Copper"],
    "Base & alloying":     ["Zinc", "Lead", "Nickel", "Tin", "Manganese",
                            "Chromium", "Molybdenum", "Vanadium", "Silicon",
                            "Magnesium", "Boron"],
    "Specialty metals":    ["Tellurium", "Indium", "Cadmium", "Silver",
                            "Niobium"],
    "Rare earth elements": ["Dysprosium", "Neodymium", "Praseodymium",
                            "Terbium", "Yttrium"],
}

# ── Create output directories ────────────────────────────────────────────────
# Importing config guarantees the output tree exists, so any script can write
# results/figures without a pre-flight mkdir.
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
FIGURES_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
