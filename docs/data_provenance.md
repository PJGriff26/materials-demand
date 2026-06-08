# Data provenance

Every input in `data/` is from a peer-reviewed or official-government source.
The repository ships the exact files the code reads, so reproduction needs no
network access. This document records what each file is and where it came from.

---

## Capacity scenarios

**`data/StdScen24_annual_national.csv`**
NREL Standard Scenarios 2024, national results, all 61 scenarios, 2026–2050 at a
3-year cadence. Source: National Renewable Energy Laboratory, Standard Scenarios
2024 (Gagnon et al., *Standard Scenarios Report*, NREL/TP-6A40-92256).
Used by: `simulate.py` (capacity input) and Figure 1.

## Material intensities

**`data/intensity_data.csv`**
Tonnes of each material per MW of each generation/storage technology, compiled
from the published material-intensity literature (multiple observations per
material–technology pair where available, which is what the distribution fitting
consumes). Used by: `simulate.py` via `src/data_ingestion.py`.

Note: the file contains the spelling "Gadium" for Gadolinium; this typo is
preserved deliberately because downstream joins key on it. See the comment in
`supply_chain/config.py` (`DEMAND_TO_RISK`).

## Supply-chain raw data

**`data/usgs_mcs_2025/`**: USGS Mineral Commodity Summaries 2025 Data Release,
DOI [10.5066/P13XCP3R](https://doi.org/10.5066/P13XCP3R). Subfolders:
- `world_data/MCS2025_World_Data.csv`: per-country production and reserves by
  commodity (global production, HHI, reserves).
- `salient_commodity/`: per-commodity US salient statistics (net import
  reliance, trade) used for thin-film byproducts and NIR.
- `industry_trends/MCS2025_Fig2_Net_Import_Reliance.csv`: published US net
  import reliance figure.

Parsed by `supply_chain/usgs_mcs2025_loader.py`; used by Figures 5, 6, 7.

**`data/oecd_crc/oecd_crc_2026.csv`**: OECD Country Risk Classification,
January 2026 release. Country risk tiers used to color the sourcing panels of
Figures 5 and 6. Source: OECD, Country Risk Classification of the Participants to
the Arrangement on Officially Supported Export Credits.

**`data/census_trade/import_shares_cache.json`**: US import-partner shares by
HTS code (averaged over 2020–2023), retrieved from the US Census Bureau
International Trade API and cached. Used by `supply_chain/census_import_shares.py`
to attribute imports to partner countries (Figures 5/6 sourcing panels). The
cache ships so the figures reproduce offline; delete it to force a live refresh
(requires `requests` and network access).

---

## Precomputed results (shipped under `outputs/data/`)

These are produced by `simulate.py` and the supply-chain step; they ship so the
figures reproduce in seconds. `python reproduce.py --simulate` regenerates them
deterministically.

- `material_demand_by_scenario.csv`: per-scenario Monte Carlo demand statistics
  (the headline result; see the column dictionary in `docs/METHODS.md` §5).
- `material_demand_summary.csv`: a compact view of the four highlighted
  scenarios (Mid_Case, Mid_Case_No_IRA, Mid_Case_100by2035, Mid_Case_95by2050),
  per scenario; a subset of `material_demand_by_scenario.csv`.
- `fitted_distributions.csv`: the fitted material-intensity distribution
  parameters, with `fit_summary.csv` and `cv_borrowing_report.csv` recording the
  fit diagnostics and the small-sample CV-borrowing decisions.
- `material_features.csv`: per-material supply-chain features (NIR, HHI, …),
  written by `supply_chain/build_material_features.py`; consumed by Figure 7.
- `table1_peak_demand.{csv,md}`: Table 1.
- `../simulation_report.txt`: a human-readable summary of the simulation run.

## Data NOT included

Several raw datasets present in the full research project are not needed to
reproduce the manuscript and were left out to keep the release lean: EIA
generator inventories, rare-earth deposit-composition PDFs, the superseded
`risk_charts_inputs.xlsx`, and the superseded MCS 2023 thin-film CSVs. See
`CONDENSATION_REPORT.md`.
