# Figure & table map

Exactly how each manuscript figure and supplementary table is produced: the
generator, the command `reproduce.py` runs, the data it consumes, and what it
writes. All paths are relative to the repository root. All figures are 300 DPI
PNG written to `outputs/figures/manuscript/`.

Script and output filenames follow the manuscript numbering (Environmental
Research: Energy, 7-23 revision): Figures 1-6 in the main text, Figures S1-S3
in the SI. Print-size Word variants come from each script's `--docx` flag and
land in `outputs/figures/manuscript/docx/`. Diagnostics (not in the SI) land in
`outputs/figures/diagnostics/`. Retired generators live in
`visualizations/retired/`.

---

## Figure 1: Annual capacity additions by technology

- **Generator:** `visualizations/fig1_capacity_additions_facet.py`
- **Command:** `python visualizations/fig1_capacity_additions_facet.py --output-dir outputs/figures/manuscript`
- **Output:** `fig1_capacity_additions_facet.png` (+ `.pdf`)
- **Reads:** `data/StdScen24_annual_national.csv` (NREL capacity; the figure works
  directly from capacity, not from the demand CSV).
- **Content:** 15-panel (5×3) facet of annual gross capacity additions on the
  PCHIP annual grid, one panel per demand-modeled technology with nonzero
  ensemble additions, alphabetical; four highlighted scenarios plus the
  min-max "Scenario ensemble range" band across all 61 scenarios.
  Ensemble-zero technologies (biomass, coal, conventional nuclear) are omitted
  (`OMIT_IF_ZERO`).

## Figure 2: Annual material demand projections

- **Generator:** `visualizations/fig2_demand_projections.py`
- **Command:** `python visualizations/fig2_demand_projections.py --envelope cross-scenario --output-dir outputs/figures/manuscript`
- **Output:** `fig2_demand_projections.png`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Small-multiples of annual demand for all 27 demand-bearing
  materials under the four highlighted scenarios, with a grey band spanning the
  min–max of the 61 scenario means. (The `_full_uncertainty` variant uses the
  within-scenario MC interval instead and is NOT the manuscript figure.)

## Figure 3: Cumulative demand by material class

- **Generator:** `visualizations/fig3_cumulative_by_family.py`
- **Command:** `python visualizations/fig3_cumulative_by_family.py`
- **Output:** `fig3_cumulative_by_family.png` (+ `.pdf`)
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Cumulative 2026–2050 demand by material class (Bulk, Base &
  alloying, Specialty, Rare earth elements) under the four highlighted scenarios.

## Figures 4 & 5: US and global supply tiers

- **Generator:** `visualizations/fig4_fig5_supply_tiers.py` (calls the split-panel
  renderer in `visualizations/supply_tiers_variants.py`; shared data/helpers in
  `visualizations/supply_tiers_shared.py`)
- **Command:** `python visualizations/fig4_fig5_supply_tiers.py --re-mode per-element --output-dir outputs/figures/manuscript`
- **Outputs:** `fig4_supply_tiers_us.png` (Figure 4) and
  `fig5_supply_tiers_global.png` (Figure 5). These two split figures are the
  only outputs (the `--legacy-colored` flag would instead emit the older
  combined 2x2 view, which the manuscript does not use).
- **Reads:** `outputs/data/material_demand_by_scenario.csv`,
  `data/usgs_mcs_2025/...`, `data/oecd_crc/oecd_crc_2026.csv`,
  `data/census_trade/import_shares_cache.json`.
- **Content:** Peak demand vs production (top) and cumulative demand vs reserves
  (bottom), for the US (Fig 4) and globally (Fig 5), with rare earths broken out
  per element in the production panels. `--re-mode per-element` selects the
  per-element REE resolution used in the manuscript.

## Figure 6: Supply-risk scatter (NIR × HHI)

- **Generator:** `visualizations/fig6_nir_vs_hhi_scatter.py` (uses
  `visualizations/peak_demand_vs_nir_scatter.py` as a library)
- **Command:** `python visualizations/fig6_nir_vs_hhi_scatter.py --scale linear`
- **Output:** `fig6_nir_vs_hhi_scatter_linear.png` (+ `.pdf`)
- **Reads:** `outputs/data/material_demand_by_scenario.csv`,
  `outputs/data/material_features.csv`,
  `data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv`.
- **Prerequisite:** `outputs/data/material_features.csv` must exist; it is written
  by `supply_chain/build_material_features.py` (which `reproduce.py` runs first,
  and which is shipped precomputed).
- **Content:** Net import reliance (x) vs production HHI (y), marker size = peak
  US demand / global production, color = material class.

## Figure S1: Material demand across electricity-supply technologies

- **Generator:** `visualizations/figS1_demand_by_technology.py` (logic in
  `visualizations/demand_by_technology.py`)
- **Command:** `python visualizations/figS1_demand_by_technology.py --output-dir outputs/figures/manuscript`
- **Outputs:** `figS1_demand_by_technology.png` (+ `.pdf`) and the traceability
  exports `outputs/data/demand_by_tech_shares_{cumulative,peak2035}.csv`
- **Reads:** `data/StdScen24_annual_national.csv`,
  `outputs/data/fitted_distributions.csv`,
  `outputs/data/material_demand_by_scenario.csv`, `src/technology_mapping.py`
- **Content:** One 100%-stacked bar per demand-bearing material (27, sorted by
  total demand, right-hand totals column); each segment is a technology's share
  of Baseline-with-IRA cumulative 2026–2050 demand, rescaled to the published
  MC mean.

## Figure S2: Per-material sensitivity

- **Generator:** `visualizations/si_figures.py`
- **Command:** `python visualizations/si_figures.py --output-dir outputs/figures/manuscript`
- **Output:** `figS2_per_material_sensitivity.png` (+ `.pdf`); the same command
  renders the two diagnostics into `outputs/figures/diagnostics/`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Per material: scenario min-max range (light bar) vs the
  within-scenario 95% intensity interval (solid bar), as mean percent deviation
  from Baseline with IRA over 2027–2050, grouped by class (manuscript palette).

## Figure S3: Demand-uncertainty sensitivity tornado

- **Generator:** `visualizations/figS3_sensitivity_tornado.py`
- **Command:** `python visualizations/figS3_sensitivity_tornado.py --compact --scenario-bounds min-max --baseline mid_case --output outputs/figures/manuscript/figS3_sensitivity_tornado.png`
- **Output:** `figS3_sensitivity_tornado.png`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Per-material-class tornado comparing the demand range from scenario
  choice (min–max across 61 scenarios) against the range from Monte Carlo
  intensity uncertainty (95% interval), as percent of the Mid_Case baseline.
  Add `--pchip` to pool over the full annual grid instead of the 8 reporting years.

## Diagnostics (not in the SI)

- **Generator:** `visualizations/si_figures.py` (`--only CONV INTERP`)
- **Outputs:** `outputs/figures/diagnostics/diag_mc_convergence.png`,
  `outputs/figures/diagnostics/diag_interpolation_robustness.png`
- **Content:** Monte Carlo running-mean convergence (seed 24601, 1% band);
  PCHIP vs mean-preserving cumulative demand (line of equality).

## Tables S1–S9

- **Generator:** `visualizations/gen_si_tables.py` (stems follow the SI docx
  numbering; see the module INVENTORY for the per-table data sources)
- **Command:** `python visualizations/gen_si_tables.py`
- **Outputs:** `outputs/data/tables/table_s1.csv` … `table_s9.csv`, plus
  `table_lifetimes.csv` and `table_fit_corrections.csv` (reproducible but no
  longer SI tables).
- **S7/S8 source:** `visualizations/table1_peak_demand.py` writes
  `outputs/data/table1_peak_demand.{csv,md}` (peak annual demand under the four
  highlighted scenarios; asserts 27 materials with positive peaks) — Table S7
  re-emits it and Table S8 derives the Baseline peak/cumulative columns from
  the same demand CSV.
