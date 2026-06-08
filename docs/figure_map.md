# Figure & table map

Exactly how each manuscript figure and Table 1 is produced: the generator, the
command `reproduce.py` runs, the data it consumes, and what it writes. All paths
are relative to the repository root. All figures are 300 DPI PNG written to
`outputs/figures/manuscript/`.

Each output PNG is named `figN_...` to match its manuscript figure number
(Figures 1-7); the generator script filenames are historical and need not match.

---

## Figure 1: Capacity additions & retirements

- **Generator:** `visualizations/fig1_capacity_additions_ensemble.py`
- **Command:** `python visualizations/fig1_capacity_additions_ensemble.py --disaggregate --output-dir outputs/figures/manuscript`
- **Output:** `fig1_capacity_additions.png`
- **Reads:** `data/StdScen24_annual_national.csv` (NREL capacity; the figure works
  directly from capacity, not from the demand CSV).
- **Content:** Mid_Case net capacity change per 3-year reporting interval by
  technology group (additions above zero, retirements below), with net-trajectory
  lines for the Net-Zero-by-2035, Net-Zero-by-2050, and No-IRA scenarios.
  `--disaggregate` splits solar into utility/distributed and wind into
  onshore/offshore. Add `--pchip` for the annual-cadence variant.

## Figure 2: Annual material demand projections

- **Generator:** `visualizations/fig1_demand_projections.py`
- **Command:** `python visualizations/fig1_demand_projections.py --envelope cross-scenario --output-dir outputs/figures/manuscript`
- **Output:** `fig2_demand_projections.png`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Small-multiples of annual demand for all 27 demand-bearing
  materials under the four highlighted scenarios, with a grey band spanning the
  min–max of the 61 scenario means.

## Figure 3: Cumulative demand by material family

- **Generator:** `visualizations/scenario_cum_by_family_tonnage.py`
- **Command:** `python visualizations/scenario_cum_by_family_tonnage.py`
- **Output:** `fig3_cumulative_by_family.png` (+ `.pdf`)
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Cumulative 2026–2050 demand by material family (Bulk, Base &
  alloying, Specialty, Rare earth elements) under the four highlighted scenarios.

## Figure 4: Demand-uncertainty sensitivity tornado

- **Generator:** `visualizations/fig4_sensitivity_tornado.py`
- **Command:** `python visualizations/fig4_sensitivity_tornado.py --compact --scenario-bounds min-max --baseline mid_case --output outputs/figures/manuscript/fig4_sensitivity_tornado.png`
- **Output:** `fig4_sensitivity_tornado.png`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Per-material-class tornado comparing the demand range from scenario
  choice (min–max across 61 scenarios) against the range from Monte Carlo
  intensity uncertainty (95% interval), as percent of the Mid_Case baseline.
  Add `--pchip` to pool over the full annual grid instead of the 8 reporting years.

## Figures 5 & 6: US and global supply tiers

- **Generator:** `visualizations/fig5_supply_chain_4panel.py` (calls the split-panel
  renderer in `visualizations/fig5_alt_variants.py`)
- **Command:** `python visualizations/fig5_supply_chain_4panel.py --re-mode per-element --output-dir outputs/figures/manuscript`
- **Outputs:** `fig5_supply_tiers_us.png` (Figure 5) and
  `fig6_supply_tiers_global.png` (Figure 6). These two split figures are the
  only outputs (the `--legacy-colored` flag would instead emit the older
  combined 2x2 view, which the manuscript does not use).
- **Reads:** `outputs/data/material_demand_by_scenario.csv`,
  `data/usgs_mcs_2025/...`, `data/oecd_crc/oecd_crc_2026.csv`,
  `data/census_trade/import_shares_cache.json`.
- **Content:** Peak demand vs production (top) and cumulative demand vs reserves
  (bottom), for the US (Fig 5) and globally (Fig 6), with rare earths broken out
  per element in the production panels. `--re-mode per-element` selects the
  per-element REE resolution used in the manuscript.

## Figure 7: Supply-risk scatter (NIR × HHI)

- **Generator:** `visualizations/nir_vs_hhi_scatter.py` (uses
  `visualizations/peak_demand_vs_nir_scatter.py` as a library)
- **Command:** `python visualizations/nir_vs_hhi_scatter.py --scale linear`
- **Output:** `fig7_nir_vs_hhi_scatter_linear.png` (+ `.pdf`)
- **Reads:** `outputs/data/material_demand_by_scenario.csv`,
  `outputs/data/material_features.csv`,
  `data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv`.
- **Prerequisite:** `outputs/data/material_features.csv` must exist; it is written
  by `supply_chain/build_material_features.py` (which `reproduce.py` runs first,
  and which is shipped precomputed).
- **Content:** Net import reliance (x) vs production HHI (y), marker size = peak
  US demand / global production, color = material class.

## Table 1: Peak annual demand by scenario

- **Generator:** `visualizations/table1_peak_demand.py`
- **Command:** `python visualizations/table1_peak_demand.py`
- **Outputs:** `outputs/data/table1_peak_demand.csv` and `.md`
- **Reads:** `outputs/data/material_demand_by_scenario.csv`
- **Content:** Peak annual demand under the four highlighted scenarios and the
  peak year, for the 27 demand-bearing materials. The script asserts 27 materials
  with positive peaks in all four scenarios before writing.
