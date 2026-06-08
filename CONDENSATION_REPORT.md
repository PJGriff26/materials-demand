# Condensation report

This release is a condensed extract of a larger research codebase, assembled so
that the manuscript's seven figures and one table reproduce from a clean
checkout with no external dependencies on the original project. This document
records exactly what was kept, what was removed, what was changed, and how the
result was verified, so the extraction is auditable.

Original working tree: ~845 MB. This release: ~20 MB.

---

## 1. What was kept (the live closure)

Starting from the seven manuscript figures and Table 1, the transitive closure of
code and data actually needed to produce them was traced and copied:

- **Simulation core** (`src/`): `stock_flow_simulation`, `data_ingestion`,
  `data_quality`, `distribution_fitting`, `cv_borrowing`, `technology_mapping`,
  `annualization`, `random_config`, and `interpolation` (the PCHIP module).
- **Supply-chain layer** (`supply_chain/`): `config`, `usgs_mcs2025_loader`,
  `census_import_shares`, `feature_engineering`, `supply_chain_analysis`, plus the
  new `build_material_features`.
- **Generators** (`visualizations/`): the seven figure generators, their shared
  helpers (`_interpolation_io`, `fig4_supply_chain`, `fig5_alt_variants`,
  `peak_demand_vs_nir_scatter`), and the table generator.
- **Entry points**: `simulate.py` (Monte Carlo) and `reproduce.py` (driver).
- **Data**: only the inputs the live code reads (see `docs/data_provenance.md`).
- **Precomputed results** and the **tests**.

## 2. What was removed, and why

None of the following is reachable from the seven figures or Table 1.

| Removed | Reason |
|---|---|
| K-means clustering + Sparse-PCA / PCA / NMF / factor-analysis (`legacy/clustering/`) | Cut from the manuscript; not a figure or table. |
| Sobol and Spearman sensitivity (`legacy/sensitivity/`) | Replaced in the manuscript by the tornado decomposition (Figure 4). |
| Diagnostic / QA scripts (`legacy/diagnostics/`) | Development-time checks, not manuscript outputs. |
| Policy-lever OLS regression (`analysis/lever_regression.py`) | The §3.4 regression was excluded from this release at the author's request. |
| Defense-talk and poster figure scripts; experimental interpolation-comparison figures | Not manuscript figures. |
| Orphan modules `src/materials_visualizations.py`, `src/historical_capacity.py` | Not imported by any kept code. |
| Unused raw data: EIA generator inventories, rare-earth deposit-composition PDFs, `risk_charts_inputs.xlsx`, MCS 2023 thin-film CSVs | Superseded or never read by the live pipeline. |
| Regenerable output trees (figure variants, interpolation/sensitivity/clustering output subdirs, diagnostic PNGs) | Regenerable from code; not needed to reproduce the manuscript. |
| Dependencies `scikit-learn`, `SALib`, `statsmodels`, `seaborn`, `openpyxl` | Only used by the removed analyses; the live code needs only numpy, pandas, scipy, matplotlib (+ requests for the optional Census refresh). |

## 3. Structural changes made (and why)

Each change preserves behavior; only file locations, import paths, and output
paths changed. Logic was not altered.

1. **Interpolation relocated.** `experimental/annual_interpolation.py` →
   `src/interpolation.py` (it is the production default, not experimental). The two
   import sites were updated; a stray module-level `mkdir` was made lazy so
   importing the module has no filesystem side effects.
2. **External generators pulled in.** The Figure 4 tornado and the Table 1
   generator lived outside the code tree (in the author's manuscript workspace).
   They are now `visualizations/fig4_sensitivity_tornado.py` and
   `visualizations/table1_peak_demand.py`, with their paths made repo-relative and
   the Table 1 generator's manuscript-markdown injection removed.
3. **Figure 7 severed from the retired clustering code.** Figure 7 previously read
   a feature CSV that only the (now-removed) clustering pipeline wrote. The needed
   columns (`import_dependency`, `production_hhi`) are computed by the live
   `feature_engineering.engineer_material_features`; a small new writer,
   `supply_chain/build_material_features.py`, dumps them to
   `outputs/data/material_features.csv`, and Figure 7 now reads that. No clustering
   code is imported anywhere in this release.
4. **Material classes relocated.** The four-class `MATERIAL_GROUPS` taxonomy that
   Figure 7 imported from the dropped `analysis/poster_policy_lever_table.py` now
   lives in `supply_chain/config.py` (the canonical copy).
5. **All paths made repo-relative.** Three absolute machine paths in the Figure 3
   generator, and the workspace-relative paths in the moved Figure 4 / Table 1
   generators, were replaced with paths resolved from each file's location. There
   are no absolute paths in the release.
6. **`config.py` trimmed** of all clustering parameters and the multi-cadence
   interpolation routing; figure output now resolves to
   `outputs/figures/manuscript/` uniformly.

## 4. Results regenerated from current code (one manuscript number changed)

A determinism check during packaging found that the original project's committed
`material_demand_by_scenario.csv` was **stale relative to its own current code**:
re-running the (deterministic, seed 24601) Monte Carlo reproduced 30 of 31
materials bit-for-bit, but **Praseodymium** differed by up to ~5% (about +2% at
its Mid_Case peak). Cause: onshore-wind Praseodymium has an extreme-tail intensity
distribution (max/median = 354.8x, above the 300x lognormal-rejection threshold),
so the current code rejects the lognormal and applies the n>=5 borrowed-CV rescue
(`src/cv_borrowing.py`); the committed CSV predated that rescue path. The
manuscript's Table 1 had been built from the stale CSV.

**Decision (author, this packaging):** regenerate all results from the current
code so the package is fully bit-reproducible and code/results/figures agree on
the current methodology. This release therefore ships freshly regenerated outputs.

**Manuscript reconciliation needed (one minor REE):** the Praseodymium values move
slightly versus the pre-regeneration manuscript. Update in the manuscript:
- **Table 1**, Praseodymium row. New values (t/yr) in the regenerated
  `outputs/data/table1_peak_demand.csv`, peak year 2031 (unchanged):
  Mid_Case 575 -> 586.7; NZ-2035 678 -> 695.4; NZ-2050 570 -> 581.0;
  No-IRA 491 -> 498.8 (about +2% each).
- **Figure 2** (Praseodymium small-multiple panel), **Figure 3** (rare-earth family
  cumulative; shift is sub-percent because Pr is a small share of the REE total),
  **Figure 4** (rare-earth class tornado; negligible), and **Figure 7** (Praseodymium
  marker position). Re-insert the regenerated PNGs from `outputs/figures/manuscript/`.

No headline finding changes (Pr is ~575 t/yr against, e.g., steel at ~14 Mt).

## 4b. Other items flagged for the author

These are properties of the **manuscript** surfaced during extraction; recorded so
the author can reconcile the manuscript text/figures.

- **Figure 3 consistency (resolved here).** In the original project the Figure 3
  generator read a separate, older PCHIP output (`outputs/data/interpolated/pchip/...`)
  that differed slightly from the canonical CSV used by every other figure. This
  release points Figure 3 at the same single canonical CSV as the rest, so all
  figures and the table are now mutually consistent.
- **Figure 4 vs §3.4 prose.** The §3.4 text in the draft describes an earlier
  tornado that plotted absolute cumulative demand (Mt) and quoted
  scenario-to-intensity ratios; the figure in this release plots percent change
  vs Mid_Case. The qualitative story is unchanged, but the specific numbers in the
  prose should be reconciled with the figure.
- **Figure 1 axis label.** The 3-year-interval variant should be labelled "net
  capacity change per 3-year reporting interval," not "annual"; the generator's
  `--pchip` variant gives a true annual-cadence figure if preferred.
- **Vanadium / tellurium values.** These post-date an offshore-wind intensity
  correction in the source data; cross-check before publishing.

## 5. Verification performed

- **Full bit-reproducibility.** The shipped results were generated by
  `python reproduce.py --simulate` (full 10,000-iteration Monte Carlo + features +
  all seven figures + Table 1). The simulation is deterministic (seed 24601): two
  independent runs are byte-identical, so re-running reproduces the shipped results
  exactly.
- **Documentation is comments-only.** Every code file was given/audited for
  comments and docstrings; an AST logic-diff against the source originals proved
  the 20 verbatim-copied files changed in comments/docstrings only (zero logic
  changes). A read-only adversarial review of all 34 files found none failing the
  well-commented / human-understandable bar.
- `python -m pytest -q`: all tests pass (unit + end-to-end; the output-file tests
  now check the shipped `material_features.csv`).
- Figures 5 and 7 were inspected and match the expected content (per-element rare
  earths; tellurium / indium as the high-stress markers).
- Grep gates confirm no absolute paths and no references to removed modules
  (clustering, Sobol, defense scripts, old paths) remain in executable code.
