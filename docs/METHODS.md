# Methods

A compact, code-anchored description of the model. Each step names the module
that implements it. For the scientific rationale of individual choices, the
module and function docstrings carry the detailed notes and literature citations.

---

## 1. Inputs

- **Capacity projections.** NREL Standard Scenarios 2024, national, all 61
  scenarios, reported every three years from 2026 to 2050
  (`data/StdScen24_annual_national.csv`). Loaded by `src/data_ingestion.py`.
- **Material intensities.** Tonnes of each material per MW of each technology
  (`data/intensity_data.csv`), with multiple literature observations per
  material–technology pair where available. Loaded and cleaned by
  `src/data_ingestion.py` + `src/data_quality.py` (unit corrections, net-exporter
  handling, outlier flags).

## 2. Material-intensity distributions

`src/distribution_fitting.py` fits a lognormal distribution to the observations
for each material–technology pair, with tail validation (rejecting fits whose
shape parameter or max/median ratio is implausible). For pairs with fewer than
five observations, `src/cv_borrowing.py` borrows the median coefficient of
variation from better-sampled pairs, preserving the central estimate while
regularizing the spread. The model is always parametric; there is no empirical
bootstrap fallback.

## 3. Capacity interpolation

NREL reports capacity on a 3-year cadence. `src/interpolation.py` lifts each
scenario's capacity *stock* onto an annual grid so demand can be reported as an
annual rate. The production default is a shape-preserving monotone cubic Hermite
(PCHIP, Fritsch & Carlson 1980): it passes through every reported value exactly
(mass-conserving) and cannot overshoot between reporting years. Piecewise-linear
and mean-preserving-spline alternatives are retained for sensitivity comparison.

## 4. Stock-flow model and Monte Carlo

`src/stock_flow_simulation.py` implements

```
Stock(t)        = Stock(t-1) + Additions(t) - Retirements(t)
Additions(t)    = max(0, Stock(t) - Stock(t-1) + Retirements(t))
Retirements(t)  = capacity reaching end of its technology lifetime
Demand(m, t)    = Additions(t) × Intensity(m, technology)
```

Technology lifetimes and the NREL-technology → intensity mapping live in
`src/technology_mapping.py`. For each scenario the simulation draws 10,000 Monte
Carlo samples of material intensity from the fitted distributions and propagates
them through the stock-flow accounting, producing a demand distribution per
(scenario, material, year). The random seed is fixed
(`src/random_config.py`, 24601), so the run is exactly reproducible.

`simulate.py` is the entry point; it writes the full per-scenario statistics
(all 61 scenarios) to `outputs/data/material_demand_by_scenario.csv`, and a
compact `material_demand_summary.csv` holding just the four highlighted
scenarios (Mid_Case, Mid_Case_No_IRA, Mid_Case_100by2035, Mid_Case_95by2050).

## 5. Annualization and the demand CSV

Because demand is accumulated per capacity interval, `src/annualization.py` adds
companion per-year columns. The demand CSV therefore carries, for each
(scenario, year, material):

| column | meaning |
|---|---|
| `mean`, `std`, `p2_5`, `p2`, `p5`, `p25`, `p50`, `p75`, `p95`, `p97_5`, `p97` | Monte Carlo statistics of demand over the reporting interval |
| `bucket_years` | width of the reporting interval for that row |
| `mean_annual`, `p2_annual`, `p50_annual`, `p97_annual`, … | the above divided by `bucket_years`, i.e. annual-rate demand |

The figures and Table 1 use the annual-rate (`*_annual`) columns. **The 95%
confidence interval is [p2.5, p97.5]** (columns `p2_5`/`p97_5`, or
`p2_annual`/`p97_annual`), not p5/p95.

## 6. Supply-chain layer

`supply_chain/` compares demand against supply benchmarks, all from machine-
readable official sources (see `data_provenance.md`):

- **Net import reliance (NIR)** and **production / reserves** per material, from
  USGS Mineral Commodity Summaries 2025 (`usgs_mcs2025_loader.py`).
- **Production concentration** as the Herfindahl-Hirschman index (HHI) of
  country production shares (Graedel et al. 2012 methodology), in
  `feature_engineering.py`.
- **Import-partner shares** by country, from the US Census International Trade
  API, cached in `data/census_trade/` (`census_import_shares.py`).
- **Country risk** via the OECD Country Risk Classification
  (`data/oecd_crc/oecd_crc_2026.csv`).

`feature_engineering.py` assembles these into the per-material feature table;
`build_material_features.py` writes the subset (`import_dependency`,
`production_hhi`, …) that Figure 7 consumes; `supply_chain_analysis.py` produces
the reserve-adequacy and CRC-sourcing quantities behind Figures 5 and 6.

## 7. Scenarios and uncertainty reporting

Four scenarios are highlighted throughout: Mid_Case (reference), Net-Zero by 2035
(`Mid_Case_100by2035`), Net-Zero by 2050 (`Mid_Case_95by2050`), and No-IRA
(`Mid_Case_No_IRA`). Two distinct uncertainty sources are reported separately:
the spread *across* the 61 scenarios, and the within-scenario 95% Monte Carlo
intensity interval. Figure 4 decomposes demand uncertainty into these two
sources by material class.

## 8. Materials

31 materials are tracked. Four (gallium, germanium, selenium, gadolinium) fall to
zero demand under the prevailing crystalline-silicon / cadmium-telluride PV mix,
leaving 27 demand-bearing materials (Table 1). Fiberglass and glass are excluded
from the supply-chain criticality analysis (no published critical-minerals
source), but their demand is still reported.
