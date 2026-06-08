# Materials Demand Model: manuscript code & data

Reproducible code, input data, and precomputed results for the manuscript

> **Monte Carlo simulation of critical material demand for US energy
> infrastructure under the NREL Standard Scenarios 2024.**
> PJ Griffiths, Dartmouth College.

---

## What the model does

For each of the 61 NREL Standard Scenarios 2024, the model takes projected
electricity-generation capacity (2026–2050), infers annual capacity *additions*
with a stock-flow accounting model, and multiplies additions by per-technology
**material intensities** drawn from fitted probability distributions. Running
10,000 Monte Carlo draws propagates material-intensity uncertainty into a
distribution of annual demand for each of 31 materials. A supply-chain layer then
compares that demand against US and global production and reserves, net import
reliance, and production concentration.

```
NREL capacity scenarios ─┐
                         ├─► stock-flow model ─► additions ─► × intensity ─► demand (10,000 MC draws)
material intensities  ───┘                                                        │
                                                                                  ▼
USGS / OECD / Census raw data ───────────────────────► supply-chain risk (NIR, HHI, reserves)
```

---

## Quick start

```bash
# 1. Install dependencies (Python 3.11+; developed on 3.12)
pip install -r requirements.txt

# 2. Reproduce every figure and table from the shipped Monte Carlo results
python reproduce.py
```

`reproduce.py` writes the seven figures to `outputs/figures/manuscript/` and
Table 1 to `outputs/data/table1_peak_demand.{csv,md}` in well under a minute,
using the precomputed Monte Carlo output that ships in `outputs/data/`.

To regenerate that Monte Carlo output from scratch (about 10–15 minutes; the
result is deterministic and reproduces the shipped CSV):

```bash
python reproduce.py --simulate
```

---

## Figures and tables

Each generator is a standalone script; `reproduce.py` runs them with the exact
flags below. Output filenames retain the generators' internal names; the mapping
to the manuscript numbering is given here and in
[docs/figure_map.md](docs/figure_map.md).

| Manuscript | Output file (`outputs/figures/manuscript/`) | Generator |
|---|---|---|
| **Figure 1** capacity additions & retirements | `fig1_capacity_additions.png` | `visualizations/fig1_capacity_additions_ensemble.py` |
| **Figure 2** annual material demand projections | `fig2_demand_projections.png` | `visualizations/fig1_demand_projections.py` |
| **Figure 3** cumulative demand by family | `fig3_cumulative_by_family.png` | `visualizations/scenario_cum_by_family_tonnage.py` |
| **Figure 4** demand-uncertainty tornado | `fig4_sensitivity_tornado.png` | `visualizations/fig4_sensitivity_tornado.py` |
| **Figure 5** US supply tiers | `fig5_supply_tiers_us.png` | `visualizations/fig5_supply_chain_4panel.py` |
| **Figure 6** global supply tiers | `fig6_supply_tiers_global.png` | `visualizations/fig5_supply_chain_4panel.py` |
| **Figure 7** supply-risk scatter (NIR × HHI) | `fig7_nir_vs_hhi_scatter_linear.png` | `visualizations/nir_vs_hhi_scatter.py` |
| **Table 1** peak annual demand | `outputs/data/table1_peak_demand.{csv,md}` | `visualizations/table1_peak_demand.py` |

Every generator also accepts a `--docx` flag that renders a print-sized variant
(absolute point sizes for direct insertion into the Word manuscript). The chart
content is identical; only the on-page font sizing changes.

---

## Repository layout

```
materials_demand_submission/
├── reproduce.py            One-command driver (figures + table; --simulate to re-run the MC)
├── simulate.py             Monte Carlo simulation entry point
├── requirements.txt        numpy, pandas, scipy, matplotlib (+ requests)
├── src/                    Core simulation
│   ├── data_ingestion.py / data_quality.py     load + clean inputs
│   ├── distribution_fitting.py / cv_borrowing.py  fit intensity distributions
│   ├── technology_mapping.py                   NREL tech → intensity + lifetimes
│   ├── interpolation.py                        3-yr capacity → annual grid (PCHIP)
│   ├── stock_flow_simulation.py                stock-flow Monte Carlo engine
│   ├── annualization.py                        per-year demand columns
│   └── random_config.py                        canonical random seed
├── supply_chain/           Supply-chain risk layer
│   ├── config.py                               paths + material taxonomies
│   ├── usgs_mcs2025_loader.py                  parse USGS MCS 2025 raw CSVs
│   ├── census_import_shares.py                 cached Census import-partner shares
│   ├── feature_engineering.py                  per-material features (NIR, HHI, …)
│   ├── supply_chain_analysis.py               reserve adequacy + CRC sourcing
│   └── build_material_features.py             write material_features.csv (for Fig 7)
├── visualizations/         Figure + table generators (see table above)
├── data/                   Raw inputs (all peer-reviewed or official; see docs/data_provenance.md)
├── outputs/
│   ├── data/               Precomputed Monte Carlo results + Table 1
│   └── figures/manuscript/ The seven figures
├── tests/                  pytest suite (unit + end-to-end)
└── docs/                   METHODS.md, data_provenance.md, figure_map.md
```

---

## Tests

```bash
python -m pytest -q
```

The suite covers data loading, the unit conversions in the USGS loader,
distribution fitting, the stock-flow equation, end-to-end determinism, and the
demand-aggregation invariants.

---

## Documentation

- [docs/METHODS.md](docs/METHODS.md): the modelling method, end to end, with pointers to the code.
- [docs/data_provenance.md](docs/data_provenance.md): every input file, its source, and its DOI/URL.
- [docs/figure_map.md](docs/figure_map.md): exact figure/table → script → data → command mapping.
- [CONDENSATION_REPORT.md](CONDENSATION_REPORT.md): what this release includes, what was left out of the full project, and why.

---

## Requirements

Python 3.11+ (tested on 3.12) and the packages in `requirements.txt`
(numpy, pandas, scipy, matplotlib; `requests` only to refresh the optional
Census cache, which ships, so reproduction is fully offline).

## License & citation

Released under the MIT License (`LICENSE`). If you use this code or data, please
cite the manuscript and this repository per `CITATION.cff`.
