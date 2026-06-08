# Raw Data Sources — Supply Chain Analysis

All supply chain data now comes from citable, machine-readable raw sources
rather than the manually-compiled `risk_charts_inputs.xlsx`.

## Primary Sources

### 1. USGS Mineral Commodity Summaries 2025 — Data Release
- **DOI:** 10.5066/P13XCP3R
- **Citation key:** `usgs_2025_data`
- **URL:** https://www.sciencebase.gov/catalog/item/677eaf95d34e760b392c4970
- **Data year:** 2020-2024 (5-year salient statistics)
- **Format:** CSV (3 ZIP archives → individual commodity CSVs)
- **Files downloaded:**
  - `salient_commodity/` — 85 commodity CSVs with US production, imports, exports, consumption, NIR, prices
  - `world_data/MCS2025_World_Data.csv` — Global production + reserves by country for 77 commodities
  - `industry_trends/MCS2025_Fig2_Net_Import_Reliance.csv` — NIR rankings + import source countries
  - `industry_trends/MCS2025_Fig3_Major_Import_Sources.csv` — Country-level import source data
- **Covers:** All 19 core materials + 6 thin-film materials (Cd, Ga, Ge, In, Se, Te)

### 2. USGS Mineral Commodity Summaries 2025 — Publication
- **DOI:** 10.3133/mcs2025
- **Citation key:** `usgs_2025`
- **URL:** https://pubs.usgs.gov/publication/mcs2025
- **Use:** Citable reference for MCS data; individual commodity PDFs for import share details

### 3. OECD Country Risk Classifications (January 2026)
- **Citation key:** `oecd_crc_2026`
- **URL:** https://www.oecd.org/en/topics/sub-issues/country-risk-classification.html
- **Data date:** Valid as of 30 January 2026
- **Format:** Parsed from PDF → `../oecd_crc/oecd_crc_2026.csv`
- **Covers:** 201 countries with CRC scores (0-7, where 0 = high-income/OECD)

### 4. USGS Historical Statistics (Data Series 140)
- **Citation key:** `kelly_2014_ds140`
- **URL:** https://www.usgs.gov/centers/national-minerals-information-center/historical-statistics-mineral-commodities-united
- **Use:** Long-run US time series (back to 1900 for many commodities)

## Material-to-File Mapping

| Material | Salient CSV | World Data Commodity |
|----------|-------------|---------------------|
| Aluminum | mcs2025-alumi_salient.csv | Aluminum |
| Boron | mcs2025-boron_salient.csv | Boron |
| Cadmium | mcs2025-cadmi_salient.csv | Cadmium |
| Cement | mcs2025-cemen_salient.csv | Cement |
| Chromium | mcs2025-chrom_salient.csv | Chromium |
| Copper | mcs2025-coppe_salient.csv | Copper |
| Gallium | mcs2025-galli_salient.csv | Gallium |
| Germanium | mcs2025-germa_salient.csv | Gemanium |
| Indium | mcs2025-indiu_salient.csv | Indium |
| Lead | mcs2025-lead_salient.csv | Lead |
| Magnesium | mcs2025-mgmet_salient.csv | Magnesium metal |
| Manganese | mcs2025-manga_salient.csv | Manganese |
| Molybdenum | mcs2025-molyb_salient.csv | Molybdenum |
| Nickel | mcs2025-nicke_salient.csv | Nickel |
| Niobium | mcs2025-niobi_salient.csv | Niobium |
| Rare Earths | mcs2025-rareee_salient.csv | Rare earths |
| Selenium | mcs2025-selen_salient.csv | Selenium |
| Silicon | mcs2025-simet_salient.csv | Silicon |
| Silver | mcs2025-silve_salient.csv | Silver |
| Steel | mcs2025-fepig_salient.csv | Iron and Steel |
| Tellurium | mcs2025-tellu_salient.csv | Tellurium |
| Tin | mcs2025-tin_salient.csv | Tin |
| Vanadium | mcs2025-vanad_salient.csv | Vanadium |
| Yttrium | mcs2025-yttri_salient.csv | (not separate) |
| Zinc | mcs2025-zinc_salient.csv | Zinc |

## Data Available Per Material

### From Salient CSVs (US data, 2020-2024):
- US production (various forms depending on commodity)
- Imports (crude + scrap/refined where applicable)
- Exports
- Apparent consumption
- Net Import Reliance (NIR %)
- Prices (various measures)
- Stocks
- Employment

### From World Data CSV (global, 2023-2024):
- Production by country (2023 actual, 2024 estimated)
- Capacity by country (where available)
- Reserves by country (2024)

### From NIR Figure Data:
- Net import reliance ranking (2024)
- Major import source countries (2020-2023 average)

### From OECD CRC:
- Country risk classification (0-7 scale)
- 201 countries covered

## Replaces

This raw data replaces the manually-compiled `risk_charts_inputs.xlsx` which contained:
- `aggregate` sheet → now from salient CSVs (production, imports, exports, consumption)
- `import_dependency` sheet → now from salient CSVs (NIR_pct columns)
- `production` sheet → now from world_data CSV (global production by country)
- `reserves` sheet → now from world_data CSV (RESERVES_2024 column)
- `import_shares` sheet → now from MCS 2025 publication PDFs + NIR figure data
- `crc` sheet → now from oecd_crc/oecd_crc_2026.csv

## Key Upgrade: 2018-2022 → 2020-2024

The old Excel had data from 2018-2022. The new MCS 2025 data covers **2020-2024**,
giving us 2 additional years of data and the most current estimates.

---

## 2026-04-08 update: clustering pipeline now consumes World Data CSV

`world_data/MCS2025_World_Data.csv` (1,250 rows × 78 commodities) is now read
directly by the clustering feature engineering pipeline via
`clustering/feature_engineering.load_mcs2025_world_data()`. Before this date,
the file was downloaded but unused — clustering relied entirely on the smaller
production sheet derived from the salient CSVs, which covers only 19 of our
31 materials. The remaining 12 materials had `production_hhi` silently
defaulted to 0 (the *least* risky value), inverting their criticality
positions.

**Materials whose production_hhi is now sourced from the World Data CSV:**
- **Cadmium** (HHI 0.21, 16 countries; China 39%)
- **Gallium** (HHI 0.97, 4 countries; China 98%)
- **Indium** (HHI 0.53, 8 countries; China 71%)
- **Selenium** (HHI 0.30, 15 countries; China 49%)
- **Tellurium** (HHI 0.60, 8 countries; China 76%)
- **Yttrium and Gadolinium** (HHI 0.50 via Rare Earths aggregate proxy; China 69%)

**Materials still using hardcoded HHI** (USGS does not publish per-country
production data for them, even in the World Data CSV):
- **Germanium** (HHI 0.65) — values withheld in CSV; qualitative description
  in MCS 2025 Germanium chapter places it as "China-dominated" (>60%)
- **Fiberglass, Glass** (HHI 0.10) — not tracked as critical commodities;
  ~100% domestic production from globally diversified silica sand inputs

These are stored in `clustering/feature_engineering.HARDCODED_PRODUCTION_HHI`
with citations.

## 2026-04-08 update: NIR priority order corrected

The `_build_import_dependency_series` function previously preferred a
trade-balance NIR computed from the aggregate sheet over the USGS-published
NIR in the import_dependency sheet. For Rare Earths, the trade-balance formula
was fooled by stage asymmetry (US exports raw concentrate from Mountain Pass,
imports refined REE oxides/metals/magnets) and produced a negative value that
got clipped to 0. The published USGS NIR for Rare Earths is ~94–100% across
2020–2024.

**The fix**: priority order is now reversed (published value preferred), with
a stage-asymmetry guard that warns when published ≥ 0.5 but trade-balance
≤ 0.05. This catches REEs and any future material with the same pattern.

See `docs/raw_data_sources.md` §3 and `supply_chain/feature_engineering.py`
docstrings for full details.
