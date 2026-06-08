"""Load supply-chain data from raw USGS MCS 2025 CSVs and OECD CRC 2026.

Parses production, reserves, per-commodity trade, and net import reliance
into a dict of tidy DataFrames consumed by the supply-chain risk analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path

from config import DATA_DIR, DEMAND_TO_RISK


# ── Paths ─────────────────────────────────────────────────────────────────────

MCS2025_DIR = DATA_DIR / "usgs_mcs_2025"
SALIENT_DIR = MCS2025_DIR / "salient_commodity"
WORLD_DATA_CSV = MCS2025_DIR / "world_data" / "MCS2025_World_Data.csv"
NIR_FIGURE_CSV = MCS2025_DIR / "industry_trends" / "MCS2025_Fig2_Net_Import_Reliance.csv"
OECD_CRC_CSV = DATA_DIR / "oecd_crc" / "oecd_crc_2026.csv"

# Map our 19 risk material names → MCS 2025 salient CSV prefixes
RISK_TO_SALIENT_PREFIX = {
    "Aluminum": "alumi",
    "Boron": "boron",
    "Cement": "cemen",
    "Chromium": "chrom",
    "Copper": "coppe",
    "Lead": "lead",
    "Magnesium": "mgmet",
    "Manganese": "manga",
    "Molybdenum": "molyb",
    "Nickel": "nicke",
    "Niobium": "niobi",
    "Rare Earths": "rareee",
    "Silicon": "simet",
    "Silver": "silve",
    "Steel": "feste",
    "Tin": "tin",
    "Vanadium": "vanad",
    "Yttrium": "yttri",
    "Zinc": "zinc",
}

# Thin-film materials also available
THIN_FILM_SALIENT_PREFIX = {
    "Cadmium": "cadmi",
    "Gallium": "galli",
    "Germanium": "germa",
    "Indium": "indiu",
    "Selenium": "selen",
    "Tellurium": "tellu",
}

# Merged prefix dict used by sheet builders that cover BOTH the 19 core
# materials AND the 6 thin-film byproducts. Keeping two dicts above
# preserves the semantic distinction (core vs. byproduct) for callers of
# load_thin_film_data_mcs2025(), while ensuring aggregate / NIR / reserves
# sheets cover all 25 materials in the pipeline's risk universe.
ALL_SALIENT_PREFIX = {**RISK_TO_SALIENT_PREFIX, **THIN_FILM_SALIENT_PREFIX}

# Map risk material names → World Data commodity names.
# Thin-films added 2026-04-19: USGS MCS 2025 World Data publishes per-
# country production for Indium/Gallium/Tellurium/Selenium/Cadmium, and
# per-country reserves for Indium/Tellurium/Selenium. Germanium has
# neither (byproduct of zinc refining, reported only as US demand).
RISK_TO_WORLD_COMMODITY = {
    "Aluminum": "Aluminum",
    "Boron": "Boron",
    "Cement": "Cement",
    "Chromium": "Chromium",
    "Copper": "Copper",
    "Lead": "Lead",
    "Magnesium": "Magnesium metal",
    "Manganese": "Manganese",
    "Molybdenum": "Molybdenum",
    "Nickel": "Nickel",
    "Niobium": "Niobium",
    "Rare Earths": "Rare earths",
    "Silicon": "Silicon",
    "Silver": "Silver",
    "Steel": "Iron and Steel",
    "Tin": "Tin",
    "Vanadium": "Vanadium",
    "Zinc": "Zinc",
    "Cadmium": "Cadmium",
    "Gallium": "Gallium",
    "Indium": "Indium",
    "Selenium": "Selenium",
    "Tellurium": "Tellurium",
}

# NIR column name varies by commodity — map to a canonical column
# Some have NIR_pct, others NIR_Metal_pct, NIR_Refined_pct, etc.
# We pick the most relevant one for each material.
_NIR_COL_OVERRIDES = {
    "Lead": "NIR_Metal_pct",
    "Nickel": "NIR_ct",          # reported as percentage despite name
    "Rare Earths": None,         # default column-finder picks NIR_Compounds-Metals_t (refined REE NIR, the policy-relevant stage)
    "Silicon": "NIR_FeSi-Si_pct",  # combined ferrosilicon + silicon metal
    "Tin": "NIR_Refined_pct",
    "Zinc": "NIR_Refined_pct",   # refined zinc NIR, not ores (US exports ore)
}

# Per-material production column preferences.
# Default: first column starting with "USprod". Override here when the
# default picks the wrong form of production (e.g., mine vs refinery).
_PROD_COL_OVERRIDES = {
    # Aluminum: primary smelter production only (excludes secondary/recycled)
    "Aluminum": ["USprod_Primary_kt"],
    # Copper: use refinery production (primary + secondary)
    "Copper": ["USprod_Refinery-primary_kt", "USprod_Refinery-secondary_kt"],
    # Lead: mine production (consistent with other mine-based commodities)
    "Lead": ["USprod_Mine_kt"],
    # Nickel: mine production only (excludes secondary)
    "Nickel": ["USprod_Mine_t"],
    # Steel: raw steel production in mmt — convert to kt (* 1000)
    "Steel": ["USprod_Steel_mmt"],
    # Zinc: use refined production
    "Zinc": ["USprod_Refined_kt"],
}

# Unit conversion factors: multiply CSV value by this to get kt.
# Default is 1.0 (CSV already in kt). Longer suffixes MUST come first in
# this dict so the `endswith` scan in `_col_unit_factor` matches `_kg`
# before `_g` and `_mmt` before `_t`.
_UNIT_TO_KT = {
    "_mmt": 1000.0,   # million metric tonnes → kt
    "_kg":  1e-6,     # kilograms → kt  (added 2026-04-19 — Germanium, Gallium)
    "_kt":  1.0,
    "_t":   0.001,    # tonnes → kt
}

# USGS published-CSV header typos. The Germanium salient CSV header
# has `Consump_g` where the metadata XML says `Consump_kg`; the column
# value (~30000 for apparent consumption) only makes physical sense
# under a kg interpretation (30 t/yr US Ge apparent consumption).
# Treat these aliases as their documented-intended unit suffix.
_COLUMN_NAME_ALIASES = {
    "Consump_g": "Consump_kg",
}

# ─────────────────────────────────────────────────────────────────────────────
# Compound-weight → metal-content conversion (added 2026-04-19)
# ─────────────────────────────────────────────────────────────────────────────
# A handful of USGS salient CSV columns publish GROSS compound weight
# rather than the metal's mass content. These must be multiplied by
# the metal's mass fraction in the compound before summing into any
# metal-equivalent total.
#
# **Important asymmetry.** Some USGS compound columns are *already*
# pre-converted to metal content (metadata XML states so explicitly,
# e.g., Germanium `Imports_GeO2_kg`: "Germanium dioxide data were
# multiplied by 69% to calculate the germanium content"). Adding such
# columns to this dict would double-apply the fraction and silently
# undercount metal-equivalent trade. The rule therefore is: list a
# column here ONLY if its USGS metadata explicitly describes the
# value as compound "gross weight" (or equivalent phrasing).
#
# Standard atomic weights — IUPAC 2021 CIAAW conventional values.
# Source: Prohaska, T. et al. (2022), "Standard atomic weights of the
# elements 2021 (IUPAC Technical Report)," Pure Appl. Chem. 94(5):
# 573–600. https://doi.org/10.1515/pac-2019-0603. Values are
# dimensionless relative atomic masses (m(12C) = 12 exactly). Parenthetical
# uncertainty is in the last digit of the reported value.
_ATOMIC_WEIGHTS_IUPAC2021 = {
    "Ga": 69.723,        # Ga: 69.723(1)
    "As": 74.921595,     # As: 74.921595(6) — monoisotopic (75As)
}


def _metal_fraction_in_compound(metal_symbol, other_symbols):
    """Return metal mass fraction in a binary/ternary compound given
    its constituent element symbols (stoichiometry 1:1:...:1).

    Uses IUPAC 2021 standard atomic weights. For compounds with
    non-unit stoichiometry (e.g., GeO2 has 1 Ge : 2 O), this helper
    is not used — pass the computed fraction directly or extend the
    signature. Currently only GaAs (1:1 binary) is needed.
    """
    m = _ATOMIC_WEIGHTS_IUPAC2021[metal_symbol]
    total = m + sum(_ATOMIC_WEIGHTS_IUPAC2021[s] for s in other_symbols)
    return m / total


# Per-column metal-mass fractions. Columns NOT listed here are summed
# at face value (correct for metal-content columns and for USGS-pre-
# converted compound columns). See _metal_fraction_in_compound and
# the citations in _ATOMIC_WEIGHTS_IUPAC2021.
#
# Current entries:
#   `Imports_GaAs_kg` — USGS metadata: "Gallium arsenide wafers
#     (gross weight)" (mcs2025-galli_meta.xml). GaAs is stoichiometric
#     1:1 Ga:As; Ga mass fraction ≈ 0.482 by IUPAC 2021 weights.
#     Before this fix, the pipeline summed GaAs wafer gross weight
#     directly into Ga trade totals, overstating Ga metal-equivalent
#     imports by ≈2× (2024: 180 t gross → 86.8 t Ga content).
#
# Do NOT add:
#   `Imports_GeO2_kg`, `Exports_GeO2_kg` — USGS metadata states
#     "Germanium dioxide data were multiplied by 69% to calculate the
#     germanium content" — ALREADY pre-converted. Adding here would
#     double-apply the 69% factor.
#   `Imports_GeCl4_kg`, `Exports_GeCl4_kg` — USGS metadata omits any
#     pre-conversion phrasing; interpretation is ambiguous. Volumes
#     are small (<1 t/yr) so the magnitude impact is immaterial;
#     scope-separate if Ge demand ever becomes non-zero.
_COMPOUND_METAL_FRACTION = {
    "Imports_GaAs_kg": _metal_fraction_in_compound("Ga", ["As"]),
}


def _col_metal_fraction(col_name):
    """Return the metal-mass fraction to apply to a USGS compound-
    weight column. Defaults to 1.0 (no conversion — column is either
    already metal content or USGS pre-converted it)."""
    effective = _COLUMN_NAME_ALIASES.get(col_name, col_name)
    return _COMPOUND_METAL_FRACTION.get(effective, 1.0)


# ── Salient data loading ──────────────────────────────────────────────────────

def _load_salient(material, prefix):
    """Load a single commodity's salient CSV. Returns DataFrame or None."""
    path = SALIENT_DIR / f"mcs2025-{prefix}_salient.csv"
    if not path.exists():
        print(f"  WARNING: {path} not found — skipping {material}")
        return None
    return pd.read_csv(path)


def _extract_nir(df, material):
    """
    Extract net import reliance (%) from a salient DataFrame.
    Returns list of (year, nir_pct) tuples.
    """
    if df is None:
        return []

    # Find the right NIR column
    override = _NIR_COL_OVERRIDES.get(material)
    if override is not None:
        if override in df.columns:
            nir_col = override
        else:
            return []
    else:
        # Default: find first column containing "NIR" and "pct"
        candidates = [c for c in df.columns if "NIR" in c and "pct" in c.lower()]
        if not candidates:
            candidates = [c for c in df.columns if "NIR" in c]
        if not candidates:
            return []
        nir_col = candidates[0]

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        val = row.get(nir_col)
        if pd.isna(year):
            continue
        s = str(val).strip()
        # Handle USGS text notation
        if s.upper() == "E" or s == "--" or s == "":
            nir = 0.0  # net exporter
        elif s.startswith("<"):
            nir = float(s[1:]) / 2
        elif s.startswith(">"):
            nir = (float(s[1:]) + 100) / 2
        else:
            try:
                nir = float(s)
            except ValueError:
                continue
        results.append((int(year), nir))
    return results


def _col_unit_factor(col_name):
    """Determine the kt conversion factor from a column name suffix.

    Resolution order:
      1. Resolve known USGS CSV-header typos via `_COLUMN_NAME_ALIASES`
         (e.g., `Consump_g` → `Consump_kg` for Germanium).
      2. Longest-suffix match in `_UNIT_TO_KT` (`_mmt` before `_kg`
         before `_kt` before `_t`), so a column named `*_kg` never
         mis-matches on `_t` or `_g`.
      3. Default: 1.0 (assume kt).
    """
    effective = _COLUMN_NAME_ALIASES.get(col_name, col_name)
    # Longest-suffix first prevents a _kg column from matching _g, or
    # a _kt column from matching _t.
    for suffix in sorted(_UNIT_TO_KT, key=len, reverse=True):
        if effective.endswith(suffix):
            return _UNIT_TO_KT[suffix]
    return 1.0  # assume kt by default


def _extract_production(df, material):
    """
    Extract US production (kt) from a salient DataFrame.
    Returns list of (year, production_kt) tuples.

    Uses _PROD_COL_OVERRIDES to pick the right columns per material
    and _UNIT_TO_KT for unit conversion.
    """
    if df is None:
        return []

    override = _PROD_COL_OVERRIDES.get(material)
    if override:
        prod_cols = [c for c in override if c in df.columns]
    else:
        prod_cols = [c for c in df.columns if c.startswith("USprod")]
        if not prod_cols:
            prod_cols = [c for c in df.columns
                         if "prod" in c.lower() and "Price" not in c
                         and "DataSource" not in c and "Commodity" not in c]
    if not prod_cols:
        return []

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        if pd.isna(year):
            continue
        total = 0.0
        any_valid = False
        for col in prod_cols:
            val_str = str(row.get(col, "")).replace(",", "").strip()
            if val_str in ("W", "XX", "--", "", "nan"):
                continue  # withheld or not available
            try:
                val = float(val_str)
                factor = _col_unit_factor(col)
                total += val * factor
                any_valid = True
            except (ValueError, TypeError):
                continue
        if any_valid:
            results.append((int(year), total))
    return results


def _extract_trade(df, material):
    """
    Extract US imports, exports, consumption from a salient DataFrame.
    Returns list of (year, imports_kt, exports_kt, consumption_kt) tuples.

    Unit handling: each column is scaled by `_col_unit_factor` (suffix
    → kt conversion) and, if applicable, by `_col_metal_fraction`
    (compound gross-weight → metal content; see
    `_COMPOUND_METAL_FRACTION`). The metal-fraction path handles
    gallium arsenide (Ga's `Imports_GaAs_kg` is USGS gross weight).
    It deliberately does NOT apply to Germanium's GeO2 columns,
    which USGS already pre-converts to Ge content per the MCS
    metadata XML — double-applying the 69% factor would silently
    undercount.
    """
    if df is None:
        return []

    # Find import columns (may be multiple: crude, scrap, refined)
    import_cols = [c for c in df.columns if c.startswith("Imports")]
    export_cols = [c for c in df.columns if c.startswith("Exports")]
    # Prefer apparent consumption
    consump_cols = [c for c in df.columns
                    if c.startswith("Consump") and ("Apprnt" in c or "Apparent" in c)]
    if not consump_cols:
        consump_cols = [c for c in df.columns
                        if c.startswith("Consump") and "Total" in c]
    if not consump_cols:
        consump_cols = [c for c in df.columns if c.startswith("Consump")]

    results = []
    for _, row in df.iterrows():
        year = row.get("Year")
        if pd.isna(year):
            continue

        def _sum_cols_kt(cols):
            total = 0.0
            for c in cols:
                v = str(row.get(c, "")).replace(",", "").strip()
                # Effective scaling = unit conversion × compound metal
                # fraction. For metal-content columns the fraction is
                # 1.0 (default). For compound gross-weight columns
                # (e.g. GaAs), it's the metal's IUPAC-derived mass
                # fraction in the compound.
                scale = _col_unit_factor(c) * _col_metal_fraction(c)
                # "Less than 1/2 unit" is USGS's smallest nonzero reporting
                # category (a left-censored value in [0, 0.5 unit]); impute its
                # interval midpoint, 0.25, scaled to kt. This is a censored-data
                # convention, not a published figure.
                if v.startswith("Less than"):
                    total += 0.25 * scale
                elif v in ("W", "XX", "--", "", "nan"):
                    continue
                else:
                    try:
                        total += float(v) * scale
                    except (ValueError, TypeError):
                        pass
            return total

        imports = _sum_cols_kt(import_cols)
        exports = _sum_cols_kt(export_cols)
        consumption = _sum_cols_kt(consump_cols) if consump_cols else 0.0

        results.append((int(year), imports, exports, consumption))
    return results


# ── Build the legacy-format sheets ────────────────────────────────────────────

def _build_aggregate_sheet():
    """
    Build a DataFrame matching the old 'aggregate' sheet format:
    columns: material, year, production, import, export, consumption, net_import

    Covers all 25 materials (19 core + 6 thin-films). Thin-film salient
    CSVs use the same column conventions (Year, USprod_*, Imports_*,
    Exports_*, Consump_*, NIR_*) so the extractors below handle them
    without special casing.
    """
    records = []
    for material, prefix in ALL_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        if df is None:
            continue

        prod_data = dict(_extract_production(df, material))
        trade_data = _extract_trade(df, material)

        for year, imports, exports, consumption in trade_data:
            production = prod_data.get(year, 0.0)
            net_import = imports - exports
            # Preserve NaN when USGS withholds consumption ("W"). The old
            # fallback (production + net_import) silently collapsed to
            # net_import whenever production was ALSO withheld, silently
            # corrupting the Form 72 back-estimation identity downstream.
            # Consumers that need consumption should check for NaN and
            # handle withheld-data cases explicitly.
            if consumption > 0:
                consump_value = consumption
            elif production > 0:
                consump_value = production + net_import
            else:
                consump_value = float("nan")
            records.append({
                "material": material,
                "year": year,
                "production": production,
                "import": imports,
                "export": exports,
                "consumption": consump_value,
                "net_import": net_import,
            })

    return pd.DataFrame(records)


def _build_import_dependency_sheet():
    """
    Build a DataFrame matching the old 'import_dependency' sheet format:
    columns: material, <year1>, <year2>, ...
    Values are NIR percentages (0-100) or "E" for net exporters.
    """
    all_years = set()
    mat_nir = {}

    for material, prefix in ALL_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        nir_data = _extract_nir(df, material)
        if nir_data:
            mat_nir[material] = dict(nir_data)
            all_years.update(y for y, _ in nir_data)

    # Rare Earths NIR comes from the salient CSV's NIR_Compounds-Metals_t column
    # (refined REE — see _NIR_COL_OVERRIDES). USGS MCS 2025 Fig 2 row 28 reports
    # the same 2024 value (80%) and is the original source for that figure.
    if "Rare Earths" not in mat_nir or not mat_nir["Rare Earths"]:
        raise ValueError(
            "Rare Earths NIR missing from salient CSV. Check that "
            "mcs2025-rareee_salient.csv exists and contains NIR_Compounds-Metals_t."
        )

    years = sorted(all_years)
    rows = []
    for material in ALL_SALIENT_PREFIX:
        row = {"material": material}
        nirs = mat_nir.get(material, {})
        for y in years:
            v = nirs.get(y)
            if v is not None:
                if v == 0:
                    row[y] = "E"
                else:
                    row[y] = v
            else:
                row[y] = np.nan
        rows.append(row)

    return pd.DataFrame(rows)


def _production_unit_factor_to_kt(unit_str):
    """Convert a USGS World Data UNIT_MEAS string to a kt multiplicative
    factor. Values in the production sheet are stored in thousand metric
    tons (kt) after this conversion, matching the legacy xlsx convention.
      "thousand metric tons" / "thousand metric dry tons" → 1.0   (already kt)
      "million metric tons"                                → 1000.0 (Mt → kt)
      "metric tons"                                        → 0.001 (t → kt)
      "kilograms"                                          → 1e-6  (kg → kt)
    """
    s = str(unit_str).lower() if unit_str is not None else ""
    if "thousand metric" in s:
        return 1.0
    if "million metric" in s:
        return 1000.0
    if "kilogram" in s:
        return 1e-6
    if "metric ton" in s:
        return 0.001
    return 1.0  # unknown — don't rescale; surfaced later if values look off


def _build_production_sheet():
    """
    Build a DataFrame matching the old 'production' sheet format:
    Global and per-country production for each material.
    Source: MCS2025_World_Data.csv. All production values are normalized
    to KILOTONS (kt) regardless of the commodity's native USGS unit, so
    downstream consumers can treat the column uniformly.
    """
    if not WORLD_DATA_CSV.exists():
        print(f"  WARNING: {WORLD_DATA_CSV} not found")
        return pd.DataFrame()

    wdf = pd.read_csv(WORLD_DATA_CSV)

    records = []
    for risk_name, world_name in RISK_TO_WORLD_COMMODITY.items():
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue
        # Some USGS commodities have multiple TYPE entries (e.g., "Iron and
        # Steel" has both "Pig iron" and "Raw steel"; "Iron Ore" has both
        # "usable ore" and "iron content"). Picking the first TYPE silently
        # would select pig iron for steel — wrong metric for Panel C
        # consistency with the aggregate salient sheet (which reports
        # crude steel). Prefer an explicitly-listed TYPE per commodity;
        # fall back to the first available.
        _PREFERRED_TYPE = {
            "Iron and Steel": "Raw steel",
            "Iron Ore":       "Mine production, usable ore",
        }
        preferred = _PREFERRED_TYPE.get(world_name)
        if preferred and (sub["TYPE"] == preferred).any():
            sub = sub[sub["TYPE"] == preferred]
        else:
            first_type = sub["TYPE"].iloc[0]
            sub = sub[sub["TYPE"] == first_type]
        # Unit per commodity (USGS mixes kt, Mt, t, and kg across commodities).
        unit = sub["UNIT_MEAS"].iloc[0] if "UNIT_MEAS" in sub.columns else ""
        unit_factor = _production_unit_factor_to_kt(unit)

        for _, row in sub.iterrows():
            country = row["COUNTRY"]
            prod_2023 = pd.to_numeric(row.get("PROD_2023"), errors="coerce")
            prod_2024 = pd.to_numeric(row.get("PROD_EST_ 2024"), errors="coerce")
            if pd.notna(prod_2023):
                prod_2023 = prod_2023 * unit_factor
            if pd.notna(prod_2024):
                prod_2024 = prod_2024 * unit_factor
            records.append({
                "material": risk_name,
                "country": country,
                "production_2023": prod_2023,
                "production_2024": prod_2024,
            })

    return pd.DataFrame(records)


def _build_reserves_sheet():
    """
    Build a DataFrame matching the old 'reserves' sheet format:
    rows = countries (with 'Global' and 'United States'),
    columns = material names, values = reserves.
    Source: MCS2025_World_Data.csv (RESERVES_2024 column)
    """
    if not WORLD_DATA_CSV.exists():
        return pd.DataFrame()

    wdf = pd.read_csv(WORLD_DATA_CSV)

    # Collect reserves by country per material
    all_countries = set()
    mat_reserves = {}

    def _parse_reserve_val(raw):
        """Parse reserve values, handling '>' prefix and commas."""
        s = str(raw).strip().replace(",", "")
        if s.startswith(">"):
            s = s[1:]
        try:
            return float(s)
        except (ValueError, TypeError):
            return np.nan

    # World data is in metric tons; old xlsx was in kt.
    # Determine unit from UNIT_MEAS column; convert to kt.
    for risk_name, world_name in RISK_TO_WORLD_COMMODITY.items():
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue

        # Determine reserve unit. USGS World Data CSV uses UNIT_MEAS for
        # production. Reserves USUALLY share the same unit, but some
        # commodities have a RESERVE_NOTES field stating the reserves use
        # a different unit (e.g., "Reserve data is thousand metric tons").
        # The XML metadata confirms RESERVE_NOTES overrides UNIT_MEAS.
        unit = sub["UNIT_MEAS"].iloc[0] if "UNIT_MEAS" in sub.columns else ""
        unit_str = str(unit).lower()

        # Check if any row for this commodity has a RESERVE_NOTES override
        has_kt_note = False
        if "RESERVE_NOTES" in sub.columns:
            notes = sub["RESERVE_NOTES"].dropna().astype(str)
            has_kt_note = notes.str.contains("thousand metric tons", case=False).any()

        if "thousand" in unit_str or has_kt_note:
            unit_factor = 1.0   # reserves already in kt
        elif "kilogram" in unit_str:
            unit_factor = 1e-6  # kg → kt
        else:
            unit_factor = 0.001  # metric tons → kt

        # Take rows with reserves data
        sub_res = sub[sub["RESERVES_2024"].notna()]
        if sub_res.empty:
            continue

        mat_reserves[risk_name] = {}
        for _, row in sub_res.iterrows():
            country = row["COUNTRY"]
            val = _parse_reserve_val(row["RESERVES_2024"])
            if pd.notna(val):
                val_kt = val * unit_factor
                if country == "United States":
                    mat_reserves[risk_name]["United States"] = val_kt
                else:
                    mat_reserves[risk_name][country] = val_kt
                all_countries.add(country)

    # Compute Global totals from "World total" rows
    for risk_name in mat_reserves:
        world_name = RISK_TO_WORLD_COMMODITY.get(risk_name)
        if not world_name:
            continue
        sub = wdf[wdf["COMMODITY"].str.strip() == world_name]
        if sub.empty:
            continue

        # Recompute unit factor using same RESERVE_NOTES logic
        unit = sub["UNIT_MEAS"].iloc[0] if "UNIT_MEAS" in sub.columns else ""
        has_kt_note = False
        if "RESERVE_NOTES" in sub.columns:
            notes = sub["RESERVE_NOTES"].dropna().astype(str)
            has_kt_note = notes.str.contains("thousand metric tons", case=False).any()
        if "thousand" in str(unit).lower() or has_kt_note:
            uf = 1.0
        elif "kilogram" in str(unit).lower():
            uf = 1e-6
        else:
            uf = 0.001

        world_rows = sub[sub["COUNTRY"].str.contains("World|Total", case=False, na=False)]
        if not world_rows.empty:
            val = _parse_reserve_val(world_rows.iloc[0]["RESERVES_2024"])
            if pd.notna(val):
                mat_reserves[risk_name]["Global"] = val * uf

        # If no global row, sum all country reserves (already converted)
        if "Global" not in mat_reserves[risk_name]:
            total = sum(v for k, v in mat_reserves[risk_name].items()
                        if k != "United States")
            us = mat_reserves[risk_name].get("United States", 0)
            total += us
            if total > 0:
                mat_reserves[risk_name]["Global"] = total

    # Build wide-format DataFrame (matching old format)
    materials = sorted(mat_reserves.keys())
    countries = ["Global", "United States"] + sorted(
        c for c in all_countries if c not in ("Global", "United States")
    )

    rows = []
    for country in countries:
        row = {"Unnamed: 0": country}
        for mat in materials:
            row[mat] = mat_reserves.get(mat, {}).get(country, np.nan)
        rows.append(row)

    return pd.DataFrame(rows)


def _build_import_shares_sheet():
    """
    Build a DataFrame matching the old 'import_shares' sheet format:
    columns: material, country, share

    Source: U.S. Census Bureau International Trade API (bilateral trade
    values by HTS code, aggregated to percentage shares per material).
    This replaces the hand-compiled import_shares sheet that was
    extracted from USGS MCS publication PDFs.

    The legacy risk_charts_inputs.xlsx fallback was removed 2026-04-19;
    any Census API failure now raises rather than silently returning
    stale hand-compiled data.
    """
    from census_import_shares import fetch_import_shares
    shares = fetch_import_shares(use_cache=True)
    if shares.empty or shares["material"].nunique() < 10:
        raise RuntimeError(
            "Census Bureau import-shares data returned incomplete "
            f"({shares['material'].nunique() if not shares.empty else 0} "
            "materials). Check network/API availability or the on-disk "
            "cache at data/census_trade/import_shares_cache.json."
        )
    print(f"  Import shares: {shares['material'].nunique()} materials "
          f"from Census Bureau data")
    return shares


def _build_crc_sheet():
    """
    Build a DataFrame matching the old 'crc' sheet format:
    columns: country, crc
    Source: OECD CRC January 2026 (parsed from PDF → CSV)
    """
    if not OECD_CRC_CSV.exists():
        print(f"  WARNING: {OECD_CRC_CSV} not found")
        return pd.DataFrame(columns=["country", "crc"])

    crc = pd.read_csv(OECD_CRC_CSV)

    # Map CRC values to match old format:
    # Old format used: "OECD" for 0, numeric 1-7, "China" handled separately
    def _map_crc(row):
        if row["crc"] == 0:
            return "OECD"
        return row["crc"]

    crc["crc_mapped"] = crc.apply(_map_crc, axis=1)

    # Rename some countries to match old format
    country_renames = {
        "China (People's Republic of)": "China",
        "Korea": "Republic of Korea",
        "Türkiye": "Turkey",
        "Viet Nam": "Vietnam",
        "Congo (Kinshasa)": "Democratic Republic of the Congo",
        "Congo (Brazzaville)": "Congo",
        "Côte d'Ivoire": "Cote d'Ivoire",
    }
    crc["country"] = crc["country"].replace(country_renames)

    result = crc[["country", "crc_mapped"]].rename(columns={"crc_mapped": "crc"})

    # Add extra columns to match old format (the old sheet had some extra cols)
    return result


# ── Public API ────────────────────────────────────────────────────────────────

def load_risk_data_mcs2025():
    """
    Load supply-chain risk data from raw USGS MCS 2025 CSVs and OECD CRC.

    Returns a dict of DataFrames with the SAME keys and structure as the
    old load_risk_data() function that read from risk_charts_inputs.xlsx:
      - aggregate: material × year trade data
      - import_dependency: material × year NIR percentages
      - production: global production by country
      - reserves: global/US reserves by country
      - import_shares: import shares by country (from the U.S. Census Bureau trade API; the old xlsx fallback was removed, so a Census failure now raises)
      - crc: country risk classifications
    """
    print("  Loading supply-chain data from USGS MCS 2025 raw CSVs...")

    sheets = {
        "aggregate": _build_aggregate_sheet(),
        "import_dependency": _build_import_dependency_sheet(),
        "production": _build_production_sheet(),
        "reserves": _build_reserves_sheet(),
        "import_shares": _build_import_shares_sheet(),
        "crc": _build_crc_sheet(),
    }

    # Report
    for name, df in sheets.items():
        print(f"    {name}: {df.shape[0]} rows × {df.shape[1]} cols")

    return sheets


def load_thin_film_data_mcs2025():
    """
    Load thin-film material data from MCS 2025 salient CSVs.

    Returns DataFrame with columns: material, production_t, nir_pct
    (same format as old load_usgs_2023_thin_film()).
    """
    records = []
    for material, prefix in THIN_FILM_SALIENT_PREFIX.items():
        df = _load_salient(material, prefix)
        if df is None:
            continue

        prod_data = _extract_production(df, material)
        nir_data = _extract_nir(df, material)

        avg_prod = np.mean([p for _, p in prod_data]) if prod_data else 0.0
        avg_nir = np.mean([n / 100.0 for _, n in nir_data]) if nir_data else 1.0

        records.append({
            "material": material,
            "production_t": avg_prod,
            "nir_pct": avg_nir,
        })

    return pd.DataFrame(records).set_index("material") if records else pd.DataFrame()
