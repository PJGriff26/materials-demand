# feature_engineering.py
"""Build the dimensionless per-material supply-chain feature table behind the
NIR vs HHI supply-risk scatter (manuscript Figure 6).

`engineer_material_features()` is called by build_material_features.py to
write outputs/data/material_features.csv; features (import dependency,
producer-country HHI, reserve adequacy, etc.) are dimensionless so they are
commensurable across materials. A scenario-feature builder is also retained.
"""

import pandas as pd
import numpy as np
from config import (
    DEMAND_FILE, NREL_SCENARIOS_FILE,
    DEMAND_TO_RISK,
)
from pathlib import Path
from usgs_mcs2025_loader import (
    load_risk_data_mcs2025,
    load_thin_film_data_mcs2025,
    _production_unit_factor_to_kt,
)

# ─────────────────────────────────────────────────────────────────────────────
# USGS MCS 2025 World Data Release CSV (added 2026-04-08)
# ─────────────────────────────────────────────────────────────────────────────
# Source: USGS Mineral Commodity Summaries 2025 — World Data Release
#   Citation: U.S. Geological Survey, 2025, Mineral Commodity Summaries 2025
#             World Data Release: U.S. Geological Survey data release,
#             https://doi.org/10.5066/P13XCP3R
#   Local copy: data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv
#               (canonical location, maintained alongside the rest of the
#                MCS 2025 data release in data/usgs_mcs_2025/)
#
# Why we need this: the original risk_charts_inputs.xlsx production sheet
# does not include byproduct/trace metals (Ga, In, Te, Se, Cd) or REEs at
# the per-country level. Without this data, _build_production_hhi defaulted
# missing materials to 0 (perfectly diversified — the *least* risky value),
# which inverted the criticality story for ~10 materials. Resolved here on
# 2026-04-08 (originally tracked as "Bug 2" in the project provenance log).

MCS2025_WORLD_CSV = Path(__file__).resolve().parents[1] / "data" / "usgs_mcs_2025" / "world_data" / "MCS2025_World_Data.csv"

# Mapping from our demand-material names → MCS 2025 commodity names.
# (USGS has a typo: "Gemanium" not "Germanium" in the World CSV.)
# Y and Gd are aggregated under "Rare earths" — we use that as a proxy
# with an explicit methods note.
MCS2025_COMMODITY_MAP = {
    "Cadmium":   "Cadmium",
    "Gallium":   "Gallium",
    "Germanium": "Gemanium",   # USGS typo in source data
    "Indium":    "Indium",
    "Selenium":  "Selenium",
    "Tellurium": "Tellurium",
    "Yttrium":   "Rare earths",      # proxy: aggregated REE production
    "Gadium":    "Rare earths",      # proxy (Gd; "Gadium" is intensity_data.csv typo)
    "Dysprosium":   "Rare earths",
    "Neodymium":    "Rare earths",
    "Praseodymium": "Rare earths",
    "Terbium":      "Rare earths",
}

# ─────────────────────────────────────────────────────────────────────────────
# REE processing-stage country shares (EU CRM 2023, Annex 7)
# ─────────────────────────────────────────────────────────────────────────────
# Source: Grohol, M. & Veeh, C. (2023). Study on the Critical Raw Materials
#   for the EU 2023 — Final Report. doi:10.2873/725585, Annex 7 (Processing
#   stage country shares for refined REO / REE-metal output).
#
# Why these constants exist here (not just in viz code):
# Per-element REE production HHI at the **mining** stage is the basket
# aggregate (~0.50) because USGS reports a single "Rare earths" world
# production row. But the binding supply constraint for Nd/Pr/Dy/Tb/Y/Gd
# is **separation/refining**, which is 85–100% Chinese. Reporting
# production_hhi at the mining stage alone understates supply concentration
# for these elements by ~2x. Following EU CRM 2023 methodology, HHI is
# computed at both stages and the more concentrated stage governs the
# canonical `production_hhi` column. See [[project_pi_meeting_2026_04_22]]
# for the per-element framing decision and [[feedback_re_publication_strategy]]
# for the aggregate-main / per-element-SI publication pattern.
#
# These distributions are also used in visualizations/supply_tiers_shared.py
# (LREE_PROCESSING / HREE_PROCESSING). That module imports from here as
# the single source of truth — supply_chain depends on no viz code.
LREE_PROCESSING_SHARES = {
    "China":    0.849,
    "Malaysia": 0.105,
    "Russia":   0.019,
    "India":    0.016,
    "Vietnam":  0.010,
    "Norway":   0.001,
}
HREE_PROCESSING_SHARES = {"China": 1.000}

# Element → processing-stage class. Heavy/light split per EU CRM 2023
# Annex 7 + Binnemans & Jones (2015) ore mineralogy classification.
# "Gadium" preserved as the intensity_data.csv typo for Gadolinium.
REE_PROCESSING_CLASS = {
    "Neodymium":    "LREE",
    "Praseodymium": "LREE",
    "Dysprosium":   "HREE",
    "Terbium":      "HREE",
    "Yttrium":      "HREE",
    "Gadium":       "HREE",
}


# Published-source values for materials where the MCS 2025 World CSV
# withholds per-country production. Every entry must be derivable from a
# peer-reviewed or official-government source — the explicit calculation
# (not a narrative fit) must appear in the comment.
PUBLISHED_PRODUCTION_HHI = {
    # Germanium: MCS 2025 World CSV rows are "W" (withheld), and the MCS
    # Germanium chapters publish no numeric country shares: "Because some
    # producers do not publicly report germanium production, global
    # production data were limited." — no per-country shares appear in the
    # 2024 or 2025 chapter. (A prior version of this entry attributed
    # China 75% / Belgium 12% / Canada 6% / Germany 3% / Russia 2% to MCS
    # 2025 chapter prose; both vintages were re-read 2026-07-07 and contain
    # no such shares.)
    # Authoritative source: EU CRM 2023 final report (doi 10.2873/725585),
    # annex "Global supply shares and trade-related variable", p. 83,
    # processing stage — the only stage the EU assesses for Ge (a zinc
    # by-product); same annex family as LREE/HREE_PROCESSING_SHARES:
    #   China 89.6%, Russia 5.4%, United States 2.1%, Japan 1.9%, Ukraine 1.0%
    # HHI (Graedel 2012 formula, SUM(s_i^2)):
    #   0.896² + 0.054² + 0.021² + 0.019² + 0.010²
    #   = 0.8028 + 0.0029 + 0.0004 + 0.0004 + 0.0001 ≈ 0.81
    # Stage note: processing-stage HHI; with no published mining distribution
    # this is canonical under the max-stage rule (matches the rare-earth
    # processing-stage handling documented in Table S5).
    "Germanium": (0.81, "EU CRM 2023 final report annex p. 83 (processing-stage country shares)"),
}

# Materials that are not tracked on any published critical-minerals list
# (USGS Critical Minerals List 2022 and 2024; EU Critical Raw Materials
# List 2023; DoE Critical Materials Strategy 2023). Per-country production,
# import-source, and governance-risk metrics are not published for these
# bulk construction materials. Including them with fabricated numbers would
# pollute the supply-risk analysis; they are kept in the demand projections
# (where they have real MC output) but **excluded from the criticality
# feature matrix**. Downstream consumers (the criticality scatter and
# reserve-adequacy metrics) skip any material in this set.
EXCLUDED_FROM_CRITICALITY = {"Fiberglass", "Glass"}


def _map_series_to_demand_names(series, all_materials, default=None):
    """
    Map a Series keyed by a mix of demand-material and risk-material names
    onto the canonical demand-material namespace.

    Resolution order per demand name:
      1. Direct hit on the demand name (e.g., "Gallium" present as-is).
      2. DEMAND_TO_RISK mapping (e.g., "Dysprosium" → "Rare Earths").
      3. `default` value if provided, otherwise leave as NaN.

    This is the reusable version of the inline 2026-04-08 production_hhi
    mapping fix. Use it for any per-material Series that may be keyed by
    either demand or risk names (notably import_hhi, import_china_frac,
    hhi_wgi, reserves-by-CRC breakdowns).

    Parameters
    ----------
    series : pd.Series
        Source series; index may mix demand and risk names.
    all_materials : Iterable[str]
        Canonical demand-material list (usually from demand output).
    default : float, optional
        Fallback value when neither direct nor mapped lookup succeeds.
        Leave as None to yield NaN (caller's fillna decides policy).

    Returns
    -------
    pd.Series
        Indexed by demand material names.
    """
    out = pd.Series(dtype=float)
    for demand_name in all_materials:
        if demand_name in series.index and pd.notna(series[demand_name]):
            out[demand_name] = series[demand_name]
            continue
        risk_name = DEMAND_TO_RISK.get(demand_name)
        if risk_name and risk_name in series.index and pd.notna(series[risk_name]):
            out[demand_name] = series[risk_name]
            continue
        if default is not None:
            out[demand_name] = default
    return out


def load_mcs2025_world_data():
    """
    Load the USGS MCS 2025 World Data Release CSV.

    Source: U.S. Geological Survey, 2025, Mineral Commodity Summaries 2025
    World Data Release, https://doi.org/10.5066/P13XCP3R

    Returns DataFrame with columns: COMMODITY, COUNTRY, TYPE, UNIT_MEAS,
    PROD_2023, PROD_EST_2024, PROD_NOTES, ... (some columns renamed for
    consistency).

    Returns empty DataFrame if the file is not present.
    """
    if not MCS2025_WORLD_CSV.exists():
        print(f"  WARNING: MCS 2025 World CSV not found at {MCS2025_WORLD_CSV}")
        return pd.DataFrame()
    df = pd.read_csv(MCS2025_WORLD_CSV)
    df["COMMODITY"] = df["COMMODITY"].str.strip()
    df["COUNTRY"] = df["COUNTRY"].str.strip()
    # Rename for tidiness
    df = df.rename(columns={"PROD_EST_ 2024": "PROD_EST_2024"})
    return df


def _mcs2025_country_production(world_df, commodity_name):
    """
    Extract per-country production for one commodity from the MCS 2025 World
    CSV, in plain METRIC TONNES. Uses 2024 estimate, falls back to 2023.
    Excludes "World total" and "Other Countries" rows (the former is the sum,
    the latter is residual). Returns DataFrame with columns: country, p.

    Unit handling (added 2026-05-21):
        Source rows mix four units across commodities (`metric tons`,
        `thousand metric tons`, `million metric tons`, `kilograms`). The
        UNIT_MEAS column is honored via `_production_unit_factor_to_kt`
        (which converts to kt) and then multiplied by 1000 to get plain
        tonnes for uniform downstream arithmetic. Prior to this patch the
        helper returned mixed-unit raw values, which silently inflated the
        `global_capacity_ratio` feature in `material_features.csv` by
        1000x for any commodity whose USGS UNIT_MEAS was kt (Lead, Aluminum,
        Copper, Cement, Zinc, Chromium, Manganese, Silicon, Boron, Steel,
        etc.) and 1e6x for million-tonne commodities (Iron ore). HHI
        consumers (`_build_production_hhi`) were unaffected because the
        per-country shares are unitless ratios that cancel.
    """
    sub = world_df[world_df["COMMODITY"] == commodity_name].copy()
    sub = sub[~sub["COUNTRY"].isin(["World total (rounded)", "Other Countries"])]
    sub["p_raw"] = sub["PROD_EST_2024"].fillna(sub["PROD_2023"])
    sub = sub.dropna(subset=["p_raw"])
    sub = sub[sub["p_raw"] > 0].copy()
    # UNIT_MEAS can vary per row even within one commodity (rare, but the
    # CSV doesn't guarantee uniformity). Apply per-row, then multiply by
    # 1000 to convert kt -> tonnes for the returned series.
    sub["p"] = sub.apply(
        lambda r: float(r["p_raw"])
        * _production_unit_factor_to_kt(r.get("UNIT_MEAS", ""))
        * 1000.0,
        axis=1,
    )
    return (
        sub[["COUNTRY", "p"]]
        .rename(columns={"COUNTRY": "country"})
        .reset_index(drop=True)
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Data loaders
# ═══════════════════════════════════════════════════════════════════════════════

def load_demand_data():
    """Load the MC demand output."""
    return pd.read_csv(DEMAND_FILE)


def load_nrel_data():
    """Load NREL StdScen24 capacity data (3 header rows to skip)."""
    return pd.read_csv(NREL_SCENARIOS_FILE, skiprows=3)



def load_risk_data():
    """
    Load supply-chain risk data from USGS MCS 2025 raw CSVs +
    Census Bureau International Trade API + OECD CRC 2026.

    Returns dict of DataFrames: aggregate, import_dependency, production,
    reserves, import_shares, crc.

    The legacy risk_charts_inputs.xlsx fallback was removed 2026-04-19:
    all sheets are now built from machine-readable raw sources. If the
    loader fails, the exception propagates rather than silently masking
    it with stale hand-compiled data.
    """
    return load_risk_data_mcs2025()


def load_usgs_2023_thin_film():
    """
    Load thin-film material data from MCS 2025 salient CSVs.
    (Cadmium, Gallium, Germanium, Indium, Selenium, Tellurium).

    Returns a DataFrame with columns: material, production_t, nir_pct
    (average across available years).

    The legacy MCS 2023 CSV fallback was removed 2026-04-19; thin-film
    data now comes exclusively from the MCS 2025 salient CSVs.
    """
    return load_thin_film_data_mcs2025()


# ═══════════════════════════════════════════════════════════════════════════════
# Risk / supply-chain helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _build_production_series(risk_data, thin_film_data):
    """
    Build a Series of average annual US production (tonnes) per demand-material
    name, combining:
      1. the aggregate sheet built by the USGS MCS 2025 loader (19 materials)
      2. USGS thin-film salient CSVs (6 byproduct metals)
    Note: values from the aggregate sheet are in kilotonnes; the caller in
    engineer_material_features converts to tonnes before use.
    """
    prod = pd.Series(dtype=float, name="production_t")

    # --- From aggregate sheet ---
    if risk_data is not None:
        agg = risk_data["aggregate"].copy()
        # Coerce non-numeric entries (e.g., '-----') to NaN
        for col in ["production", "import", "export", "consumption", "net_import"]:
            if col in agg.columns:
                agg[col] = pd.to_numeric(agg[col], errors="coerce")
        # Average production across years per material
        avg_prod = agg.groupby("material")["production"].mean()
        # Map risk material names → demand material names
        for demand_name, risk_name in DEMAND_TO_RISK.items():
            if risk_name in avg_prod.index and demand_name not in prod.index:
                prod[demand_name] = avg_prod[risk_name]

    # --- From thin-film CSVs ---
    if thin_film_data is not None and not thin_film_data.empty:
        for mat in thin_film_data.index:
            prod[mat] = thin_film_data.loc[mat, "production_t"]

    return prod


def _build_import_dependency_series(risk_data, thin_film_data):
    """
    Build a Series of import dependency (0-1 fraction) per demand-material name.

    Data sources (in priority order — REVISED 2026-04-08):
    1. import_dependency sheet: USGS-published Net Import Reliance values
       - This is what USGS Mineral Commodity Summaries reports as the
         official NIR. "E" = net exporter → 0.0.
    2. aggregate sheet: Calculate NIR = (imports - exports) / consumption
       - Used as a fallback when the published value is missing.
    3. USGS thin-film CSVs / MCS2025: Use nir_pct values for byproduct metals.

    Override path for stage-asymmetric materials:
       After computing both, if a published value is ≥ 0.5 AND the
       trade-balance value is ≤ 0.05, prefer the published value and emit
       a warning. This catches the REE class of bug where the US exports
       raw concentrate and imports refined product, so trade-balance NIR
       goes negative and gets clipped to 0 — but the published USGS value
       reflects the real (refined-product) dependency.

    Bug history (2026-04-08): Previously priority was reversed (aggregate
    first, published second). For Rare Earths, aggregate showed exports
    ~21–47 kt/yr and imports ~7–12 kt/yr (US ships Mountain Pass concentrate
    to China for separation, then imports refined REE oxides/metals/magnets),
    so NIR was negative → clipped to 0. The USGS-published value of ~97% was
    never consulted because the aggregate-derived value was always present.
    """
    dep = pd.Series(dtype=float, name="import_dependency")

    # Compute both methods first, then reconcile.
    nir_published = pd.Series(dtype=float)  # from import_dependency sheet
    nir_aggregate = pd.Series(dtype=float)  # from trade-balance aggregate sheet

    if risk_data is not None:
        # --- Method A: USGS-published NIR (import_dependency sheet) ---
        if "import_dependency" in risk_data:
            imp_df = risk_data["import_dependency"]
            year_cols = [c for c in imp_df.columns if c != "material"]
            for _, row in imp_df.iterrows():
                risk_name = row["material"]
                # Net-exporter ("E") and not-applicable ("--") years count as
                # 0 in the 5-year mean, matching
                # supply_chain_analysis._get_import_dependency so this feature
                # CSV (Fig 6 scatter) and the Fig 4/5 panels share ONE uniform
                # NIR convention: an EU CRM-style 5-year average with
                # net-exporter years set to 0. (Fixed 2026-06-06; previously
                # "E" years were dropped from the mean, which inflated volatile
                # materials, e.g. Tellurium 0.713 here vs 0.570 in the panels.
                # See DECISIONS.md.)
                vals = []
                for yc in year_cols:
                    sval = str(row[yc]).strip().upper()
                    if sval in ("E", "--"):
                        vals.append(0.0)
                    else:
                        try:
                            vals.append(float(row[yc]))
                        except (ValueError, TypeError):
                            pass
                avg = np.mean(vals) / 100.0 if vals else np.nan
                if pd.notna(avg):
                    nir_published[risk_name] = max(0.0, min(1.0, avg))

        # --- Method B: Trade-balance NIR (aggregate sheet) ---
        if "aggregate" in risk_data:
            agg = risk_data["aggregate"].copy()
            for col in ["import", "export", "consumption"]:
                if col in agg.columns:
                    agg[col] = pd.to_numeric(agg[col], errors="coerce")
            agg["nir"] = (agg["import"] - agg["export"]) / agg["consumption"]
            avg_nir_agg = agg.groupby("material")["nir"].mean()
            for risk_name in avg_nir_agg.index:
                nir_val = avg_nir_agg[risk_name]
                if pd.notna(nir_val):
                    nir_aggregate[risk_name] = max(0.0, min(1.0, nir_val))

        # --- Reconcile: published wins by default ---
        for demand_name, risk_name in DEMAND_TO_RISK.items():
            pub = nir_published.get(risk_name, np.nan)
            agg_v = nir_aggregate.get(risk_name, np.nan)

            # Stage-asymmetric guard: if published is high but aggregate is ~0,
            # the trade-balance formula was fooled by stage asymmetry. Prefer
            # published and warn.
            if pd.notna(pub) and pd.notna(agg_v):
                if pub >= 0.5 and agg_v <= 0.05:
                    print(
                        f"  NIR stage-asymmetry guard: {demand_name} "
                        f"(USGS published={pub:.2f}, trade-balance={agg_v:.2f}) "
                        f"— preferring published value"
                    )
                dep[demand_name] = pub
            elif pd.notna(pub):
                dep[demand_name] = pub
            elif pd.notna(agg_v):
                dep[demand_name] = agg_v

    # --- Method C: Thin-film materials from MCS 2025 / USGS 2023 CSVs ---
    if thin_film_data is not None and not thin_film_data.empty:
        for mat in thin_film_data.index:
            if mat not in dep.index:
                dep[mat] = thin_film_data.loc[mat, "nir_pct"]

    return dep


def _build_reserves_series(risk_data):
    """
    Build a Series of global reserves (kt) per demand-material name.
    Source: reserves sheet, row 0 = 'Global'.
    """
    res = pd.Series(dtype=float, name="reserves_kt")
    if risk_data is None:
        return res
    reserves_df = risk_data["reserves"]
    global_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]
    if global_row.empty:
        return res
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in global_row.columns:
            val = global_row[risk_name].values[0]
            try:
                res[demand_name] = float(val)
            except (ValueError, TypeError):
                pass
    return res


def _build_domestic_reserves_series(risk_data):
    """
    Build a Series of US domestic reserves (kt) per demand-material name.
    Source: reserves sheet, row where Unnamed: 0 == 'United States'.
    """
    res = pd.Series(dtype=float, name="domestic_reserves_kt")
    if risk_data is None:
        return res
    reserves_df = risk_data["reserves"]
    us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
    if us_row.empty:
        return res
    for demand_name, risk_name in DEMAND_TO_RISK.items():
        if risk_name in us_row.columns:
            val = us_row[risk_name].values[0]
            try:
                res[demand_name] = float(val)
            except (ValueError, TypeError):
                pass
    return res


def _build_reserves_by_crc(risk_data):
    """
    For each material, compute the fraction of global reserves held in
    high-risk countries (CRC 5-7 + China) vs. OECD/US.

    Returns dict of Series:
      - reserves_high_risk_frac: fraction in CRC 5-7 + China
      - reserves_oecd_frac: fraction in OECD + US
      - reserves_china_frac: fraction in China
    """
    out = {
        "reserves_high_risk_frac": pd.Series(dtype=float),
        "reserves_oecd_frac": pd.Series(dtype=float),
        "reserves_china_frac": pd.Series(dtype=float),
    }
    if risk_data is None:
        return out

    reserves_df = risk_data["reserves"]
    crc_map = risk_data["crc"].iloc[:, :2].copy()
    crc_map.columns = ["country", "crc"]

    # Get material columns (everything except Unnamed: 0)
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]

    # Skip special rows
    skip = {"Global", "United States", "Other"}
    country_res = reserves_df[~reserves_df["Unnamed: 0"].isin(skip)].copy()
    country_res = country_res.rename(columns={"Unnamed: 0": "country"})

    # Merge CRC
    country_res = country_res.merge(crc_map, on="country", how="left")
    country_res.loc[country_res["country"] == "China", "crc"] = "China"

    # Get global totals
    global_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]

    for risk_name in mat_cols:
        # Coerce to numeric
        country_res[risk_name] = pd.to_numeric(country_res[risk_name], errors="coerce").fillna(0)
        global_val = pd.to_numeric(global_row[risk_name].values[0], errors="coerce")
        if pd.isna(global_val) or global_val == 0:
            continue

        # Compute CRC group totals
        high_risk_crcs = {5, 6, 7, "China"}
        oecd_crcs = {"OECD"}

        high_risk = country_res[country_res["crc"].isin(high_risk_crcs)][risk_name].sum()
        oecd = country_res[country_res["crc"].isin(oecd_crcs)][risk_name].sum()
        # Add US to OECD
        us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
        us_val = pd.to_numeric(us_row[risk_name].values[0], errors="coerce")
        if not pd.isna(us_val):
            oecd += us_val
        china = country_res[country_res["crc"] == "China"][risk_name].sum()

        # Map to demand names
        for demand_name, rn in DEMAND_TO_RISK.items():
            if rn == risk_name:
                out["reserves_high_risk_frac"][demand_name] = high_risk / global_val
                out["reserves_oecd_frac"][demand_name] = oecd / global_val
                out["reserves_china_frac"][demand_name] = china / global_val

    return out


def _build_hhi_wgi(risk_data):
    """
    Governance-weighted HHI (HHI_WGI) per material, following the EU Critical
    Raw Materials methodology (Blengini et al., 2017; Schrijvers et al., 2020).

    Formula: HHI_WGI = SUM(s_i^2 * risk_i)
    where s_i = normalized import share (fraction summing to 1),
          risk_i = country governance risk score (0-1, higher = riskier).

    Uses OECD Country Risk Classification (CRC) as governance proxy,
    rescaled to 0-1: CRC_scaled = CRC_raw / 7.

    CRC mapping:
      United States → 0.0 (domestic, no geopolitical risk)
      OECD (high income) → 0.0
      CRC 1 → 1/7, CRC 2 → 2/7, ..., CRC 7 → 1.0
      China → 1.0 (treated as CRC 7 per OECD classification)
      Undefined → 5/7 (~0.71, conservative assumption)

    Returns Series indexed by risk material name (0-1 scale).
    Higher = more concentrated AND riskier supply sources.

    References:
      - Blengini et al. (2017) JRC Technical Report JRC106997
      - Schrijvers et al. (2020) Resour. Conserv. Recycl. 155:104617
      - Graedel et al. (2012) ES&T 46:1063-1070
    """
    hhi_wgi = pd.Series(dtype=float, name="hhi_wgi")
    if risk_data is None:
        return hhi_wgi

    # CRC → 0-1 governance risk score (higher = riskier)
    crc_risk_scores = {
        "United States": 0.0, "OECD": 0.0,
        1: 1/7, 2: 2/7, 3: 3/7, 4: 4/7, 5: 5/7, 6: 6/7, 7: 1.0,
        "China": 1.0, "Undefined": 5/7,
    }

    imp_shares = risk_data["import_shares"]  # material, country, share
    crc_map = risk_data["crc"].iloc[:, :2]
    crc_map.columns = ["country", "crc"]

    merged = imp_shares.merge(crc_map, on="country", how="left")
    merged.loc[merged["country"] == "China", "crc"] = "China"

    for mat, grp in merged.groupby("material"):
        total_share = grp["share"].sum()
        if total_share == 0:
            continue
        # Normalize shares to fractions summing to 1
        shares = grp["share"] / total_share
        # Look up governance risk score per country
        risk_scores = grp["crc"].map(
            lambda c: crc_risk_scores.get(c, 5/7)
        )
        # HHI_WGI = SUM(s_i^2 * risk_i)
        hhi_wgi[mat] = (shares ** 2 * risk_scores).sum()

    return hhi_wgi


def _build_mining_hhi(risk_data):
    """
    Mining-stage production HHI per material.

    Formula: HHI = SUM(s_i^2) where s_i = country share of global mine
    production. Scale: 0-1 (0 = perfectly distributed, 1 = single-country
    monopoly).

    Standard metric in criticality literature:
      - Graedel et al. (2012) ES&T 46:1063-1070 — original formulation
      - EU CRM 2023 methodology — uses both mining and processing stages

    For REE: USGS reports a single "Rare earths" world production row, so
    every individual REE element inherits the basket HHI (~0.50) at the
    mining stage. The binding supply concentration for HREE (Dy, Tb, Y, Gd)
    is actually at the **processing** stage — see _build_processing_hhi
    below and the orchestrating _build_production_hhi.

    Data source priority (revised 2026-04-08 to fix the missing-byproduct
    "Bug 2" described in the module-level MCS 2025 World CSV comment):

    1. PRIMARY: existing risk_data["production"] sheet (built by the USGS
       MCS 2025 raw-CSV loader)
    2. FALLBACK: USGS MCS 2025 World Data Release CSV
       (data/usgs_mcs_2025/world_data/MCS2025_World_Data.csv;
        https://doi.org/10.5066/P13XCP3R) for materials missing from #1
    3. HARDCODED: literature/qualitative values for materials USGS does
       not publish per-country production data for (germanium, fiberglass,
       — see PUBLISHED_PRODUCTION_HHI dict at module top, and
       EXCLUDED_FROM_CRITICALITY for bulk materials with no published source)

    Returns Series indexed by risk material name (or demand material name
    for entries pulled from the MCS 2025 World CSV / hardcoded dict).
    """
    hhi = pd.Series(dtype=float, name="production_hhi")

    # ── Primary: existing production sheet (risk material names) ──
    # Per-country HHI must exclude the "World total (rounded)" sentinel row
    # (the USGS-rounded global sum) and "Other Countries" residual. Including
    # the world-total row was a double-count bug that halved every share and
    # inserted a spurious 0.5^2 = 0.25 anchor term in every HHI (surfaced
    # 2026-05-21 alongside the global_capacity_ratio unit bug; see
    # [[project_global_production_unit_bug]] for the joint write-up).
    SENTINEL_ROWS_HHI = {"World total (rounded)", "Global", "World total",
                          "Other Countries", "Other"}
    if risk_data is not None:
        prod_df = risk_data.get("production")
        if prod_df is not None and not prod_df.empty:
            for mat, grp in prod_df.groupby("material"):
                prod_col = None
                for col in ["production_2024", "production_2023", "production"]:
                    if col in grp.columns:
                        prod_col = col
                        break
                if prod_col is None:
                    num_cols = grp.select_dtypes(include="number").columns
                    if len(num_cols) > 0:
                        prod_col = num_cols[0]
                    else:
                        continue
                country_col = "country" if "country" in grp.columns else None
                if country_col is not None:
                    grp = grp[~grp[country_col].isin(SENTINEL_ROWS_HHI)]
                vals = pd.to_numeric(grp[prod_col], errors="coerce").fillna(0)
                total = vals.sum()
                if total <= 0:
                    continue
                shares = vals / total
                hhi[mat] = (shares ** 2).sum()

    # ── Fallback: MCS 2025 World CSV (demand material names) ──
    world_df = load_mcs2025_world_data()
    if not world_df.empty:
        for demand_name, mcs_commodity in MCS2025_COMMODITY_MAP.items():
            # Skip if we already have a value via the primary source
            if demand_name in hhi.index:
                continue
            # Skip REE individual elements that already mapped to "Rare Earths"
            # in the primary source via DEMAND_TO_RISK
            risk_name = DEMAND_TO_RISK.get(demand_name)
            if risk_name and risk_name in hhi.index:
                continue

            sub = _mcs2025_country_production(world_df, mcs_commodity)
            if sub.empty:
                continue
            shares = sub["p"] / sub["p"].sum()
            hhi[demand_name] = (shares ** 2).sum()
            print(
                f"  mining_hhi (MCS 2025 World CSV): {demand_name} = "
                f"{hhi[demand_name]:.3f} from {len(sub)} countries "
                f"(top: {sub.loc[shares.idxmax(), 'country']} {shares.max()*100:.0f}%)"
            )

    # ── Hardcoded: materials USGS does not publish per-country data for ──
    for demand_name, (value, source) in PUBLISHED_PRODUCTION_HHI.items():
        if demand_name not in hhi.index:
            hhi[demand_name] = value
            print(f"  mining_hhi (published): {demand_name} = {value:.3f} — {source}")

    return hhi


def _build_processing_hhi(all_materials):
    """
    Processing-stage (separation / refining / metal-reduction) HHI per
    material.

    Currently only the REE elements have a published per-element processing-
    stage country breakdown (EU CRM 2023 Annex 7). Other materials return
    NaN here and the orchestrator (_build_production_hhi) falls back to
    the mining-stage HHI for those — i.e. processing stage is only
    surfaced when we have explicit refined-output country shares.

    Stages-asymmetry rationale (see module-level constants block):
      - LREE (Nd, Pr): China 84.9%, Malaysia 10.5%, Russia 1.9%, India 1.6%,
        Vietnam 1.0%, Norway 0.1% → HHI ≈ 0.733
      - HREE (Dy, Tb, Y, Gd): China 100% → HHI = 1.000
      - Mining-stage HHI for these is the basket aggregate (~0.50), so the
        processing stage governs.

    Implementation note: the EU CRM data is also persisted in
    data/eu_crm_2023_re_element_data.csv (locked 2026-05-26) for downstream
    audit, but that file only carries `processing_china_share`. The full
    per-country distribution lives in module-level LREE/HREE_PROCESSING_SHARES
    so the HHI is reproducible from peer-reviewed values without a hidden
    "everything else" residual assumption.

    Parameters
    ----------
    all_materials : Iterable[str]
        Demand-material list (used only to fix the output index — values
        are looked up by element name).

    Returns
    -------
    pd.Series
        Indexed by demand material name. NaN where no published
        processing-stage data exists (i.e. non-REE materials today).
    """
    out = pd.Series(dtype=float, name="production_hhi_processing")
    lree_hhi = sum(s ** 2 for s in LREE_PROCESSING_SHARES.values())
    hree_hhi = sum(s ** 2 for s in HREE_PROCESSING_SHARES.values())
    for mat in all_materials:
        cls = REE_PROCESSING_CLASS.get(mat)
        if cls == "LREE":
            out[mat] = lree_hhi
        elif cls == "HREE":
            out[mat] = hree_hhi
        # else: NaN — no published processing-stage shares for this material
    return out


def _build_production_hhi(risk_data, all_materials):
    """
    Production HHI orchestrator: returns the more concentrated of mining
    and processing stages per material, plus both stage components for
    transparency.

    Following EU CRM 2023 methodology (Grohol & Veeh, doi:10.2873/725585),
    HHI is computed at both mining and processing stages, and the higher
    (more concentrated) stage is treated as the binding supply-risk
    indicator. For most materials this equals the mining-stage HHI; for
    the REE elements (and any future stage-asymmetric materials with
    published refined-output shares) the processing stage dominates.

    Why this matters for REE specifically: USGS publishes only an aggregate
    "Rare earths" world production row, so per-element mining HHI is the
    basket value (~0.50). The actual chokepoint for Nd/Pr/Dy/Tb/Y/Gd
    supply is separation/refining, which is 85–100% Chinese (EU CRM 2023
    Annex 7). Using mining-stage HHI alone for these elements understates
    supply concentration by ~2x and contradicts what the Fig 5 producer-
    mix panel shows visually. The stage-resolved approach reconciles the
    feature CSV with what Fig 4/5 already display.

    Returns
    -------
    pd.DataFrame
        Indexed by material name, with three columns:
          - production_hhi_mining     : mining-stage HHI (USGS MCS)
          - production_hhi_processing : processing-stage HHI (EU CRM 2023)
          - production_hhi            : max of the two stages (NaN-safe)
    """
    mining = _build_mining_hhi(risk_data)
    processing = _build_processing_hhi(all_materials)

    # Materials in `mining` are a mix of risk-names and demand-names; the
    # mapping to demand-names happens in engineer_material_features via
    # _map_series_to_demand_names. We return the raw series here and let
    # the caller do the mapping for both stages symmetrically.
    return mining, processing


def _build_global_production_series(risk_data):
    """
    Build a Series of global annual production (TONNES) per demand-material
    name, summing per-country production from the USGS MCS production sheet.

    Data source priority (revised 2026-04-08):
    1. PRIMARY: existing risk_data["production"] sheet
       (built by `_build_production_sheet` in usgs_mcs2025_loader.py;
       values stored in KILOTONS, converted to tonnes here).
    2. FALLBACK: USGS MCS 2025 World Data Release CSV for materials missing
       from the primary source (https://doi.org/10.5066/P13XCP3R)
       (resolved via `_mcs2025_country_production`, which since 2026-05-21
       applies UNIT_MEAS conversion and returns tonnes).

    Unit history (2026-05-21): before this revision the primary path returned
    kt and the fallback returned mixed units; the consumer (`global_capacity_ratio`)
    silently divided cumulative tonnes of US demand by these inconsistent
    denominators, inflating the ratio by 1000x for most bulk commodities.
    Both paths now return tonnes. See [[project_global_production_unit_bug]].

    Note: a few materials (germanium, fiberglass, glass) have no per-country
    production data anywhere — they will simply be absent from this Series,
    which causes their global_capacity_ratio to be 0 in the consumer
    (acceptable: these materials are characterized by other features).

    Returns Series indexed by demand material name, values in TONNES.
    """
    glob = pd.Series(dtype=float, name="global_production_t")

    # ── Primary: existing production sheet (values in kt; convert to tonnes) ──
    # Aggregation logic mirrors `get_global_kt` in supply_tiers_shared.py:
    #   1. Prefer the explicit "World total (rounded)" row when present —
    #      this is USGS's authoritative rounded world total and avoids any
    #      rounding artifacts from summing per-country.
    #   2. Otherwise, sum the per-country rows including "Other Countries"
    #      (residual category) but excluding the world-total and global
    #      sentinel rows.
    # The prior implementation summed ALL rows including "World total
    # (rounded)", double-counting global production (cross-check vs USGS
    # published showed ~2x on every commodity; surfaced 2026-05-21).
    SENTINEL_ROWS = {"World total (rounded)", "Global", "World total"}
    if risk_data is not None and "production" in risk_data:
        prod_df = risk_data["production"]
        if prod_df is not None and not prod_df.empty:
            for mat, grp in prod_df.groupby("material"):
                prod_col = None
                for col in ["production_2024", "production_2023", "production"]:
                    if col in grp.columns:
                        prod_col = col
                        break
                if prod_col is None:
                    num_cols = grp.select_dtypes(include="number").columns
                    if len(num_cols) == 0:
                        continue
                    prod_col = num_cols[0]
                country_col = "country" if "country" in grp.columns else None

                total_kt = None
                if country_col is not None:
                    # 1. Prefer USGS-rounded world total
                    wt = grp[grp[country_col] == "World total (rounded)"]
                    if not wt.empty:
                        val = pd.to_numeric(wt[prod_col].iloc[0], errors="coerce")
                        if pd.notna(val) and val > 0:
                            total_kt = float(val)
                    # 2. Fallback: sum per-country (incl. Other Countries)
                    if total_kt is None:
                        per_country = grp[~grp[country_col].isin(SENTINEL_ROWS)]
                        vals = pd.to_numeric(per_country[prod_col], errors="coerce").fillna(0)
                        s = float(vals.sum())
                        if s > 0:
                            total_kt = s
                else:
                    # No country column (legacy schema?); sum naively
                    vals = pd.to_numeric(grp[prod_col], errors="coerce").fillna(0)
                    s = float(vals.sum())
                    if s > 0:
                        total_kt = s

                if total_kt and total_kt > 0:
                    total_t = total_kt * 1000.0
                    for demand_name, risk_name in DEMAND_TO_RISK.items():
                        if risk_name == mat and demand_name not in glob.index:
                            glob[demand_name] = total_t

    # ── Fallback: MCS 2025 World CSV ──
    world_df = load_mcs2025_world_data()
    if not world_df.empty:
        for demand_name, mcs_commodity in MCS2025_COMMODITY_MAP.items():
            if demand_name in glob.index:
                continue  # already from primary source
            sub = _mcs2025_country_production(world_df, mcs_commodity)
            if sub.empty:
                continue
            glob[demand_name] = float(sub["p"].sum())

    return glob


def _build_crc_sourcing_breakdown(risk_data):
    """
    For each material, compute import sourcing fractions by CRC group:
      - import_china_frac: % of imports from China
      - import_high_risk_frac: % from CRC 5-7 + China
      - import_oecd_frac: % from OECD countries
      - import_hhi: Herfindahl-Hirschman Index of import concentration

    Returns dict of Series indexed by risk material name.
    """
    out = {
        "import_china_frac": pd.Series(dtype=float),
        "import_high_risk_frac": pd.Series(dtype=float),
        "import_oecd_frac": pd.Series(dtype=float),
        "import_hhi": pd.Series(dtype=float),
    }
    if risk_data is None:
        return out

    imp_shares = risk_data["import_shares"]
    crc_map = risk_data["crc"].iloc[:, :2].copy()
    crc_map.columns = ["country", "crc"]

    merged = imp_shares.merge(crc_map, on="country", how="left")
    merged.loc[merged["country"] == "China", "crc"] = "China"

    high_risk_crcs = {5, 6, 7, "China"}
    oecd_crcs = {"OECD"}

    for mat, grp in merged.groupby("material"):
        total = grp["share"].sum()
        if total == 0:
            continue
        shares = grp["share"] / total  # normalize to fractions

        china = grp.loc[grp["crc"] == "China", "share"].sum() / total
        high_risk = grp.loc[grp["crc"].isin(high_risk_crcs), "share"].sum() / total
        oecd = grp.loc[grp["crc"].isin(oecd_crcs), "share"].sum() / total
        hhi = (shares ** 2).sum()  # 0-1 scale, higher = more concentrated

        out["import_china_frac"][mat] = china
        out["import_high_risk_frac"][mat] = high_risk
        out["import_oecd_frac"][mat] = oecd
        out["import_hhi"][mat] = hhi

    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario features
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_scenario_features(demand, nrel, risk_data=None, thin_film_data=None):
    """
    Construct features for each of the 61 scenarios.

    All features are dimensionless (percentages, fractions, or indices).
    No absolute demand values (tonnes) are used as features; they are
    incommensurable across materials.

    Feature reduction:
      - Storage capacity share dropped (compositional with solar/wind,
        smallest driver, no independent signal).
      - All scenario-level supply-chain metrics dropped (cannot be grounded
        without arbitrary cross-material weighting). Supply-chain analysis
        lives entirely at the material level.

    Parameters
    ----------
    demand : DataFrame
        MC demand output (scenario, year, material, mean, std, p2, p5, ..., p95, p97).
    nrel : DataFrame
        NREL StdScen24 capacity data.
    risk_data : dict, optional
        Risk sheets from the USGS MCS 2025 loader (unused here; the scenario
        features are demand- and capacity-only).
    thin_film_data : DataFrame, optional
        USGS thin-film salient data (unused here).

    Returns
    -------
    DataFrame with scenario index and numeric feature columns.
    """
    years = sorted(demand["year"].unique())

    # Use first year with nonzero demand as reference (2026 is zero in stock-flow model
    # because it's the starting year — demand begins at first interval, typically 2029)
    total_by_year = demand.groupby("year")["mean"].sum()
    nonzero_years = [y for y in years if total_by_year.get(y, 0) > 0]
    base_year = nonzero_years[0] if nonzero_years else years[1]

    feats = pd.DataFrame(index=sorted(demand["scenario"].unique()))
    feats.index.name = "scenario"

    # ── Demand growth features (dimensionless, annualized %) ─────────────
    # Computed per-material then averaged across materials.
    # Uses CAGR: ((demand_end / demand_start)^(1/n_years) - 1) × 100
    # Ref: Standard compound annual growth rate used in IEA (2024),
    #      Graedel et al. (2015) for characterizing demand trajectories.

    short_end = 2035   # policy target year (IRA, state RPS)
    long_end = years[-1]  # 2050
    n_short = short_end - base_year
    n_long = long_end - base_year

    # Per-material, per-scenario demand at key years
    demand_by_smy = demand.pivot_table(
        index=["scenario", "material"], columns="year", values="mean"
    ).fillna(0)

    # Annualized % growth (CAGR) — per material, then mean across materials
    def _cagr(start, end, n):
        """Compound annual growth rate, handling zero/negative starts."""
        ratio = end / (start + 1e-12)
        # Clamp extreme ratios to avoid overflow
        ratio = np.clip(ratio, 0.01, 1000)
        return (ratio ** (1.0 / n) - 1) * 100

    if base_year in demand_by_smy.columns and short_end in demand_by_smy.columns:
        cagr_short = _cagr(
            demand_by_smy[base_year], demand_by_smy[short_end], n_short
        )
        feats["growth_rate_short_pct"] = cagr_short.groupby("scenario").mean()
    else:
        feats["growth_rate_short_pct"] = 0.0

    if base_year in demand_by_smy.columns and long_end in demand_by_smy.columns:
        cagr_long = _cagr(
            demand_by_smy[base_year], demand_by_smy[long_end], n_long
        )
        feats["growth_rate_long_pct"] = cagr_long.groupby("scenario").mean()
    else:
        feats["growth_rate_long_pct"] = 0.0

    # Peak annual growth rate (max year-over-year % change)
    # Per material: max of ((demand_t+1 - demand_t) / demand_t × 100) across years
    # Start from base_year (first nonzero year) to avoid 0→nonzero explosion
    avail_years = sorted([y for y in years if y in demand_by_smy.columns and y >= base_year])
    if len(avail_years) >= 2:
        yoy_pct_changes = []
        for i in range(len(avail_years) - 1):
            y0, y1 = avail_years[i], avail_years[i + 1]
            denom = demand_by_smy[y0].clip(lower=1.0)  # floor at 1 to avoid div-by-zero
            pct_chg = (demand_by_smy[y1] - demand_by_smy[y0]) / denom * 100
            pct_chg.name = y1
            yoy_pct_changes.append(pct_chg)
        yoy_df = pd.concat(yoy_pct_changes, axis=1)

        # Short-term: years up to 2035
        short_cols = [y for y in yoy_df.columns if y <= short_end]
        if short_cols:
            feats["peak_annual_growth_short_pct"] = (
                yoy_df[short_cols].max(axis=1).groupby("scenario").mean()
            )
        else:
            feats["peak_annual_growth_short_pct"] = 0.0

        # peak_annual_growth_long_pct DROPPED 2026-04-08:
        # redundant with growth_long + peak_short, and VIF confirms (highly
        # collinear with peak_short).
    else:
        feats["peak_annual_growth_short_pct"] = 0.0

    # ── Uncertainty features (dimensionless) ─────────────────────────────

    # Mean CV across materials for this scenario
    cv_by_sm = demand.copy()
    cv_by_sm["cv"] = cv_by_sm["std"] / (cv_by_sm["mean"] + 1e-12)
    feats["mean_cv"] = cv_by_sm.groupby("scenario")["cv"].mean()

    # mean_ci_width DROPPED 2026-04-08 (VIF receipt):
    # r=0.997 with mean_cv at scenario level — they measure the same dispersion.

    # ── Technology mix fractions (dimensionless, 0-1) ────────────────────

    # Storage share dropped: smallest contributor and compositional with
    # solar+wind (residual = storage+other), so it adds no independent
    # information as a scenario feature.
    mw_cols = [c for c in nrel.columns if c.endswith("_MW")]
    solar_cols = [c for c in mw_cols if "pv" in c or "csp" in c]
    wind_cols = [c for c in mw_cols if "wind" in c]

    # 2035 capacity shares
    nrel_2035 = nrel[nrel["t"] == 2035].set_index("scenario")
    total_cap_2035 = nrel_2035[mw_cols].sum(axis=1)
    feats["solar_fraction_2035"] = nrel_2035[solar_cols].sum(axis=1) / (total_cap_2035 + 1)
    feats["wind_fraction_2035"] = nrel_2035[wind_cols].sum(axis=1) / (total_cap_2035 + 1)

    # 2050 capacity shares
    nrel_2050 = nrel[nrel["t"] == 2050].set_index("scenario")
    total_cap_2050 = nrel_2050[mw_cols].sum(axis=1)
    feats["solar_fraction_2050"] = nrel_2050[solar_cols].sum(axis=1) / (total_cap_2050 + 1)
    feats["wind_fraction_2050"] = nrel_2050[wind_cols].sum(axis=1) / (total_cap_2050 + 1)

    # ── Scenario-level supply-chain features: DROPPED ────────────────────
    # Removed: supply_chain_stress, peak_supply_chain_stress,
    #          mean_n_exceeding_production, peak_n_exceeding_production.
    # Rationale: demand-weighted CRC×import_dep at the scenario level cannot
    # be properly grounded; aggregating risk across incommensurable materials
    # requires either price weighting (which collapses under the very demand
    # shock we are modeling) or arbitrary weights. Material-level supply-chain
    # metrics remain in engineer_material_features() where the scope is clean.

    print(f"  Scenario features: {len(feats.columns)} total (all dimensionless)")

    feats = feats.fillna(0)
    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# Material features
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_material_features(demand, risk_data, thin_film_data):
    """
    Construct features for each of the 31 materials, integrating
    risk/supply-chain data from USGS MCS 2025 and OECD CRC 2026.

    All features are dimensionless (percentages, fractions, ratios, or indices).
    Absolute demand values (tonnes) are computed only as intermediates for
    ratio calculations and are NOT included as final features.

    Supply chain metrics follow established methodologies:
    - Net Import Reliance (NIR): USGS Mineral Commodity Summaries methodology
    - HHI: Herfindahl-Hirschman Index per Graedel et al. (2012)
    - CRC-weighted risk: OECD Country Risk Classification

    Feature reduction: 23 → 15 features.
      Dropped (redundant or unground­ed):
        peak_annual_growth_short_pct, peak_annual_growth_long_pct,
        mean_ci_width, mean_capacity_ratio, max_capacity_ratio,
        exceedance_frequency, reserve_consumption_pct, reserves_oecd_frac,
        import_high_risk_frac, import_oecd_frac.
      Added: us_capacity_ratio, global_capacity_ratio (clean US/global split,
      replacing the three overlapping demand-vs-production metrics).

    Returns
    -------
    DataFrame with material index and numeric feature columns.
    """
    years = sorted(demand["year"].unique())
    # Use first year with nonzero demand as reference (see scenario features comment)
    total_by_year = demand.groupby("year")["mean"].sum()
    nonzero_years = [y for y in years if total_by_year.get(y, 0) > 0]
    base_year = nonzero_years[0] if nonzero_years else years[1]
    all_materials = sorted(demand["material"].unique())

    # Build supply-chain series from risk data
    us_production = _build_production_series(risk_data, thin_film_data)
    import_dep = _build_import_dependency_series(risk_data, thin_film_data)
    global_reserves = _build_reserves_series(risk_data)
    domestic_reserves = _build_domestic_reserves_series(risk_data)
    crc_risk = _build_hhi_wgi(risk_data)
    reserves_crc = _build_reserves_by_crc(risk_data)
    sourcing = _build_crc_sourcing_breakdown(risk_data)
    mining_hhi, processing_hhi = _build_production_hhi(risk_data, all_materials)

    # Map risk-material-keyed Series onto the demand-material namespace
    # using the shared helper (2026-04-17 refactor of the 2026-04-08 fix).
    # This catches the full REE-mapping bug class: Dysprosium/Neodymium/
    # Praseodymium/Terbium/Gadium all map to "Rare Earths" and were
    # silently dropped by reindex() → fillna(0) for import_hhi,
    # import_china_frac, hhi_wgi, and the reserves-by-CRC fractions.
    crc_mapped         = _map_series_to_demand_names(crc_risk,        all_materials)
    mining_hhi_mapped  = _map_series_to_demand_names(mining_hhi,      all_materials)
    # processing_hhi is already keyed by demand-material name (only REE
    # elements have values; non-REE materials are NaN by construction).
    import_china_mapped = _map_series_to_demand_names(
        sourcing["import_china_frac"],     all_materials
    )
    import_hhi_mapped = _map_series_to_demand_names(
        sourcing["import_hhi"],            all_materials
    )
    reserves_high_risk_mapped = _map_series_to_demand_names(
        reserves_crc["reserves_high_risk_frac"], all_materials
    )
    reserves_china_mapped = _map_series_to_demand_names(
        reserves_crc["reserves_china_frac"],     all_materials
    )

    feats = pd.DataFrame(index=all_materials)
    feats.index.name = "material"

    # ── Demand growth features (dimensionless, annualized %) ─────────────
    # Use annualized % growth from baseline, not absolute demand in tonnes.

    short_end = 2035
    long_end = years[-1]  # 2050
    n_short = short_end - base_year
    n_long = long_end - base_year

    # Per-material demand at key years (mean across scenarios)
    mat_year = demand.groupby(["material", "year"])["mean"].mean().unstack(fill_value=0)

    def _cagr(start, end, n):
        """Compound annual growth rate, handling zero/negative starts."""
        ratio = end / (start + 1e-12)
        ratio = np.clip(ratio, 0.01, 1000)
        return (ratio ** (1.0 / n) - 1) * 100

    # growth_rate_short_pct DROPPED at material level 2026-04-08 (VIF receipt):
    # r=0.90 with scenario_cv (small-base materials drive both up). The
    # short vs. long horizon split is a *policy* distinction (2035 IRA/RPS
    # targets), which is only meaningful at the scenario level. At the
    # material level, the long-horizon CAGR alone characterizes the trend.

    # Annualized demand growth, base→2050 (%)
    if base_year in mat_year.columns and long_end in mat_year.columns:
        feats["growth_rate_long_pct"] = _cagr(
            mat_year[base_year], mat_year[long_end], n_long
        )
    else:
        feats["growth_rate_long_pct"] = 0.0

    # Peak annual growth (short/long) DROPPED 2026-04-08:
    # CAGR (growth_rate_short_pct, growth_rate_long_pct) already characterizes
    # material-level growth trajectories. Peak YoY adds a redundant signal that
    # is dominated by single-year noise rather than trend.

    # ── Uncertainty features (dimensionless) ─────────────────────────────

    # 5. Scenario CV: std of scenario-level total demands / mean
    scen_totals = (
        demand.groupby(["material", "scenario"])["mean"]
        .sum()
        .reset_index()
    )
    scen_cv = scen_totals.groupby("material")["mean"].agg(
        lambda x: x.std() / (x.mean() + 1e-12)
    )
    feats["scenario_cv"] = scen_cv

    # mean_ci_width DROPPED 2026-04-08: redundant with scenario_cv (r≈0.99 at
    # scenario level; same dispersion-relative-to-mean construct).

    # demand_volatility_cv DROPPED 2026-04-08 (VIF receipt):
    # r=0.86 with growth_rate_short_pct, r=0.71 with scenario_cv. Materials
    # with rapid growth from a small base have inherently large temporal CV,
    # so this re-encodes the growth signal. CAGR features carry the trend;
    # scenario_cv carries the cross-scenario uncertainty. year_pivot is still
    # built below as an intermediate for cumulative-demand calculations.
    year_pivot = demand.pivot_table(
        index=["material", "scenario"], columns="year", values="mean"
    ).fillna(0)

    # ── Supply-chain / risk features (dimensionless) ─────────────────────
    # Following USGS Mineral Commodity Summaries methodology for NIR,
    # and Graedel et al. (2012) for HHI and supply risk framework.
    #
    # Fill policy (2026-04-17): previously every supply-chain feature used
    # fillna() with an arbitrary default (NIR=1.0 worst-case, HHI_WGI=0.5
    # moderate, fractions=0). For materials with no published peer-reviewed
    # source, specifically those in EXCLUDED_FROM_CRITICALITY (Fiberglass,
    # Glass), these defaults were fabricating numeric criticality signals
    # from a data vacuum. We now propagate NaN for excluded materials and
    # drop them before supply-risk plotting.

    # 8. Import dependency / Net Import Reliance (0–1)
    #    NIR = (imports - exports) / consumption, per USGS methodology
    feats["import_dependency"] = import_dep.reindex(feats.index)

    # 9. HHI_WGI: governance-weighted import concentration (0–1 scale)
    #    HHI_WGI = SUM(s_i^2 * CRC_scaled_i), per EU CRM methodology
    #    Ref: Blengini et al. (2017), Schrijvers et al. (2020)
    feats["hhi_wgi"] = crc_mapped.reindex(feats.index)

    # Capacity ratio: annual-mean US demand / annual production (dimensionless)
    # Replaces the three overlapping ratios (mean_capacity_ratio,
    # max_capacity_ratio, exceedance_frequency) with a clean US/global split.
    #
    # Two unit fixes 2026-05-21:
    #   1. The numerator was previously `scen_totals.groupby("material")["mean"].mean()`,
    #      where `scen_totals` is the year-summed demand per (material, scenario).
    #      That construction is the *mean cumulative 25-year demand* — not an
    #      annual mean despite the variable name — so the resulting ratio had
    #      implicit units of years rather than being dimensionless. Replaced
    #      with the true annual mean (averaged over years and scenarios).
    #   2. `us_production` is reported in kilotonnes by `_build_production_series`
    #      (sourced from the aggregate sheet, which the MCS 2025 loader stores in
    #      kt). The demand numerator is in tonnes. Converting to tonnes here
    #      makes both ratios truly dimensionless.
    #
    # `global_production` is now in tonnes after the 2026-05-21 patch to
    # `_build_global_production_series`. See [[project_global_production_unit_bug]].
    mat_annual_mean = demand.groupby(["material", "year"])["mean"].mean() \
                            .groupby("material").mean()
    us_production_t = us_production * 1000.0  # kt -> tonnes
    global_production = _build_global_production_series(risk_data)

    # 8. US capacity ratio: annual-mean US demand / annual US production
    feats["us_capacity_ratio"] = (
        mat_annual_mean / us_production_t
    ).reindex(feats.index).fillna(0)

    # 9. Global capacity ratio: annual-mean US demand / annual global production
    #    Captures whether US demand alone strains the global supply base.
    feats["global_capacity_ratio"] = (
        mat_annual_mean / global_production
    ).reindex(feats.index).fillna(0)

    # ── Reserve Adequacy Features ────────────────────────────────────────
    # Cumulative demand is computed as an intermediate only (not a feature).
    # Reserve metrics use the static reserve-to-production ratio approach
    # common in USGS reporting and Graedel et al. (2012).

    # Intermediate: cumulative demand 2026-2050 (median across scenarios)
    scenario_cumulative = year_pivot.sum(axis=1).groupby("material").median()
    _cumulative_demand = scenario_cumulative.reindex(feats.index).fillna(0)

    # reserve_consumption_pct DROPPED 2026-04-08:
    # Mathematical inverse of global_reserve_coverage — keeping both is
    # redundant. Coverage form is more interpretable (years/multiples).
    global_res = global_reserves.reindex(feats.index).fillna(0)

    # Reserve coverage: redesigned 2026-04-08 after VIF receipt showed
    # domestic_reserve_coverage ↔ global_reserve_coverage at r=1.000.
    # Both shared the cumulative_demand denominator, so cross-material
    # variance was driven entirely by 1/demand and the two collapsed onto
    # the same axis. Replaced with two independent constructs:
    #
    #   global_reserve_coverage = global_reserves / cumulative_US_demand
    #     (the reserve-adequacy story; demand-dependent)
    #   domestic_reserve_share  = US_reserves / global_reserves
    #     (the geographic-self-sufficiency story; reserves-only, no demand)
    #
    # These now answer two distinct questions and are not collinear.
    domestic_res = domestic_reserves.reindex(feats.index).fillna(0)

    feats["global_reserve_coverage"] = (
        global_res / (_cumulative_demand + 1e-12)
    )
    feats.loc[_cumulative_demand == 0, "global_reserve_coverage"] = 0

    feats["domestic_reserve_share"] = domestic_res / (global_res + 1e-12)
    feats.loc[global_res == 0, "domestic_reserve_share"] = 0
    feats["domestic_reserve_share"] = feats["domestic_reserve_share"].clip(0, 1)

    # ── Reserves-by-CRC features (geographic risk of reserves) ───────────

    # 16. Fraction of global reserves in high-risk countries (CRC 5-7 + China)
    feats["reserves_high_risk_frac"] = reserves_high_risk_mapped.reindex(feats.index)

    # reserves_oecd_frac DROPPED 2026-04-08:
    # Compositional with reserves_china_frac and reserves_high_risk_frac
    # (the three sum to ~1), so it carries no independent signal.

    # Fraction of global reserves in China
    feats["reserves_china_frac"] = reserves_china_mapped.reindex(feats.index)

    # ── Import sourcing features ─────────────────────────────────────────

    # 19. Fraction of US imports from China
    feats["import_china_frac"] = import_china_mapped.reindex(feats.index)

    # import_high_risk_frac and import_oecd_frac DROPPED 2026-04-08:
    # Both overlap heavily with import_china_frac and import_hhi (China is the
    # dominant high-risk source for most materials). Keeping the China share +
    # the diversification HHI gives independent, lit-grounded signals.

    # Import source concentration — HHI (0-1 scale)
    #     HHI = sum(share_i^2), per Graedel et al. (2012)
    #     0 = perfectly diversified, 1 = single-source monopoly
    feats["import_hhi"] = import_hhi_mapped.reindex(feats.index)

    # 23. Global production concentration — HHI (0-1 scale)
    #     Stage-resolved per EU CRM 2023 methodology (Grohol & Veeh,
    #     doi:10.2873/725585): HHI computed at both mining and processing
    #     stages; the more concentrated stage governs `production_hhi`.
    #     Both stage components persisted as separate columns for
    #     transparency. For most materials the two are equal (NaN
    #     processing → mining only). For REE the processing stage
    #     dominates: HREE → 1.000, LREE → 0.733; mining stays at the
    #     basket value ~0.50. Refs: Graedel et al. (2012) ES&T; EU CRM 2023
    #     Annex 7; Nassar, Du & Graedel (2015) JIE for element-level
    #     REE-stage rationale.
    feats["production_hhi_mining"] = mining_hhi_mapped.reindex(feats.index)
    feats["production_hhi_processing"] = processing_hhi.reindex(feats.index)
    # Canonical column = max(mining, processing), NaN-safe via fillna(-inf)
    # so that materials with NaN processing fall back to mining only.
    stage_max = pd.concat(
        [feats["production_hhi_mining"], feats["production_hhi_processing"]],
        axis=1,
    ).max(axis=1, skipna=True)
    feats["production_hhi"] = stage_max

    # Replace inf with NaN (not 0) — preserves "no data" semantics
    feats = feats.replace([np.inf, -np.inf], np.nan)

    # Criticality-exclusion flag: mark materials with no published
    # peer-reviewed critical-minerals source. Downstream consumers
    # (the criticality / supply-risk scatter) must drop these rows before
    # analysis rather than imputing numeric values from a data vacuum.
    feats["excluded_from_criticality"] = feats.index.isin(
        EXCLUDED_FROM_CRITICALITY
    )

    # Zero-demand flag (added 2026-05-07): mark materials whose total
    # demand across all scenarios/years is exactly zero. These are
    # elements consumed only by technologies with 0% deployment share
    # in the current technology_mapping (CIGS=0% → Selenium, Gallium;
    # a-Si=0% → Germanium; Gadium has no consuming tech in the mapping).
    # Their MC outputs are an artifact of the tech mix, not a material
    # signal — flagged here in the raw output for transparency (consumers
    # that need a criticality-only view can filter on this column).
    total_demand = demand.groupby("material")["mean"].sum()
    feats["zero_demand"] = (
        total_demand.reindex(feats.index).fillna(0) <= 0
    )

    # Report coverage
    n_with_dep = (feats["import_dependency"] < 1.0).sum()
    n_with_crc = feats.index.isin(crc_mapped.index).sum()
    print(f"  Supply-chain coverage: import_dep={n_with_dep}/31, CRC_risk={n_with_crc}/31")
    print(f"  Material features: {len(feats.columns)} total (all dimensionless)")

    return feats


# ═══════════════════════════════════════════════════════════════════════════════
# CLI test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading data...")
    demand = load_demand_data()
    nrel = load_nrel_data()
    risk_data = load_risk_data()
    thin_film = load_usgs_2023_thin_film()

    print("\nEngineering scenario features...")
    sf = engineer_scenario_features(demand, nrel)
    print(f"  {sf.shape[0]} scenarios × {sf.shape[1]} features")
    print(sf.describe().round(2))

    print("\nEngineering material features...")
    mf = engineer_material_features(demand, risk_data, thin_film)
    print(f"  {mf.shape[0]} materials × {mf.shape[1]} features")
    print(mf.describe().round(2))

    print("\nFeature engineering complete!")
