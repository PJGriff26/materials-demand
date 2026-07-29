"""Shared supply-chain helper for manuscript Figures 5, 6, and 7.

Assembles per-material supply-chain data (load_all_data), the OECD CRC risk
grouping, the per-element rare-earth production table, and withheld-production
overrides consumed by fig4_fig5_supply_tiers.py, supply_tiers_variants.py, and
peak_demand_vs_nir_scatter.py. Run directly (--re-mode aggregate/per-element),
it also renders standalone US-supply-gap and global-squeeze panels; it is not
itself a separately numbered manuscript figure.
"""

import sys
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

from config import (
    FIGURES_MANUSCRIPT_DIR, DEMAND_TO_RISK, FIGURE_DPI,
)
from supply_chain_analysis import (
    load_data, _get_import_shares_by_crc, _get_import_dependency,
    _get_reserves_by_crc,
)
# REE processing-stage country shares (EU CRM 2023, Annex 7) live in
# supply_chain/feature_engineering.py as the single source of truth so the
# stage-resolved production_hhi feature computation and this viz code use
# identical distributions. See module-level comment in feature_engineering.py.
from feature_engineering import (
    LREE_PROCESSING_SHARES as LREE_PROCESSING,
    HREE_PROCESSING_SHARES as HREE_PROCESSING,
)

# ═══════════════════════════════════════════════════════════════════════════════
# CRC RISK GROUPING (collapsed from 11 OECD CRC categories to 5)
# ═══════════════════════════════════════════════════════════════════════════════
CRC_GROUPS = {
    "United States": "US domestic", "OECD": "Low risk",
    1: "Low risk", 2: "Low risk", 3: "Low risk",
    4: "Moderate risk", 5: "Moderate risk",
    6: "High risk", 7: "High risk",
    "China": "China", "Undefined": "Moderate risk",
}
GROUP_ORDER = ["US domestic", "Low risk", "Moderate risk", "High risk", "China"]
GROUP_COLORS = {
    "US domestic": "#1f77b4", "Low risk": "#2ca02c",
    "Moderate risk": "#ff8c00", "High risk": "#e53935", "China": "#880e4f",
}

OECD_MEMBERS = {
    "Canada", "Australia", "Japan", "South Korea", "Korea, Republic of",
    "Germany", "France", "United Kingdom", "Italy", "Netherlands", "Belgium",
    "Norway", "Sweden", "Finland", "Austria", "Switzerland", "Spain",
    "Portugal", "Ireland", "Denmark", "Poland", "Czech Republic", "Czechia",
    "Israel", "New Zealand", "Iceland", "Luxembourg", "Slovakia", "Slovenia",
    "Estonia", "Latvia", "Lithuania", "Hungary", "Greece", "Chile", "Colombia",
    "Costa Rica", "Mexico", "Turkey", "Türkiye",
}

# ═══════════════════════════════════════════════════════════════════════════════
# WITHHELD PRODUCTION OVERRIDES (USGS "W" designation)
# ═══════════════════════════════════════════════════════════════════════════════
# Boron: US is net exporter (~860 kt/yr exports), production withheld due to
#   single producer (Rio Tinto, Boron CA). Use export volume as lower bound.
# Magnesium: Primary smelters closed 2023. Use secondary production from
#   USGS MCS 2025 (mcs2025-mgmet_salient.csv): ~109 kt/yr (2023-2024 avg).
# Tellurium: USGS withholds US production ("W" in MCS 2025 salient AND world
#   data; world total note "Excludes U.S. production"). Replaces the volatile
#   Form-72 trade-balance back-estimate (~12.6 t/yr, sensitive to the swinging
#   USGS NIR; gave 8x-2000x depending on vintage). SOURCE: SCRREEN2 Tellurium
#   factsheet (2023, p.7), verbatim: "In 2022, Rio Tinto also began production
#   of an expected 20 tonnes per year at a facility in the US (Rio Tinto, 2022)."
#   This is a FLOOR (one facility; Asarco/Grupo Mexico is a second US producer,
#   unquantified), so the resulting demand/production ratio (~46x) is an upper
#   bound. 0.020 kt = 20 t/yr.
WITHHELD_OVERRIDES = {
    "Boron": {"us_prod_kt": 860.0},
    "Magnesium": {"us_prod_kt": 109.0},
    "Tellurium": {"us_prod_kt": 0.020},
}

# ═══════════════════════════════════════════════════════════════════════════════
# PER-ELEMENT RE DATA (used only in per-element mode)
# ═══════════════════════════════════════════════════════════════════════════════
# CANONICAL SOURCE (locked 2026-05-26 by PJ Griffiths): DOE 2023 Critical
# Materials Assessment, Appendix A "Historical Price and Production"
# stacked-area charts. 2022 stacked-area peak heights, in tonnes ELEMENT
# CONTENT (per DOE §4.3.4.2 / §4.3.13.2 / §4.3.17.2 / §4.3.21.2 final
# sentence: "Quantities are adjusted to be in terms of [element] content,
# rather than [oxide]").
#
# Source: U.S. Department of Energy, Critical Materials Assessment (2023-07-31),
# Appendix A per-element production charts. Global production values are reads of
# the 2022 Appendix A stacked-area total; US production values are reads of the
# 2022 US-stack layer in the same charts. (Values were cross-validated against
# USGS Mineral Commodity Summaries before being adopted as canonical.)
#
# Yttrium and Gadolinium are NOT covered by DOE 2023 (the 2023 CMA only
# has dedicated App A charts for the four "magnet REEs"). Treatment:
#   - Yttrium (Y): retained in per-element view using a directly-published
#     USGS Mineral Commodity Summaries value (see Y entry below). USGS is
#     the project standard for raw-published commodity data and is
#     independent of the Path B+ bottom-up estimate, which is explicitly
#     NOT used here.
#   - Gadolinium (Gd): no demand in the simulation (intensity entries
#     exist only for CIGS, which has 0% deployment in
#     technology_mapping.py). Excluded from per-element view; would
#     plot as a row with zero demand anyway.
# In aggregate mode all RE elements still contribute to the single
# "Rare Earths" row via the USGS REO total.
#
# Previous methodology (SCRREEN+Binnemans apportionment of USGS REO total)
# is preserved in DECISIONS.md and the prior commits of this file. The
# Path B+ deposit-weighted MC estimate (per_element_production_2024.csv)
# is an independent cross-validation and is NOT used in this dict.

RE_ELEMENT_DATA = {
    "Neodymium":    {"global_t": 49_000, "us_t": 5_000},  # 2022, DOE 2023 App A p.178 (PDF p.204)
    "Praseodymium": {"global_t": 15_500, "us_t": 1_500},  # 2022, DOE 2023 App A p.188 (PDF p.214)
    "Dysprosium":   {"global_t":  2_000, "us_t":    50},  # 2022, DOE 2023 App A p.153 (PDF p.179)
    "Terbium":      {"global_t":  1_000, "us_t":    50},  # 2022, DOE 2023 App A p.200 (PDF p.226)
    # Yttrium added 2026-05-26 per directly-published USGS source.
    # USGS Mineral Commodity Summaries (MCS) 2023/2024/2026 vintages all
    # report "World mine production of Y2O3 equivalent contained in
    # rare-earth mineral concentrates was estimated to be 10,000 to 15,000
    # tons" (consistent 3-of-5 years; MCS 2025 reported 15,000-20,000 as
    # the high-outlier year). Midpoint of 10-15 kt Y2O3 = 12,500 t Y2O3
    # = 12,500 / 1.270 = 9,840 t Y element, rounded to 10,000 t element
    # content for consistency with the DOE 2023 element-content unit
    # convention. US production reported as "NA" by USGS in every
    # vintage (NIR=100%); use 0 here per PJ instruction 2026-05-26.
    "Yttrium":      {"global_t": 10_000, "us_t":     0},  # USGS MCS 2023/2024/2026 multi-vintage median (10-15 kt Y2O3 -> 10 kt Y element rounded)
}

# LREE_PROCESSING / HREE_PROCESSING are imported from
# supply_chain.feature_engineering above as the single source of truth
# (EU CRM 2023, Annex 7 — Processing stage country shares).

RE_ELEMENTS = list(RE_ELEMENT_DATA.keys())  # 4 elements: Nd, Pr, Dy, Tb (post-2026-05-26)
HREE_LIST = ["Dysprosium", "Terbium", "Yttrium", "Gadium"]  # full HREE list, used elsewhere


# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADERS
# ═══════════════════════════════════════════════════════════════════════════════

def country_to_group(country, crc_lookup):
    """Map a producer/importer country to one of the 5 collapsed CRC risk groups.

    US and China are pinned to their own groups; otherwise the country's OECD CRC
    value (via crc_lookup) is collapsed through CRC_GROUPS. Countries with no CRC
    entry fall back to OECD membership (Low risk) or, failing that, Moderate risk
    (a conservative default for unclassified or data-sparse sources).
    """
    if country == "United States": return "US domestic"
    if country == "China": return "China"
    crc_val = crc_lookup.get(country)
    if crc_val is not None: return CRC_GROUPS.get(crc_val, "Moderate risk")
    if country in OECD_MEMBERS: return "Low risk"
    if country == "Myanmar": return "Moderate risk"
    return "Moderate risk"


def load_all_data(re_mode):
    """Load demand, risk data, and prepare per-material records."""
    demand_raw, risk = load_data()
    import_shares_crc = _get_import_shares_by_crc(risk)
    import_dep = _get_import_dependency(risk)
    reserves_crc = _get_reserves_by_crc(risk)

    crc_map_df = risk["crc"].iloc[:, :2].copy()
    crc_map_df.columns = ["country", "crc"]
    crc_lookup = dict(zip(crc_map_df["country"], crc_map_df["crc"]))

    agg = risk["aggregate"].copy()
    for col in ["production", "import", "export", "consumption", "net_import"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")
    agg_mean = agg.groupby("material")["production"].mean()

    prod_sheet = risk["production"]
    fc = prod_sheet.columns[0]

    # Detect MCS 2025 long-form schema: [material, country, production_YYYY, ...]
    # vs. legacy xlsx wide-form: [country_col, Aluminum, Aluminum.1, Boron, ...].
    _new_schema = "material" in prod_sheet.columns and "country" in prod_sheet.columns
    _new_year_cols = (
        [c for c in prod_sheet.columns if c.startswith("production_")]
        if _new_schema else []
    )

    def _new_schema_country_prod(rn, country):
        """kt production for one (material, country) row under new schema.
        Averages available year columns (NaN-skipped)."""
        sub = prod_sheet[(prod_sheet["material"] == rn) & (prod_sheet["country"] == country)]
        if sub.empty:
            return 0.0
        vals = pd.to_numeric(sub[_new_year_cols].values.ravel(), errors="coerce")
        vals = vals[~pd.isna(vals)]
        return float(np.mean(vals)) if len(vals) else 0.0

    def get_global_kt(rn):
        if _new_schema:
            # Prefer the explicit "World total (rounded)" / "Global" row.
            gtot = _new_schema_country_prod(rn, "World total (rounded)")
            if gtot > 0:
                return gtot
            gtot = _new_schema_country_prod(rn, "Global")
            if gtot > 0:
                return gtot
            # Fallback: sum over named countries.
            sub = prod_sheet[prod_sheet["material"] == rn]
            sub = sub[~sub["country"].isin(["Global", "Other", "Other Countries",
                                             "World total (rounded)", None])]
            sub = sub[~sub["country"].isna()]
            if sub.empty:
                return 0.0
            arr = sub[_new_year_cols].apply(
                lambda c: pd.to_numeric(c, errors="coerce")
            ).values
            with np.errstate(invalid="ignore"):
                row_means = np.nanmean(arr, axis=1)
            row_means = row_means[~np.isnan(row_means)]
            return float(np.sum(row_means)) if len(row_means) else 0.0

        # Legacy xlsx wide-form path.
        col1, col2 = rn, f"{rn}.1"
        cols = [c for c in [col1, col2] if c in prod_sheet.columns]
        if not cols: return 0
        gl = prod_sheet[prod_sheet[fc] == "Global"]
        if not gl.empty:
            v = [float(gl[c].values[0]) for c in cols
                 if not pd.isna(pd.to_numeric(gl[c].values[0], errors="coerce"))]
            if v and np.mean(v) > 0: return np.mean(v)
        total = 0
        for _, row in prod_sheet.iterrows():
            country = row[fc]
            if pd.isna(country) or country in ["Global", "Other", None]: continue
            v = [float(row[c]) for c in cols
                 if not pd.isna(pd.to_numeric(row[c], errors="coerce"))]
            if v: total += np.mean(v)
        return total if total > 0 else 0

    def get_grouped_import_shares(rn):
        m = import_shares_crc[import_shares_crc["material"] == rn]
        d = dict(zip(m["crc"], m["share"]))
        total = sum(d.values())
        result = {g: 0 for g in GROUP_ORDER}
        if total > 0:
            for crc_cat, share in d.items():
                result[CRC_GROUPS.get(crc_cat, "Moderate risk")] += share / total
        return result

    def get_grouped_production_shares(rn):
        result = {g: 0 for g in GROUP_ORDER}
        gkt = get_global_kt(rn)
        if gkt <= 0:
            return result

        if _new_schema:
            sub = prod_sheet[prod_sheet["material"] == rn].copy()
            for _, row in sub.iterrows():
                country = row.get("country")
                if pd.isna(country) or country in ["Global", "Other",
                                                    "Other Countries",
                                                    "World total (rounded)",
                                                    None]:
                    continue
                vals = pd.to_numeric(row[_new_year_cols].values, errors="coerce")
                vals = vals[~pd.isna(vals)]
                if len(vals) == 0:
                    continue
                mean_kt = float(np.mean(vals))
                if mean_kt > 0:
                    result[country_to_group(country, crc_lookup)] += mean_kt / gkt
            return result

        # Legacy xlsx path.
        col1, col2 = rn, f"{rn}.1"
        cols = [c for c in [col1, col2] if c in prod_sheet.columns]
        if not cols: return result
        for _, row in prod_sheet.iterrows():
            country = row[fc]
            if pd.isna(country) or country in ["Global", "Other", None]: continue
            v = [float(row[c]) for c in cols
                 if not pd.isna(pd.to_numeric(row[c], errors="coerce"))]
            if v and np.mean(v) > 0:
                result[country_to_group(country, crc_lookup)] += np.mean(v) / gkt
        return result

    def get_re_processing_shares(elem):
        """EU CRM 2023 Annex 7 processing-stage shares."""
        raw = HREE_PROCESSING if elem in HREE_LIST else LREE_PROCESSING
        result = {g: 0 for g in GROUP_ORDER}
        for country, share in raw.items():
            result[country_to_group(country, crc_lookup)] += share
        return result

    # Build records
    peak_demand = demand_raw.groupby("material")["mean"].max()
    d_2026_2050 = demand_raw[(demand_raw["year"] >= 2026) & (demand_raw["year"] <= 2050)]
    cumul = d_2026_2050.groupby(["scenario", "material"])["mean"].sum().unstack(fill_value=0).max()

    records = []
    seen = set()
    materials = sorted(set(DEMAND_TO_RISK.keys()) & set(peak_demand.index))

    for mat in materials:
        rn = DEMAND_TO_RISK[mat]
        is_re = rn == "Rare Earths" and mat != "Rare Earths"
        # Yttrium maps to its own risk_name ("Yttrium") but is physically a
        # heavy rare earth with no standalone USGS world-data row (USGS
        # publishes only the aggregate "Rare Earths" world total). It
        # therefore needs per-element handling identical to the Rare-Earths-
        # risk-name elements.
        is_y_as_re = (re_mode == "aggregate" and mat == "Yttrium")

        # "Goes through per-element data path" — true for any material listed
        # in RE_ELEMENT_DATA (Nd/Pr/Dy/Tb via is_re=True; Y via its own
        # risk_name). Centralizes the per-element membership decision.
        in_per_element_re = mat in RE_ELEMENT_DATA

        # In aggregate mode, sum all RE demands into a single "Rare Earths" row
        if re_mode == "aggregate" and (is_re or is_y_as_re):
            continue  # skip individual RE elements; they get summed below

        # In per-element mode, skip RE elements not in RE_ELEMENT_DATA —
        # standing instruction 2026-05-26: only render per-element rows for
        # elements with directly-published production data. Currently:
        # Nd/Pr/Dy/Tb (DOE 2023 App A) + Y (USGS MCS Yttrium chapter).
        # Gadium (risk_name="Rare Earths") would silently inherit the
        # whole RE aggregate if not skipped explicitly here.
        if re_mode == "per-element" and is_re and not in_per_element_re:
            continue

        peak = peak_demand.get(mat, 0)
        dep = import_dep.get(rn, 1.0)

        # Track how us_prod was obtained — readers need to distinguish a
        # directly-published USGS value from a back-estimated one, and the
        # error-bar / sensitivity story differs by source.
        us_prod_source = "usgs_direct"
        if re_mode == "per-element" and in_per_element_re:
            us_t = RE_ELEMENT_DATA[mat]["us_t"]
            global_t = RE_ELEMENT_DATA[mat]["global_t"]
            prod_shares = get_re_processing_shares(mat)
            us_prod_source = "re_per_element"
        else:
            if rn in WITHHELD_OVERRIDES:
                us_kt = WITHHELD_OVERRIDES[rn]["us_prod_kt"]
                us_prod_source = "withheld_override"
            else:
                raw = agg_mean.get(rn, np.nan)
                us_kt = raw if pd.notna(raw) and raw > 0 else 0
            # USGS "W" (withheld) designation: per-country US production is
            # reported as NaN for a handful of commodities (Silicon, Boron,
            # Magnesium, ...). When BOTH production AND consumption are
            # withheld (the common case — Silicon MCS 2025 has both "W"),
            # the aggregate sheet's `consumption` column is NaN and can't
            # drive back-estimation directly. Recover production from the
            # (always-published) trade balance + published NIR by inverting
            # USGS's Form 72 accounting identity:
            #   consumption  = production + imports − exports
            #   NIR          = (imports − exports) / consumption
            #   ⇒ consumption = net_imports / NIR
            #   ⇒ production  = consumption − net_imports
            #                 = net_imports × (1 − NIR) / NIR
            # Inputs are only USGS-published `import`, `export`, and `NIR`.
            # Mathematically equivalent to the withheld production USGS
            # already used internally to compute NIR. Requires NIR > 0 and
            # positive net-import flow (net exporters fall through to
            # WITHHELD_OVERRIDES or the demand side).
            if us_kt == 0 and 0 < dep < 0.999:
                sub = risk["aggregate"][risk["aggregate"]["material"] == rn]
                imp_mean = pd.to_numeric(sub["import"], errors="coerce").dropna().mean()
                exp_mean = pd.to_numeric(sub["export"], errors="coerce").dropna().mean()
                if pd.notna(imp_mean) and pd.notna(exp_mean):
                    net_imports = float(imp_mean) - float(exp_mean)
                    if net_imports > 0:
                        us_kt = net_imports * (1.0 - dep) / dep
                        us_prod_source = "form72_backestimated"
            us_t = us_kt * 1000
            global_t = get_global_kt(rn) * 1000
            prod_shares = get_grouped_production_shares(rn)

        mat_res_kt = reserves_crc[reserves_crc["material"] == rn]["reserves_kt"].sum()
        cumul_t = cumul.get(mat, 0)
        rc = (mat_res_kt * 1000 / cumul_t) if cumul_t > 0 and mat_res_kt > 0 else 0

        records.append({
            "material": mat, "risk_name": rn, "peak_demand_t": peak,
            "us_prod_t": us_t, "global_prod_t": global_t,
            "import_dependency": dep,
            "us_ratio": peak / us_t if us_t > 0 else np.inf,
            "global_ratio": peak / global_t if global_t > 0 else np.inf,
            "us_status": "zero" if us_t == 0 else "finite",
            "us_prod_source": us_prod_source,
            "reserve_coverage": rc,
            "is_re": False,
            "import_shares_grouped": get_grouped_import_shares(rn),
            "prod_shares_grouped": prod_shares,
        })
        seen.add(rn)

    # Aggregate mode: build a single Rare Earths row from summed element demands
    if re_mode == "aggregate":
        re_mats = [m for m, rn in DEMAND_TO_RISK.items() if rn == "Rare Earths" and m in peak_demand.index]
        # Include Yttrium in the aggregate (USGS MCS 2025 reports Y as part of REO)
        if "Yttrium" in peak_demand.index and "Yttrium" not in re_mats:
            re_mats.append("Yttrium")
        # Sum demands across ALL years/scenarios per element, then take max for "peak"
        # Actually: peak annual demand of the SUM of all RE elements
        re_demand = demand_raw[demand_raw["material"].isin(re_mats)]
        re_summed = re_demand.groupby(["scenario", "year"])["mean"].sum().reset_index()
        peak_re = re_summed["mean"].max()
        cumul_re = re_demand.groupby(["scenario", "year"])["mean"].sum().groupby("scenario").sum().max()

        rn = "Rare Earths"
        dep = import_dep.get(rn, 1.0)
        # Aggregate sheet has only separated compounds (~0.25 kt); the
        # production sheet has US Mountain Pass concentrate (~42-45 kt).
        # Use the schema-aware helper to get US Rare Earths production.
        if _new_schema:
            us_kt_concentrate = _new_schema_country_prod("Rare Earths", "United States")
        else:
            us_kt_concentrate = 0.0
            us_row = prod_sheet[prod_sheet[fc] == "United States"]
            cols = [c for c in ["Rare Earths", "Rare Earths.1"]
                    if c in prod_sheet.columns]
            if not us_row.empty and cols:
                v = [float(us_row[c].values[0]) for c in cols
                     if not pd.isna(pd.to_numeric(us_row[c].values[0],
                                                  errors="coerce"))]
                us_kt_concentrate = np.mean(v) if v else 0.0
        us_t = us_kt_concentrate * 1000
        global_t = get_global_kt("Rare Earths") * 1000

        mat_res_kt = reserves_crc[reserves_crc["material"] == rn]["reserves_kt"].sum()
        rc = (mat_res_kt * 1000 / cumul_re) if cumul_re > 0 and mat_res_kt > 0 else 0

        records.append({
            "material": "Rare Earths", "risk_name": rn, "peak_demand_t": peak_re,
            "us_prod_t": us_t, "global_prod_t": global_t,
            "import_dependency": dep,
            "us_ratio": peak_re / us_t if us_t > 0 else np.inf,
            "global_ratio": peak_re / global_t if global_t > 0 else np.inf,
            "us_status": "zero" if us_t == 0 else "finite",
            "us_prod_source": "usgs_direct",
            "reserve_coverage": rc,
            "is_re": False,  # aggregate; treated as regular material
            "import_shares_grouped": get_grouped_import_shares(rn),
            "prod_shares_grouped": get_grouped_production_shares(rn),
        })

    df = pd.DataFrame(records).set_index("material")
    df_finite = df[df["us_status"] == "finite"].sort_values("us_ratio", ascending=True)
    df_zero = df[df["us_status"] == "zero"].sort_values("import_dependency", ascending=False)
    mat_order = df_finite.index.tolist() + df_zero.index.tolist()
    finite_max = df_finite["us_ratio"].max() if len(df_finite) > 0 else 10
    inf_cap = finite_max * 1.5

    return df, mat_order, inf_cap


# ═══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_panel_a(df, mat_order, inf_cap, output_path, re_mode):
    """Render Panel A (US supply gap): peak annual demand / US production per material.

    Each horizontal bar is split into a domestic segment (width = ratio x (1 - import
    dependency)) and import segments stacked by CRC risk group. Materials with no US
    production (us_ratio == inf) are drawn at inf_cap and annotated; the x-axis is
    symlog (linthresh=1.0) so the self-sufficiency threshold at 1.0 reads cleanly.
    """
    fig, ax = plt.subplots(figsize=(11, max(9, len(mat_order) * 0.40)))
    y_pos = np.arange(len(mat_order))

    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        ratio, dep = row["us_ratio"], row["import_dependency"]
        crc_shares = row["import_shares_grouped"]
        is_inf = ratio == np.inf
        display_ratio = inf_cap if is_inf else ratio

        domestic_w = display_ratio * (1.0 - dep)
        ax.barh(i, domestic_w, color=GROUP_COLORS["US domestic"], edgecolor="white", linewidth=0.4)
        left = domestic_w
        for grp in GROUP_ORDER[1:]:
            w = display_ratio * dep * crc_shares.get(grp, 0)
            if w > 0:
                ax.barh(i, w, left=left, color=GROUP_COLORS[grp], edgecolor="white", linewidth=0.4)
                left += w
        if is_inf:
            ax.annotate("no US production \u2192", xy=(inf_cap, i), fontsize=6.5,
                        va="center", ha="right", style="italic", color="#444",
                        fontweight="bold", xytext=(-4, 0), textcoords="offset points")

    ax.axvline(1.0, color="black", ls="-", lw=2, alpha=0.9, zorder=4)
    ax.text(1.05, -0.5, "self-\nsufficiency", fontsize=8,
            ha="left", va="top", fontweight="bold", alpha=0.6)
    ax.set_xlabel("Peak annual demand / US annual production", fontsize=11, fontweight="bold")
    title_suffix = ("Rare earths aggregated" if re_mode == "aggregate"
                    else "Per-element rare earths (\u00b140% production uncertainty)")
    ax.set_title(f"A. US Supply Gap\n{title_suffix}", fontsize=12, fontweight="bold")
    ax.set_xscale("symlog", linthresh=1.0)
    ax.set_yticks(y_pos); ax.set_yticklabels(mat_order, fontsize=9)
    ax.grid(True, alpha=0.15, axis="x"); ax.set_xlim(0, inf_cap*1.15)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    leg = ax.legend(handles=handles, loc="lower right", fontsize=8, framealpha=0.95,
                    title="Demand sourcing by import origin risk (OECD CRC)", title_fontsize=8)
    leg.get_title().set_fontweight("bold")
    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


def plot_panel_b(df, mat_order_unused, output_path, re_mode):
    """Render Panel B (global production squeeze): peak demand / global production.

    Bars are stacked by the CRC risk group of the *producing* country (not import
    origin), with an end-of-bar percent label and a hatched "Unclassified" remainder
    when the country shares do not sum to 100%. A narrow right-hand column shows
    reserve coverage (reserves / max 2026-2050 demand) as a colored dot per material.
    The mat_order argument is ignored; this panel computes its own ordering so the
    largest squeeze sits at the top and "no production data" rows drop to the bottom.
    """
    # Panel B reorders by global_ratio descending (materials without production
    # data drop to the bottom). Keeps the narrative focused on the squeeze.
    df_b = df.copy()
    has_data = df_b[df_b["global_ratio"].replace(np.inf, np.nan).gt(0)].copy()
    no_data = df_b.drop(has_data.index)
    # Sort ascending so y=0 (matplotlib bottom) is the smallest ratio and the
    # largest ratio sits at the top of the plot — no axis inversion needed.
    has_data = has_data.sort_values("global_ratio", ascending=True)
    # "no production data" materials go below the smallest ratios (y=-1, -2, …)
    mat_order = no_data.index.tolist() + has_data.index.tolist()
    y_pos = np.arange(len(mat_order))

    fig = plt.figure(figsize=(12, max(9, len(mat_order) * 0.42)))
    gs = fig.add_gridspec(1, 2, width_ratios=[6.5, 1.0], wspace=0.03)
    ax = fig.add_subplot(gs[0, 0])
    axR = fig.add_subplot(gs[0, 1])  # do NOT sharey — axR tick clears would propagate

    def fmt_ratio(r):
        if r == 0 or r is None or not np.isfinite(r): return ""
        if r >= 1.0:  return f"{r*100:.0f}%"
        if r >= 0.10: return f"{r*100:.0f}%"
        if r >= 0.01: return f"{r*100:.1f}%"
        return f"{r*100:.2f}%"

    max_ratio = has_data["global_ratio"].max() if len(has_data) else 1.0
    x_right = max_ratio * 2.0

    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        ratio = row["global_ratio"]
        crc_prod = row["prod_shares_grouped"]
        display_ratio = 0 if (ratio == np.inf or ratio <= 0) else ratio
        if display_ratio == 0:
            ax.text(0.0005, i, " no production data", fontsize=7, va="center",
                    style="italic", color="#999")
            continue
        left = 0
        for grp in GROUP_ORDER:
            w = display_ratio * crc_prod.get(grp, 0)
            if w > 0:
                ax.barh(i, w, left=left, color=GROUP_COLORS[grp],
                        edgecolor="white", linewidth=0.4)
                left += w
        # Unclassified remainder: hatched gray (distinct from "no reserve data" dot)
        if left < display_ratio * 0.99 and display_ratio > 0:
            ax.barh(i, display_ratio - left, left=left, color="#d9d9d9",
                    edgecolor="white", linewidth=0.4, hatch="////")
        # End-of-bar value label
        ax.text(display_ratio * 1.08, i, fmt_ratio(display_ratio),
                fontsize=8, va="center", ha="left", color="#333", fontweight="bold")

    # Headline threshold: vertical line at 1.0 (US peak demand = world annual output)
    ax.axvline(1.0, color="black", ls="-", lw=2.2, alpha=0.95, zorder=4)

    ax.set_xscale("symlog", linthresh=0.001)
    ax.set_xlim(0, x_right)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(mat_order, fontsize=9)
    ax.set_ylim(-0.6, len(mat_order) - 0.4)
    ax.set_xlabel("Peak annual demand / global annual production",
                  fontsize=11, fontweight="bold")
    title_suffix = ("Rare earths aggregated" if re_mode == "aggregate"
                    else "Per-element rare earths (\u00b140% production uncertainty)")
    ax.set_title(f"B. Global Production Squeeze\n{title_suffix}",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="x")

    # ── Right-side reserves column ────────────────────────────────────────────
    axR.set_xlim(0, 1)
    axR.set_ylim(ax.get_ylim())
    axR.set_xticks([]); axR.set_yticks([])
    for spine in ("top", "right", "bottom", "left"):
        axR.spines[spine].set_visible(False)
    axR.set_title("Reserve coverage\n(reserves / max 2026\u20132050 demand)",
                  fontsize=9.5, fontweight="bold", loc="center", pad=8)

    def rc_color(rc):
        if rc == 0 or not np.isfinite(rc): return "#cccccc"
        if rc < 1:    return "#e53935"
        if rc < 10:   return "#ff8c00"
        return "#2ca02c"

    def rc_text(rc):
        if rc == 0 or not np.isfinite(rc): return "\u2014"
        if rc < 1:    return f"{rc:.2f}\u00d7"
        if rc < 10:   return f"{rc:.1f}\u00d7"
        if rc < 100:  return f"{rc:.0f}\u00d7"
        return f"{rc:,.0f}\u00d7"

    for i, mat in enumerate(mat_order):
        rc = df.loc[mat, "reserve_coverage"]
        axR.scatter(0.22, i, s=110, marker="o", color=rc_color(rc),
                    edgecolors="k", linewidths=0.6, zorder=5)
        axR.text(0.42, i, rc_text(rc), fontsize=8.5, va="center", ha="left",
                 color="#222", fontweight="bold")

    # ── Unified horizontal legend below plot ─────────────────────────────────
    h_prod = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    h_prod.append(mpatches.Patch(facecolor="#d9d9d9", edgecolor="white",
                                  hatch="////", label="Unclassified"))
    h_res = [
        plt.Line2D([], [], color="#2ca02c", marker="o", ls="None", ms=9,
                   mec="k", label="Reserves \u226510\u00d7"),
        plt.Line2D([], [], color="#ff8c00", marker="o", ls="None", ms=9,
                   mec="k", label="Reserves 1\u201310\u00d7"),
        plt.Line2D([], [], color="#e53935", marker="o", ls="None", ms=9,
                   mec="k", label="Reserves <1\u00d7"),
        plt.Line2D([], [], color="#cccccc", marker="o", ls="None", ms=9,
                   mec="k", label="No reserve data"),
    ]
    leg1 = fig.legend(handles=h_prod, loc="lower center",
                      bbox_to_anchor=(0.32, -0.04), ncol=6, fontsize=8,
                      frameon=False,
                      title="Production by country risk (OECD CRC)",
                      title_fontsize=8.5)
    leg1.get_title().set_fontweight("bold")
    leg2 = fig.legend(handles=h_res, loc="lower center",
                      bbox_to_anchor=(0.82, -0.04), ncol=2, fontsize=8,
                      frameon=False,
                      title="Reserve adequacy (right column)",
                      title_fontsize=8.5)
    leg2.get_title().set_fontweight("bold")

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# SI: PER-ELEMENT SCATTER WITH UNCERTAINTY BANDS
# ═══════════════════════════════════════════════════════════════════════════════

def plot_si_re_scatter(output_path):
    """SI Figure: per-element RE demand vs production with ±40% uncertainty bands.
    Provides element-level supply stress detail with explicit error bars,
    appropriate for SI placement when the main-text supply-tier figures
    (manuscript Figures 4 and 5) aggregate the rare earths into one row.
    """
    demand_raw, _ = load_data()
    elements = list(RE_ELEMENT_DATA.keys())

    rows = []
    for elem in elements:
        d = demand_raw[demand_raw["material"] == elem]
        if d.empty: continue
        peak = d["mean"].max()
        prod = RE_ELEMENT_DATA[elem]["global_t"]
        prod_lo, prod_hi = prod * 0.6, prod * 1.4  # ±40% uncertainty
        is_hree = elem in HREE_LIST
        # China processing share (EU CRM 2023, Annex 7): HREE 100%, LREE 0.849.
        # 0.849 is LREE_PROCESSING["China"] (the sourced constant imported above).
        china_share = 1.0 if is_hree else 0.849
        rows.append({"element": elem, "peak": peak, "prod": prod,
                     "prod_lo": prod_lo, "prod_hi": prod_hi,
                     "is_hree": is_hree, "china_share": china_share})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(9, 7))

    # Reference lines
    xx = np.logspace(2, 5, 100)
    ax.plot(xx, xx, "k--", lw=1, alpha=0.5, label="demand = supply")
    ax.plot(xx, xx * 0.5, color="gray", ls=":", lw=0.8, alpha=0.5, label="demand = 50% supply")
    ax.plot(xx, xx * 0.1, color="gray", ls=":", lw=0.6, alpha=0.4, label="demand = 10% supply")

    for _, r in df.iterrows():
        color = "#e53935" if r["is_hree"] else "#ff8c00"  # HREE red, LREE orange
        # Uncertainty bar (horizontal)
        ax.errorbar(r["prod"], r["peak"],
                    xerr=[[r["prod"] - r["prod_lo"]], [r["prod_hi"] - r["prod"]]],
                    fmt="o", color=color, markersize=10, capsize=4,
                    markeredgecolor="k", markeredgewidth=0.5, elinewidth=1, alpha=0.85)
        # Label
        ax.annotate(r["element"], (r["prod"], r["peak"]),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=10, fontweight="bold")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Estimated global production (t REO equivalent, 2024)", fontsize=11, fontweight="bold")
    ax.set_ylabel("US peak transition demand (t/yr)", fontsize=11, fontweight="bold")
    ax.set_title("Per-Element Rare Earth Supply Stress\n"
                 "(production estimates apportioned from USGS aggregate; horizontal bars show \u00b140% uncertainty)",
                 fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)

    # Custom legend
    h = [
        plt.Line2D([],[],marker="o",color="#e53935",ls="None",ms=10,mec="k",
                   label="HREE (China 100% processing)"),
        plt.Line2D([],[],marker="o",color="#ff8c00",ls="None",ms=10,mec="k",
                   label="LREE (China ~85% processing)"),
        plt.Line2D([],[],ls="--",color="k",label="demand = supply"),
        plt.Line2D([],[],ls=":",color="gray",label="demand = 50% / 10% supply"),
    ]
    ax.legend(handles=h, loc="upper left", fontsize=9, framealpha=0.95)

    # Methodology note
    ax.text(0.02, 0.02,
            "Sources: USGS MCS 2025 (aggregate); EU CRM 2020/SCRREEN (volume distribution);\n"
            "Binnemans & Jones 2015 (ore mineralogy); EU CRM 2023 Annex 7 (country shares).\n"
            "Cross-check: DOE 2010 Critical Materials Strategy Table 7-2 (scaled \u00d73.0).",
            transform=ax.transAxes, fontsize=7, color="#555",
            verticalalignment="bottom", style="italic")

    fig.tight_layout()
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_mode(re_mode, output_dir):
    """Build the data frame for one rare-earth handling mode and render both panels.

    Writes fig4a_<suffix>_us_supply_gap.png and fig4b_<suffix>_global_production_squeeze.png
    (suffix is "aggregate" or "per_element") into output_dir, then prints a per-material
    US-ratio / global-ratio summary table to stdout.
    """
    print(f"\n=== Generating Fig 4 in '{re_mode}' mode ===")
    df, mat_order, inf_cap = load_all_data(re_mode)

    suffix = "aggregate" if re_mode == "aggregate" else "per_element"
    out_a = output_dir / f"fig4a_{suffix}_us_supply_gap.png"
    out_b = output_dir / f"fig4b_{suffix}_global_production_squeeze.png"

    plot_panel_a(df, mat_order, inf_cap, out_a, re_mode)
    plot_panel_b(df, mat_order, out_b, re_mode)

    # Print data summary
    print(f"\n  Material rows ({len(mat_order)}):")
    for m in reversed(mat_order):
        r = df.loc[m]
        us_str = f"{r['us_ratio']:.1f}x" if r['us_ratio'] < np.inf else "[zero]"
        gl_str = f"{r['global_ratio']:.1%}" if r['global_ratio'] < np.inf else "NO DATA"
        print(f"    {m:15s}  US={us_str:>10s}  Global={gl_str:>8s}")


def main():
    """CLI entry point: parse --re-mode / --output-dir / --si-scatter and render.

    Generates the aggregate and/or per-element supply-gap panels (and, by default,
    the SI per-element scatter) into the manuscript figures directory.
    """
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--re-mode", choices=["aggregate", "per-element", "both"],
                         default="both",
                         help="Rare earth handling mode (default: both)")
    parser.add_argument("--output-dir", type=Path,
                         default=FIGURES_MANUSCRIPT_DIR,
                         help="Output directory (default: outputs/figures/manuscript/)")
    parser.add_argument("--si-scatter", action="store_true", default=True,
                         help="Also generate the per-element SI scatter figure")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.re_mode in ("aggregate", "both"):
        run_mode("aggregate", args.output_dir)
    if args.re_mode in ("per-element", "both"):
        run_mode("per-element", args.output_dir)

    if args.si_scatter:
        print(f"\n=== Generating SI per-element scatter ===")
        si_dir = args.output_dir / "si_figures"
        si_dir.mkdir(parents=True, exist_ok=True)
        plot_si_re_scatter(si_dir / "si_fig_re_per_element_scatter.png")

    print("\nDone.")


if __name__ == "__main__":
    main()
