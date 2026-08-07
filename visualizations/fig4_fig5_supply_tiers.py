"""Manuscript Figures 4 and 5: supply-chain risk (demand vs supply tiers).

The default split run writes Figure 4 (US tiers: production over reserves) and
Figure 5 (global tiers: production over reserves). It reads the Monte Carlo
demand output and the supply-chain risk tables (USGS MCS 2025, OECD CRC, Census
HTS import shares) and renders the demand-to-supply ratio bars per material,
with a magnitude sub-panel and a CRC source-mix sub-panel. Per-element
rare-earth breakdowns come from the shared helper supply_tiers_shared.py; the
--legacy-colored flag reproduces the older combined 2x2 layout.

INVENTORY:
  name: fig4_fig5_supply_tiers
  output: outputs/figures/manuscript/fig4_supply_tiers_us.png,
    outputs/figures/manuscript/fig5_supply_tiers_global.png
    (--docx variants under outputs/figures/manuscript/docx/)
  category: Manuscript main text — Figs 4 and 5
  axes:
    x: demand-to-supply ratio (symlog) on the magnitude sub-panel;
       0-100% source/reserve share on the mix sub-panel
    y: materials, stress-ranked per panel (highest ratio at top)
    color: magnitude bar neutral gray + hatched worst-case extension;
       mix bars by OECD CRC risk group; dark-gray "Unknown" filler
       (Fig 5 only) for USGS 'Other countries' aggregate shares
  data_sources:
    - outputs/data/material_demand_by_scenario.csv (via build_records)
    - USGS MCS 2025 raw CSVs, OECD CRC 2026, Census HTS import shares
      (via supply_chain_analysis.load_data / supply_tiers_shared)
  description: >
    Builds the supply-chain records and delegates the shipped split render
    to supply_tiers_variants.plot_option3a_midcase(split=True). Each figure
    stacks panel a (peak-annual-demand vs production ratio + share bars)
    over panel b (cumulative-demand vs reserves ratio + share bars).
    2026-08-07 (PI request): V1 x-axis labels reworded to "US peak annual
    demand / US cumulative demand" phrasing in both the shipped split path
    (supply_tiers_variants.PANEL_TITLES) and the legacy --legacy-colored
    V1 f-strings here; V2 (supply-over-demand) labels unchanged.
    2026-08-07 numbering cleanup: docstring/help updated to the current
    Fig 4/5 numbering; --legacy-colored variants now write to
    fig4_fig5_variants/ (was fig5_4panel_variants/) with fig4_fig5__ stems.
    No rendering changes.
END_INVENTORY
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.transforms import blended_transform_factory

# Poster-styling constants (used when styling="poster")
REE_HIGHLIGHT_SET = {
    "Dysprosium", "Neodymium", "Praseodymium", "Terbium",
    "Yttrium", "Gadium", "Rare Earths",
}
REE_HIGHLIGHT_COLOR = "#880e4f"
POSTER_NO_DATA_COLOR = "#7b1d3f"

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))
sys.path.insert(0, str(BASE_DIR / "visualizations"))

# Pre-import so supply_chain/config.py resolves paths against --interpolation.
from _interpolation_io import (  # noqa: E402
    setup_interpolation_env, add_interpolation_arg,
)
_INTERP_METHOD = setup_interpolation_env()

from config import FIGURES_MANUSCRIPT_DIR, DEMAND_TO_RISK, FIGURE_DPI  # noqa: E402
from supply_tiers_shared import (  # noqa: E402
    load_all_data, CRC_GROUPS, GROUP_ORDER, GROUP_COLORS, OECD_MEMBERS,
    country_to_group, WITHHELD_OVERRIDES,
)
from supply_chain_analysis import load_data  # noqa: E402
from src.scenario_config import REFERENCE_SCENARIO  # noqa: E402


# ═══════════════════════════════════════════════════════════════════════════════
# Material-scope for reserves: which materials lack a per-country reserves
# concept in USGS MCS 2025.
#
# Historical (pre-audit) note: Aluminum, Steel, Indium, Selenium, and
# Tellurium used to be excluded here. A 2026-04 USGS data audit established:
#   - Aluminum reserves are reportable via Bauxite × 0.25 Bayer+Hall-Héroult
#     yield (USGS Bauxite and Alumina metadata documents this factor).
#   - Steel reserves are reportable directly from USGS Iron Ore "iron
#     content" rows (no conversion needed).
#   - Indium, Selenium, Tellurium DO have per-country reserves in USGS
#     MCS 2025 World Data (metric-ton metal content). The earlier claim
#     that thin-film byproducts lack per-country reporting was wrong.
# These five materials are now loaded via `augment_reserves_from_usgs_world_data`
# in supply_chain/supply_chain_analysis.py.
#
# Genuinely missing (USGS does not publish per-country reserves):
#   - Cadmium, Gallium, Germanium — byproduct metals of Zn, bauxite, Cu.
#   - Cement, Silicon — feedstocks (limestone, silica) not tracked as
#     bounded "reserves."
#   - Fiberglass, Glass — not on any critical-minerals list.
# ═══════════════════════════════════════════════════════════════════════════════
NO_SEPARATE_RESERVES = {
    "Cement", "Silicon", "Fiberglass", "Glass",
    "Cadmium", "Gallium", "Germanium",
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def _mid_case_and_worst_scenario(demand_raw, mid_case_name=REFERENCE_SCENARIO):
    """
    For the mid-case-variant framing: per material, return:
      - peak_mid_case : peak-year MC mean annual demand under Mid_Case
      - peak_worst_case : max across all 61 scenarios of peak-year mean annual demand
      - cum_mid_case  : 2026–2050 cumulative MC mean under Mid_Case
      - cum_worst_case: max across scenarios of cumulative MC mean

    Peaks use the `_annual` columns so the magnitude is annual demand
    regardless of whether the underlying MC ran on the PCHIP annual grid
    (default since 2026-05-04) or the legacy 3-yr bucket. Cumulative uses
    raw `mean` because each row is a period total under both cadences.

    No MC intensity error propagation here — the mid-case variant replaces
    the envelope-style error bars with a grey "scenario-pessimism" segment
    that extends from the Mid_Case bar end to the worst-case scenario value.
    """
    d = demand_raw

    # Peak per scenario × material (max over years of annualized MC mean)
    peak_col = "mean_annual" if "mean_annual" in d.columns else "mean"
    peak_by_scen = d.groupby(["scenario", "material"])[peak_col].max().unstack(fill_value=0)
    cum_window = d[(d["year"] >= 2026) & (d["year"] <= 2050)]
    cum_by_scen = cum_window.groupby(["scenario", "material"])["mean"].sum().unstack(fill_value=0)

    materials = peak_by_scen.columns
    out = pd.DataFrame(index=materials)
    out.index.name = "material"

    if mid_case_name in peak_by_scen.index:
        out["peak_mid_case"]  = peak_by_scen.loc[mid_case_name]
        out["cum_mid_case"]   = cum_by_scen.loc[mid_case_name]
    else:
        # Fall back to median scenario if Mid_Case is absent
        out["peak_mid_case"]  = peak_by_scen.median()
        out["cum_mid_case"]   = cum_by_scen.median()
    out["peak_worst_case"] = peak_by_scen.max()
    out["cum_worst_case"]  = cum_by_scen.max()
    return out


def _median_and_range_demand(demand_raw):
    """
    Per material, return:
      - peak_median : median (across scenarios) of peak-year scenario mean
      - peak_min_p025 : outer p2.5 lower bound combining scenario & MC
      - peak_max_p975 : outer p97.5 upper bound combining scenario & MC
      - cum_median / cum_min_p025 / cum_max_p975 : same for 2026–2050
        cumulative demand

    Why outer p2.5/p97.5 rather than scenario min/max (as in v1):
      The paper's methodological contribution is Monte Carlo intensity
      uncertainty on top of scenario uncertainty. Error bars need to
      show BOTH. Scenario min/max ignores the 10,000 MC draws per
      (scenario × year), so the MC intensity uncertainty must appear
      somewhere in the visualization set.

      Formula (outer 95% CI, pooled):
        peak_min_p025 = min over scenarios of peak-year p2
        peak_max_p975 = max over scenarios of peak-year p97
      Because both are stored per (scenario, year, material) in the
      demand CSV, no extra sampling is needed.
    """
    d = demand_raw.copy()

    # Peak-year statistics per (scenario, material): take the year with
    # the largest annualized mean as the "peak year", then read
    # annualized p2/p97 at that year. The `_annual` columns make peak
    # magnitude consistent across PCHIP (annual) and legacy (3-yr-bucket)
    # cadences.
    mean_col = "mean_annual" if "mean_annual" in d.columns else "mean"
    p2_col   = "p2_annual"   if "p2_annual"   in d.columns else "p2"
    p97_col  = "p97_annual"  if "p97_annual"  in d.columns else "p97"

    def _scenario_peak(grp):
        peak_idx = grp[mean_col].idxmax()
        return pd.Series({
            "peak_mean": grp.loc[peak_idx, mean_col],
            "peak_p2":   grp.loc[peak_idx, p2_col]  if p2_col  in grp.columns else grp.loc[peak_idx, mean_col],
            "peak_p97":  grp.loc[peak_idx, p97_col] if p97_col in grp.columns else grp.loc[peak_idx, mean_col],
        })

    peaks = d.groupby(["scenario", "material"], group_keys=False).apply(_scenario_peak)

    # Cumulative 2026–2050 per (scenario, material) — sum the per-row
    # period totals (raw `mean`/`p2`/`p97`) so the result is total
    # tonnes regardless of cadence.
    cum_window = d[(d["year"] >= 2026) & (d["year"] <= 2050)]
    cum_mean = cum_window.groupby(["scenario", "material"])["mean"].sum()
    cum_p2   = cum_window.groupby(["scenario", "material"])["p2"].sum() if "p2" in cum_window.columns else cum_mean
    cum_p97  = cum_window.groupby(["scenario", "material"])["p97"].sum() if "p97" in cum_window.columns else cum_mean

    materials = peaks.index.get_level_values("material").unique()
    out = pd.DataFrame(index=materials)
    out.index.name = "material"
    out["peak_median"]  = peaks.groupby("material")["peak_mean"].median()
    out["peak_min_p025"] = peaks.groupby("material")["peak_p2"].min()   # lowest lower bound
    out["peak_max_p975"] = peaks.groupby("material")["peak_p97"].max()  # highest upper bound
    out["cum_median"]   = cum_mean.groupby("material").median()
    out["cum_min_p025"] = cum_p2.groupby("material").min()
    out["cum_max_p975"] = cum_p97.groupby("material").max()
    return out


def _alias_yttrium_to_rare_earths(out):
    """USGS MCS 2025 reports Yttrium reserves only as part of the aggregate
    "Rare Earths" total, not as a separate column. Without this alias the
    Yttrium row gets reserves=0 and falsely flags as ∞-stress in Panel D,
    while Dy/Tb/Nd/Pr correctly inherit the aggregate via risk_name=
    "Rare Earths". No source publishes per-element REE reserves, so by design
    every rare-earth element's reserve ratio is measured against the aggregate
    total-rare-earth-oxide reserve base (an optimistic upper bound); the paper
    reports this as the chosen reserve basis (see Table S2). Per-element
    apportionment is documented as context only and is not used here.
    """
    if "Rare Earths" in out and "Yttrium" not in out:
        out["Yttrium"] = out["Rare Earths"]


def _get_us_reserves(risk):
    """Return {risk_material_name: tonnes} for US reserves."""
    reserves_df = risk["reserves"]
    us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]
    out = {}
    if us_row.empty:
        return out
    for mc in mat_cols:
        val = pd.to_numeric(us_row[mc].values[0], errors="coerce")
        out[mc] = val * 1000 if pd.notna(val) else 0  # kt → t
    _alias_yttrium_to_rare_earths(out)
    return out


def _get_global_reserves(risk):
    reserves_df = risk["reserves"]
    gl_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]
    out = {}
    if gl_row.empty:
        return out
    for mc in mat_cols:
        val = pd.to_numeric(gl_row[mc].values[0], errors="coerce")
        out[mc] = val * 1000 if pd.notna(val) else 0
    _alias_yttrium_to_rare_earths(out)
    return out


def _reserves_by_country_group(risk, crc_lookup):
    """For each risk material, return {group: fraction_of_global_reserves}.

    Shares are fractions of the GLOBAL total (the "Global"/"World total" row),
    not of the sum of listed countries, so any unattributed reserves (the USGS
    "Other Countries" row, plus rounding) read as the Unknown remainder in the
    bar — mirroring how the production-share path works. Three data hazards are
    handled explicitly:
      - aggregate rows ("Global", "World total (rounded)") are the denominator,
        never a country (otherwise they'd be classified as Moderate risk and
        double the totals);
      - "Other"/"Other Countries" are unattributed and excluded from the CRC
        split (they become the Unknown remainder);
      - duplicate country rows (e.g. "Russia" and "Russia " with a trailing
        space, which the augment step can introduce) are collapsed by
        normalized name so a country is counted once and classified once.
    """
    reserves_df = risk["reserves"].copy()
    reserves_df["_c"] = reserves_df["Unnamed: 0"].astype(str).str.strip()
    mat_cols = [c for c in reserves_df.columns if c not in ("Unnamed: 0", "_c")]

    out = {}
    for mc in mat_cols:
        global_total = 0.0
        by_country = {}
        for _, row in reserves_df.iterrows():
            country = row["_c"]
            low = country.lower()
            val = pd.to_numeric(row[mc], errors="coerce")
            if pd.isna(val) or val <= 0:
                continue
            if low == "global" or "total" in low:
                global_total = max(global_total, val)     # denominator
            elif low in ("other", "other countries", "none", "nan"):
                continue                                   # -> Unknown remainder
            else:
                by_country[country] = max(by_country.get(country, 0.0), val)  # de-dup
        denom = max(global_total, sum(by_country.values()))
        shares = {g: 0.0 for g in GROUP_ORDER}
        if denom > 0:
            for country, val in by_country.items():
                shares[country_to_group(country, crc_lookup)] += val / denom
        out[mc] = shares
    _alias_yttrium_to_rare_earths(out)
    return out


REE_PER_ELEMENT_LIST = ["Dysprosium", "Neodymium", "Praseodymium",
                        "Terbium", "Yttrium"]


def apply_s9d_style_order(df, mat_order, re_mode, n=10, sort_panel="A"):
    """Top-N by sort_panel ratio (descending) + REE block forced to the
    bottom as a contiguous group, applied consistently across all panels so
    the highest supply-stress materials lead and the rare earths read as one
    block. (Figures 4 and 5 pass n = all materials, so the top-N cap is not
    binding there; it just fixes the ordering.)

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `build_records`. Indexed by material; carries
        `us_prod_t`, `global_prod_t`, `us_reserves_t`, `global_reserves_t`,
        `peak_mid_case`, and `cumul_mid_case` columns.
    mat_order : list[str]
        Materials to consider. Typically the full list from `build_records`.
    re_mode : {"aggregate", "per-element"}
        Aggregate: REE block = single "Rare Earths" row.
        Per-element: REE block = Nd/Pr/Dy/Tb/Y (Gd dropped — zero demand).
    n : int
        Number of non-REE materials to retain (sorted by sort_panel ratio,
        descending). REE block is always retained in full.
    sort_panel : {"A", "B", "C", "D"}
        Panel A = US production (default; matches the existing poster
        styling convention). The same material order is then applied to
        all 4 panels so the rows align visually.

    Returns
    -------
    list[str]
        New mat_order, ready to pass to plot_option3a_midcase. Order is
        in matplotlib's bottom-up convention: row 0 = bottom of chart.
        REE block ends up at the bottom; highest-ratio non-REE at the top.
    """
    panel_keys = {
        "A": ("peak_mid_case", "us_prod_t"),
        "B": ("cumul_mid_case", "us_reserves_t"),
        "C": ("peak_mid_case", "global_prod_t"),
        "D": ("cumul_mid_case", "global_reserves_t"),
    }
    num_col, den_col = panel_keys[sort_panel]

    def _ratio(mat):
        if mat not in df.index:
            return None
        row = df.loc[mat]
        num = row.get(num_col, 0) or 0
        den = row.get(den_col, 0) or 0
        if den <= 0:
            # No-denominator (e.g., no US production) ⇒ effectively
            # infinite ratio. Treat as highest for sort purposes.
            return float("inf") if num > 0 else None
        return num / den

    # REE membership depends on re_mode.
    if re_mode == "aggregate":
        re_set = {"Rare Earths"}
    else:
        re_set = set(REE_PER_ELEMENT_LIST)

    # Sort non-REE materials by Panel A ratio descending. None-ratios
    # (no demand or missing data) drop to the very end of the non-REE
    # block (i.e., closest to the REE block in the final order).
    non_re_present = [m for m in mat_order
                      if m not in re_set and m != "Rare Earths"
                      and m not in REE_PER_ELEMENT_LIST]
    non_re_ratios = [(m, _ratio(m)) for m in non_re_present]
    # Split: ratio not None, then ratio None
    finite = [m for m, r in non_re_ratios if r is not None]
    finite.sort(key=lambda m: _ratio(m), reverse=True)
    # Take the top-N non-REE materials.
    non_re_top = finite[:n]

    # REE block — sort by Panel A ratio ascending so the highest-ratio
    # REE sits at the TOP of the block (visually adjacent to the
    # lowest-ratio non-REE). Lowest-ratio REE goes to the bottom.
    re_present = [m for m in mat_order if m in re_set]
    re_present.sort(
        key=lambda m: _ratio(m) if _ratio(m) is not None else 0,
        reverse=True,
    )

    # matplotlib horizontal-bar charts plot row 0 at the BOTTOM of the
    # plot, so we want the lowest-ratio rows at index 0. Final order
    # from bottom to top: lowest-ratio REE, ..., highest-ratio REE,
    # lowest-ratio non-REE, ..., highest-ratio non-REE.
    re_block_reversed = list(reversed(re_present))  # lowest at row 0
    non_re_top_reversed = list(reversed(non_re_top))  # lowest at row N-1

    return re_block_reversed + non_re_top_reversed


# Material families for row grouping, from the canonical taxonomy
# (src/material_classes.py, Option 2 placements). The REE family is extended
# with the 'Gadium' data spelling and the aggregate 'Rare Earths' row.
from src.material_classes import CLASS_MEMBERS as _CM, CLASS_ORDER as FAMILY_ORDER  # noqa: E402
MATERIAL_FAMILIES = {c: list(ms) for c, ms in _CM.items()}
MATERIAL_FAMILIES["Rare earth elements"] = (
    MATERIAL_FAMILIES["Rare earth elements"] + ["Gadium", "Rare Earths"])


def family_grouped_order(df, mat_order, re_mode, sort_panel="C"):
    """Row order for Figs 4/5: group materials by family (bulk -> base &
    alloying -> specialty -> rare earths, reading top to bottom), and within
    each family sort by the sort_panel demand/supply ratio (descending).
    Rare earths stay a contiguous block at the bottom (so the B/D aggregation
    and the REE highlight band keep working). Returns matplotlib bottom-up
    order (row 0 = bottom of chart)."""
    panel_keys = {
        "A": ("peak_mid_case", "us_prod_t"),
        "B": ("cum_mid_case", "us_reserves_t"),
        "C": ("peak_mid_case", "global_prod_t"),
        "D": ("cum_mid_case", "global_reserves_t"),
    }
    num_col, den_col = panel_keys[sort_panel]

    def _ratio(m):
        if m not in df.index:
            return -1.0
        row = df.loc[m]
        num = row.get(num_col, 0) or 0
        den = row.get(den_col, 0) or 0
        if den <= 0:
            return float("inf") if num > 0 else -1.0
        return num / den

    fam_of = {m: fam for fam, mats in MATERIAL_FAMILIES.items() for m in mats}
    # Unmapped materials (shouldn't occur for the manuscript set) are slotted
    # into "Base & alloying" so the families stay clean and contiguous.
    leftover = sorted((m for m in mat_order if m not in fam_of),
                      key=_ratio, reverse=True)
    top_to_bottom = []
    for fam in FAMILY_ORDER:
        members = [m for m in mat_order if fam_of.get(m) == fam]
        if fam == "Base & alloying":
            members += leftover
        members.sort(key=_ratio, reverse=True)
        top_to_bottom += members
    # matplotlib barh draws row 0 at the bottom; reverse so "Bulk commodities"
    # sits at the top of the chart and the REE block at the bottom.
    return list(reversed(top_to_bottom))


def build_records(re_mode="aggregate"):
    """Compose full per-material data for all 4 panels."""
    df, mat_order, _ = load_all_data(re_mode)
    demand_raw, risk = load_data()

    dm = _median_and_range_demand(demand_raw)
    mc = _mid_case_and_worst_scenario(demand_raw)  # per-material Mid_Case + worst-scenario values

    # Carry median + range over to df using the risk-name aggregation already
    # computed in load_all_data. For the "Rare Earths" aggregate row, sum the
    # RE element stats.
    rn_for_mat = dict(DEMAND_TO_RISK)
    # REE aggregate bucket
    ree_elements = [m for m, rn in rn_for_mat.items() if rn == "Rare Earths"]
    if "Yttrium" in demand_raw["material"].unique() and "Yttrium" not in ree_elements:
        ree_elements.append("Yttrium")

    def peak_stats(mat):
        """Return (median, outer p2.5, outer p97.5) peak demand across
        scenarios, combining scenario uncertainty and MC intensity
        uncertainty (the outer 95% CI pooled across the 61 scenarios)."""
        if mat == "Rare Earths":
            # Sum across RE elements per (scenario, year) to get the RE-
            # aggregate MC mean trajectory, then compute peak-year statistics.
            # Use `_annual` columns for cadence-invariant annual rates.
            # For p2/p97, pooling element-level CIs by summing is conservative
            # (assumes comonotone intensities) but is the only defensible
            # aggregation without a joint MC over elements — which we do not
            # retain in the CSV. Document this in caption.
            re_d = demand_raw[demand_raw["material"].isin(ree_elements)]
            mean_col = "mean_annual" if "mean_annual" in re_d.columns else "mean"
            p2_col   = "p2_annual"   if "p2_annual"   in re_d.columns else "p2"
            p97_col  = "p97_annual"  if "p97_annual"  in re_d.columns else "p97"
            by_sy_mean = re_d.groupby(["scenario", "year"])[mean_col].sum()
            by_sy_p2   = re_d.groupby(["scenario", "year"])[p2_col].sum()  if p2_col  in re_d.columns else by_sy_mean
            by_sy_p97  = re_d.groupby(["scenario", "year"])[p97_col].sum() if p97_col in re_d.columns else by_sy_mean
            def _peak_triplet(scen):
                m_s = by_sy_mean.loc[scen]; py = m_s.idxmax()
                return m_s.loc[py], by_sy_p2.loc[scen].loc[py], by_sy_p97.loc[scen].loc[py]
            triplets = [_peak_triplet(s) for s in by_sy_mean.index.get_level_values(0).unique()]
            means, p2s, p97s = zip(*triplets)
            return float(np.median(means)), float(min(p2s)), float(max(p97s))
        if mat in dm.index:
            return (dm.loc[mat, "peak_median"],
                    dm.loc[mat, "peak_min_p025"],
                    dm.loc[mat, "peak_max_p975"])
        return 0.0, 0.0, 0.0

    def cum_stats(mat):
        if mat == "Rare Earths":
            re_d = demand_raw[(demand_raw["material"].isin(ree_elements))
                              & (demand_raw["year"] >= 2026)
                              & (demand_raw["year"] <= 2050)]
            by_s_mean = re_d.groupby("scenario")["mean"].sum()
            by_s_p2   = re_d.groupby("scenario")["p2"].sum()  if "p2"  in re_d.columns else by_s_mean
            by_s_p97  = re_d.groupby("scenario")["p97"].sum() if "p97" in re_d.columns else by_s_mean
            return float(by_s_mean.median()), float(by_s_p2.min()), float(by_s_p97.max())
        if mat in dm.index:
            return (dm.loc[mat, "cum_median"],
                    dm.loc[mat, "cum_min_p025"],
                    dm.loc[mat, "cum_max_p975"])
        return 0.0, 0.0, 0.0

    def mid_case_stats(mat):
        """Return (mid_case_peak, worst_case_peak, mid_case_cum, worst_case_cum)."""
        if mat == "Rare Earths":
            re_d = demand_raw[demand_raw["material"].isin(ree_elements)]
            mean_col = "mean_annual" if "mean_annual" in re_d.columns else "mean"
            by_sy = re_d.groupby(["scenario", "year"])[mean_col].sum()
            peak_by_scen = by_sy.groupby("scenario").max()
            # Cumulative still uses raw `mean` (period totals).
            cum_re = re_d[(re_d["year"] >= 2026) & (re_d["year"] <= 2050)].groupby("scenario")["mean"].sum()
            mid_peak = peak_by_scen.get(REFERENCE_SCENARIO, peak_by_scen.median())
            mid_cum  = cum_re.get(REFERENCE_SCENARIO, cum_re.median())
            return float(mid_peak), float(peak_by_scen.max()), float(mid_cum), float(cum_re.max())
        if mat in mc.index:
            return (float(mc.loc[mat, "peak_mid_case"]),
                    float(mc.loc[mat, "peak_worst_case"]),
                    float(mc.loc[mat, "cum_mid_case"]),
                    float(mc.loc[mat, "cum_worst_case"]))
        return 0.0, 0.0, 0.0, 0.0

    for mat in df.index:
        med, lo, hi = peak_stats(mat)
        df.loc[mat, "peak_median"] = med
        df.loc[mat, "peak_min"]    = lo   # outer p2.5
        df.loc[mat, "peak_max"]    = hi   # outer p97.5
        cm, cl, ch = cum_stats(mat)
        df.loc[mat, "cum_median"]  = cm
        df.loc[mat, "cum_min"]     = cl
        df.loc[mat, "cum_max"]     = ch
        # Mid_Case scenario + worst-case envelope (for scenario_mode="mid_case")
        mpk, wpk, mcm, wcm = mid_case_stats(mat)
        df.loc[mat, "peak_mid_case"]    = mpk
        df.loc[mat, "peak_worst_case"]  = wpk
        df.loc[mat, "cum_mid_case"]     = mcm
        df.loc[mat, "cum_worst_case"]   = wcm

    # Reserves
    us_res  = _get_us_reserves(risk)
    gl_res  = _get_global_reserves(risk)
    crc_map_df = risk["crc"].iloc[:, :2].copy()
    crc_map_df.columns = ["country", "crc"]
    crc_lookup = dict(zip(crc_map_df["country"], crc_map_df["crc"]))
    res_shares = _reserves_by_country_group(risk, crc_lookup)

    for mat in df.index:
        rn = df.loc[mat, "risk_name"]
        df.loc[mat, "us_reserves_t"]     = us_res.get(rn, 0)
        df.loc[mat, "global_reserves_t"] = gl_res.get(rn, 0)
        df.loc[mat, "has_reserves"]      = mat not in NO_SEPARATE_RESERVES

    # Reserve sourcing shares (for stacked bars)
    df["reserve_shares_grouped"] = [res_shares.get(df.loc[m, "risk_name"], {g: 0 for g in GROUP_ORDER}) for m in df.index]

    # Yttrium import-share alias (Panel A & B mix bars). Census Bureau publishes
    # Y under its own HTS code, so Y's import_shares_grouped comes back as
    # ~75% Low risk + 23% China — visibly different from the other four
    # per-element REEs (Nd/Pr/Dy/Tb), which share the "Rare Earths" aggregate
    # at ~35% Low risk + 61% China. Until a per-element REE import-source
    # series is available, force Y to inherit the aggregate so the REE block
    # reads as a contiguous supply-stress story rather than a bar with a
    # different palette dropped at the bottom.
    if "Yttrium" in df.index:
        donor = next((m for m in df.index
                      if df.loc[m, "risk_name"] == "Rare Earths"
                      and df.loc[m, "import_shares_grouped"]), None)
        if donor is not None:
            df.at["Yttrium", "import_shares_grouped"] = dict(
                df.loc[donor, "import_shares_grouped"]
            )

    # Cross-panel consistency pass: reconcile Panel A's us_prod_t (from the
    # aggregate salient sheet) with Panel C's prod_shares_grouped (from the
    # per-country World Data sheet). When USGS withholds US in the per-
    # country data ("W"), the per-country shares exclude US and sum over
    # the rest-of-world. Meanwhile us_prod_t has a nonzero estimate
    # (either directly published in the salient sheet, WITHHELD_OVERRIDES,
    # or the Form 72 trade-balance back-estimation). Injecting us_prod_t
    # into Panel C's mix and expanding global_prod_t to include it keeps
    # the figure internally consistent: Silicon shows the same ~600 kt
    # domestic slice in Panel A and Panel C (instead of 600 kt / 0%).
    for mat in df.index:
        us_p = df.loc[mat, "us_prod_t"] or 0
        shares = df.loc[mat, "prod_shares_grouped"]
        if not isinstance(shares, dict):
            continue
        us_share_existing = shares.get("US domestic", 0) or 0
        gl_p = df.loc[mat, "global_prod_t"] or 0
        # Only inject when US is genuinely missing from Panel C (existing
        # share < 0.5%) AND aggregate us_prod is material (> 1 t/yr).
        if us_share_existing < 0.005 and us_p > 1000 and gl_p > 0:
            true_global = gl_p + us_p
            us_share_new = us_p / true_global
            scale = gl_p / true_global   # = 1 − us_share_new
            # Rescale existing non-US shares.
            new_shares = {g: (shares.get(g, 0) or 0) * scale for g in shares}
            new_shares["US domestic"] = us_share_new
            df.at[mat, "prod_shares_grouped"] = new_shares
            df.at[mat, "global_prod_t"] = true_global

    # Drop materials with zero transition-driven demand across all scenarios
    # and years. These are materials that exist in our intensity database
    # but only for technologies not present in the UPV tech-mix (e.g.,
    # Selenium is CIGS-only, and CIGS receives 0% of UPV capacity per the
    # Seel et al. 2025 LBNL mapping → Se demand = 0 everywhere). Rather
    # than show empty rows with "no modeled demand" markers, exclude them
    # from the figure entirely. Reserves/supply data is retained in df
    # for any future analysis that adds these materials back to the mix.
    zero_demand_mats = [
        m for m in mat_order
        if (df.loc[m, "peak_median"] or 0) <= 0
        and (df.loc[m, "cum_median"] or 0) <= 0
    ]
    if zero_demand_mats:
        mat_order = [m for m in mat_order if m not in zero_demand_mats]
        df = df.drop(zero_demand_mats)

    return df, mat_order


# ═══════════════════════════════════════════════════════════════════════════════
# PANEL DRAW HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_sourcing_bar(ax, i, ratio_median, ratio_min, ratio_max, shares,
                       gap_beyond=None, linthresh=1.0,
                       extension_mode="error_bar"):
    """
    Draw a horizontal stacked bar at y=i:
      - Bar width = ratio_median
      - Segments colored by `shares` (dict: group → fraction of ratio)
      - If `gap_beyond` is given, a gray hatched segment extends from
        (ratio_median - gap_beyond) to ratio_median showing the portion
        of demand that exceeds historical/current supply chain capacity.

    extension_mode:
      - "error_bar" (default): scenario min/max shown as an error-bar whisker
      - "scenario_extension": ratio_median → ratio_max drawn as a solid light-grey
        bar extension (used for the Mid_Case variant, where the whisker
        would represent the Mid_Case→worst-case scenario gap, not MC noise)
    """
    if not np.isfinite(ratio_median) or ratio_median <= 0:
        return
    left = 0.0
    # Domestic first (if present)
    for grp in GROUP_ORDER:
        frac = shares.get(grp, 0)
        if frac <= 0:
            continue
        w = ratio_median * frac
        ax.barh(i, w, left=left, color=GROUP_COLORS[grp],
                edgecolor="white", linewidth=0.4, zorder=3)
        left += w
    # "Unclassified" remainder: USGS production / reserves data sometimes
    # includes an "Other countries" catch-all that can't be CRC-classified,
    # so the per-group shares don't always sum to 1. Fill the gap between
    # (left) and (ratio_median) with the Undefined color so the colored
    # stack always extends to the bar's nominal end, and any remaining
    # scenario-extension / error-bar / demand-gap rendering is visually
    # contiguous with the bar.
    unclassified = ratio_median - left
    if unclassified > ratio_median * 0.005:  # ignore sub-0.5% rounding noise
        ax.barh(i, unclassified, left=left,
                color=GROUP_COLORS.get("Undefined", "#999999"),
                edgecolor="white", linewidth=0.4, zorder=3)
        left = ratio_median
    # Optional "gap" shaded segment inside the bar
    if gap_beyond is not None and gap_beyond > 0:
        gap_start = max(0, ratio_median - gap_beyond)
        ax.barh(i, gap_beyond, left=gap_start, color="#d9d9d9",
                edgecolor="white", linewidth=0.4, hatch="////", zorder=4,
                alpha=0.95)
    # Upper extension (scenario worst case) or min–max error bars
    if extension_mode == "scenario_extension":
        if np.isfinite(ratio_max) and ratio_max > ratio_median:
            ax.barh(i, ratio_max - ratio_median, left=ratio_median,
                    color="#bdbdbd", edgecolor="white", linewidth=0.4,
                    alpha=0.85, zorder=2)
    elif np.isfinite(ratio_min) and np.isfinite(ratio_max) and ratio_max > ratio_min:
        ax.errorbar(
            ratio_median, i,
            xerr=[[max(0, ratio_median - ratio_min)], [max(0, ratio_max - ratio_median)]],
            fmt="none", color="black", lw=1.0, capsize=3, zorder=6, alpha=0.8,
        )


def _mat_labels_with_flags(mat_order, df, panel):
    """Return plain material labels.

    Previously appended a ⊕ glyph to Panel A labels for materials whose
    us_prod_t was back-estimated from the USGS Form 72 identity. Removed
    because the glyph rendered as tofu (□) under some font fallbacks.
    Disclosure of the Form 72 back-estimation method now lives in the
    figure caption / methods text. Function retained for call-site
    stability.
    """
    return list(mat_order)


def _fmt_axis(ax, title, xlabel, mat_order, linthresh=1.0,
              xlim_right=None, xlim_left=None, invert_x=False,
              threshold_note=None, xscale="symlog",
              tick_labels=None):
    """
    threshold_note : short string describing what ratio = 1 means in this
        specific panel. Folded into the panel title as a third line
        (non-bold italic) so the semantic of the shared vertical line is
        unambiguous in each of the four panels (the line at 1 means four
        different things across panels A–D).
    xscale : "symlog" (default) or "linear". Linear scale is appropriate
        for comparative views where the range fits within ~2 orders of
        magnitude per panel; symlog is better when the demand/supply span
        is 3+ orders.
    """
    ax.axvline(1.0, color="black", ls="-", lw=1.8, alpha=0.9, zorder=5)
    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=linthresh)
    # linear is the matplotlib default; nothing to call
    ax.set_yticks(range(len(mat_order)))
    ax.set_yticklabels(tick_labels if tick_labels is not None else mat_order,
                       fontsize=9)
    ax.set_ylim(-0.6, len(mat_order) - 0.4)
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold")
    if threshold_note:
        title = f"{title}\n{threshold_note}"
    # Split title into heading (first line, bold) + subtitle (remaining lines,
    # smaller italic grey) for visual hierarchy. Push the bold title up via
    # `pad` so the italic subtitle has clear vertical separation; anchor the
    # subtitle with va="top" just above the axes so it can't crash into the
    # axes tick labels below.
    if "\n" in title:
        heading, _, subtitle = title.partition("\n")
        ax.set_title(heading, fontsize=11, fontweight="bold", pad=22)
        ax.text(0.5, 1.025, subtitle, transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=8, style="italic", color="#555555")
    else:
        ax.set_title(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15, axis="x")
    if xlim_right is not None and xlim_left is not None:
        ax.set_xlim(xlim_left, xlim_right)
    elif xlim_right is not None:
        ax.set_xlim(0, xlim_right)
    if invert_x:
        ax.invert_xaxis()


def _dominant_source_color(shares):
    """Return the CRC-group color of the dominant (largest-share) source."""
    if not shares:
        return "#888888"
    top = max(shares.items(), key=lambda kv: kv[1])
    return GROUP_COLORS.get(top[0], "#888888")


def _draw_simple_bar(ax, i, value_median, value_min, value_max, color,
                     label_text=None, extension_mode="error_bar"):
    """
    Draw a simple single-color horizontal bar (used for the
    supply-over-demand framing, where the stacked-by-source decomposition
    does not apply cleanly).

    extension_mode:
      - "error_bar" (default) — black whisker for [value_min, value_max]
      - "scenario_extension" — auto-detects direction:
          * value_max > value_median: RIGHT extension (grey bar appended)
            [used by V1, demand÷supply, where worst-case = bigger ratio]
          * value_min < value_median: LEFT OVERLAY (grey hatched rectangle
            layered on top of the colored bar, from value_min to value_median)
            [used by V2, supply÷demand, where worst-case = smaller ratio —
             the overlay flags the "scenario-sensitive, at-risk" portion]
    """
    if not np.isfinite(value_median) or value_median <= 0:
        return
    ax.barh(i, value_median, left=0, color=color,
            edgecolor="white", linewidth=0.4, zorder=3)
    if extension_mode == "scenario_extension":
        if np.isfinite(value_max) and value_max > value_median:
            # V1-style right extension: grey after the bar
            ax.barh(i, value_max - value_median, left=value_median,
                    color="#bdbdbd", edgecolor="white", linewidth=0.4,
                    alpha=0.85, zorder=2)
        elif np.isfinite(value_min) and 0 < value_min < value_median:
            # V2-style left overlay: grey hatched on top of the colored bar
            # from value_min (worst-case) to value_median (mid-case). Signals
            # "this portion is scenario-sensitive — under worst case the bar
            # would end at value_min."
            # Dark-hatch overlay with no border — matches the colored bar's
            # vertical extent exactly, lets the underlying color show through,
            # and reads as "this portion is scenario-sensitive" without
            # visually suggesting a separate/taller rectangle.
            ax.barh(i, value_median - value_min, left=value_min,
                    facecolor="none", edgecolor="#222222", linewidth=0,
                    hatch="/////", zorder=5)
    elif np.isfinite(value_min) and np.isfinite(value_max) and value_max > value_min:
        ax.errorbar(
            value_median, i,
            xerr=[[max(0, value_median - value_min)], [max(0, value_max - value_median)]],
            fmt="none", color="black", lw=1.0, capsize=3, zorder=6, alpha=0.8,
        )
    if label_text is not None:
        ax.text(value_median * 1.05, i, label_text,
                fontsize=7, va="center", ha="left", color="#333")


def _draw_us_plus_foreign_bar(ax, i, value_median, value_min, value_max,
                              us_fraction, dominant_foreign_color,
                              extension_mode="error_bar"):
    """
    Two-segment bar (V2 Panels C & D): blue US segment + one foreign
    segment colored by the dominant non-US CRC group. Makes US presence
    visible on every bar even when US is not the dominant producer.

    us_fraction ∈ [0, 1] is the fraction of the total bar width attributable
    to the United States (e.g., US_production / global_production for
    Panel C; US_reserves / global_reserves for Panel D).

    extension_mode: see _draw_simple_bar. In V2 mid_case mode the worst-case
    position is to the LEFT of value_median, so the scenario-sensitive
    portion is rendered as a hatched overlay on top of the colored bar
    (from value_min to value_median).
    """
    if not np.isfinite(value_median) or value_median <= 0:
        return
    us_w = max(0.0, min(1.0, us_fraction)) * value_median
    foreign_w = value_median - us_w
    if us_w > 0:
        ax.barh(i, us_w, left=0, color=GROUP_COLORS["US domestic"],
                edgecolor="white", linewidth=0.4, zorder=3)
    if foreign_w > 0:
        ax.barh(i, foreign_w, left=us_w, color=dominant_foreign_color,
                edgecolor="white", linewidth=0.4, zorder=3)
    if extension_mode == "scenario_extension":
        if np.isfinite(value_max) and value_max > value_median:
            ax.barh(i, value_max - value_median, left=value_median,
                    color="#bdbdbd", edgecolor="white", linewidth=0.4,
                    alpha=0.85, zorder=2)
        elif np.isfinite(value_min) and 0 < value_min < value_median:
            # Dark-hatch overlay with no border — matches the colored bar's
            # vertical extent exactly, lets the underlying color show through,
            # and reads as "this portion is scenario-sensitive" without
            # visually suggesting a separate/taller rectangle.
            ax.barh(i, value_median - value_min, left=value_min,
                    facecolor="none", edgecolor="#222222", linewidth=0,
                    hatch="/////", zorder=5)
    elif np.isfinite(value_min) and np.isfinite(value_max) and value_max > value_min:
        ax.errorbar(
            value_median, i,
            xerr=[[max(0, value_median - value_min)], [max(0, value_max - value_median)]],
            fmt="none", color="black", lw=1.0, capsize=3, zorder=6, alpha=0.8,
        )


def _dominant_non_us_color(shares):
    """Return the CRC-group color of the largest non-US supply source."""
    if not shares:
        return "#888888"
    non_us = {g: v for g, v in shares.items() if g != "US domestic" and v > 0}
    if not non_us:
        return GROUP_COLORS.get("US domestic", "#888888")
    top = max(non_us.items(), key=lambda kv: kv[1])
    return GROUP_COLORS.get(top[0], "#888888")


# ═══════════════════════════════════════════════════════════════════════════════
# 4-PANEL FIGURE
# ═══════════════════════════════════════════════════════════════════════════════

def _headline_for_panel(df, which):
    """Build a short data-driven headline for the subtitle of each panel."""
    if which == "A":
        # Count materials where US peak demand exceeds US production.
        over = []
        for m, r in df.iterrows():
            us = r.get("us_prod_t", 0)
            if us and us > 0:
                ratio = r["peak_median"] / us
                if ratio > 1:
                    over.append((m, ratio))
        over.sort(key=lambda kv: -kv[1])
        if over:
            m0, r0 = over[0]
            return (f"{len(over)}/{(df['us_prod_t'] > 0).sum()} materials exceed US annual production — "
                    f"{m0} peaks at {r0:.0f}×")
        return "No material exceeds US annual production"
    if which == "B":
        zeros = []
        over = []
        for m, r in df.iterrows():
            has = r.get("has_reserves", False)
            res = r.get("us_reserves_t", 0) or 0
            if not has:
                continue
            if res <= 0:
                zeros.append(m)
                continue
            ratio = r["cum_median"] / res
            if ratio > 1:
                over.append((m, ratio))
        parts = []
        if zeros:
            parts.append(f"{len(zeros)} materials have no US reserves: {', '.join(zeros[:4])}")
        if over:
            m0, r0 = max(over, key=lambda kv: kv[1])
            parts.append(f"{len(over)} consume >100% of US reserves (peak: {m0} {r0:.1f}×)")
        return "; ".join(parts) if parts else "US reserves cover all materials"
    if which == "C":
        pairs = []
        for m, r in df.iterrows():
            gl = r.get("global_prod_t", 0)
            if gl and gl > 0:
                pairs.append((m, r["peak_median"] / gl))
        pairs.sort(key=lambda kv: -kv[1])
        m0, r0 = pairs[0] if pairs else ("", 0)
        over = [x for x in pairs if x[1] > 0.1]
        return (f"{len(over)} materials need >10% of global output at peak — "
                f"{m0} {r0*100:.0f}%")
    if which == "D":
        pairs = []
        for m, r in df.iterrows():
            has = r.get("has_reserves", False)
            gl = r.get("global_reserves_t", 0) or 0
            if has and gl > 0:
                pairs.append((m, r["cum_median"] / gl))
        pairs.sort(key=lambda kv: -kv[1])
        if pairs:
            m0, r0 = pairs[0]
            over10 = [x for x in pairs if x[1] > 0.1]
            return (f"{len(over10)} materials: US transition consumes >10% of global reserves — "
                    f"{m0} {r0*100:.0f}%")
        return "Global reserves amply cover US transition"
    return ""


def plot_4panel(df, mat_order, output_path, framing="demand-over-supply",
                xscale="symlog", scenario_mode="median", styling="manuscript",
                split=False):
    """
    framing:
      "demand-over-supply"  (default, matches literature — Graedel 2012,
         Habib & Wenzel 2014): bar = demand / supply, threshold at 1.0.
         Bars above 1 are vulnerability signals. Segments stacked by
         import-source or production-country CRC group.
      "supply-over-demand" (more interpretable — Habib 2019): bar =
         supply / demand. For reserves, this reads as "years of coverage
         at peak demand" (multiply by 1 for fraction-of-transition).
         Values below 1 are vulnerability signals. Bars single-colored
         by the dominant supply-source group.
    """
    if framing not in ("demand-over-supply", "supply-over-demand"):
        raise ValueError(f"Unknown framing: {framing}")

    inverse = (framing == "supply-over-demand")

    # Scenario-mode view: for scenario_mode="mid_case", swap the per-material
    # central-value and upper-bound with Mid_Case peak and worst-case-scenario
    # peak. The existing bar-drawing code will read peak_median/cum_median
    # (now = Mid_Case) and draw error bars from peak_min→peak_max; to make
    # the upper bar render as a grey "scenario-pessimism" extension instead
    # of an error-bar whisker, we post-process below.
    if scenario_mode == "mid_case":
        df = df.copy()
        # Substitute the central value with Mid_Case scenario's peak/cumulative.
        # Lower bound = central (no downside whisker in mid_case mode).
        # Upper bound = worst-case scenario → becomes the grey extension.
        df["peak_median"] = df["peak_mid_case"]
        df["peak_min"]    = df["peak_mid_case"]
        df["peak_max"]    = df["peak_worst_case"]
        df["cum_median"]  = df["cum_mid_case"]
        df["cum_min"]     = df["cum_mid_case"]
        df["cum_max"]     = df["cum_worst_case"]
        ext_mode = "scenario_extension"
    else:
        ext_mode = "error_bar"

    # Precompute headline subtitles (same facts for both framings;
    # framing-specific wording handled in each panel title below).
    headline_a = _headline_for_panel(df, "A")
    headline_b = _headline_for_panel(df, "B")
    headline_c = _headline_for_panel(df, "C")
    headline_d = _headline_for_panel(df, "D")

    # Poster styling: re-sort mat_order by Panel-A Mid_Case ratio ascending,
    # with "no US production" rows pinned LAST in mat_order (= TOP of plot,
    # since matplotlib draws bottom-up from y=0).
    if styling == "poster":
        def _poster_sort_key(mat):
            row = df.loc[mat]
            us_p = row.get("us_prod_t", 0) or 0
            peak = row.get("peak_mid_case", row.get("peak_median", 0)) or 0
            if us_p <= 0:
                return (1, float("inf"))
            return (0, peak / us_p if us_p > 0 else 0)
        mat_order = sorted(mat_order, key=_poster_sort_key)

    # Use one material order for all 4 panels, driven by US-production ratio
    n_mat = len(mat_order)
    if split:
        # Two stacked figures: US tiers (A over B) and global tiers (C over D).
        # The downstream panel-drawing code references only the axis handles
        # below, so it is unchanged; only setup, banners, and save are split-aware.
        fig_us = plt.figure(figsize=(10, max(13, n_mat * 0.62)))
        gs_us = fig_us.add_gridspec(2, 1, hspace=0.42)
        ax_usp = fig_us.add_subplot(gs_us[0, 0])
        ax_usr = fig_us.add_subplot(gs_us[1, 0])
        fig_gl = plt.figure(figsize=(10, max(13, n_mat * 0.62)))
        gs_gl = fig_gl.add_gridspec(2, 1, hspace=0.42)
        ax_glp = fig_gl.add_subplot(gs_gl[0, 0])
        ax_glr = fig_gl.add_subplot(gs_gl[1, 0])
        fig = fig_us  # fallback handle; banners/legend/save are split-aware below
    else:
        fig_us = fig_gl = None
        fig = plt.figure(figsize=(18, max(11, n_mat * 0.4)))
        gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.58)
        ax_usp = fig.add_subplot(gs[0, 0])
        ax_usr = fig.add_subplot(gs[0, 1])
        ax_glp = fig.add_subplot(gs[1, 0])
        ax_glr = fig.add_subplot(gs[1, 1])

    # Poster styling: row banding and clean spines on all 4 axes.
    if styling == "poster":
        for _ax in (ax_usp, ax_usr, ax_glp, ax_glr):
            for _i in range(len(mat_order)):
                if _i % 2 == 1:
                    _ax.axhspan(_i - 0.5, _i + 0.5, color="#f2f2f2",
                                zorder=0, alpha=0.6)
            _ax.spines[["top", "right"]].set_visible(False)

    def _poster_no_data_label(ax, i, text):
        """Render a centered bold maroon 'no data' label for poster styling."""
        trans = blended_transform_factory(ax.transAxes, ax.transData)
        ax.text(0.5, i, text, transform=trans, va="center", ha="center",
                fontsize=11, fontweight="bold", color=POSTER_NO_DATA_COLOR,
                style="normal", zorder=4)

    # Row-level semantic banners (2026-04-18 research-scientist audit):
    # the top-row decomposition uses Census HTS US-actual import partners,
    # while the bottom-row decomposition uses USGS MCS global production /
    # reserves geography. These are different data sources answering
    # different questions; flag the split visually so readers don't assume
    # the color semantics are identical across all four panels.
    if split:
        fig_us.text(
            0.5, 0.965,
            "US-actual sourcing (Census HTS import partners, 2020-2023)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50", style="italic",
        )
        fig_gl.text(
            0.5, 0.965,
            "Global supply geography (USGS MCS 2025 per-country production / reserves)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50", style="italic",
        )
    else:
        fig.text(
            0.5, 0.935,
            "Top row: US-actual sourcing (Census HTS import partners, 2020-2023)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50",
            style="italic",
        )
        fig.text(
            0.5, 0.495,
            "Bottom row: Global supply geography (USGS MCS 2025 per-country production / reserves)",
            ha="center", va="bottom",
            fontsize=10, fontweight="bold", color="#2c3e50",
            style="italic",
        )

    # ── Panel A: US Production ────────────────────────────────────────────
    max_ratio_a = 0
    min_ratio_a = np.inf
    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        us_t = row["us_prod_t"]
        if not (us_t and us_t > 0):
            if styling == "poster":
                _poster_no_data_label(ax_usp, i, "No US production")
            else:
                ax_usp.text(0.02, i, " no US production", fontsize=7,
                            va="center", style="italic", color="#888", zorder=3)
            continue
        if not inverse:
            r_med = row["peak_median"] / us_t
            r_lo  = row["peak_min"]    / us_t
            r_hi  = row["peak_max"]    / us_t
        else:
            # supply / demand = US_production / peak_demand
            r_med = us_t / row["peak_median"] if row["peak_median"] > 0 else 0
            # flipped bounds (supply fixed, demand varies)
            r_lo  = us_t / row["peak_max"]    if row["peak_max"]    > 0 else 0
            r_hi  = us_t / row["peak_min"]    if row["peak_min"]    > 0 else 0
        max_ratio_a = max(max_ratio_a, r_hi)
        if r_med > 0: min_ratio_a = min(min_ratio_a, r_lo)

        nir = row["import_dependency"]
        crc_shares = row["import_shares_grouped"]

        if not inverse:
            # (demand / supply) framing. Bar = peak demand / US production.
            # Revised baseline 2026-04-18 (PJ review): the "gap" threshold is
            # now US PRODUCTION (ratio = 1), not current consumption
            # (ratio = 1/(1-NIR)). The older consumption-based baseline
            # correctly reflected "does transition demand exceed today's total
            # consumption", but only 3 materials broke through — understating
            # the bottleneck story. The production-based baseline asks the
            # simpler policy question: "does transition demand exceed what
            # the US can make domestically?"
            #
            #   US-domestic segment          = min(1.0, r_med)    [US can make at most its annual production]
            #   Import-group segments        = (r_med − 1) × crc_share[grp]  if r_med > 1
            #   Gray "new-demand gap"        = portion of import-covered region above 1/(1-NIR) of production
            #                                  (i.e., new imports beyond current import tonnage)
            #
            # The result:
            #   Materials with r_med ≤ 1: entire bar is blue (US can supply).
            #   Materials with 1 < r_med ≤ 1/(1-NIR): blue up to 1, imports
            #     (CRC colors) from 1 to r_med. Covered by current chain
            #     scaled proportionally.
            #   Materials with r_med > 1/(1-NIR): blue up to 1, imports
            #     from 1 to 1/(1-NIR) (current tonnage), gray hatched from
            #     1/(1-NIR) to r_med (new imports needed beyond today's
            #     import flows).
            # Capped-blue stacking for Panel A (2026-04-18 audit correction):
            # the blue US-domestic segment must never exceed 1.0 in ratio
            # units because US annual production IS the denominator. Segment
            # math:
            #   r_med ≤ 1: blue = r_med (US production covers the demand)
            #   r_med > 1: blue = 1 (capped at US production capacity)
            #              imports = r_med − 1 (distributed by Census CRC shares)
            # All segments sum to r_med, so the bar is always contiguous.
            shares = {grp: 0.0 for grp in GROUP_ORDER}
            if r_med <= 1.0:
                # US production alone covers peak demand.
                shares["US domestic"] = 1.0
            else:
                # US production capped at 1 unit; imports fill the remainder.
                # Share is segment_width ÷ r_med, so a blue segment of width
                # exactly 1.0 corresponds to share = 1/r_med.
                shares["US domestic"] = 1.0 / r_med
                import_band = r_med - 1.0
                for grp in GROUP_ORDER[1:]:
                    shares[grp] = (import_band / r_med) * crc_shares.get(grp, 0)
            covered = r_med

            if covered > 0:
                _draw_sourcing_bar(ax_usp, i, covered, r_lo, r_hi, shares,
                                   linthresh=1.0, extension_mode=ext_mode)

            if r_med < 0.01:
                ax_usp.text(
                    0.015, i,
                    f" net exporter (demand/US prod ≈ {r_med:.1e})",
                    fontsize=6.5, va="center", style="italic", color="#2a7",
                    zorder=3,
                )
        else:
            # (supply / demand) framing: simple single-color bar, colored by
            # the dominant-demand-sourcing group (so the viewer still reads
            # which region would bear the brunt of scaling imports).
            # Build implied demand sourcing shares (same logic as left panel)
            if nir < 0.999:
                hist_cover = 1.0 / (1.0 - nir)
            else:
                hist_cover = 1.0
            shares = {grp: 0.0 for grp in GROUP_ORDER}
            shares["US domestic"] = 1.0 / hist_cover
            for grp in GROUP_ORDER[1:]:
                shares[grp] = (1 - 1.0 / hist_cover) * crc_shares.get(grp, 0)
            color = _dominant_source_color(shares)
            _draw_simple_bar(ax_usp, i, r_med, r_lo, r_hi, color,
                              extension_mode=ext_mode)

    scale_lbl = "log" if xscale == "symlog" else "linear"
    panel_a_labels = _mat_labels_with_flags(mat_order, df, "A")
    if not inverse:
        _fmt_axis(ax_usp,
                  "A. US production\ndeficit split by 2020–23 US import partners (CRC group)",
                  f"US peak annual demand ÷ US annual production ({scale_lbl})",
                  mat_order, linthresh=0.01, xscale=xscale,
                  xlim_right=max_ratio_a * 1.1 if max_ratio_a > 0 else None,
                  tick_labels=panel_a_labels)
    else:
        _fmt_axis(ax_usp,
                  "A. US production\nbar colored by dominant 2020–23 US import partner (CRC group)",
                  f"US annual production ÷ peak-year demand ({scale_lbl})",
                  mat_order, linthresh=0.1, xscale=xscale,
                  xlim_right=max(100, max_ratio_a * 1.2) if max_ratio_a > 0 else 100,
                  tick_labels=panel_a_labels)

    # ── Panel B: US Reserves ──────────────────────────────────────────────
    max_ratio_b = 0
    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        us_res = row.get("us_reserves_t", 0) or 0
        has_res = row.get("has_reserves", False)
        if not has_res:
            if styling == "poster":
                _poster_no_data_label(ax_usr, i, "No separate reserves concept")
            else:
                ax_usr.text(0.02, i, " no separate reserves concept",
                            fontsize=7, va="center", style="italic", color="#888", zorder=3)
            continue
        if us_res <= 0:
            if styling == "poster":
                _poster_no_data_label(ax_usr, i, "No US reserves (100% import-dependent)")
            else:
                ax_usr.text(0.02, i, " no US reserves (100% import-dependent)",
                            fontsize=7, va="center", style="italic", color="#c0392b", zorder=3)
            continue

        if not inverse:
            r_med = row["cum_median"] / us_res
            r_lo  = row["cum_min"]    / us_res
            r_hi  = row["cum_max"]    / us_res
        else:
            # reserves / cumulative_demand = multiple of 2026–2050 transition
            # covered by current US reserves. Equivalently, (reserves / peak
            # annual demand) gives "years of coverage at peak annual rate".
            r_med = us_res / row["cum_median"] if row["cum_median"] > 0 else 0
            r_lo  = us_res / row["cum_max"]    if row["cum_max"]    > 0 else 0
            r_hi  = us_res / row["cum_min"]    if row["cum_min"]    > 0 else 0
        max_ratio_b = max(max_ratio_b, r_hi)

        nir = row["import_dependency"]
        crc_shares = row["import_shares_grouped"]
        # Capped-blue stacking for Panel B (2026-04-18 audit, same rationale
        # as Panel A): US reserves IS the denominator, so the blue segment
        # must not exceed 1.0 in ratio units. Where r_med > 1 (cumulative
        # demand exceeds US reserves), imports cover the remainder, scaled
        # by historical Census import shares.
        shares = {grp: 0.0 for grp in GROUP_ORDER}
        if r_med <= 1.0:
            shares["US domestic"] = 1.0
        else:
            shares["US domestic"] = 1.0 / r_med
            import_band = r_med - 1.0
            for grp in GROUP_ORDER[1:]:
                shares[grp] = (import_band / r_med) * crc_shares.get(grp, 0)

        if not inverse:
            _draw_sourcing_bar(ax_usr, i, r_med, r_lo, r_hi, shares,
                               linthresh=1.0, extension_mode=ext_mode)
        else:
            _draw_simple_bar(ax_usr, i, r_med, r_lo, r_hi,
                              _dominant_source_color(shares),
                              extension_mode=ext_mode)

    if not inverse:
        _fmt_axis(ax_usr,
                  "B. US reserves\ndeficit split by 2020–23 US import partners (CRC group)",
                  f"US cumulative demand ÷ US reserves ({scale_lbl})",
                  mat_order, linthresh=0.01, xscale=xscale,
                  xlim_right=max_ratio_b * 1.2 if max_ratio_b > 0 else None)
    else:
        _fmt_axis(ax_usr,
                  "B. US reserves\nbar colored by dominant 2020–23 US import partner (CRC group)",
                  f"US reserves ÷ cumulative 2026–2050 demand ({scale_lbl})",
                  mat_order, linthresh=0.1, xscale=xscale,
                  xlim_right=max(100, max_ratio_b * 1.2) if max_ratio_b > 0 else 100)

    # ── Panel C: Global Production ────────────────────────────────────────
    max_ratio_c = 0
    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        gl_t = row["global_prod_t"]
        if not (gl_t and gl_t > 0):
            if styling == "poster":
                _poster_no_data_label(ax_glp, i, "No production data")
            else:
                ax_glp.text(0.0005, i, " no production data",
                            fontsize=7, va="center", style="italic", color="#888", zorder=3)
            continue
        if not inverse:
            r_med = row["peak_median"] / gl_t
            r_lo  = row["peak_min"]    / gl_t
            r_hi  = row["peak_max"]    / gl_t
        else:
            r_med = gl_t / row["peak_median"] if row["peak_median"] > 0 else 0
            r_lo  = gl_t / row["peak_max"]    if row["peak_max"]    > 0 else 0
            r_hi  = gl_t / row["peak_min"]    if row["peak_min"]    > 0 else 0
        max_ratio_c = max(max_ratio_c, r_hi)

        shares = row["prod_shares_grouped"]
        if not inverse:
            _draw_sourcing_bar(ax_glp, i, r_med, r_lo, r_hi, shares,
                               linthresh=0.01, extension_mode=ext_mode)
        else:
            # Two-segment bar: US + dominant foreign CRC group. Without this
            # the US would be invisible in Panel C everywhere it is not the
            # top global producer (e.g., essentially all critical materials
            # except Boron), losing context for the reader.
            us_frac = (row["us_prod_t"] / gl_t) if gl_t > 0 else 0.0
            _draw_us_plus_foreign_bar(
                ax_glp, i, r_med, r_lo, r_hi,
                us_fraction=us_frac,
                dominant_foreign_color=_dominant_non_us_color(shares),
                extension_mode=ext_mode,
            )

    if not inverse:
        _fmt_axis(ax_glp,
                  "C. Global production\nbar split by producer-country share (USGS MCS 2025, CRC group)",
                  f"US peak annual demand ÷ global annual production ({scale_lbl})",
                  mat_order, linthresh=0.01, xscale=xscale,
                  xlim_right=max(1.2, max_ratio_c * 1.5) if max_ratio_c > 0 else 1.2)
    else:
        _fmt_axis(ax_glp,
                  "C. Global production\nUS slice + dominant non-US producer (USGS MCS 2025, CRC group)",
                  f"Global annual production ÷ US peak-year demand ({scale_lbl})",
                  mat_order, linthresh=0.1, xscale=xscale,
                  xlim_right=max(100, max_ratio_c * 1.2) if max_ratio_c > 0 else 100)

    # ── Panel D: Global Reserves ──────────────────────────────────────────
    max_ratio_d = 0
    for i, mat in enumerate(mat_order):
        row = df.loc[mat]
        gl_res = row.get("global_reserves_t", 0) or 0
        has_res = row.get("has_reserves", False)
        if not has_res:
            if styling == "poster":
                _poster_no_data_label(ax_glr, i, "No separate reserves concept")
            else:
                ax_glr.text(0.0005, i, " no separate reserves concept",
                            fontsize=7, va="center", style="italic", color="#888", zorder=3)
            continue
        if gl_res <= 0:
            if styling == "poster":
                _poster_no_data_label(ax_glr, i, "No reserve data")
            else:
                ax_glr.text(0.0005, i, " no reserve data",
                            fontsize=7, va="center", style="italic", color="#888", zorder=3)
            continue
        if not inverse:
            r_med = row["cum_median"] / gl_res
            r_lo  = row["cum_min"]    / gl_res
            r_hi  = row["cum_max"]    / gl_res
        else:
            r_med = gl_res / row["cum_median"] if row["cum_median"] > 0 else 0
            r_lo  = gl_res / row["cum_max"]    if row["cum_max"]    > 0 else 0
            r_hi  = gl_res / row["cum_min"]    if row["cum_min"]    > 0 else 0
        max_ratio_d = max(max_ratio_d, r_hi)
        shares = row.get("reserve_shares_grouped", {g: 0 for g in GROUP_ORDER})
        if not inverse:
            _draw_sourcing_bar(ax_glr, i, r_med, r_lo, r_hi, shares,
                               linthresh=0.01, extension_mode=ext_mode)
        else:
            us_res = row.get("us_reserves_t", 0) or 0
            us_frac = (us_res / gl_res) if gl_res > 0 else 0.0
            _draw_us_plus_foreign_bar(
                ax_glr, i, r_med, r_lo, r_hi,
                us_fraction=us_frac,
                dominant_foreign_color=_dominant_non_us_color(shares),
                extension_mode=ext_mode,
            )

    if not inverse:
        _fmt_axis(ax_glr,
                  "D. Global reserves\nbar split by reserve-holder country share (USGS MCS 2025, CRC group)",
                  f"US cumulative demand ÷ global reserves ({scale_lbl})",
                  mat_order, linthresh=0.01, xscale=xscale,
                  xlim_right=max(1.2, max_ratio_d * 1.5) if max_ratio_d > 0 else 1.2)
    else:
        _fmt_axis(ax_glr,
                  "D. Global reserves\nUS slice + dominant non-US reserve holder (USGS MCS 2025, CRC group)",
                  f"Global reserves ÷ US cumulative 2026–2050 demand ({scale_lbl})",
                  mat_order, linthresh=0.1, xscale=xscale,
                  xlim_right=max(100, max_ratio_d * 1.2) if max_ratio_d > 0 else 100)

    # Poster styling: bold + maroon REE labels on all 4 axes' y-tick labels.
    if styling == "poster":
        for _ax in (ax_usp, ax_usr, ax_glp, ax_glr):
            for _lab, _mat in zip(_ax.get_yticklabels(), mat_order):
                if _mat in REE_HIGHLIGHT_SET:
                    _lab.set_fontweight("bold")
                    _lab.set_color(REE_HIGHLIGHT_COLOR)

    # ── Unified legend ────────────────────────────────────────────────────
    handles_prod = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    if not inverse:
        threshold_label = "Self-sufficiency threshold (ratio = 1.0)"
    else:
        threshold_label = "Adequacy threshold (supply ÷ demand = 1.0)"

    if scenario_mode == "mid_case":
        if not inverse:
            worst_case_handle = mpatches.Patch(
                facecolor="#bdbdbd", edgecolor="white",
                label="Mid Case → worst-case scenario extension")
        else:
            worst_case_handle = mpatches.Patch(
                facecolor="#d5d5d5", edgecolor="none", hatch="/////",
                label="Scenario-sensitive portion (shrinks to worst-case)")
        handles_err = [
            worst_case_handle,
            plt.Line2D([], [], color="black", ls="-", lw=2, label=threshold_label),
        ]
    else:
        handles_err = [
            plt.Line2D([], [], color="black", marker="|", ls="-", markersize=10,
                       label="Scenario min–max (error bars)"),
            plt.Line2D([], [], color="black", ls="-", lw=2, label=threshold_label),
        ]

    # Publication-ready figure title (split-aware: applied per figure below in split mode).
    suptitle = "Figure 6. Critical-material demand versus supply under the US clean-energy transition, 2026-2050"
    if not split:
        fig.suptitle(suptitle, fontsize=13, fontweight="bold", y=0.995)

    # Short, concise legend title.
    if not inverse:
        bar_legend_title = "Supply-source country risk group (OECD CRC 2026)"
    else:
        bar_legend_title = (
            "Supply-source country risk group (OECD CRC 2026). "
            "Panels A, B: dominant source. Panels C, D: US segment + "
            "dominant foreign group."
        )

    leg2_title = ("Worst-case extension & threshold" if scenario_mode == "mid_case"
                  else "Error bars & threshold")
    if not split:
        leg1 = fig.legend(handles=handles_prod,
                          loc="lower center", bbox_to_anchor=(0.32, -0.06),
                          ncol=5, fontsize=9, frameon=False,
                          title=bar_legend_title, title_fontsize=8.5)
        leg1.get_title().set_fontweight("bold")
        leg2 = fig.legend(handles=handles_err,
                          loc="lower center", bbox_to_anchor=(0.82, -0.06),
                          ncol=1, fontsize=9, frameon=False,
                          title=leg2_title, title_fontsize=9)
        leg2.get_title().set_fontweight("bold")

    # Methodological footer: the static R/P (reserve-to-production) framing
    # in panels B and D is known to overstate scarcity (Graedel et al.
    # 2015, PNAS 112:4257; Tilton 2003, Resources Policy 29:195). Flagging
    # it in-figure prevents a misread of "reserves = finite atoms".
    # In-figure caption removed (2026-04-18): methodological notes belong in
    # the manuscript figure-caption / Methods §2.5, not on the figure itself.
    # The dual-semantic row banners (Census vs USGS) and the legend carry
    # the minimum information needed to read the figure standalone.

    # Back-estimation footnote removed 2026-04-29 — Form 72 back-estimation
    # disclosure now lives in the figure caption / methods text rather than
    # an in-figure glyph (which had tofu-rendering issues in some font
    # fallbacks and lacked a footer on the alt-variants poster).

    if split:
        out_us = output_path.parent / "fig4_supply_tiers_us.png"
        out_gl = output_path.parent / "fig5_supply_tiers_global.png"
        for _fig, _ttl, _out in (
            (fig_us, "Figure 5. US critical-material demand versus domestic supply, 2026-2050", out_us),
            (fig_gl, "Figure 6. US critical-material demand versus global supply, 2026-2050", out_gl),
        ):
            _fig.suptitle(_ttl, fontsize=12, fontweight="bold", y=0.997)
            _l1 = _fig.legend(handles=handles_prod, loc="lower center",
                              bbox_to_anchor=(0.5, -0.03), ncol=5, fontsize=8,
                              frameon=False, title=bar_legend_title, title_fontsize=8)
            _l1.get_title().set_fontweight("bold")
            _l2 = _fig.legend(handles=handles_err, loc="lower center",
                              bbox_to_anchor=(0.5, -0.075), ncol=3, fontsize=8,
                              frameon=False, title=leg2_title, title_fontsize=8)
            _l2.get_title().set_fontweight("bold")
            _fig.tight_layout(rect=[0, 0.05, 1, 0.95])
            _fig.savefig(_out, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(_fig)
            print(f"  Saved {_out.name}")
    else:
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    add_interpolation_arg(parser)
    parser.add_argument("--output-dir", type=Path,
                         default=FIGURES_MANUSCRIPT_DIR,
                        help="Top-level output directory. A 'fig4_fig5_variants/' "
                             "subfolder will be created inside it; both framings × "
                             "both x-axis scales are written there. The literature-"
                             "standard (demand÷supply, log-scale) copy is also "
                             "written at the top level for the pipeline default.")
    parser.add_argument("--re-mode", choices=["aggregate", "per-element"],
                         default="aggregate")
    parser.add_argument(
        "--framing",
        choices=["demand-over-supply", "supply-over-demand", "both"],
        default="both",
        help="demand-over-supply: bar = demand/supply (literature-standard, "
             "Graedel 2012). supply-over-demand: bar = supply/demand "
             "(interpretable: 'years of coverage' / 'coverage fraction').",
    )
    parser.add_argument(
        "--scale",
        choices=["log", "linear", "both"],
        default="log",
        help="Log (symlog, default) is the publication-quality choice — "
             "critical materials span 3+ orders of magnitude in demand/supply. "
             "Linear is available for poster/presentation use but not recommended "
             "for the main-text figure.",
    )
    parser.add_argument(
        "--scenario-mode",
        choices=["median", "mid_case", "both"],
        default="median",
        help="median (default): bar = median peak across 61 scenarios; error "
             "bars = outer p2.5/p97.5 envelope (combines scenario + MC intensity "
             "uncertainty). mid_case: bar = Mid_Case scenario peak alone; grey "
             "extension = Mid_Case → worst-case scenario across the 61 NREL "
             "scenarios. 'both' writes both variants side-by-side.",
    )
    parser.add_argument(
        "--styling",
        choices=["manuscript", "poster"],
        default="manuscript",
        help="manuscript (default): standard publication styling. "
             "poster: row banding, REE labels in maroon, larger inline 'no US production' "
             "text, sort by Panel-A Mid_Case ratio ascending. Compatible with all framings.",
    )
    parser.add_argument(
        "--legacy-colored",
        action="store_true",
        help="Render the pre-2026-05-21 color-by-CRC layout (single bar "
             "per material with CRC segments) instead of the new split "
             "magnitude|mix layout. Use to reproduce older manuscript "
             "figures or for poster/slide variants that need the compact "
             "single-bar form.",
    )
    parser.add_argument(
        "--docx",
        action="store_true",
        default=False,
        help="Typography-only: uniformly enlarge all in-plot text in the "
             "split fig4/fig5 (US-tiers / global-tiers) figures so they read "
             "legibly when inserted into a Word docx at a small insert width. "
             "Scales every text artist by a single factor, preserving the "
             "designed hierarchy. Default output (no --docx) is unchanged.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df, mat_order = build_records(re_mode=args.re_mode)

    default_v1 = args.output_dir / "fig4_fig5_supply_tiers.png"
    default_v2 = args.output_dir / "fig4_fig5_supply_tiers_supply_coverage.png"

    if args.legacy_colored:
        # Legacy color-by-CRC layout for reproducing pre-2026-05-21 figures.
        framings_to_run = (
            ["demand-over-supply", "supply-over-demand"] if args.framing == "both"
            else [args.framing]
        )
        scales_to_run = (
            ["symlog", "linear"] if args.scale == "both"
            else (["symlog"] if args.scale == "log" else ["linear"])
        )
        scenario_modes_to_run = (
            ["median", "mid_case"] if args.scenario_mode == "both"
            else [args.scenario_mode]
        )

        variants_dir = args.output_dir / "fig4_fig5_variants"
        variants_dir.mkdir(parents=True, exist_ok=True)

        styling_token = "_poster" if args.styling == "poster" else ""
        for fr in framings_to_run:
            fr_token = "demand_over_supply" if fr == "demand-over-supply" else "supply_over_demand"
            for sc in scales_to_run:
                sc_token = "log" if sc == "symlog" else "linear"
                for sm in scenario_modes_to_run:
                    sm_token = "median_envelope" if sm == "median" else "mid_case_worst"
                    out = variants_dir / f"fig4_fig5__{fr_token}__{sc_token}__{sm_token}{styling_token}.png"
                    plot_4panel(df, mat_order, out,
                                framing=fr, xscale=sc, scenario_mode=sm,
                                styling=args.styling)

        plot_4panel(df, mat_order, default_v1,
                    framing="demand-over-supply", xscale="symlog",
                    scenario_mode="median", styling=args.styling)
        plot_4panel(df, mat_order, default_v2,
                    framing="supply-over-demand", xscale="symlog",
                    scenario_mode="median", styling=args.styling)
        return

    # Default (2026-05-21): split mag|mix layout via the opt3a-midcase
    # renderer in supply_tiers_variants. The gray magnitude bar shows the
    # Mid_Case scenario mean, the hatched extension reaches the worst-case
    # scenario, and the thin whiskers are the within-Mid_Case MC 95% CI.
    # The mix sub-panel is a 0-100% CRC / HTS stacked bar.
    #
    # 2026-05-27 update: applies S9d-style row selection — top-10
    # non-REE materials by Panel A (US production) ratio descending,
    # plus the REE block forced to the bottom as a contiguous group
    # with pink background banding and maroon-bold REE labels.
    from supply_tiers_variants import plot_option3a_midcase  # noqa: E402

    # Manuscript Figures 4 and 5: the split supply-tier figures (per the
    # outline comment "split into 2 figs, stack"). split=True writes two files
    # into args.output_dir:
    #   Figure 5 -> fig4_supply_tiers_us.png      (Panel A US production over
    #                                              Panel B US reserves)
    #   Figure 6 -> fig5_supply_tiers_global.png  (Panel C global production over
    #                                              Panel D global reserves)
    # All materials are shown. Each panel is ordered independently from highest
    # to lowest demand-to-supply stress in the renderer; the gray magnitude bar
    # is the Mid_Case scenario mean, the hatched extension reaches the worst-case
    # scenario, and the thin whiskers are the within-Mid_Case MC 95% interval;
    # the mix sub-panel is a 0-100% CRC / HTS stacked bar. show_flow_fallback=
    # False drops the diagonal-hatch fallback overlay (noted in the caption / SI).
    # family_grouped_order supplies the material set; the renderer re-sorts each
    # panel by stress.
    mat_order_all = family_grouped_order(df, mat_order, args.re_mode,
                                         sort_panel="C")
    plot_option3a_midcase(df, mat_order_all, default_v1,
                          styling=args.styling, ree_band=True,
                          show_flow_fallback=False, split=True,
                          docx=args.docx)


if __name__ == "__main__":
    main()
