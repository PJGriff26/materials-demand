"""Supply-chain split-panel renderer for manuscript Figures 4 and 5.

The shipped Figure 4 (US demand vs domestic supply, panels A/B) and Figure 5
(US demand vs global supply, panels C/D) are produced by
plot_option3a_midcase(..., split=True), which writes fig4_supply_tiers_us.png
and fig5_supply_tiers_global.png. The other plot_option* functions are earlier
rendering experiments kept for reference. Supply-chain inputs are assembled by
build_records() in fig4_fig5_supply_tiers.py; this module only renders them.

INVENTORY:
  name: supply_tiers_variants
  output: outputs/figures/manuscript/fig4_supply_tiers_us.png,
    outputs/figures/manuscript/fig5_supply_tiers_global.png
    (renderer; invoked by fig4_fig5_supply_tiers.py, --docx variants under
    outputs/figures/manuscript/docx/)
  category: Manuscript main text — Figs 4 and 5 (renderer module)
  axes:
    x: demand-to-supply ratio (symlog, magnitude sub-panel); 0-100% share
       (mix sub-panel)
    y: materials, stress-ranked per panel
    color: OECD CRC risk groups (mix bars); MAG_BAR_COLOR #a0a0a0 magnitude
       bar; WORST_CASE_FACE #d9d9d9 hatched extension; UNKNOWN_COLOR
       #4a4a4a for unclassifiable USGS 'Other countries' shares
  data_sources:
    - build_records() in fig4_fig5_supply_tiers.py (see that INVENTORY)
  description: >
    plot_option3a_midcase(split=True) is the shipped renderer for Figs 4/5.
    X-axis labels come from PANEL_TITLES ("US peak annual demand ÷ ...
    production (log)", "US cumulative demand ÷ ... reserves (log)"). The
    dark-gray UNKNOWN_COLOR mix segment carries an "Unknown" legend entry on
    Figure 5 only. Standalone runs write to fig4_fig5_variants/past_iterations/.
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

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))
sys.path.insert(0, str(BASE_DIR / "visualizations"))

from _interpolation_io import (  # noqa: E402
    setup_interpolation_env, add_interpolation_arg,
)
_INTERP_METHOD = setup_interpolation_env()

from config import FIGURES_MANUSCRIPT_DIR, FIGURE_DPI  # noqa: E402
from supply_tiers_shared import GROUP_ORDER, GROUP_COLORS, country_to_group  # noqa: E402
from fig4_fig5_supply_tiers import (  # noqa: E402
    build_records, NO_SEPARATE_RESERVES,
)
from supply_chain_analysis import load_data as _load_demand_raw  # noqa: E402
from config import DEMAND_TO_RISK  # noqa: E402
sys.path.insert(0, str(BASE_DIR))
from src.scenario_config import REFERENCE_SCENARIO  # noqa: E402

# Dark gray for the "Unknown / 'Other countries'" segment in mix bars: USGS
# reports part of global production/reserves in an 'Other countries' bucket
# with no per-country breakdown, so those tonnes can't be CRC-classified.
# Only Panels C and D draw it — the Panel A/B Census HTS import shares and
# partner-reserve shares always sum to 1. Kept visually distinct from
# MAG_BAR_COLOR (#a0a0a0, neutral magnitude bar) and WORST_CASE_FACE (pale
# gray worst-case scenario extension).
UNKNOWN_COLOR = "#4a4a4a"
UNKNOWN_LABEL = "Unknown (USGS 'Other countries')"

# Worst-case extension: very light-gray with a hatch overlay so it's not
# confused with the (now lighter) solid magnitude bar.
WORST_CASE_FACE  = "#d9d9d9"
WORST_CASE_EDGE  = "white"
WORST_CASE_HATCH = "/////"
WORST_CASE_LABEL = "Mid Case (with IRA) → highest-demand scenario extension"

# Neutral magnitude-bar color. Lightened from #4a4a4a so the
# black MC whiskers read clearly on top of it; kept away from the CRC palette
# blues/greens so the bar can't be mistaken for a colored mix segment.
MAG_BAR_COLOR = "#a0a0a0"

# Uniform text-scale factor applied only under the opt-in --docx path of the
# split fig4/fig5 render (see plot_option3a_midcase). Tuned so that, at the
# manuscript insert width (5.0 in) and the bbox-tight saved pixel width, the
# in-plot text lands in the target legibility band. The override hook is for
# tuning sweeps only and is None in normal use.
_DOCX_SCALE = 2.0
_DOCX_SCALE_OVERRIDE = None


# ─────────────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────────────

def _scale_all_text(fig, S):
    """Uniformly enlarge every text artist in `fig` by factor S.

    Used only by the opt-in --docx path so the figure's in-plot text lands
    in a legible on-page band when inserted into a Word docx at a small
    insert width. Preserves the designed type hierarchy (titles vs. ticks
    vs. annotations) because every artist is multiplied by the same S.
    Does not touch any data, layout, or color.
    """
    for _ax in fig.get_axes():
        for _t in ([_ax.title, _ax.xaxis.label, _ax.yaxis.label]
                   + list(_ax.get_xticklabels()) + list(_ax.get_yticklabels())
                   + list(_ax.texts)):
            if _t is not None and _t.get_fontsize():
                _t.set_fontsize(_t.get_fontsize() * S)
        _lg = _ax.get_legend()
        if _lg:
            for _txt in _lg.get_texts():
                _txt.set_fontsize(_txt.get_fontsize() * S)
    for _t in list(fig.texts):           # covers suptitle + fig.text() annotations
        _t.set_fontsize(_t.get_fontsize() * S)
    for _lg in list(fig.legends):        # figure-level legends
        for _txt in _lg.get_texts():
            _txt.set_fontsize(_txt.get_fontsize() * S)


def _deficit_shares(r_med, crc_shares):
    """Panel A/B deficit stacking: US capped at 1.0, deficit split by
    historical import-partner CRC shares. Returns dict group->segment width
    (in ratio units) that sum to r_med."""
    seg = {grp: 0.0 for grp in GROUP_ORDER}
    if r_med <= 0:
        return seg
    if r_med <= 1.0:
        seg["US domestic"] = r_med
        return seg
    seg["US domestic"] = 1.0
    deficit = r_med - 1.0
    for grp in GROUP_ORDER[1:]:
        seg[grp] = deficit * crc_shares.get(grp, 0)
    return seg


def _global_shares(r_med, prod_shares):
    """Panel C/D stacking: bar fully partitioned by producer/reserve-holder
    CRC shares (no domestic cap — the denominator is global, not US)."""
    seg = {grp: 0.0 for grp in GROUP_ORDER}
    if r_med <= 0:
        return seg
    for grp in GROUP_ORDER:
        seg[grp] = r_med * prod_shares.get(grp, 0)
    return seg


def _draw_stacked_linear(ax, i, r_med, r_lo, r_hi, seg_widths, xlim_right,
                         show_errbar=True):
    """Linear-axis stacked bar with overflow marker when r_med > xlim_right."""
    drawn_total = 0.0
    capped = r_med > xlim_right

    left = 0.0
    for grp in GROUP_ORDER:
        w = seg_widths.get(grp, 0)
        if w <= 0:
            continue
        # Clip this segment at the panel's right edge.
        remaining = xlim_right - left
        if remaining <= 0:
            break
        w_draw = min(w, remaining)
        ax.barh(i, w_draw, left=left, color=GROUP_COLORS[grp],
                edgecolor="white", linewidth=0.4, zorder=3)
        left += w_draw
        drawn_total += w_draw
        if left >= xlim_right - 1e-9:
            break

    # Unclassified filler (USGS "Other countries" not CRC-tagged) so the
    # colored stack always covers to the bar's nominal end.
    if not capped and drawn_total < r_med - r_med * 0.005 and drawn_total < xlim_right:
        fill_w = min(r_med - drawn_total, xlim_right - drawn_total)
        if fill_w > 0:
            ax.barh(i, fill_w, left=drawn_total, color=UNKNOWN_COLOR,
                    edgecolor="white", linewidth=0.4, zorder=3)

    if capped:
        # Arrow/marker at right edge + numeric label
        ax.annotate(
            f"  {r_med:.0f}×→" if r_med >= 10 else f"  {r_med:.1f}×→",
            xy=(xlim_right, i), xytext=(xlim_right * 0.985, i),
            fontsize=7, va="center", ha="right", color="#222",
            fontweight="bold", zorder=5,
        )

    if show_errbar and np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo:
        r_lo_clip = min(r_lo, xlim_right)
        r_hi_clip = min(r_hi, xlim_right)
        r_mid_clip = min(r_med, xlim_right)
        if r_hi_clip > r_lo_clip:
            ax.errorbar(
                r_mid_clip, i,
                xerr=[[max(0, r_mid_clip - r_lo_clip)],
                      [max(0, r_hi_clip - r_mid_clip)]],
                fmt="none", color="black", lw=1.0, capsize=3,
                zorder=6, alpha=0.8,
            )


def _draw_mix_bar(ax, i, shares, y=None, height=0.8):
    """100%-stacked linear bar. shares: dict group -> fraction (summing ≤1).
    The remainder (if shares don't sum to 1) is drawn in UNKNOWN_COLOR and
    represents the USGS 'Other countries' aggregation in panel C.
    `y`/`height` override the row centre/thickness (general hook)."""
    yy = i if y is None else y
    left = 0.0
    # render the risk/country colours slightly translucent so
    # they read less boldly (alpha 0.82).
    for grp in GROUP_ORDER:
        w = shares.get(grp, 0)
        if w <= 0:
            continue
        ax.barh(yy, w, left=left, height=height, color=GROUP_COLORS[grp],
                alpha=0.82, edgecolor="white", linewidth=0.4, zorder=3)
        left += w
    if left < 0.995:
        ax.barh(yy, 1 - left, left=left, height=height, color=UNKNOWN_COLOR,
                alpha=0.82, edgecolor="white", linewidth=0.4, zorder=3)


def _overlay_flow_fallback_hatch(ax, i):
    """Render a diagonal-hatch overlay on a Panel B mix bar to flag that
    the shares fell back to Panel A's flow-based decomposition because
    partner-reserve coverage was below threshold. The viewer must not
    read this bar as a reserves decomposition."""
    ax.barh(i, 1.0, left=0.0,
            facecolor="none", edgecolor="#222222", linewidth=0,
            hatch="////", zorder=6)


def _mat_labels_with_backest_glyph(mat_order, df, panel):
    """Return plain material labels.

    Previously appended a ⊕ glyph to Panel A labels for materials whose
    us_prod_t was back-estimated from the USGS Form 72 identity. The glyph
    rendered as tofu (□) in some font contexts and lacked an in-figure
    legend on the alt variants. Disclosure of the back-estimation method
    now lives in the figure caption / methods text, not on the y-axis.
    The helper is retained (rather than inlined at call sites) so a future
    flag-based version can be reintroduced without touching the renderers.
    """
    return list(mat_order)


def _draw_mag_bar(ax, i, r_med, r_lo, r_hi, color="#3a6fb0"):
    """Single-color magnitude bar (log axis). Used in option 3a left columns."""
    if not np.isfinite(r_med) or r_med <= 0:
        return
    ax.barh(i, r_med, color=color, edgecolor="white", linewidth=0.4, zorder=3)
    if np.isfinite(r_lo) and np.isfinite(r_hi) and r_hi > r_lo:
        ax.errorbar(
            r_med, i,
            xerr=[[max(0, r_med - r_lo)], [max(0, r_hi - r_med)]],
            fmt="none", color="black", lw=1.0, capsize=3, zorder=6, alpha=0.8,
        )


def _draw_mag_bar_midcase(ax, i, r_mid, r_mid_lo, r_mid_hi, r_worst,
                          color="#3a6fb0", y=None, height=0.8):
    """Mid_Case main bar + within-Mid_Case MC p2.5/p97.5 whiskers + grey
    extension to worst-case scenario's peak/cumulative.

    `y`/`height` override the bar's row centre and thickness (a general hook;
    the aggregated REE reserves entry now uses a normal-height bar on its
    centre row rather than one block-spanning bar)."""
    yy = i if y is None else y
    if not np.isfinite(r_mid) or r_mid <= 0:
        # When demand is zero (no intensity records for this material in
        # our technology set — e.g., Selenium), note it so the empty row
        # isn't confusing.
        if r_mid == 0:
            ax.text(0.02, yy, " no modeled demand (intensity data unavailable)",
                    transform=ax.get_yaxis_transform(),
                    fontsize=7, va="center", style="italic",
                    color="#888", zorder=3)
        return
    ax.barh(yy, r_mid, height=height, color=color, edgecolor="white",
            linewidth=0.4, zorder=3)
    # Scenario-envelope extension: hatched light gray. Hatching keeps it
    # visually distinct from the darker solid "Unknown" mix-bar segments.
    if np.isfinite(r_worst) and r_worst > r_mid:
        ax.barh(yy, r_worst - r_mid, left=r_mid, height=height,
                facecolor=WORST_CASE_FACE, edgecolor=WORST_CASE_EDGE,
                linewidth=0.4, hatch=WORST_CASE_HATCH,
                alpha=0.95, zorder=2)
    # Within-Mid_Case MC uncertainty whiskers (p2.5–p97.5)
    if (np.isfinite(r_mid_lo) and np.isfinite(r_mid_hi) and
            r_mid_hi > r_mid_lo):
        ax.errorbar(
            r_mid, yy,
            xerr=[[max(0, r_mid - r_mid_lo)], [max(0, r_mid_hi - r_mid)]],
            fmt="none", color="black", lw=1.0, capsize=3,
            zorder=6, alpha=0.85,
        )


def _build_partner_reserve_shares():
    """For each risk-material, compute partner-reserve-weighted CRC shares:
      share_grp[mat] = sum_{country ∈ partners ∩ group grp} reserves(country, mat)
                       / sum_{country ∈ partners} reserves(country, mat)

    Returns (shares, coverage, partner_total_kt):
      shares[risk_name]           -> {group: fraction}
      coverage[risk_name]         -> fraction of import volume with reserves data
      partner_total_kt[risk_name] -> total partner reserves in kt (used by
        Panel B's reserve-based US-domestic fraction). Unit matches the
        reserves sheet (kt).
    """
    _demand_raw, risk = _load_demand_raw()
    imp = risk["import_shares"]
    res = risk["reserves"].copy()
    res = res.rename(columns={res.columns[0]: "country"})
    crc_df = risk["crc"].iloc[:, :2].copy()
    crc_df.columns = ["country", "crc"]
    crc_lookup = dict(zip(crc_df["country"], crc_df["crc"]))

    shares_out = {}
    coverage_out = {}
    partner_total_out = {}
    materials = [m for m in res.columns if m != "country"]

    for mat in materials:
        res_col = pd.to_numeric(res[mat], errors="coerce").fillna(0)
        res_by_country = dict(zip(res["country"], res_col))

        partners = imp[imp["material"] == mat][["country", "share"]]
        partners = partners[~partners["country"].astype(str).str.lower()
                            .isin(["other", "global", "nan"])]
        if partners.empty:
            continue

        partner_res = []
        import_total_with_res = 0.0
        import_total = float(partners["share"].sum())
        for _, prow in partners.iterrows():
            c = prow["country"]
            r = float(res_by_country.get(c, 0) or 0)
            partner_res.append((c, float(prow["share"]), r))
            if r > 0:
                import_total_with_res += float(prow["share"])

        total_partner_res = sum(r for _, _, r in partner_res)
        shares = {g: 0.0 for g in GROUP_ORDER}
        if total_partner_res > 0:
            for c, _s, r in partner_res:
                if r <= 0:
                    continue
                grp = country_to_group(c, crc_lookup)
                shares[grp] += r / total_partner_res

        shares_out[mat] = shares
        coverage_out[mat] = (import_total_with_res / import_total) if import_total > 0 else 0.0
        partner_total_out[mat] = total_partner_res  # kt, same unit as reserves sheet

    # Yttrium reserves alias: USGS MCS 2025 reports Y reserves only as part of
    # the aggregate "Rare Earths" total. Without this alias the Y row falls
    # through to the flow-based fallback in the Panel B mix bar, which uses
    # Y-specific HTS import shares (heavy in Low-risk OECD imports) instead
    # of the partner-reserve-weighted shares the other four per-element REEs
    # get — creating a visually inconsistent green-vs-purple split in the REE
    # block. Aliases all three outputs (shares, coverage, partner_total)
    # so Y mirrors the aggregate Rare Earths treatment. Companion alias to
    # _alias_yttrium_to_rare_earths in fig4_fig5_supply_tiers.py. Placeholder
    # until per-element REE reserves are sourced (USGS MCS 2025 publishes only
    # the aggregate TREO reserve, not a per-element split).
    if "Rare Earths" in shares_out and "Yttrium" not in shares_out:
        shares_out["Yttrium"]        = shares_out["Rare Earths"]
        coverage_out["Yttrium"]      = coverage_out["Rare Earths"]
        partner_total_out["Yttrium"] = partner_total_out["Rare Earths"]

    return shares_out, coverage_out, partner_total_out


def _augment_midcase_mc_ci(df):
    """Extend df in place with Mid_Case MC p2.5/p97.5 bounds (within-scenario
    Monte Carlo uncertainty, as distinct from the across-scenario envelope
    already stored as peak_worst_case / cum_worst_case).

    Adds columns:
      peak_mid_case_p2, peak_mid_case_p97
      cum_mid_case_p2,  cum_mid_case_p97
    """
    demand_raw, _risk = _load_demand_raw()
    has_p = "p2" in demand_raw.columns and "p97" in demand_raw.columns

    mid = demand_raw[demand_raw["scenario"] == REFERENCE_SCENARIO]

    # REE aggregate: sum across RE elements per year (conservative — assumes
    # comonotone intensity draws across elements).
    ree_elements = [m for m, rn in DEMAND_TO_RISK.items() if rn == "Rare Earths"]
    if "Yttrium" in demand_raw["material"].unique() and "Yttrium" not in ree_elements:
        ree_elements.append("Yttrium")

    for mat in df.index:
        if mat == "Rare Earths":
            m = mid[mid["material"].isin(ree_elements)]
            if m.empty:
                df.loc[mat, "peak_mid_case_p2"]  = df.loc[mat, "peak_mid_case"]
                df.loc[mat, "peak_mid_case_p97"] = df.loc[mat, "peak_mid_case"]
                df.loc[mat, "cum_mid_case_p2"]   = df.loc[mat, "cum_mid_case"]
                df.loc[mat, "cum_mid_case_p97"]  = df.loc[mat, "cum_mid_case"]
                continue
            by_year_mean = m.groupby("year")["mean"].sum()
            by_year_p2   = m.groupby("year")["p2"].sum()  if has_p else by_year_mean
            by_year_p97  = m.groupby("year")["p97"].sum() if has_p else by_year_mean
            peak_year = by_year_mean.idxmax()
            df.loc[mat, "peak_mid_case_p2"]  = float(by_year_p2.loc[peak_year])
            df.loc[mat, "peak_mid_case_p97"] = float(by_year_p97.loc[peak_year])
            window = (by_year_mean.index >= 2026) & (by_year_mean.index <= 2050)
            df.loc[mat, "cum_mid_case_p2"]   = float(by_year_p2[window].sum())
            df.loc[mat, "cum_mid_case_p97"]  = float(by_year_p97[window].sum())
            continue

        m = mid[mid["material"] == mat]
        if m.empty:
            df.loc[mat, "peak_mid_case_p2"]  = df.loc[mat, "peak_mid_case"]
            df.loc[mat, "peak_mid_case_p97"] = df.loc[mat, "peak_mid_case"]
            df.loc[mat, "cum_mid_case_p2"]   = df.loc[mat, "cum_mid_case"]
            df.loc[mat, "cum_mid_case_p97"]  = df.loc[mat, "cum_mid_case"]
            continue
        m = m.set_index("year")
        peak_year = m["mean"].idxmax()
        df.loc[mat, "peak_mid_case_p2"]  = float(m.loc[peak_year, "p2"])  if has_p else float(m.loc[peak_year, "mean"])
        df.loc[mat, "peak_mid_case_p97"] = float(m.loc[peak_year, "p97"]) if has_p else float(m.loc[peak_year, "mean"])
        window = (m.index >= 2026) & (m.index <= 2050)
        df.loc[mat, "cum_mid_case_p2"]   = float(m.loc[window, "p2"].sum())  if has_p else float(m.loc[window, "mean"].sum())
        df.loc[mat, "cum_mid_case_p97"]  = float(m.loc[window, "p97"].sum()) if has_p else float(m.loc[window, "mean"].sum())
    return df


def _midcase_ratios_for_panel(df, mat_order, panel):
    """Mid_Case central value + within-Mid_Case MC p2.5/p97.5 + worst-case-
    scenario upper bound per material.
    Returns list of (mat, r_mid, r_mid_lo, r_mid_hi, r_worst, skip_reason)."""
    out = []
    for mat in mat_order:
        row = df.loc[mat]
        if panel == "A":
            us = row.get("us_prod_t", 0) or 0
            if us <= 0:
                out.append((mat, None, None, None, None, "no US production")); continue
            denom = us
            num_mid, num_lo, num_hi, num_w = (row["peak_mid_case"],
                                               row.get("peak_mid_case_p2", row["peak_mid_case"]),
                                               row.get("peak_mid_case_p97", row["peak_mid_case"]),
                                               row["peak_worst_case"])
        elif panel == "B":
            if not row.get("has_reserves", False):
                out.append((mat, None, None, None, None, "no reserves concept")); continue
            denom = row.get("us_reserves_t", 0) or 0
            if denom <= 0:
                out.append((mat, None, None, None, None, "no US reserves")); continue
            num_mid, num_lo, num_hi, num_w = (row["cum_mid_case"],
                                               row.get("cum_mid_case_p2", row["cum_mid_case"]),
                                               row.get("cum_mid_case_p97", row["cum_mid_case"]),
                                               row["cum_worst_case"])
        elif panel == "C":
            denom = row.get("global_prod_t", 0) or 0
            if denom <= 0:
                out.append((mat, None, None, None, None, "no global prod data")); continue
            num_mid, num_lo, num_hi, num_w = (row["peak_mid_case"],
                                               row.get("peak_mid_case_p2", row["peak_mid_case"]),
                                               row.get("peak_mid_case_p97", row["peak_mid_case"]),
                                               row["peak_worst_case"])
        elif panel == "D":
            if not row.get("has_reserves", False):
                out.append((mat, None, None, None, None, "no reserves concept")); continue
            denom = row.get("global_reserves_t", 0) or 0
            if denom <= 0:
                out.append((mat, None, None, None, None, "no reserve data")); continue
            num_mid, num_lo, num_hi, num_w = (row["cum_mid_case"],
                                               row.get("cum_mid_case_p2", row["cum_mid_case"]),
                                               row.get("cum_mid_case_p97", row["cum_mid_case"]),
                                               row["cum_worst_case"])
        else:
            raise ValueError(panel)
        out.append((mat,
                    num_mid / denom,
                    num_lo  / denom,
                    num_hi  / denom,
                    num_w   / denom,
                    None))
    return out


def _ratios_for_panel(df, mat_order, panel):
    """Return list of (mat, r_med, r_lo, r_hi, seg_widths_full) per material.
    seg_widths_full: dict group->ratio-unit width for the *full* r_med bar.
    Skipped materials yield r_med=None."""
    out = []
    for mat in mat_order:
        row = df.loc[mat]
        if panel == "A":
            us = row.get("us_prod_t", 0) or 0
            if us <= 0:
                out.append((mat, None, None, None, None, "no US production"))
                continue
            r_med = row["peak_median"] / us if us > 0 else 0
            r_lo  = row["peak_min"]    / us if us > 0 else 0
            r_hi  = row["peak_max"]    / us if us > 0 else 0
            seg = _deficit_shares(r_med, row["import_shares_grouped"])
        elif panel == "B":
            if not row.get("has_reserves", False):
                out.append((mat, None, None, None, None, "no reserves concept"))
                continue
            us_res = row.get("us_reserves_t", 0) or 0
            if us_res <= 0:
                out.append((mat, None, None, None, None, "no US reserves"))
                continue
            r_med = row["cum_median"] / us_res
            r_lo  = row["cum_min"]    / us_res
            r_hi  = row["cum_max"]    / us_res
            seg = _deficit_shares(r_med, row["import_shares_grouped"])
        elif panel == "C":
            gl = row.get("global_prod_t", 0) or 0
            if gl <= 0:
                out.append((mat, None, None, None, None, "no global prod data"))
                continue
            r_med = row["peak_median"] / gl
            r_lo  = row["peak_min"]    / gl
            r_hi  = row["peak_max"]    / gl
            seg = _global_shares(r_med, row["prod_shares_grouped"])
        elif panel == "D":
            if not row.get("has_reserves", False):
                out.append((mat, None, None, None, None, "no reserves concept"))
                continue
            gl_res = row.get("global_reserves_t", 0) or 0
            if gl_res <= 0:
                out.append((mat, None, None, None, None, "no reserve data"))
                continue
            r_med = row["cum_median"] / gl_res
            r_lo  = row["cum_min"]    / gl_res
            r_hi  = row["cum_max"]    / gl_res
            seg = _global_shares(r_med, row.get("reserve_shares_grouped",
                                                 {g: 0 for g in GROUP_ORDER}))
        else:
            raise ValueError(panel)
        out.append((mat, r_med, r_lo, r_hi, seg, None))
    return out


def _mix_shares_for_panel(df, mat_order, panel,
                          partner_reserve_shares=None,
                          partner_reserve_coverage=None,
                          partner_reserve_total_kt=None,
                          coverage_threshold=0.20):
    """Shares-only (sum to 1) for 100%-stacked mix bars (option 3).

    Panel A (US flow / production): US-domestic slice = 1 − NIR (flow).
      Import slice split by CRC using HTS import-partner shares.
    Panel B (US reserves / stock): US-domestic slice = us_reserves /
      (us_reserves + partner_reserves_total), a RESERVE-BASED fraction.
      Using 1 − NIR here would be a category error — flow metric for
      a stock question (would draw phantom blue bars for materials like
      Tin where US has positive production from imported concentrate
      but zero domestic reserves). Import slice split by partner-
      reserve-weighted CRC shares.
    Panel C: global production split by producer-country CRC shares.
    Panel D: global reserves split by holder-country CRC shares.

    For Panel B, if partner_reserve coverage < coverage_threshold,
    falls back to Panel A's flow-based mix (the only other available
    signal). The fallback flag is surfaced in the returned tuple so
    downstream renderers can visually mark these bars — a silent
    fallback recreates the category error this function was designed
    to prevent.

    Returns: list of (mat, shares, flags) triples.
      shares : {crc_group: fraction}  or  None (skip row)
      flags  : dict with keys:
        "panel_b_flow_fallback": True when Panel B silently reverted to
          flow-based US fraction because reserve-coverage was below
          `coverage_threshold`.
        "panel_b_reserve_based": True when Panel B used the intended
          reserve-based US fraction.
    """
    # Show the mix wherever sourcing data exists — decoupled from whether
    # the magnitude bar has a valid denominator. E.g., a material with
    # "no reserves concept" can still show an import-partner mix in
    # Panel B; a material with no US production can still show a
    # global-production mix in Panel C.
    def _nonzero(d):
        return d is not None and sum(v for v in d.values() if v) > 0

    out = []
    empty_flags = {"panel_b_flow_fallback": False, "panel_b_reserve_based": False}
    for mat in mat_order:
        row = df.loc[mat]
        if panel in ("A", "B"):
            # Panel B is about reserves. Materials with no reserves concept
            # (Silicon, Cement, Fiberglass, etc. — silica, limestone, etc.
            # are effectively unbounded feedstocks, not tracked as reserves)
            # should skip the mix bar just as the magnitude bar does.
            # Otherwise the flow-based fallback would draw a phantom US
            # slice from 1 − NIR, recreating the category error.
            if panel == "B" and not row.get("has_reserves", True):
                out.append((mat, None, dict(empty_flags)))
                continue
            nir_raw = row.get("import_dependency", np.nan)
            if pd.isna(nir_raw):
                out.append((mat, None, dict(empty_flags)))
                continue
            nir = float(nir_raw)
            rn = row.get("risk_name", mat)

            use_reserve_fraction = False
            if panel == "B" and partner_reserve_shares is not None:
                cov = (partner_reserve_coverage or {}).get(rn, 0.0)
                pr_shares = partner_reserve_shares.get(rn)
                if pr_shares and cov >= coverage_threshold and sum(pr_shares.values()) > 0:
                    crc = pr_shares
                    use_reserve_fraction = True
                else:
                    crc = row.get("import_shares_grouped") or {}
            else:
                crc = row.get("import_shares_grouped") or {}

            # If NIR is 1 (fully import-dependent) and we have no partner
            # mix at all, we can't draw anything meaningful — skip.
            if nir >= 0.999 and not _nonzero(crc):
                out.append((mat, None, dict(empty_flags)))
                continue

            shares = {grp: 0.0 for grp in GROUP_ORDER}
            flags = dict(empty_flags)
            if use_reserve_fraction:
                # Panel B: reserve-based US fraction. Converts us_reserves_t
                # (tonnes) and partner_reserves (kt) to the same unit, then
                # computes US share of the (US + partner) reserve pool.
                us_res_kt = (row.get("us_reserves_t", 0) or 0) / 1000.0
                partner_total_kt = (partner_reserve_total_kt or {}).get(rn, 0.0)
                denom = us_res_kt + partner_total_kt
                us_fraction = (us_res_kt / denom) if denom > 0 else 0.0
                shares["US domestic"] = us_fraction
                for grp in GROUP_ORDER[1:]:
                    shares[grp] = (1.0 - us_fraction) * crc.get(grp, 0)
                flags["panel_b_reserve_based"] = True
            else:
                # Panel A and Panel B fallback: flow-based.
                shares["US domestic"] = max(0.0, 1.0 - nir)
                for grp in GROUP_ORDER[1:]:
                    shares[grp] = nir * crc.get(grp, 0)
                if panel == "B":
                    # Flagged so the renderer can visually mark the bar —
                    # a silent fallback in a reserves panel reintroduces
                    # the exact category error this function prevents.
                    flags["panel_b_flow_fallback"] = True
            out.append((mat, shares, flags))
        elif panel == "C":
            prod = row.get("prod_shares_grouped") or {}
            if not _nonzero(prod):
                out.append((mat, None, dict(empty_flags)))
                continue
            out.append((mat, dict(prod), dict(empty_flags)))
        elif panel == "D":
            res_shares = row.get("reserve_shares_grouped") or {}
            if not _nonzero(res_shares):
                out.append((mat, None, dict(empty_flags)))
                continue
            out.append((mat, dict(res_shares), dict(empty_flags)))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — linear x-axis
# ─────────────────────────────────────────────────────────────────────────────

# Manuscript x-axis wording; the split renderer appends " (log)".
PANEL_TITLES = {
    "A": ("A. US production",
          "US peak annual demand ÷ US annual production",
          "deficit split by 2020–23 US import partners (CRC group)"),
    "B": ("B. US reserves",
          "US cumulative demand ÷ US reserves",
          "deficit split by 2020–23 US import partners (CRC group)"),
    "C": ("C. Global production",
          "US peak annual demand ÷ global annual production",
          "bar split by producer-country share (USGS MCS 2025, CRC group)"),
    "D": ("D. Global reserves",
          "US cumulative demand ÷ global reserves",
          "bar split by reserve-holder country share (USGS MCS 2025, CRC group)"),
}

# Mix-panel subtitles used in split variants (3a, 3a-midcase). The
# CRC-source decomposition lives on the mix sub-panel, not the single-color
# magnitude bar.
MIX_SUBTITLES = {
    "A": "share by 2020–23 US import partners (CRC group, Census HTS)",
    "B": "share by 2020–23 US import partners (CRC group, Census HTS)",
    "C": "share by producer country (USGS MCS 2025, CRC group)",
    "D": "share by reserve-holder country (USGS MCS 2025, CRC group)",
}

# Consolidated mix x-axis labels: short primary label + CRC-descriptor
# carried from the old image subtitles (citations live in the caption).
# Panel A and B use identical data but different semantics (peak-year
# import flow vs. cumulative 2026–2050 flow) — labeled to match.
MIX_X_LABELS = {
    "A": "Source share",
    "B": "Reserve share",
    "C": "Global production\nshare",
    "D": "Global reserve\nshare",
}

# Short italic grey subtitle describing each panel's sourcing basis —
# placed below the x-axis label in each mix panel.
MIX_X_SUBTITLES = {
    "A": "share by US import partners (CRC group)",
    "B": "share by partner reserve depth (CRC group)",
    "C": "share by producer country (CRC group)",
    "D": "share by reserve-holder country (CRC group)",
}

# Magnitude-panel subtitles — describe what the bar/whisker show,
# NOT the CRC decomposition (which lives on the mix panel).
MAG_SUBTITLES_MC = {
    "A": "bar = scenario median peak; whiskers = MC p2.5–p97.5 (pooled across scenarios)",
    "B": "bar = scenario median cumulative; whiskers = MC p2.5–p97.5 (pooled across scenarios)",
    "C": "bar = scenario median peak; whiskers = MC p2.5–p97.5 (pooled across scenarios)",
    "D": "bar = scenario median cumulative; whiskers = MC p2.5–p97.5 (pooled across scenarios)",
}
MAG_SUBTITLES_MIDCASE = {
    "A": "bar = Mid Case (with IRA) peak; whiskers = MC p2.5–p97.5; grey = worst-case scenario extension",
    "B": "bar = Mid Case (with IRA) cumulative; whiskers = MC p2.5–p97.5; grey = worst-case scenario extension",
    "C": "bar = Mid Case (with IRA) peak; whiskers = MC p2.5–p97.5; grey = worst-case scenario extension",
    "D": "bar = Mid Case (with IRA) cumulative; whiskers = MC p2.5–p97.5; grey = worst-case scenario extension",
}

# Variant 2a: fixed per-panel upper bounds chosen to show the reference line
# at 1.0 prominently while clipping tails (Niobium's ~10^5 in panel A etc.).
FIXED_CAPS = {"A": 3.0, "B": 5.0, "C": 1.2, "D": 1.2}


def _axes_setup(ax, title_line_1, xlabel, subtitle, mat_order, xlim_right):
    if xlim_right >= 1.0:
        ax.axvline(1.0, color="black", ls="-", lw=1.8, alpha=0.9, zorder=5)
    else:
        # Threshold falls off-scale for this panel. Flag it so readers
        # don't assume the absence of a vertical line means "no reference".
        ax.annotate(
            "→ 1.0",
            xy=(xlim_right, -0.55), xytext=(xlim_right * 0.97, -0.55),
            fontsize=7.5, va="top", ha="right",
            color="#555", style="italic", zorder=5,
        )
        xlabel = xlabel + "  [threshold 1.0 off-scale →]"
    ax.set_yticks(range(len(mat_order)))
    ax.set_yticklabels(mat_order, fontsize=9)
    ax.set_ylim(-0.6, len(mat_order) - 0.4)
    ax.set_xlim(0, xlim_right)
    ax.set_xlabel(xlabel, fontsize=9.5, fontweight="bold")
    ax.set_title(title_line_1, fontsize=11, fontweight="bold", pad=22)
    ax.text(0.5, 1.025, subtitle, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=8,
            style="italic", color="#555555")
    ax.grid(True, alpha=0.15, axis="x")


def _skip_text(ax, i, msg):
    ax.text(0.02, i, f" {msg}", fontsize=7, va="center",
            style="italic", color="#888", zorder=3,
            transform=ax.get_yaxis_transform())


def _auto_cap(ratios, p_quantile=0.90, padding=1.3, floor=1.5):
    """Data-driven per-panel cap.
    - Small-magnitude panels (data max < 0.5): tight fit at max × 1.12
      so the visible axis matches the data range. The threshold at 1.0
      falls off-scale and is flagged in the x-label instead.
    - Larger panels: keep the p90 × padding heuristic, floored so the
      threshold at 1.0 always has breathing room.
    """
    vals = [r for r in ratios if r is not None and r > 0]
    if not vals:
        return floor
    data_max = max(vals)
    if data_max < 0.5:
        return data_max * 1.12
    q = np.quantile(vals, p_quantile)
    return max(floor, q * padding)


def plot_option2(df, mat_order, output_path, mode="clipped"):
    """Linear x-axis variants.
      mode="clipped": fixed per-panel caps (see FIXED_CAPS).
      mode="autorange": per-panel cap = max(floor, p90 × 1.3).
    """
    fig = plt.figure(figsize=(18, max(11, len(mat_order) * 0.4)))
    gs = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.58)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
    }


    panel_caps = {}
    for p in ("A", "B", "C", "D"):
        rows = _ratios_for_panel(df, mat_order, p)
        if mode == "clipped":
            cap = FIXED_CAPS[p]
        else:
            cap = _auto_cap([r[1] for r in rows])
        panel_caps[p] = cap

        ax = axes[p]
        title, xlab, sub = PANEL_TITLES[p]
        _axes_setup(ax, title, xlab + f" (linear, clipped at {cap:g}×)",
                    sub, mat_order, cap)

        for i, (mat, r_med, r_lo, r_hi, seg, skip_reason) in enumerate(rows):
            if r_med is None:
                _skip_text(ax, i, skip_reason)
                continue
            if r_med <= 0:
                continue
            _draw_stacked_linear(ax, i, r_med, r_lo, r_hi, seg,
                                 xlim_right=cap, show_errbar=True)

    # Legend
    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    handles.append(mpatches.Patch(color=UNKNOWN_COLOR, label=UNKNOWN_LABEL))
    handles += [plt.Line2D([], [], color="black", lw=1.8,
                           label="Self-sufficiency threshold (ratio = 1.0)")]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=7, fontsize=9, frameon=False,
               title="Supply-source country risk group (OECD CRC 2026). "
                     "Bars beyond panel cap show `→N×` marker.",
               title_fontsize=8.5)

    suptitle = ("Figure 5 (Variant 2" + ("a" if mode == "clipped" else "b") +
                "). Linear-axis sourcing bars — honest segment proportions, "
                f"{'fixed per-panel caps' if mode == 'clipped' else 'auto-ranged caps'}")
    fig.suptitle(suptitle, fontsize=12, fontweight="bold", y=0.995)

    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3a — split per-panel: log-mag bar + linear mix bar (side by side)
# ─────────────────────────────────────────────────────────────────────────────

def plot_option3a(df, mat_order, output_path):
    """Each of 4 panels is split into two sub-axes: log ratio (magnitude) +
    linear 100%-stacked (mix). Magnitude uses a neutral single color; mix
    carries the CRC-group semantics."""
    fig = plt.figure(figsize=(22, max(12, len(mat_order) * 0.44)))
    # Nested gridspec: outer 2×2 with wide wspace; each cell inner-split into
    # (mag | mix). Prevents mix-panel bars from crashing into the next
    # panel's y-tick labels.
    outer = fig.add_gridspec(2, 2, wspace=0.28, hspace=0.55)
    inner = {
        "A": outer[0, 0].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.08),
        "B": outer[0, 1].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.08),
        "C": outer[1, 0].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.08),
        "D": outer[1, 1].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.08),
    }


    for p in ("A", "B", "C", "D"):
        ax_mag = fig.add_subplot(inner[p][0, 0])
        ax_mix = fig.add_subplot(inner[p][0, 1], sharey=ax_mag)

        title, xlab, _ = PANEL_TITLES[p]
        mix_sub = MIX_SUBTITLES[p]
        # Magnitude: log, single color. Bar/whisker semantics are carried by
        # the bottom legend, so no magnitude-panel subtitle is needed.
        ax_mag.axvline(1.0, color="black", ls="-", lw=1.8, alpha=0.9, zorder=5)
        ax_mag.set_yticks(range(len(mat_order)))
        ax_mag.set_yticklabels(mat_order, fontsize=9)
        ax_mag.set_ylim(-0.6, len(mat_order) - 0.4)
        # Tighter linthresh for Panel A/B (1e-4) so small-ratio materials
        # (Boron ~4e-4, Magnesium ~7e-2) get visible bars instead of being
        # compressed into the linear band. Panel C/D use 1e-4 too since
        # US demand / global anything is typically small.
        ax_mag.set_xscale("symlog", linthresh=1e-4)
        ax_mag.set_xlabel(xlab + " (log)", fontsize=9.5, fontweight="bold")
        ax_mag.set_title(title, fontsize=10.5, fontweight="bold", pad=10)
        ax_mag.grid(True, alpha=0.15, axis="x")

        ratios = _ratios_for_panel(df, mat_order, p)
        # Upper bound driven by the p97.5 whisker (r_hi, index 3), not the
        # median — otherwise the MC uncertainty bars get clipped at the right
        # edge. Extra 1.25× padding so whisker caps aren't flush with the axis.
        max_hi = max([r[3] for r in ratios if r[3] is not None] + [0])
        max_med = max([r[1] for r in ratios if r[1] is not None] + [0])
        upper = max(max_hi, max_med) * 1.25
        if p in ("A", "B"):
            ax_mag.set_xlim(right=max(10, upper))
        else:
            ax_mag.set_xlim(right=max(1.5, upper))

        for i, (mat, r_med, r_lo, r_hi, _seg, skip_reason) in enumerate(ratios):
            if r_med is None:
                _skip_text(ax_mag, i, skip_reason)
                continue
            _draw_mag_bar(ax_mag, i, r_med, r_lo, r_hi,
                          color="#3a6fb0" if p in ("A", "B") else "#4a8b52")

        # Mix: linear 0–1
        mix = _mix_shares_for_panel(df, mat_order, p)
        ax_mix.set_xlim(0, 1)
        ax_mix.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_mix.set_xticklabels(["0", ".25", ".5", ".75", "1"], fontsize=8)
        ax_mix.tick_params(axis="y", left=False, labelleft=False)
        ax_mix.set_xlabel("Source mix (fraction)", fontsize=9, fontweight="bold")
        ax_mix.set_title("Sourcing mix", fontsize=10.5, fontweight="bold", pad=22)
        ax_mix.text(0.5, 1.025, mix_sub, transform=ax_mix.transAxes,
                    ha="center", va="bottom", fontsize=7.5,
                    style="italic", color="#555555")
        ax_mix.grid(True, alpha=0.15, axis="x")

        for i, (mat, shares, flags) in enumerate(mix):
            if shares is None:
                continue
            _draw_mix_bar(ax_mix, i, shares)
            if flags.get("panel_b_flow_fallback"):
                _overlay_flow_fallback_hatch(ax_mix, i)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    handles.append(mpatches.Patch(color=UNKNOWN_COLOR, label=UNKNOWN_LABEL))
    handles += [
        mpatches.Patch(facecolor="none", edgecolor="#222222", hatch="////",
                       label="Panel B: flow-based fallback (reserve coverage <20%)"),
        plt.Line2D([], [], color="black", lw=1.8,
                   label="Self-sufficiency threshold (ratio = 1.0)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.035), ncol=7, fontsize=9,
               frameon=False)

    fig.suptitle(
        "Figure 5. Supply coverage and sourcing mix, 2026–2050",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3a-midcase — same as 3a but mag bars show Mid_Case + worst-case
# scenario grey extension (instead of MC p2.5–p97.5 whiskers)
# ─────────────────────────────────────────────────────────────────────────────

def plot_option3a_midcase(df, mat_order, output_path, styling="manuscript",
                          show_flow_fallback=True, ree_band=True, split=False,
                          docx=False):
    """Same split layout as 3a, but each magnitude bar now carries BOTH
    uncertainty types:
      - colored bar = Mid_Case scenario mean
      - MC whiskers = within-Mid_Case p2.5/p97.5 Monte Carlo envelope
      - grey extension = Mid_Case → worst-case-scenario peak/cumulative
    The layout uses nested gridspec so (mag | mix) pairs have tight internal
    spacing but the pairs themselves are separated by a wide gap — that
    avoids mix-panel bars crashing into the next panel's y-tick labels.

    Parameters
    ----------
    styling : {"manuscript", "poster"}
        "manuscript" (default) is the styling used for the shipped figures.
        "poster" applies a higher-contrast presentation variant (REE label
        highlight, alternating row banding, REE-by-Panel-A-ratio sort with
        no-US-production pinned to the top, prominent "No US production" tags,
        and de-cluttered spines). The poster variant is a presentation
        alternative and is not part of the manuscript figure set.
    ree_band : bool
        If True, draws a soft-pink background band behind contiguous REE
        rows (Nd/Pr/Dy/Tb/Y/Gd/Rare Earths) and bolds the REE labels in
        maroon. Used by the manuscript default to highlight the REE block
        when the mat_order has REEs as a bottom-contiguous group.
        Independent of `styling` so the band appears in both the manuscript
        and poster renderings.
    """
    REE_HIGHLIGHT_SET = {"Dysprosium", "Neodymium", "Praseodymium",
                         "Terbium", "Yttrium", "Gadium", "Rare Earths"}
    # REE highlight must NOT match the China bar colour
    # (#880e4f). Use a dark navy for the REE labels and a pale blue-grey band.
    REE_HIGHLIGHT_COLOR = "#0d3b66"
    REE_BAND_FACE = "#eef2f8"  # pale blue-grey background banding for the REE block
    NO_PROD_COLOR = "#0d3b66"

    _augment_midcase_mc_ci(df)  # adds peak/cum_mid_case_p2/_p97

    if styling == "poster":
        def _sort_key(mat):
            row = df.loc[mat]
            us_p = row.get("us_prod_t", 0) or 0
            peak = row.get("peak_mid_case", 0) or 0
            if us_p <= 0:
                return (1, float('inf'))  # last in list = top of plot
            return (0, peak / us_p if us_p > 0 else 0)
        mat_order = sorted(mat_order, key=_sort_key)
    # Partner-reserve-weighted CRC shares for Panel B (genuine reserves
    # analog, vs. Panel A's import-flow shares). Falls back silently for
    # refiner-driven chains where partner-reserve coverage is too low.
    partner_res_shares, partner_res_cov, partner_res_total_kt = _build_partner_reserve_shares()

    if split:
        # Two stacked figures: US tiers (A over B) and global tiers (C over D).
        # Each panel keeps its (magnitude | mix) inner pair; only the outer
        # layout, the per-panel parent figure, and the save block change.
        #
        # docx mode: render AT PRINT SIZE so on-page pt == the absolute pt
        # set later (the figure is saved with bbox_inches="tight", so any
        # font-scaling-at-large-figsize approach is cancelled by the canvas
        # growing in lockstep — the correct fix is to shrink the figure to
        # the insert width and set absolute pt). Print width = 6.5 in
        # (full-page portrait); keep the original aspect, capped at 8.6 in
        # tall so it fits a portrait page.
        # use the full text width (7.2 in) and a wider gap
        # between the two stacked panels so a and b never overlap. The
        # magnitude sub-panel takes a larger width share so the demand bars use
        # the available horizontal space.
        _orig_w = 11.5
        _orig_h = max(11, len(mat_order) * 0.52)
        _mag_ratio = 2.7
        if docx:
            _w = 7.2
            _h = min(_w * (_orig_h / _orig_w), 9.4)
            _hspace, _wspace = 0.52, 0.05
        else:
            _w, _h = _orig_w, _orig_h
            _hspace, _wspace = 0.50, 0.06
        fig_us = plt.figure(figsize=(_w, _h))
        outer_us = fig_us.add_gridspec(2, 1, hspace=_hspace)
        fig_gl = plt.figure(figsize=(_w, _h))
        outer_gl = fig_gl.add_gridspec(2, 1, hspace=_hspace)
        inner = {
            "A": outer_us[0, 0].subgridspec(1, 2, width_ratios=[_mag_ratio, 1.0], wspace=_wspace),
            "B": outer_us[1, 0].subgridspec(1, 2, width_ratios=[_mag_ratio, 1.0], wspace=_wspace),
            "C": outer_gl[0, 0].subgridspec(1, 2, width_ratios=[_mag_ratio, 1.0], wspace=_wspace),
            "D": outer_gl[1, 0].subgridspec(1, 2, width_ratios=[_mag_ratio, 1.0], wspace=_wspace),
        }
        panel_fig = {"A": fig_us, "B": fig_us, "C": fig_gl, "D": fig_gl}
        fig = fig_us  # fallback handle; save block is split-aware below
    else:
        fig_us = fig_gl = None
        fig = plt.figure(figsize=(19, max(10, len(mat_order) * 0.36)))
        # Outer gridspec: 2 rows × 2 panel-pairs, with a wide gap between pairs.
        outer = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.42)
        # Per-panel inner gridspec splits each cell into (mag | mix) with tight
        # wspace so the two halves read as a single unit.
        inner = {
            "A": outer[0, 0].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.06),
            "B": outer[0, 1].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.06),
            "C": outer[1, 0].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.06),
            "D": outer[1, 1].subgridspec(1, 2, width_ratios=[2.2, 1.0], wspace=0.06),
        }
        panel_fig = {"A": fig, "B": fig, "C": fig, "D": fig}


    # Precompute REE row indices (contiguous-block detector). For ree_band
    # we want the lowest-index REE row through the highest-index REE row
    # painted as one continuous background; this looks right even if a
    # single non-REE accidentally sits between two REEs (no holes).
    ree_row_indices = [i for i, m in enumerate(mat_order)
                       if m in REE_HIGHLIGHT_SET]

    # (fig, ax, label) for split-figure panel titles, placed after tight_layout.
    _panel_titles = []
    # (fig, ax_mag, ax_mix) per panel, used after tight_layout to vertically
    # align each panel's two x-axis titles to a common baseline.
    _panel_pairs = []

    for p in ("A", "B", "C", "D"):
        _pf = panel_fig[p]
        ax_mag = _pf.add_subplot(inner[p][0, 0])
        ax_mix = _pf.add_subplot(inner[p][0, 1], sharey=ax_mag)
        _panel_pairs.append((_pf, ax_mag, ax_mix))

        # Panel labels: lowercase letters, reset per split figure (Fig 4 = a/b
        # US tiers; Fig 5 = a/b global tiers).
        if split:
            title = {"A": "a. U.S. production", "B": "b. U.S. reserves",
                     "C": "a. Global production", "D": "b. Global reserves"}[p]
            xlab = PANEL_TITLES[p][1]
        else:
            title, xlab, _ = PANEL_TITLES[p]

        # Assemble one entry per material for this panel: the demand-to-supply
        # ratio triplet (median, low, high), the worst-case-scenario value, and
        # the source-mix shares. The reserves panels (B, D) collapse the
        # per-element rare earths into a single "Rare Earths" entry, because USGS
        # publishes only the aggregate TREO reserve (no per-element reserves); the
        # aggregate ratio is the sum of the per-element demand-to-reserve ratios,
        # which share one denominator. The production panels (A, C) keep the
        # per-element breakdown.
        mid_rows = _midcase_ratios_for_panel(df, mat_order, p)
        mix_rows = _mix_shares_for_panel(
            df, mat_order, p,
            partner_reserve_shares=partner_res_shares,
            partner_reserve_coverage=partner_res_cov,
            partner_reserve_total_kt=partner_res_total_kt,
        )
        entries = []
        for (mat, r_mid, r_lo, r_hi, r_w, skip_reason), (_m, shares, flags) in zip(
                mid_rows, mix_rows):
            entries.append({"mat": mat, "r_mid": r_mid, "r_lo": r_lo, "r_hi": r_hi,
                            "r_w": r_w, "skip": skip_reason, "shares": shares,
                            "flags": flags or {}})

        if p in ("B", "D") and sum(1 for e in entries
                                   if e["mat"] in REE_HIGHLIGHT_SET) > 1:
            ree_es = [e for e in entries if e["mat"] in REE_HIGHLIGHT_SET]

            def _sum(key, _es=ree_es):
                vals = [_e[key] for _e in _es if _e[key] is not None]
                return sum(vals) if vals else None

            agg = {"mat": "Rare Earths", "r_mid": _sum("r_mid"), "r_lo": _sum("r_lo"),
                   "r_hi": _sum("r_hi"), "r_w": _sum("r_w"), "skip": None,
                   "shares": next((e["shares"] for e in ree_es if e["shares"]), None),
                   "flags": {}}
            entries = [e for e in entries
                       if e["mat"] not in REE_HIGHLIGHT_SET] + [agg]

        # Order this panel from highest to lowest stress (demand-to-supply
        # ratio). matplotlib draws row 0 at the bottom, so finite ratios are
        # sorted ascending (top row = highest stress); materials with no
        # denominator (e.g. no US production, no separate reserves) collect at
        # the bottom as labelled rows, grouped by skip reason so the two
        # categories (no reserves concept vs no domestic/global reserves) sit
        # in contiguous blocks rather than interleaving.
        finite = sorted((e for e in entries if e["r_mid"] is not None),
                        key=lambda e: e["r_mid"])
        nodata = sorted((e for e in entries if e["r_mid"] is None),
                        key=lambda e: e["skip"] or "")
        ordered = nodata + finite
        n_rows = len(ordered)

        # Magnitude axis: log scale, neutral grey bar + black whiskers, thin
        # dotted sufficiency threshold at ratio 1.0 (explained in the caption).
        ax_mag.axvline(1.0, color="black", ls=":", lw=1.2, alpha=0.85, zorder=5)
        ax_mag.set_yticks(range(n_rows))
        # No-data rows are flagged with an asterisk placed ON THE PLOT at the
        # row's left edge (not appended to the y-axis label), explained in the
        # caption: "*" = no separately reported reserves; "**" = no US
        # production (panel a) / no US reserves (panel b). Keeping the asterisk
        # off the tick label leaves the material names clean and uniform.
        def _marker(e):
            if e["r_mid"] is not None:
                return ""
            sk = (e["skip"] or "").lower()
            return "*" if ("concept" in sk or "data" in sk) else "**"
        ax_mag.set_yticklabels([e["mat"] for e in ordered], fontsize=9)
        # Rare-earth labels in bold navy so they read as a set even though each
        # element now sits at its own stress-ranked row.
        for lab, e in zip(ax_mag.get_yticklabels(), ordered):
            if e["mat"] in REE_HIGHLIGHT_SET:
                lab.set_fontweight("bold")
                lab.set_color(REE_HIGHLIGHT_COLOR)
        ax_mag.set_ylim(-0.6, n_rows - 0.4)
        ax_mag.set_xscale("symlog", linthresh=1e-4)
        ax_mag.set_xlabel(xlab + " (log)", fontsize=9.5, fontweight="bold")
        if split:
            # Panel-title placement is deferred to after tight_layout (in the
            # save block) so it can be left-aligned to the figure's left content
            # margin (the leftmost y-tick label) instead of the plot area.
            _panel_titles.append((_pf, ax_mag, title))
        else:
            ax_mag.set_title(title, loc="left", fontsize=11.5,
                             fontweight="bold", pad=8)
        ax_mag.grid(True, alpha=0.15, axis="x")
        # Remove x and y axis tick marks (keep the labels).
        ax_mag.tick_params(axis="both", which="both", length=0)

        hi_candidates = [v for e in finite for v in (e["r_mid"], e["r_hi"], e["r_w"])
                         if v is not None and np.isfinite(v)]
        upper = (max(hi_candidates) * 1.25) if hi_candidates else 1.0
        ax_mag.set_xlim(right=max(10, upper) if p in ("A", "B") else max(1.5, upper))

        for i, e in enumerate(ordered):
            if e["r_mid"] is None:
                if styling == "poster":
                    trans = blended_transform_factory(ax_mag.transAxes,
                                                       ax_mag.transData)
                    ax_mag.text(0.5, i, e["skip"] if e["skip"] else "No data",
                                transform=trans, fontsize=11, fontweight="bold",
                                ha="center", va="center", color=NO_PROD_COLOR,
                                zorder=4)
                else:
                    # Manuscript styling: asterisk marker drawn ON THE PLOT at
                    # the row's left edge (x in axes fraction via the blended
                    # y-axis transform, y in data coords) rather than appended
                    # to the material name on the y-axis.
                    _mk = _marker(e)
                    if _mk:
                        ax_mag.text(0.01, i, _mk,
                                    transform=ax_mag.get_yaxis_transform(),
                                    ha="left", va="center", fontsize=9,
                                    fontweight="bold", color="#555555",
                                    zorder=6)
                continue
            _draw_mag_bar_midcase(ax_mag, i, e["r_mid"], e["r_lo"], e["r_hi"],
                                  e["r_w"], color=MAG_BAR_COLOR)

        # Source-mix axis: 100%-stacked CRC/HTS bars (translucent), 50% guide.
        ax_mix.set_xlim(0, 1)
        ax_mix.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax_mix.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        # Dashed 50% reference so readers can anchor on more-than-half /
        # less-than-half without counting ticks.
        ax_mix.axvline(0.5, color="#222222", ls="--", lw=1.0, alpha=0.85, zorder=7)
        ax_mix.tick_params(axis="y", left=False, labelleft=False)
        # Remove x tick marks too (keep the % labels).
        ax_mix.tick_params(axis="x", which="both", length=0)
        ax_mix.set_xlabel(MIX_X_LABELS[p], fontsize=8.5, fontweight="bold")
        # Share-basis subtitles removed from the figure (explained in the
        # caption instead).
        ax_mix.grid(True, alpha=0.15, axis="x")

        # Bring the plot-area border (spines = the box outline + the x/y
        # baselines) to the front so the data bars (zorder 3) do not paint over
        # the axis lines. Without this the left baseline and the box edges are
        # hidden wherever a bar runs up against them.
        for _ax in (ax_mag, ax_mix):
            for _sp in _ax.spines.values():
                _sp.set_zorder(10)

        for i, e in enumerate(ordered):
            if e["shares"] is None:
                continue
            _draw_mix_bar(ax_mix, i, e["shares"])
            if show_flow_fallback and e["flags"].get("panel_b_flow_fallback"):
                _overlay_flow_fallback_hatch(ax_mix, i)

        if styling == "poster":
            for ax in (ax_mag, ax_mix):
                for i in range(n_rows):
                    if i % 2 == 1:
                        ax.axhspan(i - 0.5, i + 0.5, color="#f2f2f2",
                                   zorder=0, alpha=0.6)
                ax.spines[["top", "right"]].set_visible(False)

    # the legend shows the supply-source country-risk groups
    # (translucent to match the bars, "US domestic" relabelled
    # "United States"). The highest-demand-scenario hatched extension, the
    # MC p2.5-p97.5 whiskers, and the sufficiency threshold line are
    # described in the figure caption instead of the legend.
    # The dark-gray "Unknown" mix segment gets a legend entry on the global
    # figure only; the US panels never draw that filler.
    GROUP_LEGEND_LABELS = {"US domestic": "United States"}
    handles = [mpatches.Patch(color=GROUP_COLORS[g], alpha=0.82,
                              label=GROUP_LEGEND_LABELS.get(g, g))
               for g in GROUP_ORDER]
    unknown_handle = mpatches.Patch(color=UNKNOWN_COLOR, alpha=0.82,
                                    label="Unknown")
    if split:
        out_us = output_path.parent / "fig4_supply_tiers_us.png"
        out_gl = output_path.parent / "fig5_supply_tiers_global.png"
        for _fig, _ttl, _out in (
            (fig_us, "Figure 4. US critical-material demand vs domestic supply, 2026-2050 (Mid Case (with IRA), MC + scenario uncertainty)", out_us),
            (fig_gl, "Figure 5. US critical-material demand vs global supply, 2026-2050 (Mid Case (with IRA), MC + scenario uncertainty)", out_gl),
        ):
            # Under docx the figure is at the 6.5 in print width, so the long
            # single-line title and the wide 4-column legend overflow the
            # canvas and bbox_inches="tight" would save a >6.5 in image. Wrap
            # the title to two lines and stack the legend into 3 columns so
            # the saved width stays at the 6.5 in print target. Layout-only;
            # title words and legend entries are unchanged.
            _ncol = 3 if docx else 4
            # Anchor by the legend's UPPER edge and place it below the figure so
            # it grows downward into clear space, rather than the old lower-edge
            # anchor at y=-0.02 that let the top legend row grow up into the
            # x-axis labels / italic mix subtitle. The subtitle (mix-axis
            # transAxes y=-0.20) lands just below the figure bottom edge, so the
            # legend top must sit lower than that to leave a visible gap. Tuned
            # per render size (the docx canvas is short, so the gap is a larger
            # figure-fraction than on the tall original canvas).
            _leg_top_y = 0.012 if docx else 0.049
            # Global figure carries the extra "Unknown" entry (its mix bars
            # are the only ones that draw the dark-gray unclassified filler).
            _handles = handles + [unknown_handle] if _fig is fig_gl else handles
            _fig.legend(handles=_handles, loc="upper center",
                        bbox_to_anchor=(0.5, _leg_top_y), ncol=_ncol,
                        fontsize=9, frameon=False)
            # No figure title on the figure (the caption carries it).
            if docx:
                # Render-at-print-size docx mode: the figure is already at the
                # 6.5 in print width, so absolute point sizes set here equal the
                # on-page pt when inserted at the saved (bbox-tight) width. Set
                # by ROLE — this is a dense full-page figure, so these are the
                # largest sizes that fit (small text is accepted to keep fig4/5
                # as single full-page figures). Fonts are set BEFORE
                # tight_layout so the layout solver packs at the final sizes.
                _SUP, _AXTITLE, _AXLABEL, _TICK, _LEG, _ANNOT = (
                    11, 9.5, 9, 7.5, 7.5, 7)
                if getattr(_fig, "_suptitle", None) is not None:
                    _fig._suptitle.set_fontsize(_SUP)
                for _ax in _fig.get_axes():
                    if _ax.get_title():
                        _ax.title.set_fontsize(_AXTITLE)
                    if _ax.get_xlabel():
                        _ax.xaxis.label.set_fontsize(_AXLABEL)
                    if _ax.get_ylabel():
                        _ax.yaxis.label.set_fontsize(_AXLABEL)
                    for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                        _t.set_fontsize(_TICK)
                    # In-axes text annotations (MIX_X_SUBTITLE derivation note,
                    # any inline "no US production / no reserves" skip labels)
                    # are the smallest annotations — set them to the side-share
                    # / smallest-annotation target.
                    for _t in _ax.texts:
                        _t.set_fontsize(_ANNOT)
                    _lg = _ax.get_legend()
                    if _lg:
                        for _x in _lg.get_texts():
                            _x.set_fontsize(_LEG)
                for _lg in _fig.legends:
                    for _x in _lg.get_texts():
                        _x.set_fontsize(_LEG)
                _fig.tight_layout(rect=[0, 0.045, 1, 0.95])
            else:
                _fig.tight_layout(rect=[0, 0.045, 1, 0.95])

            # Panel titles: left-aligned to the figure's left content margin
            # (the leftmost y-tick label), placed after tight_layout so the
            # final tick-label extents are known. Using a figure-level text (not
            # the axes title) avoids matplotlib re-centering it over the plot.
            _fig.canvas.draw()
            _rend = _fig.canvas.get_renderer()

            # Vertically align each panel's two x-axis titles (magnitude bar +
            # source-mix). The magnitude axis carries tall log-exponent tick
            # labels while the mix axis carries short "%" labels, so
            # matplotlib's auto-placement drops the two xlabels to different
            # heights. Pin both just below the lower of the two tick-label
            # bottoms; the sub-axes share the panel row, so an identical
            # axes-fraction y lands the labels at the same figure height.
            _pad_px = (3.0 / 72.0) * _fig.dpi
            for _pf2, _axm_p, _axx_p in _panel_pairs:
                if _pf2 is not _fig:
                    continue
                _bottoms = [t.get_window_extent(_rend).y0
                            for _ax in (_axm_p, _axx_p)
                            for t in _ax.get_xticklabels() if t.get_text()]
                if not _bottoms:
                    continue
                _y_disp = min(_bottoms) - _pad_px
                for _ax in (_axm_p, _axx_p):
                    _yf = _ax.transAxes.inverted().transform((0, _y_disp))[1]
                    _ax.xaxis.set_label_coords(0.5, _yf)

            _tfs = 9.5 if docx else 11.5
            _fig_panels = [(a, l) for f, a, l in _panel_titles if f is _fig]
            # Common left edge = leftmost y-tick label across ALL panels in the
            # figure, so the panel titles (a., b.) align with each other and with
            # the figure's left content margin (panels have different materials,
            # so their individual leftmost labels differ).
            _all_xs = [t.get_window_extent(_rend).x0
                       for _axm, _ in _fig_panels
                       for t in _axm.get_yticklabels() if t.get_text()]
            _left_disp = min(_all_xs) if _all_xs else 0.0
            _left_fig = _fig.transFigure.inverted().transform((_left_disp, 0))[0]
            for _axm, _lbl in _fig_panels:
                _top_fig = _axm.get_position().y1
                _fig.text(_left_fig, _top_fig + 0.006, _lbl, ha="left",
                          va="bottom", fontsize=_tfs, fontweight="bold")

            _fig.savefig(_out, dpi=FIGURE_DPI, bbox_inches="tight")
            plt.close(_fig)
            print(f"  Saved {_out.name}")
    else:
        fig.legend(handles=handles, loc="lower center",
                   bbox_to_anchor=(0.5, -0.035), ncol=4, fontsize=9,
                   frameon=False)
        fig.suptitle(
            "Figure 5. Supply coverage and sourcing mix, 2026-2050 (Mid Case (with IRA) with MC + scenario uncertainty)",
            fontsize=12, fontweight="bold", y=0.995,
        )
        fig.tight_layout(rect=[0, 0.04, 1, 0.93])
        fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant 3b — mix-only companion figure (to pair with existing log-magnitude)
# ─────────────────────────────────────────────────────────────────────────────

def plot_option3b(df, mat_order, output_path):
    """4-panel 100%-stacked sourcing mix, linear axis only."""
    fig = plt.figure(figsize=(15, max(11, len(mat_order) * 0.4)))
    gs = fig.add_gridspec(2, 2, wspace=0.22, hspace=0.58)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
    }


    panel_xlabels = {
        "A": "US demand sourcing mix (fraction)",
        "B": "US demand sourcing mix (fraction)",
        "C": "Global producer-country mix (fraction)",
        "D": "Global reserve-holder-country mix (fraction)",
    }

    for p, ax in axes.items():
        title, _, sub = PANEL_TITLES[p]
        ax.set_yticks(range(len(mat_order)))
        ax.set_yticklabels(_mat_labels_with_backest_glyph(mat_order, df, p),
                           fontsize=9)
        ax.set_ylim(-0.6, len(mat_order) - 0.4)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.set_xlabel(panel_xlabels[p], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=22)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8,
                style="italic", color="#555555")
        ax.grid(True, alpha=0.15, axis="x")

        mix = _mix_shares_for_panel(df, mat_order, p)
        for i, (mat, shares, flags) in enumerate(mix):
            if shares is None:
                continue
            _draw_mix_bar(ax, i, shares)
            if flags.get("panel_b_flow_fallback"):
                _overlay_flow_fallback_hatch(ax, i)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    handles.append(mpatches.Patch(color=UNKNOWN_COLOR, label=UNKNOWN_LABEL))
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=6, fontsize=9, frameon=False,
               title="Source country risk group (OECD CRC 2026)",
               title_fontsize=8.5)

    fig.suptitle(
        "Figure 5 (Variant 3b). Sourcing-mix companion — shows only WHERE supply comes from, "
        "to pair with the log-scale magnitude figure",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Variant 4 — Raw tonnage forest plot (supply bar + demand marker, log axis)
# ─────────────────────────────────────────────────────────────────────────────

def _tonnage_for_panel(df, mat_order, panel):
    """Return per-material raw-tonnage data for the forest plot.
    Yields (mat, supply_t, supply_shares, demand_med, demand_lo, demand_hi, skip_reason).
    supply_shares sums to supply_t (in absolute tonnes, CRC-partitioned)."""
    rows = []
    for mat in mat_order:
        row = df.loc[mat]
        if panel == "A":
            sup = row.get("us_prod_t", 0) or 0
            if sup <= 0:
                rows.append((mat, None, None, None, None, None, "no US production"))
                continue
            shares_abs = {"US domestic": sup}
            for g in GROUP_ORDER[1:]:
                shares_abs[g] = 0.0
            dmed, dlo, dhi = row["peak_median"], row["peak_min"], row["peak_max"]
        elif panel == "B":
            if not row.get("has_reserves", False):
                rows.append((mat, None, None, None, None, None, "no reserves concept"))
                continue
            sup = row.get("us_reserves_t", 0) or 0
            if sup <= 0:
                rows.append((mat, None, None, None, None, None, "no US reserves"))
                continue
            shares_abs = {"US domestic": sup}
            for g in GROUP_ORDER[1:]:
                shares_abs[g] = 0.0
            dmed, dlo, dhi = row["cum_median"], row["cum_min"], row["cum_max"]
        elif panel == "C":
            sup = row.get("global_prod_t", 0) or 0
            if sup <= 0:
                rows.append((mat, None, None, None, None, None, "no global prod data"))
                continue
            frac = row["prod_shares_grouped"]
            shares_abs = {g: sup * frac.get(g, 0) for g in GROUP_ORDER}
            dmed, dlo, dhi = row["peak_median"], row["peak_min"], row["peak_max"]
        elif panel == "D":
            if not row.get("has_reserves", False):
                rows.append((mat, None, None, None, None, None, "no reserves concept"))
                continue
            sup = row.get("global_reserves_t", 0) or 0
            if sup <= 0:
                rows.append((mat, None, None, None, None, None, "no reserve data"))
                continue
            frac = row.get("reserve_shares_grouped", {g: 0 for g in GROUP_ORDER})
            shares_abs = {g: sup * frac.get(g, 0) for g in GROUP_ORDER}
            dmed, dlo, dhi = row["cum_median"], row["cum_min"], row["cum_max"]
        else:
            raise ValueError(panel)
        rows.append((mat, sup, shares_abs, dmed, dlo, dhi, None))
    return rows


def _draw_supply_stack_log(ax, i, shares_abs, total):
    """Log-axis stacked supply bar at y=i-0.18 (upper half of row)."""
    if total <= 0:
        return
    # Log-axis stacked bars are drawn in data units but need to respect the
    # log transform — we draw each segment as a rect from log-space-correct
    # left/right positions. matplotlib's barh on a log axis is fine as long
    # as we set left > 0; the first segment must start at left > 0, so we
    # use a tiny epsilon based on the bar's total.
    # However, log-stacked bars ARE the distortion problem we were trying to
    # solve. Here we accept it as a display trade-off because the dominant
    # story is "how big is the supply pool", not "what's the CRC mix" —
    # the CRC mix is overlaid but visually compressed. Readers compare to
    # the companion mix figure for exact proportions.
    left = max(total * 1e-6, 1e-3)
    for grp in GROUP_ORDER:
        w = shares_abs.get(grp, 0)
        if w <= 0:
            continue
        ax.barh(i - 0.18, w, height=0.32, left=left,
                color=GROUP_COLORS[grp],
                edgecolor="white", linewidth=0.4, zorder=3)
        left += w


def _draw_demand_marker(ax, i, dmed, dlo, dhi):
    if not np.isfinite(dmed) or dmed <= 0:
        return
    ax.scatter([dmed], [i + 0.18], marker="D", s=38,
               facecolor="#c0392b", edgecolor="white",
               linewidths=0.8, zorder=6)
    if np.isfinite(dlo) and np.isfinite(dhi) and dhi > dlo:
        ax.errorbar(dmed, i + 0.18,
                    xerr=[[max(0, dmed - dlo)], [max(0, dhi - dmed)]],
                    fmt="none", color="#c0392b", lw=1.1, capsize=3,
                    alpha=0.85, zorder=6)


def plot_option4(df, mat_order, output_path):
    """Raw tonnage forest plot. Each panel: log x-axis in tonnes.
      - Supply bar (CRC-stacked, upper half of row)
      - Demand marker (diamond, lower half of row) with MC p2.5-p97.5 whisker
      - Shortfall is visible when demand diamond lies right of supply bar end.
    """
    fig = plt.figure(figsize=(19, max(12, len(mat_order) * 0.44)))
    gs = fig.add_gridspec(2, 2, wspace=0.30, hspace=0.58)
    axes = {
        "A": fig.add_subplot(gs[0, 0]),
        "B": fig.add_subplot(gs[0, 1]),
        "C": fig.add_subplot(gs[1, 0]),
        "D": fig.add_subplot(gs[1, 1]),
    }


    panel_xlabels = {
        "A": "Tonnes — US annual production (bar) vs US peak-year demand (♦)",
        "B": "Tonnes — US reserves (bar) vs cumulative 2026–2050 demand (♦)",
        "C": "Tonnes — global annual production (bar) vs US peak-year demand (♦)",
        "D": "Tonnes — global reserves (bar) vs US cumulative 2026–2050 demand (♦)",
    }
    panel_titles = {
        "A": ("A. US production vs US peak demand",
              "supply bar (blue) = US annual production; diamond = US peak-year demand (♦, MC p2.5–p97.5)"),
        "B": ("B. US reserves vs cumulative demand",
              "supply bar (blue) = US reserves; diamond = cumulative 2026–2050 US demand"),
        "C": ("C. Global production vs US peak demand",
              "supply bar = global annual production, stacked by producer country (CRC)"),
        "D": ("D. Global reserves vs cumulative US demand",
              "supply bar = global reserves, stacked by reserve-holder country (CRC)"),
    }

    # Compute a shared x-range per panel
    for p, ax in axes.items():
        rows = _tonnage_for_panel(df, mat_order, p)
        all_vals = []
        for _, sup, _, dm, dl, dh, _skip in rows:
            for v in (sup, dm, dl, dh):
                if v is not None and np.isfinite(v) and v > 0:
                    all_vals.append(v)
        if not all_vals:
            xmin, xmax = 1e0, 1e6
        else:
            xmin = max(10 ** np.floor(np.log10(min(all_vals))), 1e-1)
            xmax = 10 ** np.ceil(np.log10(max(all_vals)))

        title, sub = panel_titles[p]
        ax.set_xscale("log")
        ax.set_yticks(range(len(mat_order)))
        ax.set_yticklabels(mat_order, fontsize=9)
        ax.set_ylim(-0.6, len(mat_order) - 0.4)
        ax.set_xlim(xmin, xmax * 1.2)
        ax.set_xlabel(panel_xlabels[p], fontsize=9.5, fontweight="bold")
        ax.set_title(title, fontsize=11, fontweight="bold", pad=22)
        ax.text(0.5, 1.025, sub, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=8,
                style="italic", color="#555555")
        ax.grid(True, which="major", alpha=0.18, axis="x")
        ax.grid(True, which="minor", alpha=0.08, axis="x")

        for i, (mat, sup, shares_abs, dm, dl, dh, skip_reason) in enumerate(rows):
            if sup is None:
                _skip_text(ax, i, skip_reason)
                continue
            _draw_supply_stack_log(ax, i, shares_abs, sup)
            _draw_demand_marker(ax, i, dm, dl, dh)

    handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in GROUP_ORDER]
    handles.append(mpatches.Patch(color=UNKNOWN_COLOR, label=UNKNOWN_LABEL))
    handles += [
        plt.Line2D([], [], color="#c0392b", marker="D", markersize=8,
                   linestyle="none", markeredgecolor="white",
                   label="US demand (♦ median, whiskers = MC p2.5–p97.5)"),
    ]
    fig.legend(handles=handles, loc="lower center",
               bbox_to_anchor=(0.5, -0.04), ncol=4, fontsize=9, frameon=False,
               title="Supply = bar (CRC-partitioned in C/D). Demand = ♦. Shortfall → demand diamond lies right of bar end.",
               title_fontsize=8.5)

    fig.suptitle(
        "Figure 5 (Variant 4). Raw tonnage forest plot — absolute supply pool vs MC demand (log t)",
        fontsize=12, fontweight="bold", y=0.995,
    )
    fig.tight_layout(rect=[0, 0.04, 1, 0.93])
    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    add_interpolation_arg(ap)
    ap.add_argument("--output-dir", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir) if args.output_dir else \
        FIGURES_MANUSCRIPT_DIR / "fig4_fig5_variants" / "past_iterations"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data…")
    df, mat_order = build_records(re_mode="aggregate")

    print("Rendering Variant 2a — linear, fixed caps")
    plot_option2(df, mat_order, out_dir / "fig5_alt__opt2a_linear_clipped.png",
                 mode="clipped")

    print("Rendering Variant 2b — linear, autorange caps")
    plot_option2(df, mat_order, out_dir / "fig5_alt__opt2b_linear_autorange.png",
                 mode="autorange")

    print("Rendering Variant 3a — split magnitude + mix")
    plot_option3a(df, mat_order, out_dir / "fig5_alt__opt3a_split_mag_mix.png")

    print("Rendering Variant 3a-midcase — Mid_Case + worst-case scenario extension")
    plot_option3a_midcase(df, mat_order, out_dir / "fig5_alt__opt3a_midcase_extension.png")

    print("Rendering Variant 3a-midcase POSTER — split mag+mix with poster styling")
    plot_option3a_midcase(df, mat_order, out_dir / "fig5_alt__opt3a_midcase_poster.png", styling="poster")

    print("Rendering Variant 3a-midcase POSTER (no flow-fallback callout)")
    plot_option3a_midcase(df, mat_order,
                          out_dir / "fig5_alt__opt3a_midcase_poster_no_callout.png",
                          styling="poster", show_flow_fallback=False)

    print("Rendering Variant 3b — mix-only companion")
    plot_option3b(df, mat_order, out_dir / "fig5_alt__opt3b_mix_companion.png")

    print("Rendering Variant 4 — raw tonnage forest plot")
    plot_option4(df, mat_order, out_dir / "fig5_alt__opt4_raw_tonnage.png")

    print("Done.")


if __name__ == "__main__":
    main()
