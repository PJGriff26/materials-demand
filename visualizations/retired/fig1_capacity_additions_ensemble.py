"""Manuscript Figure 1: annual capacity additions and retirements.

Year-over-year stacked view of additions (positive axis) and retirements
(negative axis) by technology group across the 61 NREL Standard Scenarios
2024, exposing the build-out peak. Reads StdScen24_annual_national.csv.
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

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

from config import FIGURES_MANUSCRIPT_DIR, FIGURE_DPI, NREL_SCENARIOS_FILE


# ═══════════════════════════════════════════════════════════════════════════════
# TECHNOLOGY GROUPING
# ═══════════════════════════════════════════════════════════════════════════════

TECH_GROUPS = {
    "Utility solar":     ["upv_MW"],
    "Distributed solar": ["distpv_MW"],
    "Wind":     ["wind_onshore_MW", "wind_offshore_MW"],
    "Storage":  ["battery_4_MW", "battery_8_MW"],
    "Nuclear":  ["nuclear_MW", "nuclear_smr_MW"],
    "Hydro":    ["hydro_MW"],
    "Gas":      ["gas_cc_MW", "gas_ct_MW", "gas_cc_ccs_MW"],
    "Coal":     ["coal_MW", "coal_ccs_MW"],
    "Other":    ["csp_MW", "bio_MW", "bio-ccs_MW", "geo_MW"],
}

# Stacking order for additions (bottom to top): Solar/Wind at bottom, rare at top.
ADDITION_ORDER = ["Utility solar", "Distributed solar", "Wind", "Storage", "Gas", "Nuclear", "Other", "Hydro"]
# Coal typically retires, but show it in retirements; place in retirement order.
RETIREMENT_ORDER = ["Coal", "Gas", "Nuclear", "Hydro", "Other", "Utility solar", "Distributed solar", "Wind", "Storage"]

GROUP_COLORS = {
    "Utility solar":     "#FFD700",
    "Distributed solar": "#FFEC99",
    "Wind":     "#4169E1",
    "Storage":  "#32CD32",
    "Nuclear":  "#9370DB",
    "Hydro":    "#00CED1",
    "Gas":      "#808080",
    "Coal":     "#3a3a3a",
    "Other":    "#D2B48C",
}

# ── Disaggregated technology view (opt-in via --disaggregate). Every individual
# technology becomes its own stacked band + legend entry (e.g. onshore vs
# offshore wind, gas CC vs CT vs CC-CCS) instead of the aggregated families
# above. Colors are family-shaded so the grouping still reads. Technologies with
# negligible capacity (CSP, biomass, biomass-CCS, and any all-zero tech) are
# auto-skipped at plot time so they never clutter the legend.
DISAGG_TECH_GROUPS = {
    "Utility solar":     ["upv_MW"],
    "Distributed solar": ["distpv_MW"],
    "Onshore wind":      ["wind_onshore_MW"],
    "Offshore wind":     ["wind_offshore_MW"],
    "Storage":           ["battery_4_MW", "battery_8_MW"],
    "Gas":               ["gas_cc_MW", "gas_ct_MW", "gas_cc_ccs_MW"],
    "Nuclear":           ["nuclear_MW", "nuclear_smr_MW"],
    "Coal":              ["coal_MW", "coal_ccs_MW"],
    "Hydro":             ["hydro_MW"],
    "Geothermal":        ["geo_MW"],
    "CSP":               ["csp_MW"],
    "Biomass":           ["bio_MW", "bio-ccs_MW"],
}
DISAGG_ADDITION_ORDER = [
    "Utility solar", "Distributed solar", "Onshore wind", "Offshore wind",
    "Storage", "Gas", "Nuclear", "Coal", "Hydro", "Geothermal",
    "CSP", "Biomass",
]
DISAGG_RETIREMENT_ORDER = [
    "Coal", "Gas", "Nuclear", "Hydro", "Geothermal", "Onshore wind",
    "Offshore wind", "Utility solar", "Distributed solar", "Storage",
    "CSP", "Biomass",
]
DISAGG_GROUP_COLORS = {
    "Utility solar":     "#FFD700",   # gold
    "Distributed solar": "#FFEC99",   # pale gold
    "Onshore wind":      "#3B6CB7",   # royal blue
    "Offshore wind":     "#9CC3E8",   # light blue
    "Storage":           "#32CD32",   # green
    "Gas":               "#808080",   # grey
    "Nuclear":           "#9370DB",   # purple
    "Coal":              "#3A3A3A",   # near-black
    "Hydro":             "#00CED1",   # cyan
    "Geothermal":        "#C97B3C",   # burnt orange
    "CSP":               "#E6B800",   # amber
    "Biomass":           "#6B8E23",   # olive
}


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def load_nrel():
    """Load the NREL Standard Scenarios 2024 national capacity table.

    The first three rows are NREL header/metadata lines (skiprows=3); the
    remaining rows hold capacity (MW) by scenario, year (column "t"), and
    technology column.
    """
    df = pd.read_csv(NREL_SCENARIOS_FILE, skiprows=3)
    return df


def compute_yoy_changes(nrel):
    """
    For each scenario × year × technology group, compute year-over-year change
    in capacity (positive = net addition, negative = net retirement).

    Returns
    -------
    yoy : DataFrame indexed by (scenario, year) with columns = tech groups
          (values in GW).
    """
    # Sort by scenario, year
    df = nrel.sort_values(["scenario", "t"]).copy()

    # For each tech group, sum the MW columns and compute diff within each scenario
    records = []
    for scen, group in df.groupby("scenario"):
        group = group.sort_values("t").reset_index(drop=True)
        for tech_name, cols in TECH_GROUPS.items():
            cols_present = [c for c in cols if c in group.columns]
            if not cols_present:
                continue
            total_mw = group[cols_present].sum(axis=1)
            diff_gw = total_mw.diff() / 1000.0  # MW → GW
            for i, year in enumerate(group["t"].values):
                if i == 0:
                    continue
                records.append({
                    "scenario": scen,
                    "year": int(year),
                    "tech": tech_name,
                    "delta_gw": diff_gw.iloc[i],
                })

    yoy = pd.DataFrame(records)
    return yoy


def scenario_view(yoy, scenario="Mid_Case"):
    """Per-scenario tech×year additions / retirements + scenario net by year.

    Returns
    -------
    tech_year_adds : DataFrame [tech, year, add_gw]   (positive-only)
    tech_year_rets : DataFrame [tech, year, ret_gw]   (negative-only)
    net_by_year    : Series indexed by year, total net delta (GW)
    """
    sub = yoy[yoy["scenario"] == scenario].copy()
    sub["add_gw"] = sub["delta_gw"].clip(lower=0)
    sub["ret_gw"] = sub["delta_gw"].clip(upper=0)
    tech_year_adds = sub.groupby(["tech", "year"])["add_gw"].sum().reset_index()
    tech_year_rets = sub.groupby(["tech", "year"])["ret_gw"].sum().reset_index()
    net_by_year = sub.groupby("year")["delta_gw"].sum()
    return tech_year_adds, tech_year_rets, net_by_year


def find_extreme_net_scenarios(yoy):
    """Identify the scenarios with the highest and lowest cumulative net
    capacity change across 2026–2050.

    Returns
    -------
    (max_scen, min_scen) : tuple of scenario names
    """
    cum_net = (yoy.groupby(["scenario", "year"])["delta_gw"].sum()
                  .groupby("scenario").sum())
    return cum_net.idxmax(), cum_net.idxmin()


def scenario_net_by_year(yoy, scenario):
    """Net delta (GW) per year for a single named scenario."""
    return (yoy[yoy["scenario"] == scenario]
            .groupby("year")["delta_gw"].sum())


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT
# ═══════════════════════════════════════════════════════════════════════════════

def plot(tech_year_adds, tech_year_rets, net_by_year,
         max_net_name, max_net_series,
         min_net_name, min_net_series,
         output_path, scenario_label="Mid_Case",
         extra_net_name=None, extra_net_series=None, docx=False,
         disaggregate=False, pchip=False):
    years = sorted(net_by_year.index.tolist())
    x = np.arange(len(years))

    fig, ax = plt.subplots(
        figsize=((6.5, 5.7) if disaggregate else (6.5, 4.8)) if docx else (13, 7))

    width = 0.9

    # Skip technologies with negligible capacity so the (disaggregated) legend
    # stays clean: any tech whose total |additions| + |retirements| over the
    # horizon is under 0.5 GW (e.g. CSP, biomass, biomass-CCS, all-zero techs).
    _add_tot = (tech_year_adds.groupby("tech")["add_gw"].sum()
                if len(tech_year_adds) else pd.Series(dtype=float))
    _ret_tot = (tech_year_rets.groupby("tech")["ret_gw"].sum()
                if len(tech_year_rets) else pd.Series(dtype=float))

    def _active(t):
        return abs(float(_add_tot.get(t, 0.0))) + abs(float(_ret_tot.get(t, 0.0))) >= 0.5

    # ── Positive stack (additions) ──
    bottom_pos = np.zeros(len(years))
    for tech in ADDITION_ORDER:
        if not _active(tech):
            continue
        sub = tech_year_adds[tech_year_adds["tech"] == tech].set_index("year").reindex(years)
        vals = sub["add_gw"].fillna(0).values
        ax.bar(x, vals, width=width, bottom=bottom_pos,
               color=GROUP_COLORS[tech], label=tech, linewidth=0)
        bottom_pos += vals

    # ── Negative stack (retirements) ──
    bottom_neg = np.zeros(len(years))
    for tech in RETIREMENT_ORDER:
        if not _active(tech):
            continue
        sub = tech_year_rets[tech_year_rets["tech"] == tech].set_index("year").reindex(years)
        vals = sub["ret_gw"].fillna(0).values  # negative numbers
        # Full opacity so the retirement-bar color exactly matches the legend
        # swatch (a 75% alpha rendered near-black Coal as mid-grey ~ Gas, which
        # read as the wrong technology). Above/below the zero line already
        # distinguishes additions from retirements.
        ax.bar(x, vals, width=width, bottom=bottom_neg,
               color=GROUP_COLORS[tech], linewidth=0)
        bottom_neg += vals

    # Zero line (reference)
    ax.axhline(0, color="black", lw=0.8, alpha=0.7)

    # Net trajectory for the bar-scenario (Mid_Case).
    net_main = net_by_year.reindex(years).fillna(0).values
    ax.plot(x, net_main, color="black", lw=2.2, marker="o", markersize=5,
            markerfacecolor="white", markeredgecolor="black",
            markeredgewidth=1.2, zorder=6,
            label=("Net: Mid Case (with IRA)" if scenario_label == "Mid_Case"
                   else f"Net: {scenario_label}"))

    # Max-net and min-net scenario envelopes: the bars represent the Mid_Case
    # scenario, and the two flanking lines show the highest- and
    # lowest-cumulative-net scenarios in the 61-scenario ensemble (here the two
    # decarbonization-pace endpoints, net-zero by 2035 and net-zero by 2050).
    net_hi = max_net_series.reindex(years).fillna(0).values
    net_lo = min_net_series.reindex(years).fillna(0).values
    ax.plot(x, net_hi, color="#a83232", lw=1.8, ls="--",
            marker="^", markersize=5, zorder=6,
            label="Net: Net-Zero by 2035")
    ax.plot(x, net_lo, color="#1f5fa8", lw=1.8, ls="--",
            marker="v", markersize=5, zorder=6,
            label="Net: Net-Zero by 2050")
    if extra_net_series is not None:
        net_extra = extra_net_series.reindex(years).fillna(0).values
        ax.plot(x, net_extra, color="#888888", lw=1.8, ls=":",
                marker="s", markersize=4, zorder=6,
                label="Net: Mid Case (no IRA)")

    _step = 3 if pchip else 2   # ~8 ticks for the 24-year annual grid, else every other 3-yr point
    ax.set_xticks(x[::_step])
    ax.set_xticklabels([str(y) for y in years[::_step]], rotation=0, fontsize=9)
    ax.set_xlabel("Year", fontsize=11, fontweight="bold")
    if pchip:
        ylabel_text = ("Annual capacity change\n(GW/yr): additions above 0, retirements below"
                       if docx else
                       "Annual capacity change (GW/yr): additions above 0, retirements below")
    else:
        ylabel_text = ("Capacity change per 3-yr reporting interval\n(GW): additions above 0, retirements below"
                       if docx else
                       "Capacity change per 3-yr reporting interval (GW): additions above 0, retirements below")
    ax.set_ylabel(ylabel_text, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.2, axis="y")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # De-duplicate legend (tech labels stored once)
    handles, labels = ax.get_legend_handles_labels()
    seen = set()
    keep = []
    for h, l in zip(handles, labels):
        if l not in seen:
            keep.append((h, l))
            seen.add(l)
    if docx:
        # Place the legend BELOW the axes (horizontal, frameless) so it never
        # obscures the bars or the late-period net lines. The figure is wide and
        # short, so expanding downward keeps the full plot width.
        ax.legend([h for h, l in keep], [l for h, l in keep],
                  loc="upper center", bbox_to_anchor=(0.5, -0.13),
                  fontsize=8, frameon=False, ncol=4, columnspacing=1.2,
                  handletextpad=0.5, title="Technology group",
                  title_fontsize=8).get_title().set_fontweight("bold")
    else:
        ax.legend([h for h, l in keep], [l for h, l in keep],
                  loc="lower right", fontsize=8, frameon=True, framealpha=0.92,
                  ncol=2, title="Technology group",
                  title_fontsize=8).get_title().set_fontweight("bold")

    fig.tight_layout()

    if docx:
        # Render-at-print-size approach: figure is drawn at ~6.5 in print width,
        # so absolute point sizes set here equal on-page pt when inserted at the
        # saved width. (Replaces the broken font-scaling block, which grew the
        # tight-bbox canvas by the same factor and left on-page sizes invariant.)
        AXTITLE = 10.5   # any axis/figure title
        AXLABEL = 8.5    # x/y axis labels
        TICK = 7.0       # tick labels
        LEG = 7.0        # legend entries (incl. net-line labels) + legend title
        ANNOT = 7.0      # in-plot text annotations / net-line labels
        if getattr(fig, "_suptitle", None) is not None:
            fig._suptitle.set_fontsize(AXTITLE)
        for _ax in fig.get_axes():
            if _ax.get_title():
                _ax.title.set_fontsize(AXTITLE)
            _ax.xaxis.label.set_fontsize(AXLABEL)
            _ax.yaxis.label.set_fontsize(AXLABEL)
            for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                _t.set_fontsize(TICK)
            for _t in _ax.texts:
                _t.set_fontsize(ANNOT)
            _lg = _ax.get_legend()
            if _lg:
                for _x in _lg.get_texts():
                    _x.set_fontsize(LEG)
                _lgt = _lg.get_title()
                if _lgt is not None:
                    _lgt.set_fontsize(LEG)
        for _lg in fig.legends:
            for _x in _lg.get_texts():
                _x.set_fontsize(LEG)

    fig.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {output_path.name}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path,
                         default=FIGURES_MANUSCRIPT_DIR)
    parser.add_argument("--docx", action="store_true", default=False,
                        help="Render at print width (6.5 in) with absolute point sizes for "
                             "legible insertion into a Word docx.")
    parser.add_argument("--disaggregate", action="store_true", default=False,
                        help="Break each technology family into individual technologies "
                             "(e.g. onshore vs offshore wind, gas CC vs CT vs CC-CCS) as "
                             "separate stacked bands. Negligible techs are auto-skipped.")
    parser.add_argument("--pchip", action="store_true", default=False,
                        help="Interpolate the NREL 3-yr-cadence capacity stock to an annual "
                             "grid via monotone PCHIP (Fritsch & Carlson 1980, the pipeline's "
                             "default), so the figure shows ANNUAL additions instead of 3-yr "
                             "interval totals. Matches the annual cadence of Figs 2/5/6.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.disaggregate:
        # Swap the module-level group definitions to the disaggregated set;
        # compute_yoy_changes() and plot() read these globals at call time.
        global TECH_GROUPS, ADDITION_ORDER, RETIREMENT_ORDER, GROUP_COLORS
        TECH_GROUPS = DISAGG_TECH_GROUPS
        ADDITION_ORDER = DISAGG_ADDITION_ORDER
        RETIREMENT_ORDER = DISAGG_RETIREMENT_ORDER
        GROUP_COLORS = DISAGG_GROUP_COLORS

    nrel = load_nrel()
    print(f"Loaded NREL: {nrel.shape[0]:,} rows  "
          f"({nrel['scenario'].nunique()} scenarios, "
          f"{nrel['t'].min()}–{nrel['t'].max()})")

    if args.pchip:
        # PCHIP-interpolate the 3-yr-cadence capacity STOCK to a 1-year grid
        # (Fritsch & Carlson 1980), using the project's canonical routine, then
        # let compute_yoy_changes() take annual Δstock as annual additions.
        from src.interpolation import interpolate_capacity_pchip
        nrel = (interpolate_capacity_pchip(nrel.rename(columns={"t": "year"}))
                .rename(columns={"year": "t"}))
        print(f"  PCHIP-interpolated to annual grid: {nrel['t'].min()}–{nrel['t'].max()} "
              f"({nrel['t'].nunique()} years)")

    yoy = compute_yoy_changes(nrel)
    adds, rets, net_main = scenario_view(yoy, scenario="Mid_Case")
    # Comparison net lines: the two decarbonization-pace scenarios, so the
    # figure shows the 2035 net-zero amplification and the slower 95%-by-2050 path.
    max_name, min_name = "Mid_Case_100by2035", "Mid_Case_95by2050"
    print(f"  Bar scenario: Mid_Case")
    print(f"  Comparison net lines: {max_name} (Net-Zero by 2035), {min_name} (Net-Zero by 2050)")
    max_net = scenario_net_by_year(yoy, max_name)
    min_net = scenario_net_by_year(yoy, min_name)
    no_ira_net = scenario_net_by_year(yoy, "Mid_Case_No_IRA")

    out = args.output_dir / "fig1_capacity_additions.png"
    plot(adds, rets, net_main, max_name, max_net, min_name, min_net, out,
         scenario_label="Mid_Case",
         extra_net_name="Mid_Case_No_IRA", extra_net_series=no_ira_net,
         docx=args.docx, disaggregate=args.disaggregate, pchip=args.pchip)


if __name__ == "__main__":
    main()
