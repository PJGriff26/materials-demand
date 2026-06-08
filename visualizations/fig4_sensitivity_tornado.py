"""Manuscript Figure 4: demand-uncertainty sensitivity tornado.

Paired-bar tornado comparing demand sensitivity to scenario choice
against material-intensity uncertainty, per material class, pooled across
reporting years 2029-2050. Reads
outputs/data/material_demand_by_scenario.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo-relative paths (this file lives in visualizations/, so the repository
# root is one level up). The single shipped demand CSV is already PCHIP-
# annualized, so the default and the --pchip path read the same file; they
# differ only in which years are pooled (8 reporting years vs the full annual
# grid). See the --pchip help below.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
PCHIP_INPUT = DEFAULT_INPUT
DEFAULT_OUTPUT = (
    PROJECT_ROOT / "outputs" / "figures" / "manuscript"
    / "fig4_sensitivity_tornado.png"
)

EXCLUDED = {"Fiberglass", "Glass"}
ZERO_DEMAND = {"Gadium", "Selenium", "Germanium", "Gallium"}

MATERIAL_GROUPS = {
    "Bulk commodities":    ["Cement", "Steel", "Aluminum", "Copper"],
    "Base & alloying":     ["Zinc", "Lead", "Nickel", "Tin", "Manganese",
                            "Chromium", "Molybdenum", "Vanadium", "Silicon",
                            "Magnesium", "Boron"],
    "Specialty metals":    ["Tellurium", "Indium", "Cadmium", "Silver", "Niobium"],
    "Rare earth elements": ["Dysprosium", "Neodymium", "Praseodymium",
                            "Terbium", "Yttrium"],
}
CATEGORY_COLORS = {
    "Bulk commodities":     "#00693E",
    "Base & alloying":      "#3B6E8F",
    "Specialty metals":     "#C97B3A",
    "Rare earth elements":  "#8E3A62",
}
SCENARIO_COLOR = "#00693E"
INTENSITY_COLOR = "#C97B3A"

# Skip year 2026 (all scenarios = 0 additions by construction).
# For the default 3-yr-bucket CSV, POOLED_YEARS are the 8 NREL reporting
# years covering 2027-2050. For the PCHIP-annualized CSV, POOLED_YEARS
# expands to every year 2027-2050 (24 annual values).
POOLED_YEARS = [2029, 2032, 2035, 2038, 2041, 2044, 2047, 2050]
POOLED_YEARS_ANNUAL = list(range(2027, 2051))


SCENARIO_BOUNDS = {
    "p2.5-p97.5": {
        "scen_low":  lambda arr: float(np.quantile(arr, 0.025)),
        "scen_high": lambda arr: float(np.quantile(arr, 0.975)),
        "label":     "P2.5 / P97.5 across 61 NREL scenarios",
        "short":     "P2.5/P97.5",
        "note":      "(central 95% of scenario envelope; less biased tail estimator for n=61)",
    },
    "min-max": {
        "scen_low":  lambda arr: float(np.min(arr)),
        "scen_high": lambda arr: float(np.max(arr)),
        "label":     "min / max across 61 NREL scenarios",
        "short":     "min/max",
        "note":      "(full scenario envelope; scenarios are equally-plausible discrete futures, not a sample)",
    },
}

# Baseline selection: which row serves as the 0% reference for both bars.
#   median   — the scenario whose mean is the median of the 61 per-scenario means
#              (baseline tracks the ensemble center; varies by material/year)
#   mid_case — NREL's Mid_Case reference scenario (fixed scenario ID across cells)
#   no_ira   — Mid_Case_No_IRA policy-rollback counterfactual
BASELINES = {
    "median": {
        "label":  "MEDIAN-SCENARIO\nBASELINE",
        "short":  "median-scenario",
        "desc":   "baseline = scenario closest to the median of the 61 per-scenario means "
                  "(tracks ensemble center; identity varies by material/year)",
        "fixed_id": None,  # picked dynamically per cell
    },
    "mid_case": {
        "label":  "Mid Case\n(with IRA)",
        "short":  "Mid Case (with IRA)",
        "desc":   "baseline = NREL Mid_Case reference scenario (fixed scenario across all cells)",
        "fixed_id": "Mid_Case",
    },
    "no_ira": {
        "label":  "NO-IRA\nBASELINE",
        "short":  "Mid Case (no IRA)",
        "desc":   "baseline = Mid_Case_No_IRA policy-rollback counterfactual (fixed scenario)",
        "fixed_id": "Mid_Case_No_IRA",
    },
}


def _pick_baseline_row(g: pd.DataFrame, means: np.ndarray, baseline: str):
    """Return the row of `g` to serve as the baseline, or None if unavailable."""
    cfg = BASELINES[baseline]
    if cfg["fixed_id"] is None:
        median_mean = float(np.median(means))
        idx = int(np.argmin(np.abs(means - median_mean)))
        return g.iloc[idx]
    match = g[g["scenario"] == cfg["fixed_id"]]
    if match.empty:
        return None
    return match.iloc[0]


def per_cell_ranges(df: pd.DataFrame, years: list[int],
                    scenario_bounds: str = "p2.5-p97.5",
                    baseline: str = "median",
                    timebase: str = "pooled") -> pd.DataFrame:
    """For each material (or material × year) yield scenario & intensity ranges.

    scenario_bounds : "p2.5-p97.5" or "min-max" — how to bound the scenario
        envelope across the 61 per-scenario means. Intensity always uses
        p2.5/p97.5 (the CSV doesn't expose raw MC draws).
    baseline : "median", "mid_case", or "no_ira" — 0% reference for both bars.
    timebase : "pooled" (per-year ranges, one record per (material, year))
        or "cumulative" (sum across years first, one record per material).
        For cumulative: because intensity is sampled once per MC iteration
        and held across years (correlated-in-time by construction of the
        model), cumulative p2.5/p97.5 are approximated as the sum of the
        per-year p2.5/p97.5 values — the perfect-correlation bound.
    """
    if scenario_bounds not in SCENARIO_BOUNDS:
        raise ValueError(f"scenario_bounds must be one of {sorted(SCENARIO_BOUNDS)}")
    if baseline not in BASELINES:
        raise ValueError(f"baseline must be one of {sorted(BASELINES)}")
    if timebase not in ("pooled", "cumulative"):
        raise ValueError(f"timebase must be 'pooled' or 'cumulative'")
    bounds = SCENARIO_BOUNDS[scenario_bounds]
    sub = df[df.year.isin(years) & ~df.material.isin(EXCLUDED | ZERO_DEMAND)]
    records = []
    skipped = 0

    if timebase == "cumulative":
        # Collapse years: per (scenario, material) sum the bucket totals.
        agg_cols = {"mean": "sum", "p2": "sum", "p97": "sum"}
        cum = sub.groupby(["scenario", "material"], as_index=False).agg(agg_cols)
        for material, g in cum.groupby("material"):
            g = g[g["mean"] > 0]
            if len(g) < 5:
                continue
            means = g["mean"].to_numpy()
            base_row = _pick_baseline_row(g, means, baseline)
            if base_row is None:
                skipped += 1
                continue
            base_mean = float(base_row["mean"])
            if base_mean <= 0:
                skipped += 1
                continue
            scen_low = bounds["scen_low"](means)
            scen_high = bounds["scen_high"](means)
            records.append({
                "material": material,
                "year": "cumulative",
                "scen_p2_rel": 100.0 * (scen_low - base_mean) / base_mean,
                "scen_p97_rel": 100.0 * (scen_high - base_mean) / base_mean,
                "int_p2_rel": 100.0 * (float(base_row["p2"]) - base_mean) / base_mean,
                "int_p97_rel": 100.0 * (float(base_row["p97"]) - base_mean) / base_mean,
            })
    else:
        for (material, year), g in sub.groupby(["material", "year"]):
            g = g[g["mean"] > 0]
            if len(g) < 5:
                continue
            means = g["mean"].to_numpy()
            base_row = _pick_baseline_row(g, means, baseline)
            if base_row is None:
                skipped += 1
                continue
            base_mean = float(base_row["mean"])
            if base_mean <= 0:
                skipped += 1
                continue
            scen_low = bounds["scen_low"](means)
            scen_high = bounds["scen_high"](means)
            records.append({
                "material": material,
                "year": year,
                "scen_p2_rel": 100.0 * (scen_low - base_mean) / base_mean,
                "scen_p97_rel": 100.0 * (scen_high - base_mean) / base_mean,
                "int_p2_rel": 100.0 * (float(base_row["p2"]) - base_mean) / base_mean,
                "int_p97_rel": 100.0 * (float(base_row["p97"]) - base_mean) / base_mean,
            })
    if skipped:
        print(f"  note: skipped {skipped} cells with no valid baseline")
    return pd.DataFrame(records)


def class_aggregate(per_cell: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-(material, year) ranges into one tornado pair per material class.

    For each class in MATERIAL_GROUPS, equal-weight average the per-cell
    scenario and intensity relative ranges across all member materials and
    years. Adds span columns (P97.5 - P2.5) for each source and the
    intensity/scenario span ratio used to sort the classes top-to-bottom.
    """
    rows = []
    for cat, members in MATERIAL_GROUPS.items():
        sub = per_cell[per_cell.material.isin(members)]
        if sub.empty:
            continue
        rows.append({
            "category": cat,
            "n_materials": int(sub.material.nunique()),
            "n_cells": int(len(sub)),
            "scen_p2": float(sub["scen_p2_rel"].mean()),
            "scen_p97": float(sub["scen_p97_rel"].mean()),
            "int_p2": float(sub["int_p2_rel"].mean()),
            "int_p97": float(sub["int_p97_rel"].mean()),
        })
    out = pd.DataFrame(rows)
    out["scen_span"] = out["scen_p97"] - out["scen_p2"]
    out["int_span"] = out["int_p97"] - out["int_p2"]
    out["ratio"] = out["int_span"] / out["scen_span"]
    return out


def plot(input_csv: Path, output_path: Path,
         scenario_bounds: str = "p2.5-p97.5",
         baseline: str = "median",
         compact: bool = False,
         timebase: str = "pooled",
         annual_grid: bool = False,
         docx: bool = False) -> None:
    """Build and save the manuscript Figure 4 sensitivity tornado.

    Reads the per-scenario demand CSV, computes per-cell scenario and
    intensity ranges, aggregates them by material class, and draws the
    paired-bar tornado (scenario bar above, intensity bar below each class).
    Saves a 300 DPI PNG to output_path and prints the per-class summary
    table. Flags control the bounds, baseline, time pooling, and layout;
    see main() / the argparse help for each option.
    """
    df = pd.read_csv(input_csv)
    years = POOLED_YEARS_ANNUAL if annual_grid else POOLED_YEARS
    per_cell = per_cell_ranges(df, years,
                               scenario_bounds=scenario_bounds,
                               baseline=baseline,
                               timebase=timebase)
    bounds_cfg = SCENARIO_BOUNDS[scenario_bounds]
    baseline_cfg = BASELINES[baseline]
    if per_cell.empty:
        raise SystemExit("No data after filtering")
    agg = class_aggregate(per_cell)
    # Sort classes by ratio ascending → scenario-dominated on top,
    # intensity-dominated on bottom (regime gradient)
    agg = agg.sort_values("ratio", ascending=True).reset_index(drop=True)

    n = len(agg)
    if compact:
        # Slightly more square than the original 12.0 x ~6.9 (ratio 1.74)
        # — now 10.5 x ~7.3 (ratio ~1.44)
        _compact_w, _compact_h = 10.5, 1.15 * n + 2.7
        if docx:
            # Render at print width so on-page pt == the absolute pt set below
            # (the figure is saved with bbox_inches="tight"; rendering at print
            # size is the only way to control on-page text size). Aspect kept.
            _docx_w = 6.0
            _docx_h = _docx_w * (_compact_h / _compact_w)
            fig = plt.figure(figsize=(_docx_w, _docx_h), dpi=300)
        else:
            fig = plt.figure(figsize=(_compact_w, _compact_h), dpi=300)
        if docx:
            # Manuscript layout: headline removed (more top room) and the
            # methods subtitle moved below the x-axis label (more bottom room).
            gs = fig.add_gridspec(1, 1, left=0.17, right=0.97, top=0.90, bottom=0.22)
        else:
            gs = fig.add_gridspec(1, 1, left=0.17, right=0.97, top=0.85, bottom=0.13)
    else:
        fig = plt.figure(figsize=(13.0, 1.55 * n + 3.6), dpi=300)
        gs = fig.add_gridspec(1, 1, left=0.13, right=0.94, top=0.78, bottom=0.15)
    ax = fig.add_subplot(gs[0, 0])

    # For each class we draw two bars: scenario above, intensity below.
    # y-position: class center at integer i; bars at i+0.22 (scen) and i-0.22 (int)
    bar_h = 0.34

    # Inverted y-axis so top of page = first sorted class (scenario-dominated)
    y_centers = np.arange(n)[::-1]

    xmax = 0
    for yi, (_, r) in zip(y_centers, agg.iterrows()):
        # Scenario bar (upper)
        ax.barh(yi + 0.22, r["scen_p97"] - r["scen_p2"],
                left=r["scen_p2"], height=bar_h,
                color=SCENARIO_COLOR, edgecolor="white", linewidth=0.8, zorder=3)
        # Intensity bar (lower)
        ax.barh(yi - 0.22, r["int_p97"] - r["int_p2"],
                left=r["int_p2"], height=bar_h,
                color=INTENSITY_COLOR, edgecolor="white", linewidth=0.8, zorder=3)

        # Bar endpoint labels (P2.5 and P97.5)
        ax.text(r["scen_p2"] - 5, yi + 0.22, f"{r['scen_p2']:+.0f}%",
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=SCENARIO_COLOR, gid="tornado_value")
        ax.text(r["scen_p97"] + 5, yi + 0.22, f"{r['scen_p97']:+.0f}%",
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=SCENARIO_COLOR, gid="tornado_value")
        ax.text(r["int_p2"] - 5, yi - 0.22, f"{r['int_p2']:+.0f}%",
                ha="right", va="center", fontsize=9, fontweight="bold",
                color=INTENSITY_COLOR, gid="tornado_value")
        ax.text(r["int_p97"] + 5, yi - 0.22, f"{r['int_p97']:+.0f}%",
                ha="left", va="center", fontsize=9, fontweight="bold",
                color=INTENSITY_COLOR, gid="tornado_value")

        xmax = max(xmax, r["scen_p97"], r["int_p97"])

    # Baseline line at 0%
    ax.axvline(0, color="#2E2A26", linewidth=1.3, alpha=0.9, zorder=4)

    # Right-margin regime label (suppressed in compact mode)
    if not compact:
        for yi, (_, r) in zip(y_centers, agg.iterrows()):
            ratio = r["ratio"]
            color = INTENSITY_COLOR if ratio >= 1 else SCENARIO_COLOR
            label = (f"I/S = {ratio:.2f}×\n"
                     f"{'intensity-dominated' if ratio >= 1 else 'scenario-dominated'}")
            ax.text(1.02, yi, label,
                    transform=ax.get_yaxis_transform(),
                    ha="left", va="center", fontsize=10, fontweight="bold",
                    color=color, linespacing=1.3)

    # Class labels and source sub-labels
    ax.set_yticks(y_centers)
    ax.set_yticklabels(agg["category"].tolist(), fontsize=13, fontweight="bold")
    for tick, cat in zip(ax.get_yticklabels(), agg["category"]):
        tick.set_color(CATEGORY_COLORS[cat])

    # Inline source labels inside bars where space allows
    for yi, (_, r) in zip(y_centers, agg.iterrows()):
        scen_w = r["scen_p97"] - r["scen_p2"]
        int_w = r["int_p97"] - r["int_p2"]
        if scen_w > 80:
            ax.text((r["scen_p2"] + r["scen_p97"]) / 2, yi + 0.22,
                    "Scenario", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=5,
                    gid="tornado_category")
        if int_w > 80:
            ax.text((r["int_p2"] + r["int_p97"]) / 2, yi - 0.22,
                    "Intensity", ha="center", va="center",
                    fontsize=10, fontweight="bold", color="white", zorder=5,
                    gid="tornado_category")

    # Axis styling — fixed x range so P2.5/P97.5 and min-max variants
    # render on the same scale for visual comparison.
    ax.set_xlim(-125 if docx else -115, 300)
    ax.set_ylim(-0.7, n - 0.3)
    xlabel_prefix = (
        "Cumulative demand change vs"
        if timebase == "cumulative"
        else "Annual demand change vs"
    )
    xlabel_suffix = (
        "(%, 2027–2050 cumulative)"
        if timebase == "cumulative"
        else "(%, 2027–2050 average)"
    )
    ax.set_xlabel(
        f"{xlabel_prefix} {baseline_cfg['short']} {xlabel_suffix}",
        fontsize=12, labelpad=8, color="#444444",
    )
    tick_vals = np.arange(-100, 301, 100)
    ax.set_xticks(tick_vals)
    ax.set_xticklabels([f"{int(t):+d}%" if t != 0 else "0%" for t in tick_vals],
                       fontsize=10)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5, zorder=0)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_color("#888888")
    ax.spines["bottom"].set_color("#888888")

    # Baseline label (MID_CASE / median-scenario anchor)
    ax.text(0, n - 0.15, baseline_cfg["label"],
            ha="center", va="bottom",
            fontsize=9, color="#555555", fontweight="bold", linespacing=1.1,
            gid="tornado_anchor")

    if compact:
        _headline = fig.text(
            0.14, 0.945,
            "Demand sensitivity to scenario choice vs material intensity",
            ha="left", va="center",
            fontsize=16, fontweight="bold", color="#2E2A26",
        )
        # Methods subtitle: what the bars actually represent
        bounds_phrase = (
            "min–max across 61 NREL scenarios"
            if scenario_bounds == "min-max"
            else "95% CI (P2.5–P97.5) across 61 NREL scenarios"
        )
        _subtitle = fig.text(
            0.14, 0.905,
            f"Intensity: 95% CI (p2.5–p97.5)   ·   Scenario: {bounds_phrase}",
            ha="left", va="center",
            fontsize=10.5, color="#666666", style="italic",
        )
    if not compact:
        # Title block
        fig.text(
            0.13, 0.945,
            "Traditional sensitivity tornado: which uncertainty source swings demand more?",
            ha="left", va="center",
            fontsize=17, fontweight="bold", color="#2E2A26",
        )
        fig.text(
            0.13, 0.908,
            "Each bar = P2.5-to-P97.5 range of demand attributable to one uncertainty source, "
            "pooled across 2029–2050 and equal-weighted across class members.",
            ha="left", va="center",
            fontsize=11.5, color="#444444", style="italic",
        )

        # Legend
        fig.patches.append(
            plt.Rectangle((0.13, 0.855), 0.022, 0.012,
                          facecolor=SCENARIO_COLOR, edgecolor="none",
                          transform=fig.transFigure, figure=fig)
        )
        fig.text(0.158, 0.862,
                 "SCENARIO CHOICE  ", fontsize=11, fontweight="bold",
                 color=SCENARIO_COLOR, va="center")
        fig.text(0.158, 0.842,
                 f"upper bar · {bounds_cfg['short']} of 61 per-scenario means",
                 fontsize=9.5, color="#666666", style="italic", va="center")

        fig.patches.append(
            plt.Rectangle((0.50, 0.855), 0.022, 0.012,
                          facecolor=INTENSITY_COLOR, edgecolor="none",
                          transform=fig.transFigure, figure=fig)
        )
        fig.text(0.528, 0.862,
                 "INTENSITY UNCERTAINTY  ", fontsize=11, fontweight="bold",
                 color=INTENSITY_COLOR, va="center")
        fig.text(0.528, 0.842,
                 "lower bar · p2.5/p97.5 of 10,000 MC draws, median scenario",
                 fontsize=9.5, color="#666666", style="italic", va="center")

        # Methods footer
        fig.text(
            0.01, 0.025,
            f"Baseline: {baseline_cfg['desc']}.  "
            f"Scenario bounds: {bounds_cfg['label']} {bounds_cfg['note']}.  "
            f"Each range is computed per (material, year), expressed as percent of the {baseline_cfg['short']} mean, "
            "and equal-weighted across class members and 8 reporting years (2029-2050; 2026 excluded).  "
            "I/S = intensity span ÷ scenario span.  "
            "After Deutsch et al. (U. Alberta CCG) Fig 4.4.",
            fontsize=8, color="#666666",
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if docx:
        # Correct docx approach: the figure is already rendered at print width
        # (6.0 in) above, so setting in-plot text to ABSOLUTE point sizes here
        # makes on-page pt == these targets when the saved PNG is inserted at
        # its saved width in Word. (bbox_inches="tight" trims margins but does
        # not rescale text, so this is the only way to fix on-page size.)
        SUP, AXTITLE, AXLABEL, TICK, ANNOT, LEG = 12, 12, 10, 8.5, 8.5, 9
        ANCHOR_PT, CATEGORY_PT, VALUE_PT = 8.5, 9, 8.5
        if getattr(fig, "_suptitle", None):
            fig._suptitle.set_fontsize(SUP)
        for _ax in fig.get_axes():
            if _ax.get_title():
                _ax.title.set_fontsize(AXTITLE)
            _ax.xaxis.label.set_fontsize(AXLABEL)
            _ax.yaxis.label.set_fontsize(AXLABEL)
            for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                _t.set_fontsize(TICK)
            for _t in _ax.texts:
                # Size axes-level annotations by role (gid set at creation).
                _role = _t.get_gid()
                if _role == "tornado_anchor":
                    _t.set_fontsize(ANCHOR_PT)
                elif _role == "tornado_category":
                    _t.set_fontsize(CATEGORY_PT)
                elif _role == "tornado_value":
                    _t.set_fontsize(VALUE_PT)
                else:
                    _t.set_fontsize(ANNOT)
            _lg = _ax.get_legend()
            if _lg:
                for _x in _lg.get_texts():
                    _x.set_fontsize(LEG)
        for _lg in fig.legends:
            for _x in _lg.get_texts():
                _x.set_fontsize(LEG)
        # Manuscript layout: remove the headline entirely and move the methods
        # subtitle to BELOW the x-axis label, centered.
        if compact:
            _headline.set_visible(False)
            _subtitle.set_fontsize(9)
            _subtitle.set_position((0.5, 0.035))
            _subtitle.set_ha("center")
            _subtitle.set_va("center")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Wrote {output_path}")
    print()
    print(f"{'Class':<22s}  {'scen P2.5':>10s} {'scen P97.5':>10s} {'int P2.5':>10s} {'int P97.5':>10s} {'ratio':>7s}")
    for _, r in agg.iterrows():
        print(f"  {r['category']:<20s}  "
              f"{r['scen_p2']:>+9.1f}% {r['scen_p97']:>+9.1f}% "
              f"{r['int_p2']:>+9.1f}% {r['int_p97']:>+9.1f}% "
              f"{r['ratio']:>6.2f}×")


def main() -> None:
    """Parse command-line arguments and render Figure 4.

    With no flags, reads the default demand CSV and writes
    outputs/figures/manuscript/fig4_sensitivity_tornado.png. When --output
    is unset, non-default variant flags are appended to the filename so the
    canonical figure is never overwritten by an alternative variant.
    """
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    ap.add_argument("--output", type=Path, default=None,
                    help="If unset, output filename includes the variant tags.")
    ap.add_argument("--scenario-bounds", choices=list(SCENARIO_BOUNDS),
                    default="p2.5-p97.5",
                    help="How to bound the scenario envelope (default: p2.5-p97.5).")
    ap.add_argument("--baseline", choices=list(BASELINES),
                    default="median",
                    help="Which scenario serves as the 0%% baseline (default: median).")
    ap.add_argument("--compact", action="store_true",
                    help="Condensed canvas, no title/subtitle/legend/footer.")
    ap.add_argument("--timebase", choices=("pooled", "cumulative"),
                    default="pooled",
                    help="'pooled' averages per-year ranges; 'cumulative' sums demand "
                         "across 2027-2050 first, then computes one range per material.")
    ap.add_argument("--pchip", action="store_true",
                    help="Pool over the full 24-year annual grid (2027-2050) of "
                         "the PCHIP-annualized demand CSV, instead of the default "
                         "8 NREL reporting years. Both read the same shipped CSV "
                         "(outputs/data/material_demand_by_scenario.csv); the flag "
                         "only changes which years are pooled.")
    ap.add_argument("--docx", action="store_true",
                    help="Render at print width (6.0 in) with absolute point "
                         "sizes so in-plot text is legible when the figure is "
                         "inserted into a Word docx at its saved width. "
                         "Does not change data, series, or colors.")
    args = ap.parse_args()
    if args.pchip and args.input == DEFAULT_INPUT:
        args.input = PCHIP_INPUT
    if args.output is None:
        tags = []
        if args.baseline != "median":
            tags.append(args.baseline)
        if args.scenario_bounds != "p2.5-p97.5":
            tags.append(args.scenario_bounds)
        if args.pchip:
            tags.append("pchip")
        if args.timebase != "pooled":
            tags.append(args.timebase)
        if args.compact:
            tags.append("compact")
        suffix = ("_" + "_".join(tags)) if tags else ""
        args.output = DEFAULT_OUTPUT.with_name(
            DEFAULT_OUTPUT.stem + suffix + DEFAULT_OUTPUT.suffix
        )
    plot(args.input, args.output,
         scenario_bounds=args.scenario_bounds,
         baseline=args.baseline,
         compact=args.compact,
         timebase=args.timebase,
         annual_grid=args.pchip,
         docx=args.docx)


if __name__ == "__main__":
    main()
