"""Manuscript Figure 3: cumulative demand by material family.

2x2 panel line chart of cumulative material demand over 2026-2050 by
scenario in absolute tonnage, one panel per material family on its own
auto-scaled axis. Reads the PCHIP-annualized
outputs/data/material_demand_by_scenario.csv; excludes Fiberglass and
Glass (no criticality source).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Repo-relative paths (this file lives in visualizations/, so the repository
# root is two levels up). Reads the same canonical PCHIP-annualized demand CSV
# every other figure uses, and writes into the manuscript figure directory.
CODEBASE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(CODEBASE_ROOT))
from src.scenario_config import SCENARIO_LABELS, SCENARIO_COLORS  # noqa: E402
from src.material_classes import CLASS_MEMBERS, CLASS_LABELS, CLASS_ORDER  # noqa: E402
DATA_PATH = (
    CODEBASE_ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
)
OUT_DIR = CODEBASE_ROOT / "outputs" / "figures" / "manuscript"

EXCL: set = set()  # 2026-07-28: Glass/Fiberglass now included in Bulk (Option 2)

# Panel title -> members, from the canonical taxonomy (Option 2 placements).
FAMILIES = {CLASS_LABELS[c]: list(CLASS_MEMBERS[c]) for c in CLASS_ORDER}

# Canonical scenario palette, shared with Fig 1 (capacity additions) and Fig 2
# (annual demand): NZ2035 red, NZ2050 purple, Mid-IRA blue,
# no-IRA orange. All lines solid (2026-07-28 revision: the no-IRA dash is out,
# matching Fig 2's all-solid treatment; color alone separates the curves).
# Draw order (baseline over no-IRA, net-zero curves on top); (linestyle,
# linewidth) per scenario, all solid per the 2026-07-28 revision. Keys,
# labels, and colors come from src/scenario_config.py.
_DRAW_STYLE = [
    ("Mid_Case_No_IRA",    "-", 1.8),
    ("Mid_Case",           "-", 2.2),
    ("Mid_Case_CO2e_95by2050",  "-", 2.0),
    ("Mid_Case_CO2e_100by2035", "-", 2.0),
]
SCENARIOS = [(k, SCENARIO_LABELS[k], SCENARIO_COLORS[k], ls, lw)
             for k, ls, lw in _DRAW_STYLE]

# Legend entry order, matching Fig 2's legend top-to-bottom (2026-07-28
# revision: consistency across the figure set).
LEGEND_ORDER = ["Net-Zero by 2035", "Net-Zero by 2050",
                "Baseline with IRA", "Baseline without IRA"]


def pick_unit(max_tonnes: float) -> tuple[float, str]:
    """Choose a per-panel display unit from the panel's largest value.

    Returns (divisor, label) so each family's y-axis is readable on its
    own scale: megatonnes (Mt) for bulk commodities, kilotonnes (kt) for
    specialty metals, and tonnes for the smallest (REE) families.
    """
    if max_tonnes >= 1e6:
        return 1e6, "Mt"
    if max_tonnes >= 1e3:
        return 1e3, "kt"
    return 1.0, "tonnes"


def compute_family_trajectories() -> dict:
    """Build per-family, per-scenario cumulative-tonnage trajectories.

    Loads the annualized demand CSV, drops the uncriticized materials
    (Fiberglass, Glass) and any scenario not in SCENARIOS, then for each
    family sums the annual demand of its member materials across scenario
    and year and takes the running cumulative sum per scenario. Returns a
    dict mapping family name to a DataFrame with columns
    [scenario, year, mean_annual, cum]. mean_annual is the per-year
    family total; cum is the cumulative-to-date family total.
    """
    df = pd.read_csv(DATA_PATH)
    df = df[~df.material.isin(EXCL)]
    keep = {s for s, _, _, _, _ in SCENARIOS}
    df = df[df.scenario.isin(keep)]

    out = {}
    for fam, mats in FAMILIES.items():
        sub = df[df.material.isin(mats)]
        family_annual = (
            sub.groupby(["scenario", "year"])["mean_annual"]
            .sum()
            .reset_index()
            .sort_values(["scenario", "year"])
        )
        family_annual["cum"] = family_annual.groupby("scenario")["mean_annual"].cumsum()
        out[fam] = family_annual
    return out


def main(docx: bool = False) -> None:
    """Render the 2x2 figure and print a per-family endpoint table.

    Draws one panel per material family (Bulk, Base & alloying,
    Specialty, REE), one line per scenario, with a shared bottom-center
    scenario legend. Saves PNG (300 dpi) and PDF into the manuscript
    figure directory and echoes a small per-family pivot table to stdout
    for a sanity check. Pass docx=True to scale all text up for full-page
    print insertion; the default output is unchanged.
    """
    data = compute_family_trajectories()

    # Keep the original large canvas in BOTH modes; --docx only scales the text
    # up by F (below) for legible full-page insertion, without respacing panels.
    F = 1.4 if docx else 1.0
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), dpi=150, sharex=True)
    axes_flat = axes.ravel()

    for panel_idx, (ax, (fam, traj)) in enumerate(zip(axes_flat, data.items())):
        max_cum = traj["cum"].max()
        divisor, unit = pick_unit(max_cum)

        endpoints = []
        for scen, label, color, ls, lw in SCENARIOS:
            s = traj[traj.scenario == scen].sort_values("year").copy()
            s["y"] = s["cum"] / divisor
            # No point markers on the lines (2026-07-23 review, Fig 3 comment).
            ax.plot(
                s.year, s.y,
                color=color, linestyle=ls, linewidth=lw, label=label,
            )
            end = s.iloc[-1]
            endpoints.append((float(end.y), label, color))

        y_max = max(e[0] for e in endpoints)

        ax.set_title(fam, fontsize=11.5, fontweight="bold", pad=4)
        # Quantity is the mean across MC iterations; unit stays per-panel
        # (Mt/kt/tonnes) because a single kt axis would force 6-digit bulk
        # values and sub-unit REE values (deviation from the literal "(kt)"
        # in the 2026-07-23 review comment).
        ax.set_ylabel(f"Average cumulative demand ({unit})", fontsize=10)
        ax.set_xticks(range(2030, 2051, 5))   # every 5 years: 2030, 2035, 2040, 2045, 2050
        # No right-margin reserve anymore: it existed only for the endpoint
        # labels, which the legend replaces (2026-07-28).
        ax.set_xlim(2025.5, 2051)
        ax.set_ylim(top=y_max * (1.12 if docx else 1.08))
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        # Review item C18: keep the x-axis line from extending past 2050 (the
        # endpoint labels still sit in the reserved right margin), and drop the
        # x/y tick marks while keeping the tick labels.
        ax.spines["bottom"].set_bounds(2026, 2050)
        ax.tick_params(axis="both", length=0)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    # Review item C18: no "Year" axis titles (the caption states the span).
    for ax in axes[-1, :]:
        ax.set_xlabel("")

    # Bottom-center figure legend, borderless, one row (2026-07-28 revision:
    # replaces the upper-right in-panel legend for consistency with Fig 2's
    # bottom legend; entry order mirrors Fig 2 top-to-bottom). tight_layout's
    # rect reserves the bottom strip so the legend never overlaps the year
    # tick labels at either text scale.
    handle_by_label = dict(zip(
        [lbl for _, lbl, _, _, _ in SCENARIOS], axes_flat[0].get_lines()))
    fig.legend(
        [handle_by_label[lbl] for lbl in LEGEND_ORDER], LEGEND_ORDER,
        loc="lower center", bbox_to_anchor=(0.5, 0.0), ncol=4,
        frameon=False, fontsize=10 * F, handlelength=1.8,
        columnspacing=1.6,
    )

    # No figure title (the caption carries it).
    fig.tight_layout(rect=[0, 0.05 * F, 1, 0.99])

    if docx:
        # Keep the original canvas (above) and scale ALL text up by F for
        # legible full-page insertion, without respacing the panels. Typography
        # only; no data, series, or colors touched. Default (no --docx) output
        # is byte-for-byte unchanged.
        def _bump(t):
            if t is not None and t.get_fontsize():
                t.set_fontsize(t.get_fontsize() * F)

        _bump(getattr(fig, "_suptitle", None))
        for _ax in fig.get_axes():
            _bump(_ax.title)
            _bump(_ax.xaxis.label)
            _bump(_ax.yaxis.label)
            for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                _bump(_t)
            # Legend text is NOT bumped here: the figure-level legend is
            # already created at 10 * F, and its entries don't live in
            # _ax.texts anyway.
            for _t in _ax.texts:
                _bump(_t)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png = OUT_DIR / "fig3_cumulative_by_family.png"
    out_pdf = OUT_DIR / "fig3_cumulative_by_family.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote: {out_png}")
    print(f"Wrote: {out_pdf}")

    show_years = [2026, 2030, 2035, 2040, 2045, 2050]
    scen_cols = [s for s, _, _, _, _ in SCENARIOS]
    for fam, traj in data.items():
        max_cum = traj["cum"].max()
        divisor, unit = pick_unit(max_cum)
        print(f"\n=== {fam}: cumulative tonnage ({unit}) ===")
        disp = traj.copy()
        disp["disp"] = disp["cum"] / divisor
        pivot = disp.pivot(index="year", columns="scenario", values="disp")
        print(pivot.loc[show_years, scen_cols].round(2).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--docx", action="store_true", default=False)
    args = parser.parse_args()
    main(docx=args.docx)
