"""Manuscript Figure 3: cumulative demand by material family.

2x2 panel line chart of cumulative material demand over 2026-2050 by
scenario in absolute tonnage, one panel per material family on its own
auto-scaled axis. Reads the PCHIP-annualized
outputs/data/material_demand_by_scenario.csv; excludes Fiberglass and
Glass (no criticality source).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Repo-relative paths (this file lives in visualizations/, so the repository
# root is two levels up). Reads the same canonical PCHIP-annualized demand CSV
# every other figure uses, and writes into the manuscript figure directory.
CODEBASE_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = (
    CODEBASE_ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
)
OUT_DIR = CODEBASE_ROOT / "outputs" / "figures" / "manuscript"

EXCL = {"Fiberglass", "Glass"}

FAMILIES = {
    "Bulk commodities":     ["Cement", "Steel", "Aluminum", "Copper"],
    "Base & alloying":      ["Zinc", "Lead", "Nickel", "Tin", "Manganese",
                             "Chromium", "Molybdenum", "Vanadium", "Silicon",
                             "Magnesium", "Boron"],
    "Specialty metals":     ["Tellurium", "Indium", "Cadmium", "Silver", "Niobium"],
    "Rare earth elements":  ["Dysprosium", "Neodymium", "Praseodymium",
                             "Terbium", "Yttrium"],
}

SCENARIOS = [
    ("Mid_Case_No_IRA",                   "Mid Case (no IRA)",          "#888888", "--", 1.8),
    ("Mid_Case",                          "Mid Case (with IRA)",        "#333333", "-",  2.2),
    ("Mid_Case_95by2050",                 "Net-Zero by 2050",           "#59A14F", "-",  2.0),
    ("Mid_Case_100by2035",                "Net-Zero by 2035",           "#F28E2B", "-",  2.0),
]


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
    Specialty, REE), one line per scenario, with leader-line endpoint
    labels at 2050. Saves PNG (300 dpi) and PDF into the manuscript
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

    for ax, (fam, traj) in zip(axes_flat, data.items()):
        max_cum = traj["cum"].max()
        divisor, unit = pick_unit(max_cum)

        endpoints = []
        for scen, label, color, ls, lw in SCENARIOS:
            s = traj[traj.scenario == scen].sort_values("year").copy()
            s["y"] = s["cum"] / divisor
            ax.plot(
                s.year, s.y,
                color=color, linestyle=ls, linewidth=lw,
                marker="o", markersize=3, label=label,
            )
            end = s.iloc[-1]
            endpoints.append((float(end.y), label, color))

        endpoints.sort(key=lambda t: t[0])
        y_max = max(e[0] for e in endpoints)
        # At print width (--docx) each panel is physically much shorter, so the
        # four endpoint labels need a larger vertical de-collision step; the
        # reserved right margin (tight_layout rect below) gives the labels room
        # so they never overlap each other or clip.
        # Scale the endpoint-label de-collision step with the label font (F) so
        # the four labels never overlap once the text is scaled up.
        y_min_step = y_max * 0.055 * F
        label_x = 2050.6
        last_y = -np.inf
        for y_val, label, color in endpoints:
            y_text = max(y_val, last_y + y_min_step)
            ax.annotate(
                label,
                xy=(2050, y_val),
                xytext=(label_x, y_text),
                textcoords="data",
                fontsize=8.5, color=color, va="center", fontweight="bold",
                arrowprops=dict(arrowstyle="-", color=color, lw=0.6, alpha=0.6)
                if abs(y_text - y_val) > y_min_step * 0.3 else None,
            )
            last_y = y_text

        ax.set_title(fam, fontsize=11.5, fontweight="bold", pad=4)
        ax.set_ylabel(f"Cumulative tonnage ({unit})", fontsize=10)
        ax.set_xticks(range(2030, 2051, 5))   # every 5 years: 2030, 2035, 2040, 2045, 2050
        # Under --docx the endpoint labels are larger; widen the x-limit so they
        # sit inside the panel's own right margin instead of spilling into the
        # gutter and overlapping the next column's y-axis label.
        ax.set_xlim(2025.5, 2064 if docx else 2061)
        ax.set_ylim(top=y_max * (1.24 if docx else 1.18))
        ax.set_ylim(bottom=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    for ax in axes[-1, :]:
        ax.set_xlabel("Year", fontsize=10.5)

    fig.suptitle(
        "Cumulative material demand over time, by family and scenario (absolute tonnage)",
        fontsize=12, y=1.0, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

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
            for _t in _ax.texts:        # includes the endpoint scenario labels
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
