"""
SI Figure S2 (per-material sensitivity) + two diagnostics figures
(S1 demand-by-technology is figS1_demand_by_technology.py; S3 tornado is
figS3_sensitivity_tornado.py).

INVENTORY:
  name: si_figures
  output: outputs/figures/manuscript/figS2_per_material_sensitivity.{png,pdf}
          outputs/figures/diagnostics/diag_mc_convergence.{png,pdf}
          outputs/figures/diagnostics/diag_interpolation_robustness.{png,pdf}
  category: Manuscript (SI) + diagnostics
  data_sources:
    - material_demand_by_scenario.csv (canonical = PCHIP annual, published MC)
    - fitted_distributions.csv (lognormal params, for the convergence proxy)
    - outputs/data/interpolated/mean_preserving/material_demand_by_scenario.csv
      (produced by: python simulate.py --interpolation mean-preserving)
  description: >
    S2  Per-material sensitivity: scenario min-max range (light) vs the 95%
        intensity interval (solid), % deviation from Baseline with IRA, averaged
        2027-2050, by material and class. Class colors use the manuscript
        palette (REE navy, 2026-07-13 convention).
    CONV  (diagnostic, not in SI) Monte Carlo convergence for representative
        materials (seed 24601): running mean settling into the 1% band.
    INTERP  (diagnostic, not in SI) Interpolation robustness: PCHIP vs
        mean-preserving cumulative demand (max deviation under 1%).
END_INVENTORY

Usage:
    python visualizations/si_figures.py
    python visualizations/si_figures.py --output-dir outputs/figures/manuscript
    python visualizations/si_figures.py --only S2      # SI figure only
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "outputs" / "data"
DEMAND_FILE = DATA / "material_demand_by_scenario.csv"          # canonical = PCHIP
FIT_FILE = DATA / "fitted_distributions.csv"
MP_FILE = DATA / "interpolated" / "mean_preserving" / "material_demand_by_scenario.csv"

from src.scenario_config import REFERENCE_SCENARIO  # noqa: E402  (Baseline with IRA)
from src.material_classes import (  # noqa: E402
    CLASS_COLORS as CLASS_COLOR, CLASS_ORDER, CLASS_MEMBERS, CLASS_LABELS)
CLASS_OF = {m: c for c, ms in CLASS_MEMBERS.items() for m in ms}

# House figure style (2026-07-28): sans-serif to match every other generator.
# The serif setting was inherited from thesis/sections/gen_appendix_figures.py
# when this figure was ported in, and made Figure S2 the only serif figure in
# the paper. Grid matches the dotted, low-alpha convention used by Figure 3.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9, "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
    "axes.grid": True, "grid.color": "#B0B0B0", "grid.linestyle": ":",
    "grid.linewidth": 0.6, "grid.alpha": 0.35,
    "savefig.dpi": 300, "figure.dpi": 150, "pdf.fonttype": 42, "ps.fonttype": 42,
})


def _save(fig, out_dir: Path, stem: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white")
        print(f"  wrote {p}")
    plt.close(fig)


def fig_s2(mds: pd.DataFrame, out_dir: Path):
    """Per-material sensitivity: scenario range (light) vs 95% intensity interval."""
    years = sorted(mds["year"].unique())
    win = [y for y in years if y >= 2027]
    rows = []
    for m in mds.material.unique():
        if m not in CLASS_OF:
            continue
        mm = mds[(mds.material == m) & (mds.year.isin(win))]
        midm = mm[mm.scenario == REFERENCE_SCENARIO].set_index("year")["mean"]
        if midm.sum() <= 0:
            continue
        mp = mm[mm.scenario == REFERENCE_SCENARIO].set_index("year")
        lo_i = ((mp["p2_5"] - midm) / midm).mean() * 100
        hi_i = ((mp["p97_5"] - midm) / midm).mean() * 100
        piv = mm.pivot_table(index="year", columns="scenario", values="mean")
        lo_s = ((piv.min(axis=1) - midm) / midm).mean() * 100
        hi_s = ((piv.max(axis=1) - midm) / midm).mean() * 100
        rows.append((m, CLASS_OF[m], lo_i, hi_i, lo_s, hi_s))
    rows.sort(key=lambda r: (CLASS_ORDER.index(r[1]), -(r[5] - r[4])))
    labels = [r[0] for r in rows]
    y = np.arange(len(rows))
    # Layout (2026-07-28): the legend used to sit inside the axes at lower
    # right, where it covered the last two materials; it now sits below the
    # x-label in a single row. Row pitch widened slightly, explicit y-limits
    # remove the dead band above the first material, and an x-margin keeps the
    # longest bar (terbium) off the right spine.
    ROW_H = 0.34                       # inches of canvas per material
    fig, ax = plt.subplots(figsize=(7.2, ROW_H * len(rows) + 1.15))
    for i, (m, cls, lo_i, hi_i, lo_s, hi_s) in enumerate(rows):
        c = CLASS_COLOR[cls]
        ax.barh(i + 0.20, hi_s - lo_s, left=lo_s, height=0.34, color=c,
                alpha=0.45, edgecolor="black", linewidth=0.3)
        ax.barh(i - 0.20, hi_i - lo_i, left=lo_i, height=0.34, color=c,
                alpha=0.95, edgecolor="black", linewidth=0.3)
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7.5)
    ax.set_ylim(len(rows) - 0.45, -0.55)      # inverts and trims the padding
    ax.margins(x=0.03)
    ax.grid(axis="x")                          # value axis only, like Figure 3
    ax.grid(axis="y", visible=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel("Demand deviation from Baseline with IRA (%), averaged 2027-2050",
                  labelpad=8)
    # No in-figure title: the docx caption carries it, as in every other figure.
    handles = [plt.Rectangle((0, 0), 1, 1, color=CLASS_COLOR[c]) for c in CLASS_ORDER]
    ax.legend(handles, [CLASS_LABELS[c] for c in CLASS_ORDER], fontsize=8,
              frameon=False, ncol=4, loc="upper center",
              bbox_to_anchor=(0.5, -0.045 - 24.0 / (ROW_H * len(rows) * 72)),
              handlelength=1.6, columnspacing=1.6, borderpad=0.0)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "figS2_per_material_sensitivity")


def diag_mc_convergence(out_dir: Path):
    """Monte Carlo convergence proxy for representative materials (seed 24601)."""
    fd = pd.read_csv(FIT_FILE)
    reps = ["Steel", "Copper", "Aluminum", "Neodymium"]
    rng = np.random.default_rng(24601)
    N = 10000
    it = np.arange(1, N + 1)
    fig, ax = plt.subplots(figsize=(6.6, 3.6))
    ax.axhspan(0.99, 1.01, color="#B00020", alpha=0.10, zorder=0)
    ax.axhline(1.0, color="#999999", lw=0.7, zorder=1)
    for m in reps:
        sub = fd[fd.material == m]
        if not len(sub):
            continue
        draws = np.zeros(N)
        for _, r in sub.iterrows():
            draws += stats.lognorm.rvs(float(r.param_s), loc=0,
                                       scale=float(r.param_scale), size=N,
                                       random_state=rng)
        draws /= len(sub)
        run = np.cumsum(draws) / it
        ax.plot(it, run / run[-1], lw=1.1, label=f"{m} ({len(sub)} pairs)")
    ax.text(N, 1.012, "1% tolerance band", color="#B00020", fontsize=7.5,
            ha="right", va="bottom")
    ax.set_xlim(200, N); ax.set_ylim(0.95, 1.05)
    ax.set_xlabel("Monte Carlo iterations")
    ax.set_ylabel("Running mean / final estimate")
    ax.set_title("Monte Carlo convergence (representative materials, seed 24601)",
                 fontsize=9.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.set_axisbelow(True)
    _save(fig, out_dir, "diag_mc_convergence")


# Compact cumulative comparison cached here (tracked) so INTERP reproduces from a
# clone without shipping the full 16 MB mean-preserving run (which stays
# git-ignored under outputs/data/interpolated/).
S3_CACHE = DATA / "s3_interpolation_cumulative.csv"


def diag_interpolation_robustness(out_dir: Path):
    """Interpolation robustness: canonical PCHIP vs mean-preserving cumulative.

    Prefers the full mean-preserving run if present (refreshing the small cache);
    otherwise falls back to the committed cache so a clone can still render S3.
    """
    def cum(df):
        d = df[df.scenario == REFERENCE_SCENARIO]
        return d.groupby("material")["mean"].sum()

    if MP_FILE.exists():
        cp, cm = cum(pd.read_csv(DEMAND_FILE)), cum(pd.read_csv(MP_FILE))
        common = [m for m in cp.index if m in cm.index and cp[m] > 0]
        cache = pd.DataFrame({"material": common,
                              "pchip_cumulative_t": [cp[m] for m in common],
                              "mean_preserving_cumulative_t": [cm[m] for m in common]})
        cache.to_csv(S3_CACHE, index=False)
    elif S3_CACHE.exists():
        cache = pd.read_csv(S3_CACHE)
        common = list(cache["material"])
        print(f"  (INTERP using cached comparison {S3_CACHE.name}; "
              f"run --interpolation mean-preserving to refresh)")
    else:
        print(f"  SKIP INTERP: neither {MP_FILE} nor {S3_CACHE.name} found. Generate "
              f"with `python simulate.py --interpolation mean-preserving`.")
        return

    xp = cache["pchip_cumulative_t"].to_numpy()
    ym = cache["mean_preserving_cumulative_t"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    cols = [CLASS_COLOR.get(CLASS_OF.get(m), "#888888") for m in common]
    ax.scatter(xp, ym, c=cols, s=22, edgecolor="black", linewidth=0.3, zorder=3)
    lims = [min(xp.min(), ym.min()) * 0.7, max(xp.max(), ym.max()) * 1.4]
    ax.plot(lims, lims, color="#B00020", lw=1.0, ls="--", zorder=2)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lims); ax.set_ylim(lims)
    maxdiff = np.max(np.abs(ym - xp) / xp) * 100
    ax.set_xlabel("Cumulative Baseline with IRA demand, PCHIP (t)")
    ax.set_ylabel("Cumulative Baseline with IRA demand, mean-preserving (t)")
    ax.set_title(f"Interpolation robustness\n(max deviation {maxdiff:.1f}%; "
                 "y = x dashed)", fontsize=9.5)
    ax.set_axisbelow(True)
    _save(fig, out_dir, "diag_interpolation_robustness")


def main():
    ap = argparse.ArgumentParser(
        description="Render SI Figure S2 and the two diagnostics figures.")
    ap.add_argument("--output-dir", default="outputs/figures/manuscript")
    ap.add_argument("--diag-dir", default="outputs/figures/diagnostics")
    ap.add_argument("--only", nargs="*", choices=["S2", "CONV", "INTERP"],
                    help="render only these (default: all)")
    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    diag_dir = Path(args.diag_dir)
    want = set(args.only) if args.only else {"S2", "CONV", "INTERP"}

    if "S2" in want:
        fig_s2(pd.read_csv(DEMAND_FILE), out_dir)
    if "CONV" in want:
        diag_mc_convergence(diag_dir)
    if "INTERP" in want:
        diag_interpolation_robustness(diag_dir)


if __name__ == "__main__":
    main()
