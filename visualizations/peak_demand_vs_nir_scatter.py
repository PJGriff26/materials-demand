"""Manuscript Figure 6 library: data assembly and rendering for the supply-risk
scatter (Net Import Reliance vs demand/production ratio, colored by production
HHI).

fig6_nir_vs_hhi_scatter.py imports the table-assembly helpers here (assemble_table,
load_global_production) to build Figure 6; running this file directly emits a
standalone scatter plus its companion CSV. The y-axis defaults to the Mid_Case
NREL Standard Scenario MC-mean; pass --scenario to override.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Rectangle

try:
    from adjustText import adjust_text
    _HAVE_ADJUST_TEXT = True
except ImportError:
    _HAVE_ADJUST_TEXT = False

# Reuse the canonical commodity map + global-production helpers from the
# supply-chain feature module so we never diverge from the production
# numbers used elsewhere in the manuscript.
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "supply_chain"))
sys.path.insert(0, str(BASE_DIR))
from src.scenario_config import REFERENCE_SCENARIO  # noqa: E402
from feature_engineering import (  # noqa: E402
    load_mcs2025_world_data,
    _mcs2025_country_production,
    MCS2025_COMMODITY_MAP,
    EXCLUDED_FROM_CRITICALITY,
)

# Alias kept for backwards-compatibility with v1 of this script. The
# central `_mcs2025_country_production` now applies UNIT_MEAS conversion
# and returns tonnes (patched 2026-05-21), so the local helper is no
# longer needed.
_country_production_tonnes = _mcs2025_country_production

DEMAND_BY_SCENARIO_CSV = BASE_DIR / "outputs" / "data" / "material_demand_by_scenario.csv"
DEFAULT_SCENARIO = REFERENCE_SCENARIO
# Per-material supply-chain features (import_dependency, production_hhi),
# written by supply_chain/build_material_features.py from the live
# feature_engineering functions. (Previously read from a CSV emitted by the
# now-retired clustering pipeline; see CONDENSATION_REPORT.md.)
FEATURES_CSV = BASE_DIR / "outputs" / "data" / "material_features.csv"
OUT_DIR = BASE_DIR / "outputs" / "figures" / "manuscript"

# Materials sharing aggregated REE production (flagged in CSV)
REE_AGGREGATE = {"Yttrium", "Gadium", "Dysprosium", "Neodymium",
                 "Praseodymium", "Terbium"}

# Demand-material → MCS 2025 World CSV commodity name for materials that
# do not match by exact string (Steel is "Iron and Steel" in the CSV;
# Magnesium is split into "Magnesium metal" vs "Magnesium Compounds" —
# use metal, the energy-relevant primary production).
EXTRA_COMMODITY_MAP = {
    "Steel": "Iron and Steel",
    "Magnesium": "Magnesium metal",
}

# Chemical-symbol labels for tick clutter reduction. Cement, Steel keep
# their words since they have no standard one- or two-letter equivalent.
LABEL_SYMBOLS = {
    "Aluminum": "Al", "Boron": "B", "Cadmium": "Cd", "Chromium": "Cr",
    "Copper": "Cu", "Dysprosium": "Dy", "Gallium": "Ga", "Germanium": "Ge",
    "Indium": "In", "Lead": "Pb", "Magnesium": "Mg", "Manganese": "Mn",
    "Molybdenum": "Mo", "Neodymium": "Nd", "Nickel": "Ni", "Niobium": "Nb",
    "Praseodymium": "Pr", "Selenium": "Se", "Silicon": "Si", "Silver": "Ag",
    "Tellurium": "Te", "Terbium": "Tb", "Tin": "Sn", "Vanadium": "V",
    "Yttrium": "Y", "Zinc": "Zn",
}

# Cosmetic x-axis nudges for REE elements that share basket-level NIR.
# Census Bureau trade data doesn't split REE imports per element, so all of
# Dy/Tb/Nd/Pr inherit NIR = 0.945 from the "Rare Earths" aggregate. Without
# a small horizontal offset, Dy and Tb plot on top of each other (their peak
# ratios differ by <1% — invisible on the log y-axis) and Nd/Pr collide on
# x even though their ratios separate them by ~4x on y. These are render-
# only offsets — the data CSV and axis labels are unchanged.
REE_MARKER_X_OFFSETS = {
    "Dysprosium":   -0.018,   # Dy ←
    "Terbium":       0.018,   # Tb →
    "Neodymium":    -0.018,   # Nd ←  (y already separates from Pr but x-nudge keeps the visual grouping tidy)
    "Praseodymium":  0.018,   # Pr →
}


# ──────────────────────────────────────────────────────────────────────────
# Data assembly
# ──────────────────────────────────────────────────────────────────────────

def load_peak_demand(scenario=DEFAULT_SCENARIO):
    """Per-material peak-year annualized demand for a single NREL scenario,
    plus the within-MC 95% CI bounds at that year.

    Reads `material_demand_by_scenario.csv` (all 61 scenarios), not
    `material_demand_summary.csv` (only the four highlighted scenarios), so any
    scenario can be selected and each y-value is the MC-mean for THIS scenario's
    peak year. Default scenario is Mid_Case to match the 4-panel mid-case bar;
    override with `scenario=...`.
    """
    df = pd.read_csv(DEMAND_BY_SCENARIO_CSV)
    df = df[df["scenario"] == scenario]
    if df.empty:
        raise ValueError(
            f"Scenario {scenario!r} not found in {DEMAND_BY_SCENARIO_CSV.name}."
        )
    df = df.dropna(subset=["mean_annual"])
    df = df[df["mean_annual"] > 0]
    idx = df.groupby("material")["mean_annual"].idxmax()
    peak = df.loc[idx, ["material", "year", "mean_annual",
                        "p2_annual", "p97_annual"]].reset_index(drop=True)
    peak["scenario"] = scenario
    return peak.rename(columns={
        "year": "peak_year",
        "mean_annual": "peak_demand_t_per_yr",
        "p2_annual": "peak_demand_p2_5",
        "p97_annual": "peak_demand_p97_5",
    })


def load_global_production():
    """Sum per-country MCS 2025 production per demand-material name.

    Per-element rare earth override (2026-05-27): the upstream
    MCS2025_COMMODITY_MAP collapses every REE element to the aggregate
    "Rare earths" 390 kt REO total, which makes Y, Nd, Pr, Dy, Tb all
    plot against the same denominator and badly understates per-element
    supply stress. The override below swaps in the canonical per-element
    production values from supply_tiers_shared.RE_ELEMENT_DATA (DOE 2023
    Critical Materials Assessment Appendix A for Nd/Pr/Dy/Tb; USGS MCS
    multi-vintage median for Y; Gadium dropped because it has zero demand
    in the simulation per technology_mapping CIGS=0%). Values are tonnes
    ELEMENT content, matching DOE 2023 §4.3 convention.
    """
    world_df = load_mcs2025_world_data()
    rows = []
    for demand_name, mcs_commodity in MCS2025_COMMODITY_MAP.items():
        # Skip REE elements here; they're filled from RE_ELEMENT_DATA below.
        if demand_name in REE_AGGREGATE:
            continue
        sub = _country_production_tonnes(world_df, mcs_commodity)
        if not sub.empty:
            rows.append((demand_name, float(sub["p"].sum()), mcs_commodity))

    # Also pick up any commodity not in MCS2025_COMMODITY_MAP whose name
    # matches the demand-material directly (Aluminum, Cement, Lead, etc.).
    # For materials whose CSV commodity name differs from the demand name
    # (Steel / Iron and Steel, Magnesium / Magnesium metal), the explicit
    # mapping in EXTRA_COMMODITY_MAP takes priority over the direct match.
    seen = {r[0] for r in rows}
    demand_materials = pd.read_csv(DEMAND_BY_SCENARIO_CSV,
                                    usecols=["material"])["material"].unique()
    for mat in demand_materials:
        if mat in seen or mat in REE_AGGREGATE:
            continue
        mcs_name = EXTRA_COMMODITY_MAP.get(mat, mat)
        sub = _country_production_tonnes(world_df, mcs_name)
        if not sub.empty:
            rows.append((mat, float(sub["p"].sum()), mcs_name))

    # Per-element REE override (DOE 2023 App A + USGS Y).
    # Import here to avoid circular-import surprises at module load.
    sys.path.insert(0, str(BASE_DIR / "visualizations"))
    from supply_tiers_shared import RE_ELEMENT_DATA  # noqa: E402
    for re_elem, data in RE_ELEMENT_DATA.items():
        rows.append((re_elem, float(data["global_t"]),
                     "RE_ELEMENT_DATA (DOE 2023 App A / USGS MCS)"))

    return pd.DataFrame(rows, columns=["material", "global_production_t",
                                       "mcs_commodity"])


def assemble_table(scenario=DEFAULT_SCENARIO):
    peak = load_peak_demand(scenario=scenario)
    feats = pd.read_csv(FEATURES_CSV)[
        ["material", "import_dependency", "production_hhi"]
    ]
    glob = load_global_production()

    df = peak.merge(feats, on="material", how="left") \
             .merge(glob, on="material", how="left")

    df["ratio"] = df["peak_demand_t_per_yr"] / df["global_production_t"]
    df["ratio_lo"] = df["peak_demand_p2_5"] / df["global_production_t"]
    df["ratio_hi"] = df["peak_demand_p97_5"] / df["global_production_t"]
    df["ree_aggregate"] = df["material"].isin(REE_AGGREGATE)

    # Drop excluded + anything missing the three plotting axes
    df = df[~df["material"].isin(EXCLUDED_FROM_CRITICALITY)]
    df = df.dropna(subset=["ratio", "import_dependency", "production_hhi"])
    df = df[df["ratio"] > 0]
    return df.sort_values("ratio", ascending=False).reset_index(drop=True)


def assemble_cumulative_table(scenario=DEFAULT_SCENARIO, horizon_years=25):
    """Cumulative-demand companion to assemble_table().

    y-axis ratio = cumulative 2026-2050 US demand (sum of the annual mean)
    divided by horizon_years of current global production (annual production
    x 25), i.e. the fraction of a quarter-century of global output the US
    transition would consume. Same NIR (x) and production-HHI (color) axes
    as the peak variant. CI bounds are summed annual p2.5/p97.5, which
    ignores cross-year correlation and therefore OVERSTATES the cumulative
    interval (conservative envelope; see citation_support / methods note).
    """
    dem = pd.read_csv(DEMAND_BY_SCENARIO_CSV)
    dem = dem[dem["scenario"] == scenario].reset_index(drop=True)
    if dem.empty:
        raise ValueError(
            f"Scenario {scenario!r} not found in {DEMAND_BY_SCENARIO_CSV.name}."
        )
    dem = dem[(dem["year"] >= 2026) & (dem["year"] <= 2050)]
    dem = dem[dem["mean_annual"] > 0]
    grp = dem.groupby("material")
    cum = pd.DataFrame({
        "cum_demand_t": grp["mean_annual"].sum(),
        "cum_demand_p2_5": grp["p2_annual"].sum(),
        "cum_demand_p97_5": grp["p97_annual"].sum(),
    })
    peak_idx = grp["mean_annual"].idxmax()
    cum["peak_year"] = dem.loc[peak_idx].set_index("material")["year"]
    cum = cum.reset_index()
    cum["scenario"] = scenario

    feats = pd.read_csv(FEATURES_CSV)[
        ["material", "import_dependency", "production_hhi"]
    ]
    glob = load_global_production()
    df = cum.merge(feats, on="material", how="left") \
            .merge(glob, on="material", how="left")

    denom = df["global_production_t"] * horizon_years
    df["ratio"] = df["cum_demand_t"] / denom
    df["ratio_lo"] = df["cum_demand_p2_5"] / denom
    df["ratio_hi"] = df["cum_demand_p97_5"] / denom
    df["ree_aggregate"] = df["material"].isin(REE_AGGREGATE)

    df = df[~df["material"].isin(EXCLUDED_FROM_CRITICALITY)]
    df = df.dropna(subset=["ratio", "import_dependency", "production_hhi"])
    df = df[df["ratio"] > 0]
    return df.sort_values("ratio", ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────────────────

def _label_offsets(df):
    """Greedy collision-free label placement in (NIR, log10(ratio)) space.

    For each material we try eight compass offsets and pick the one that
    keeps the label farthest from any other point in this rescaled
    coordinate system (NIR is 0-1, log-ratio spans ~5 orders of magnitude,
    so we normalize before scoring).
    """
    xs = df["import_dependency"].to_numpy(dtype=float)
    ys = np.log10(df["ratio"].to_numpy(dtype=float))
    # Normalize to [0, 1] for fair distance scoring
    x_range = xs.max() - xs.min() + 1e-9
    y_range = ys.max() - ys.min() + 1e-9
    xs_n = (xs - xs.min()) / x_range
    ys_n = (ys - ys.min()) / y_range

    candidates = [
        (0.025, 0.10), (-0.025, 0.10), (0.025, -0.10), (-0.025, -0.10),
        (0.04, 0.0), (-0.04, 0.0), (0.0, 0.18), (0.0, -0.18),
    ]
    chosen = []
    taken = []  # list of (xn, yn) of placed labels
    for i in range(len(df)):
        best, best_score = candidates[0], -np.inf
        for dx, dy in candidates:
            lx = xs_n[i] + dx / x_range
            ly = ys_n[i] + dy / y_range
            # Score = min distance to any other point + any already-placed label
            others = np.delete(np.column_stack([xs_n, ys_n]), i, axis=0)
            d_pts = np.min(np.hypot(others[:, 0] - lx, others[:, 1] - ly))
            d_lbls = (np.min([np.hypot(lx - tx, ly - ty)
                              for tx, ty in taken])
                      if taken else np.inf)
            # Penalize labels that go off-axis
            off_axis_pen = 0.0
            real_x = xs[i] + dx
            if real_x < -0.02 or real_x > 1.02:
                off_axis_pen = 0.5
            score = min(d_pts, d_lbls) - off_axis_pen
            if score > best_score:
                best, best_score = (dx, dy), score
        chosen.append(best)
        taken.append((xs_n[i] + best[0] / x_range,
                      ys_n[i] + best[1] / y_range))
    return chosen


def plot_scatter(df, out_dir, show_errorbars=False,
                 ylabel=None, title=None, out_stem="peak_demand_vs_nir_scatter"):
    """Render the supply-risk scatter.

    show_errorbars defaults to False because the 95% CI bars span several
    log decades and visually dominate the central estimate. MC uncertainty
    is noted in the caption and lives in the companion CSV instead.

    ylabel / title / out_stem override the default peak-variant labels and
    output filename so the same renderer can draw the cumulative variant.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 8.5))

    # HHI colour range spans the actual data distribution (~0.14 to 1.00
    # after the 2026-05-27 stage-resolved production_hhi refactor — HREE
    # elements Dy/Tb/Y/Gd now sit at exactly 1.000 because EU CRM 2023
    # Annex 7 reports HREE processing as 100% Chinese). Graedel et al.
    # (2012) critical-concentration threshold (HHI = 0.4) sits roughly
    # mid-ramp. Upper bound previously 0.9 (pre-stage-resolution); raised
    # to 1.0 so HREE values don't clip to the same darkest color as
    # arbitrary 0.91+ values.
    #
    # Single-hue gradient (light gray -> deep red): the multi-hue magma
    # ramp introduced spurious categorical reading at the yellow/orange
    # transition; a single-hue lightness ramp reads as one ordered axis
    # and ties visually to the "high concentration = high risk" framing.
    hhi_vmin, hhi_vmax = 0.1, 1.0
    norm = Normalize(vmin=hhi_vmin, vmax=hhi_vmax)
    cmap = plt.get_cmap("Reds")

    # Optional MC uncertainty bars (off by default)
    if show_errorbars:
        for _, r in df.iterrows():
            ax.plot(
                [r["import_dependency"], r["import_dependency"]],
                [max(r["ratio_lo"], 1e-6), r["ratio_hi"]],
                color="lightgray", linewidth=0.9, zorder=1,
            )

    # Per-marker x-offset for REE elements that share basket NIR (cosmetic
    # only — see REE_MARKER_X_OFFSETS comment). Applied to the *plotting*
    # x-coordinate, not the underlying data.
    plot_x = df.apply(
        lambda r: r["import_dependency"] + REE_MARKER_X_OFFSETS.get(r["material"], 0.0),
        axis=1,
    )

    # Points coloured by HHI
    sc = ax.scatter(
        plot_x, df["ratio"],
        c=df["production_hhi"], cmap=cmap, norm=norm,
        s=150, edgecolors="black", linewidth=0.7, zorder=3,
    )

    # Material labels with manual collision-avoiding offsets (chemical
    # symbols where unambiguous; full names for Cement / Steel which lack
    # standard one-or-two-letter equivalents). adjustText fights the log
    # y-axis (inflates saved bbox to >100k px), so we use a manual greedy.
    offsets = _label_offsets(df)
    for (dx, dy_log), (_, r) in zip(offsets, df.iterrows()):
        label = LABEL_SYMBOLS.get(r["material"], r["material"])
        ree_dx = REE_MARKER_X_OFFSETS.get(r["material"], 0.0)
        ax.annotate(
            label,
            xy=(r["import_dependency"] + ree_dx, r["ratio"]),
            xytext=(r["import_dependency"] + ree_dx + dx,
                    r["ratio"] * (10 ** dy_log)),
            fontsize=10, color="black", zorder=5,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy_log >= 0 else "top",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5)
            if (abs(dx) > 0.04 or abs(dy_log) > 0.2) else None,
        )

    # Cosmetics — minimalist. No reference threshold lines, no shaded
    # quadrants, no Graedel marker. Only the data and the axes.
    ax.set_yscale("log")
    ax.set_xlim(-0.04, 1.04)
    ax.set_ylim(1e-5, 8)

    # Suppress log minor tick marks.
    ax.yaxis.set_minor_locator(plt.NullLocator())

    # Surface the NREL scenario name and peak-year horizon in the title so
    # the figure is self-describing in a slide deck or SI caption. Materials
    # peak at different years (per-material argmax of mean_annual within the
    # scenario), so we report the modal peak year with a hint at the spread.
    scenario_name = df["scenario"].iloc[0] if "scenario" in df.columns else REFERENCE_SCENARIO
    yrs = df["peak_year"].astype(int)
    modal_yr = int(yrs.mode().iloc[0])
    n_modal = int((yrs == modal_yr).sum())
    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    if yr_min == yr_max:
        yr_phrase = f"peak year {modal_yr}"
    else:
        yr_phrase = f"peak year per material, {modal_yr} ({n_modal}/{len(df)}); range {yr_min}–{yr_max}"
    ax.set_xlabel("Net Import Reliance", fontsize=11)
    ax.set_ylabel(ylabel or "Peak US demand / global production", fontsize=11)
    ax.set_title(
        title or
        f"Supply risk: demand burden vs import dependence, NREL {scenario_name} ({yr_phrase})",
        fontsize=11, pad=8, loc="left",
    )
    ax.grid(True, which="major", axis="both", linestyle=":",
            linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                        shrink=0.85, pad=0.04)
    cbar.set_label("Production HHI", fontsize=10)

    plt.subplots_adjust(left=0.08, right=0.88, top=0.93, bottom=0.10)
    png = out_dir / f"{out_stem}.png"
    pdf = out_dir / f"{out_stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help=f"NREL Standard Scenario name. Default: {DEFAULT_SCENARIO}. "
                         "Pass any name from material_demand_by_scenario.csv.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Peak-year variant (default).
    df = assemble_table(scenario=args.scenario)
    df.to_csv(OUT_DIR / "peak_demand_vs_nir_data.csv", index=False)
    png, pdf = plot_scatter(df, OUT_DIR, show_errorbars=False)
    print(f"  scenario: {args.scenario}")
    print(f"  peak variant rows plotted: {len(df)}")
    print(f"  saved: {png.relative_to(BASE_DIR)}")
    print(f"  saved: {pdf.relative_to(BASE_DIR)}")
    print(f"  saved: {(OUT_DIR / 'peak_demand_vs_nir_data.csv').relative_to(BASE_DIR)}")

    # Cumulative 2026-2050 variant (companion figure).
    dfc = assemble_cumulative_table(scenario=args.scenario)
    dfc.to_csv(OUT_DIR / "cumulative_demand_vs_nir_data.csv", index=False)
    png_c, pdf_c = plot_scatter(
        dfc, OUT_DIR, show_errorbars=False,
        ylabel="Cumulative 2026-2050 US demand / (25 yr of global production)",
        title=f"Supply risk: cumulative demand burden vs import dependence, NREL {args.scenario} (2026-2050)",
        out_stem="cumulative_demand_vs_nir_scatter",
    )
    print(f"  cumulative variant rows plotted: {len(dfc)}")
    print(f"  saved: {png_c.relative_to(BASE_DIR)}")
    print(f"  saved: {pdf_c.relative_to(BASE_DIR)}")
    print(f"  saved: {(OUT_DIR / 'cumulative_demand_vs_nir_data.csv').relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
