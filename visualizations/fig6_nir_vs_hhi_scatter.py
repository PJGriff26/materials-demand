"""Manuscript Figure 6: supply-risk scatter (Net Import Reliance vs production
HHI), marker size = peak US demand / global production, color = material class.

Reuses the table-assembly helpers in peak_demand_vs_nir_scatter.py (per-material
peak demand, global production, NIR, and HHI) and reshapes the visual encoding
into the NIR-by-HHI supply-risk plane. Writes fig6_nir_vs_hhi_scatter.{png,pdf}.

INVENTORY:
  name: fig6_nir_vs_hhi_scatter
  output: outputs/figures/manuscript/fig6_nir_vs_hhi_scatter{,_linear}.{png,pdf}
  category: Manuscript main text — Fig 6
  axes:
    x: Net Import Reliance (%, USGS MCS 2025)
    y: Herfindahl–Hirschman Index of global production concentration (0–1)
    size: US peak annual demand / global production (log or linear area map,
      floor-clipped below 1%)
    color: 4 material classes (locked manuscript palette, src.material_classes)
  data_sources:
    - assemble_table() in peak_demand_vs_nir_scatter.py (per-material peak
      demand from the MC output, USGS MCS 2025 global production / NIR, and
      production HHI via supply_chain feature table).
  description: >
    Supply-risk scatter placing each demand-bearing material in the NIR-by-HHI
    plane; marker area encodes the share of global production that US peak
    annual demand would represent. Both in-axes legends share a left edge
    with left-aligned titles; legend paddings scale with 11pt/_LEG_FONT so
    the physical spacing is invariant to the legend font size.
END_INVENTORY
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "visualizations"))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

# The four material classes live in supply_chain/config.py (the canonical
# copy). (Originally imported from analysis/poster_policy_lever_table.py,
# which was dropped with the policy-lever regression; see CONDENSATION_REPORT.md.)
from config import MATERIAL_GROUPS  # noqa: E402
from peak_demand_vs_nir_scatter import (  # noqa: E402
    assemble_table,
    DEFAULT_SCENARIO,
    LABEL_SYMBOLS,
    REE_MARKER_X_OFFSETS as _REE_OFFSETS_FRAC,
    OUT_DIR,
)

# X-axis is rendered in percent (0-100).
# Dy and Tb share an identical NIR (94.5%, basket-level Census trade
# data — see comment in peak_demand_vs_nir_scatter.REE_MARKER_X_OFFSETS),
# so they need a cosmetic horizontal nudge to read as distinct points.
# The required spread differs between size scales: on the linear scale
# Dy/Tb markers are very large (ratio ≈ 0.43 sits high in the linear
# range) and need wider separation; on the log scale they're moderate.
REE_MARKER_X_OFFSETS = {
    "log": {
        "Dysprosium":   -1.8, "Terbium":       1.8,
        "Neodymium":    -1.8, "Praseodymium":  1.8,
    },
    "linear": {
        # Slight overlap between Dy and Tb (both at NIR=94.5%) — visually
        # signals they share the same basket-level NIR rather than
        # implying two distinct values.
        "Dysprosium":   -1.5, "Terbium":       1.5,
        "Neodymium":    -1.8, "Praseodymium":  1.8,
    },
}

# Manual per-label nudges applied on top of the greedy label-placement
# offsets. Per-scale because marker sizes (and thus collision geometry)
# differ. Values in (NIR%, HHI) data units. Positive dx = right.
LABEL_NUDGES = {
    "log":    {},
    "linear": {
        "Tellurium":   (1.5, 0.0),
    },
}

# Absolute label offsets — override the greedy algorithm entirely for
# materials whose placement is too constrained for collision-avoidance
# to find a clean spot. Values are (dx, dy) from the marker, same units
# as the data axes. Per-scale because marker sizes differ.
LABEL_OVERRIDES = {
    "log":    {},
    "linear": {
        # Dy / Tb / Y cluster at HHI≈1.0 with Y at the right edge.
        # All three labels read out toward the upper edges of the plot;
        # Y goes right (into the small xlim margin) since left would
        # crowd Tb.
        "Dysprosium":  (-3.0, 0.05),   # "Dy" up-left of its (big) marker
        "Terbium":     (-2.0, 0.05),   # "Tb" up-left of its marker
        "Yttrium":     (2.0, 0.0),     # "Y" right of its marker
    },
}


# Categorical palette: the locked manuscript material-class palette, used
# consistently wherever materials are colored by class (this figure and the
# SI tornado, figS3_sensitivity_tornado.CATEGORY_COLORS; keep in sync).
# 2026-07-13 figure revision: REE now carries the same navy used for the
# REE labels in Figs 4/5 (supply_tiers_variants.REE_HIGHLIGHT_COLOR) so the REE
# set reads consistently across figures; Base & alloying takes the plum that
# REE freed up (navy would have collided with the old steel blue).
from src.material_classes import CLASS_COLORS, CLASS_LABELS  # noqa: E402

# Display-only legend labels from the canonical taxonomy; MATERIAL_GROUPS
# keys stay canonical (src/material_classes.py via supply_chain/config.py).
LEGEND_LABELS = dict(CLASS_LABELS)

# Continuous log->area sizing with a floor clip at 1%. Materials below
# 1% (Steel, Cement, Cr, Mn, Mg, Ni, Y, B) all render at the same small
# floor marker — fine-grained ordering down at 0.001 vs 0.0001 isn't
# meaningful for the supply-risk story. From 1% to 100% the marker
# area scales linearly with log10(ratio), giving two decades of
# perceptible range across the materials that actually consume a
# meaningful share of global supply.
SIZE_FLOOR_RATIO = 0.01        # 1% — anything below clips to the floor
SIZE_FLOOR_PT2 = 40.0          # marker area for clipped values
SIZE_MAX_PT2 = 1800.0          # marker area at ratio = 1.0
SIZE_MIN_PT2_AT_FLOOR = 100.0  # marker area at ratio = SIZE_FLOOR_RATIO

# Legend anchors for the size scale. None -> the clipped "< 1%" floor
# marker; the rest are sample ratio values to render at full size.
SIZE_LEGEND_RATIOS = {
    "log":    [None, 0.01, 0.10, 0.50, 1.00],
    "linear": [None, 0.01, 0.25, 0.50, 0.75, 1.00],
}


def material_to_class():
    """Flat dict: material name -> class label."""
    return {m: cls for cls, members in MATERIAL_GROUPS.items() for m in members}


def size_from_ratio(ratio, scale="log"):
    """Map demand ratio -> matplotlib scatter `s` (pt^2).

    Ratios below SIZE_FLOOR_RATIO clip to SIZE_FLOOR_PT2. Above the
    floor the mapping depends on `scale`:

      - "log":    marker area linear in log10(ratio) between
                  (SIZE_FLOOR_RATIO -> SIZE_MIN_PT2_AT_FLOOR) and
                  (1.0 -> SIZE_MAX_PT2). Gives perceptible separation
                  across the 1%-100% range.
      - "linear": marker area linear in ratio over the same end
                  anchors. Compresses the low end (1%-30% all look
                  small) and emphasizes truly outlier values like
                  Tellurium near 100%.
    """
    arr = np.asarray(ratio, dtype=float)
    out = np.full_like(arr, SIZE_FLOOR_PT2, dtype=float)

    above = arr >= SIZE_FLOOR_RATIO
    if np.any(above):
        if scale == "log":
            log_floor = np.log10(SIZE_FLOOR_RATIO)  # e.g., -2
            slope = (SIZE_MAX_PT2 - SIZE_MIN_PT2_AT_FLOOR) / (0.0 - log_floor)
            log_r = np.log10(np.clip(arr[above], SIZE_FLOOR_RATIO, None))
            out[above] = SIZE_MIN_PT2_AT_FLOOR + (log_r - log_floor) * slope
        elif scale == "linear":
            slope = (SIZE_MAX_PT2 - SIZE_MIN_PT2_AT_FLOOR) / (1.0 - SIZE_FLOOR_RATIO)
            r = np.clip(arr[above], SIZE_FLOOR_RATIO, 1.0)
            out[above] = SIZE_MIN_PT2_AT_FLOOR + (r - SIZE_FLOOR_RATIO) * slope
        else:
            raise ValueError(f"scale must be 'log' or 'linear', got {scale!r}")
    return out


def _label_offsets(xs, ys):
    """Greedy collision-free label placement in (NIR %, HHI) space.

    xs are in percent (0-100), ys are on the HHI scale (0-1), so we
    normalize before scoring distances. Offset candidates are expressed
    in the *data* coordinate system the caller passes in.
    """
    x_span = max(xs) - min(xs) + 1e-9
    y_span = max(ys) - min(ys) + 1e-9
    xs_n = (np.asarray(xs) - min(xs)) / x_span
    ys_n = (np.asarray(ys) - min(ys)) / y_span

    # Candidate offsets are tuples in (NIR%, HHI) data space.
    candidates = [
        (1.8, 0.030), (-1.8, 0.030), (1.8, -0.030), (-1.8, -0.030),
        (2.8, 0.0), (-2.8, 0.0), (0.0, 0.045), (0.0, -0.045),
    ]
    # Approx label half-width in NIR-percent units. Labels are mostly
    # 1-3 chars ("Mo", "Te", "Cement") at fontsize=10; ~4% NIR is a
    # conservative width that keeps even the longest chemical-symbol
    # labels inside the axes.
    LABEL_WIDTH_X = 4.0

    chosen = []
    taken = []
    for i in range(len(xs)):
        best, best_score = candidates[0], -np.inf
        for dx, dy in candidates:
            anchor_x = xs[i] + dx
            anchor_y = ys[i] + dy
            lx_n = xs_n[i] + dx / x_span
            ly_n = ys_n[i] + dy / y_span
            others = np.delete(np.column_stack([xs_n, ys_n]), i, axis=0)
            d_pts = np.min(np.hypot(others[:, 0] - lx_n, others[:, 1] - ly_n))
            d_lbls = (np.min([np.hypot(lx_n - tx, ly_n - ty)
                              for tx, ty in taken])
                      if taken else np.inf)

            # Off-axis penalty accounting for which side the text extends
            # to. ha="right" when dx<0 (text extends LEFT of anchor),
            # ha="left" otherwise (text extends RIGHT of anchor).
            off_axis_pen = 0.0
            if dx < 0:
                if anchor_x - LABEL_WIDTH_X < 0:
                    off_axis_pen += 100.0  # effectively forbid
            else:
                if anchor_x + LABEL_WIDTH_X > 100:
                    off_axis_pen += 100.0
            if anchor_y < -0.02 or anchor_y > 1.04:
                off_axis_pen += 100.0

            score = min(d_pts, d_lbls) - off_axis_pen
            if score > best_score:
                best, best_score = (dx, dy), score
        chosen.append(best)
        taken.append((xs_n[i] + best[0] / x_span,
                      ys_n[i] + best[1] / y_span))
    return chosen


def plot_scatter(df, out_dir, scale="log", docx=False):
    out_dir.mkdir(parents=True, exist_ok=True)

    mat_class = material_to_class()
    df = df.copy()
    df["material_class"] = df["material"].map(mat_class)
    # Anything not in the 4 published groups gets dropped from this
    # visualization — keeps the legend honest. Currently no orphans in
    # the Mid_Case data CSV.
    df = df.dropna(subset=["material_class"]).reset_index(drop=True)

    df["color"] = df["material_class"].map(CLASS_COLORS)

    # --docx: this is a bubble chart whose marker areas ENCODE data (absolute
    # pt^2), so shrinking the canvas to a small print width would force the
    # bubbles to overlap. Instead — like Figs 2 and 3 — keep the figure at its
    # natural large canvas (markers untouched, no overlap) and only enlarge the
    # text, so it stays legible when the figure is inserted into Word at page
    # width. Default (no --docx) keeps the original sizes.
    _FULL_W, _FULL_H = 11.0, 8.5
    _fig_w, _fig_h = _FULL_W, _FULL_H
    _F = 1.35 if docx else 1.0   # print variant: text-only enlargement

    # Right margins leave room for the two in-axes legends; any tighter and
    # the legend titles clip at the canvas edge.
    _MARGINS = dict(left=0.08, right=(0.75 if docx else 0.74),
                    top=0.93, bottom=0.10)

    df["size"] = size_from_ratio(df["ratio"], scale=scale)
    fig, ax = plt.subplots(figsize=(_fig_w, _fig_h))

    # Render NIR on a 0-100 percent axis.
    # Underlying CSV is fractional, so multiply by 100 at render time
    # only; the data table on disk stays in 0-1 space.
    df["nir_pct"] = df["import_dependency"] * 100.0

    # Per-marker x-offset for REE elements that share basket NIR
    # (cosmetic only). Scale-keyed because marker sizes differ.
    ree_offsets = REE_MARKER_X_OFFSETS[scale]
    plot_x = df.apply(
        lambda r: r["nir_pct"] + ree_offsets.get(r["material"], 0.0),
        axis=1,
    ).to_numpy()
    plot_y = df["production_hhi"].to_numpy()

    ax.scatter(
        plot_x, plot_y,
        s=df["size"], c=df["color"],
        edgecolors="black", linewidth=0.7, alpha=0.85, zorder=3,
    )

    # Material labels. Materials in LABEL_OVERRIDES bypass the greedy
    # algorithm entirely; everything else uses algorithm + (optional)
    # additive nudge.
    overrides = LABEL_OVERRIDES[scale]
    # Exclude overridden points from the algorithm input so they don't
    # bias the collision scoring for remaining points.
    algo_mask = [m not in overrides for m in df["material"]]
    algo_offsets = _label_offsets(plot_x[algo_mask], plot_y[algo_mask])
    algo_iter = iter(algo_offsets)

    nudges = LABEL_NUDGES[scale]
    for (_, r), px, py in zip(df.iterrows(), plot_x, plot_y):
        label = LABEL_SYMBOLS.get(r["material"], r["material"])
        if r["material"] in overrides:
            dx, dy = overrides[r["material"]]
        else:
            dx, dy = next(algo_iter)
            nudge_x, nudge_y = nudges.get(r["material"], (0.0, 0.0))
            dx += nudge_x
            dy += nudge_y
        ax.annotate(
            label,
            xy=(px, py),
            xytext=(px + dx, py + dy),
            fontsize=13 * _F, color="black", zorder=5,
            ha="left" if dx >= 0 else "right",
            va="bottom" if dy >= 0 else "top",
            arrowprops=dict(arrowstyle="-", color="gray", lw=0.4, alpha=0.5)
            if (abs(dx) > 0.025 or abs(dy) > 0.035) else None,
        )

    # Cosmetics
    ax.set_xlim(-4, 104)
    ax.set_ylim(-0.04, 1.08)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
    ax.set_xlabel("Net Import Reliance", fontsize=15 * _F)
    ax.set_ylabel("Herfindahl–Hirschman Index", fontsize=15 * _F)
    ax.tick_params(axis="both", labelsize=14 * _F)

    scenario_name = df["scenario"].iloc[0] if "scenario" in df.columns else DEFAULT_SCENARIO
    yrs = df["peak_year"].astype(int)
    modal_yr = int(yrs.mode().iloc[0])
    n_modal = int((yrs == modal_yr).sum())
    yr_min, yr_max = int(yrs.min()), int(yrs.max())
    if yr_min == yr_max:
        yr_phrase = f"peak year {modal_yr}"
    else:
        yr_phrase = f"peak year per material, {modal_yr} ({n_modal}/{len(df)}); range {yr_min}–{yr_max}"
    # No figure title (the caption carries it).
    ax.grid(True, which="major", axis="both", linestyle=":",
            linewidth=0.4, alpha=0.4)
    ax.set_axisbelow(True)

    # --- Class color legend (top right inside axes) ---
    class_handles = [
        Line2D([0], [0], marker="o", linestyle="",
               markerfacecolor=col, markeredgecolor="black",
               markersize=10 * _F, label=LEGEND_LABELS.get(cls, cls))
        for cls, col in CLASS_COLORS.items()
    ]
    # Legend paddings are in font-height units; _PAD rescales them so the
    # physical spacing stays fixed as _LEG_FONT changes (calibrated at 11 pt).
    _LEG_FONT = 13            # legend entry size (title is +1)
    _PAD = 11.0 / _LEG_FONT   # padding calibration was done at 11 pt
    leg1 = ax.legend(
        handles=class_handles, title="Material class",
        loc="upper left", bbox_to_anchor=(1.01, 1.0),
        frameon=False, fontsize=_LEG_FONT * _F,
        title_fontsize=(_LEG_FONT + 1) * _F,
        handletextpad=0.6 * _PAD, labelspacing=0.5 * _PAD,
        borderaxespad=0.0,
    )
    # Left-align the legend title over its entries (matches leg2; both
    # legends share the same left edge via bbox_to_anchor x=1.01).
    leg1._legend_box.align = "left"
    ax.add_artist(leg1)

    # --- Size legend (continuous with a clipped floor) ---
    size_handles = []
    for r_val in SIZE_LEGEND_RATIOS[scale]:
        if r_val is None:
            size_pt2 = SIZE_FLOOR_PT2
            label = f"< {SIZE_FLOOR_RATIO * 100:.0f}%"
        else:
            size_pt2 = float(size_from_ratio(np.array([r_val]), scale=scale).item())
            label = f"{r_val * 100:.0f}%"
        ms = np.sqrt(size_pt2)
        size_handles.append(
            Line2D([0], [0], marker="o", linestyle="",
                   markerfacecolor="lightgray", markeredgecolor="black",
                   markersize=ms, label=label)
        )
    # Pad the row spacing so the biggest markers don't overlap their
    # neighbors — Line2D `labelspacing` is in units of font height,
    # which is too tight when the diameter ratio between adjacent
    # legend dots exceeds ~3x (happens on the linear scale top end).
    # `labelspacing` is in font-height units, so the enlarged --docx text would
    # make all six rows taller and push the bottom (100%) marker off the
    # canvas. Divide by _F so the ABSOLUTE row spacing matches the full-size
    # figure (the markers are the same natural size in both modes, so the
    # legend keeps the full-size vertical extent and all rows stay on-canvas).
    leg2 = ax.legend(
        handles=size_handles, title="US peak annual demand ÷\nglobal production (%)",
        loc="upper left", bbox_to_anchor=(1.01, 0.78),
        frameon=False, fontsize=_LEG_FONT * _F,
        title_fontsize=(_LEG_FONT + 1) * _F,
        handletextpad=1.8 * _PAD, labelspacing=4.2 * _PAD / _F,
        borderaxespad=0.0,
    )
    leg2._legend_box.align = "left"

    # Under --docx the right margin is pulled in slightly (see _MARGINS) to
    # reserve room for the enlarged legend labels; the non-docx layout is
    # byte-for-byte unchanged.
    plt.subplots_adjust(**_MARGINS)
    suffix = "" if scale == "log" else f"_{scale}"
    png = out_dir / f"fig6_nir_vs_hhi_scatter{suffix}.png"
    pdf = out_dir / f"fig6_nir_vs_hhi_scatter{suffix}.pdf"

    # --docx: set in-plot text to ABSOLUTE point sizes by role. Because the
    # figure is rendered at the 5.5 in print width above, these absolute pt
    # equal the on-page pt when the saved PNG is inserted at its saved width
    # in Word. Default (no --docx) output is byte-for-byte unchanged.
    if docx:
        from matplotlib.legend import Legend as _Legend

        # Absolute print-size roles (pt); hierarchy mirrors the screen render.
        _SUP = 14.0       # (no suptitle in this fig; kept for parity)
        _AXTITLE = 14.0   # axis title ("Supply risk scatter ...")
        _AXLABEL = 12.0   # x/y axis labels
        _TICK = 10.5      # tick labels
        _LEG = 10.5       # legend entry + title text
        _ANNOT = 10.0     # per-point material labels (ax.texts)

        def _size_legend(_lg):
            if _lg is None:
                return
            if _lg.get_title() is not None:
                _lg.get_title().set_fontsize(_LEG)
            for _txt in _lg.get_texts():
                _txt.set_fontsize(_LEG)

        if getattr(fig, "_suptitle", None):
            fig._suptitle.set_fontsize(_SUP)
        for _ax in fig.get_axes():
            if _ax.get_title():
                _ax.title.set_fontsize(_AXTITLE)
            _ax.xaxis.label.set_fontsize(_AXLABEL)
            _ax.yaxis.label.set_fontsize(_AXLABEL)
            for _t in _ax.get_xticklabels() + _ax.get_yticklabels():
                _t.set_fontsize(_TICK)
            for _t in _ax.texts:
                _t.set_fontsize(_ANNOT)
            # Both the active legend (size legend, ax.get_legend) and any
            # extra legends parked via ax.add_artist (the class-color leg1).
            _size_legend(_ax.get_legend())
            for _art in list(_ax.artists):
                if isinstance(_art, _Legend):
                    _size_legend(_art)
        for _lg in fig.legends:               # figure-level legends (none here)
            _size_legend(_lg)
    # Override the project's matplotlibrc default of savefig.bbox='tight'
    # — 'tight' was clipping the legend's text past its frameless bbox
    # and truncating "Rare earth elements".
    with matplotlib.rc_context({"savefig.bbox": "standard"}):
        fig.savefig(png, dpi=300)
        fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenario", default=DEFAULT_SCENARIO,
                    help=f"NREL Standard Scenario name. Default: {DEFAULT_SCENARIO}.")
    ap.add_argument("--scale", choices=("log", "linear"), default="log",
                    help="Marker-size mapping for demand ratio. 'log' (default) "
                         "spreads 1%%-100%% across two log decades; 'linear' is "
                         "linear in ratio and emphasizes outliers like Tellurium.")
    ap.add_argument("--docx", action="store_true", default=False,
                    help="Render at the 5.5 in print width with absolute "
                         "point sizes by role so the figure stays legible when "
                         "inserted into a Word docx at its saved width. Does not "
                         "change data, series, or colors.")
    args = ap.parse_args()

    df = assemble_table(scenario=args.scenario)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png, pdf = plot_scatter(df, OUT_DIR, scale=args.scale, docx=args.docx)
    print(f"  scenario: {args.scenario}")
    print(f"  size scale: {args.scale}")
    print(f"  rows plotted: {len(df.dropna(subset=['production_hhi', 'import_dependency']))}")
    print(f"  saved: {png.relative_to(BASE_DIR)}")
    print(f"  saved: {pdf.relative_to(BASE_DIR)}")


if __name__ == "__main__":
    main()
