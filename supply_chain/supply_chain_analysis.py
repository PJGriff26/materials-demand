# supply_chain_analysis.py
"""Generator for the supporting supply-chain risk figures: CRC sourcing
breakdown, reserve-adequacy bars, and the global/US supply-risk scatter
matrices.

Layers OECD CRC country-risk weighting, Census import-partner shares, and
USGS MCS 2025 production/reserves onto the Monte Carlo demand output. The
canonical 4-panel supply-tier figures (manuscript Figures 4 and 5) come from
visualizations/fig4_fig5_supply_tiers.py; the fig3/fig4/fig5A/fig5B labels
here are retained only as stable filenames.
"""

from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import (
    DEMAND_FILE, FIGURES_MANUSCRIPT_DIR,
    DEMAND_TO_RISK, FIGURE_DPI, FIGURE_FORMAT,
)
from usgs_mcs2025_loader import load_risk_data_mcs2025

# ─────────────────────────────────────────────────────────────────────────────
# USGS MCS 2025 World Data — reserves augmentation for materials where the
# per-commodity reserves sheet lacks data but the World Data CSV publishes it.
# ─────────────────────────────────────────────────────────────────────────────

# Path to the USGS MCS 2025 World Data CSV. Resolved relative to the repository
# root (supply_chain/ -> parent -> data/usgs_mcs_2025/world_data/...).
USGS_WORLD_DATA_CSV = (
    Path(__file__).resolve().parent.parent
    / "data" / "usgs_mcs_2025" / "world_data" / "MCS2025_World_Data.csv"
)

# USGS MCS 2025 Bauxite and Alumina metadata (mcs2025-bauxi_meta.xml)
# explicitly documents the Bayer + Hall-Héroult yield:
#   "As a general rule, 4 tons of dried bauxite is required to produce
#    2 tons of alumina, which, in turn, produces 1 ton of aluminum."
# → 1 t dry bauxite yields 0.25 t aluminum metal.
# Source: USGS National Minerals Information Center, Mineral Commodity
# Summaries 2025, Bauxite and Alumina chapter.
BAUXITE_TO_ALUMINUM_FACTOR = 0.25

# Mapping: our reserves-sheet column name → USGS World Data query info.
#   (usgs_commodity, type_filter, conversion_factor_to_metal_tonnes)
# A type_filter is a substring of the TYPE column that selects the right
# row (e.g., the USGS Iron Ore commodity has both "usable ore" and "iron
# content" variants; we want the latter).
_USGS_RESERVES_AUGMENT = {
    # Thin-film byproducts: USGS reports metal-content reserves directly.
    "Indium":    ("Indium",    None,             1.0),
    "Selenium":  ("Selenium",  None,             1.0),
    "Tellurium": ("Tellurium", None,             1.0),
    # Iron/Steel: USGS reports both "usable ore" and "iron content" — use
    # iron content (directly comparable to steel mass; ~1:1 Fe → Steel).
    "Steel":     ("Iron Ore",  "iron content",   1.0),
    # Aluminum: USGS only reports reserves as Bauxite (dry ore). Apply the
    # USGS-documented Bayer+Hall-Héroult factor (4 bauxite → 1 Al).
    "Aluminum":  ("Bauxite",   None,             BAUXITE_TO_ALUMINUM_FACTOR),
}


def _usgs_world_to_kt(value, unit_str, reserve_notes=None):
    """Convert a USGS World Data reserves value to kilotons (kt).

    Unit precedence: RESERVE_NOTES (when present and parseable) > UNIT_MEAS.
    USGS uses UNIT_MEAS to label the *production* columns; reserves can
    use a different unit, which is then stated in RESERVE_NOTES. Iron Ore
    is the canonical case: UNIT_MEAS="thousand metric tons" (production),
    RESERVE_NOTES="Iron Content, million metric tons" (reserves). Reading
    only UNIT_MEAS underreports iron-content reserves by 1000× and put
    Steel above the global-reserves line in the cumulative-demand-vs-reserves
    panel of the global supply-tier figure, now Figure 5 (audit 2026-04-29,
    then panel D of the combined 4-panel figure).

    Recognized units (in either column):
      million metric tons                              → ×1000
      thousand metric tons / thousand metric dry tons  → as-is
      metric tons                                      → /1000
      kilograms                                        → /1_000_000
    """
    if pd.isna(value):
        return np.nan
    notes = str(reserve_notes).lower() if reserve_notes and pd.notna(reserve_notes) else ""
    unit  = str(unit_str).lower() if unit_str else ""
    # Prefer notes; fall back to UNIT_MEAS.
    for src in (notes, unit):
        if "million metric" in src:
            return float(value) * 1000.0
        if "thousand metric" in src:
            return float(value)
        if "metric ton" in src:
            return float(value) / 1000.0
        if "kilogram" in src:
            return float(value) / 1_000_000.0
    return float(value)  # fall back — treat as kt if unit string is unfamiliar


def augment_reserves_from_usgs_world_data(risk, world_data_path=USGS_WORLD_DATA_CSV):
    """Augment `risk['reserves']` in place with per-country reserves from
    USGS MCS 2025 World Data, for materials where the legacy
    risk_charts_inputs.xlsx reserves sheet has no data but USGS publishes
    it directly or via a documented ore-to-metal conversion.

    Applied to:
      - Indium, Selenium, Tellurium (direct metal-content reserves)
      - Steel (via iron-content reserves from Iron Ore commodity)
      - Aluminum (via Bauxite × 0.25 Bayer+Hall-Héroult yield, per
        USGS MCS 2025 Bauxite and Alumina metadata)

    Units: all values are written in kt (thousand metric tons) to match
    the existing sheet convention. USGS country names are joined to the
    reserves sheet country column ('Unnamed: 0') by exact string match;
    unmatched USGS countries are aggregated into 'Other'.

    After augmentation, `risk['reserves']` has additional non-zero columns
    for the augmented materials (or existing zero columns are overwritten).
    """
    if not Path(world_data_path).exists():
        return risk  # silent no-op if USGS CSV is missing

    wd = pd.read_csv(world_data_path, low_memory=False)
    wd["COMMODITY"] = wd["COMMODITY"].astype(str).str.strip()
    wd["COUNTRY"]   = wd["COUNTRY"].astype(str).str.strip()

    res = risk["reserves"].copy()
    country_col = res.columns[0]  # typically "Unnamed: 0"

    for our_name, (usgs_name, type_filter, factor) in _USGS_RESERVES_AUGMENT.items():
        sub = wd[wd["COMMODITY"] == usgs_name]
        if type_filter is not None:
            sub = sub[sub["TYPE"].astype(str).str.contains(type_filter, case=False, na=False)]
        if sub.empty:
            continue

        # Build per-country reserves in kt (metal-equivalent).
        country_reserves_kt = {}
        for _, row in sub.iterrows():
            country = row["COUNTRY"]
            if country in ("World total (rounded)", "nan"):
                continue
            res_raw = row.get("RESERVES_2024")
            res_kt = _usgs_world_to_kt(
                res_raw, row.get("UNIT_MEAS"), row.get("RESERVE_NOTES")
            )
            if pd.isna(res_kt) or res_kt <= 0:
                continue
            metal_kt = res_kt * factor
            # Normalize USGS country-name trailing whitespace to match the
            # legacy sheet (e.g., 'Canada ' vs 'Canada').
            country_norm = country.rstrip()
            country_reserves_kt[country_norm] = (
                country_reserves_kt.get(country_norm, 0.0) + metal_kt
            )

        # Aggregate 'Other Countries' into existing 'Other' row if present,
        # otherwise leave under its own name.
        if "Other Countries" in country_reserves_kt and "Other" in res[country_col].values:
            country_reserves_kt["Other"] = country_reserves_kt.get("Other", 0.0) \
                                         + country_reserves_kt.pop("Other Countries")

        # Upsert the column.
        if our_name not in res.columns:
            res[our_name] = 0.0
        else:
            res[our_name] = pd.to_numeric(res[our_name], errors="coerce").fillna(0.0)

        for country, kt in country_reserves_kt.items():
            mask = res[country_col].astype(str).str.strip() == country
            if mask.any():
                res.loc[mask, our_name] = float(kt)
            else:
                # Country not in the legacy sheet — append a new row with
                # zeros for all other materials and the reserves_kt here.
                new_row = {c: 0.0 for c in res.columns if c != country_col}
                new_row[country_col] = country
                new_row[our_name] = float(kt)
                res = pd.concat([res, pd.DataFrame([new_row])], ignore_index=True)

        # Ensure there is a 'Global' row that matches the USGS world total.
        # USGS World Data often has two "World total (rounded)" rows per
        # commodity (one for production stats, one for reserves); pick
        # whichever has a non-null RESERVES_2024.
        world_rows = sub[
            sub["COUNTRY"].astype(str).str.startswith("World total")
            & sub["RESERVES_2024"].notna()
        ]
        if not world_rows.empty:
            wr = world_rows.iloc[0]
            world_val = _usgs_world_to_kt(
                wr.get("RESERVES_2024"), wr.get("UNIT_MEAS"), wr.get("RESERVE_NOTES")
            )
            if pd.notna(world_val):
                metal_world_kt = world_val * factor
                gmask = res[country_col].astype(str).str.strip() == "Global"
                if gmask.any():
                    res.loc[gmask, our_name] = float(metal_world_kt)

    risk["reserves"] = res
    return risk

# ── CRC colour scheme (matches manuscript) ────────────────────────────────────
CRC_ORDER = ["United States", "OECD", 1, 2, 3, 4, 5, 6, 7, "China", "Undefined"]
CRC_COLORS = {
    "United States": "#1f77b4",   # Blue
    "OECD": "#2ca02c",            # Green
    1: "#98df8a",                  # Light green
    2: "#c0ca33",                  # Yellow-green (shifted from #d4e157)
    3: "#fdd835",                  # Golden yellow (shifted from #ffeb3b)
    4: "#ffb300",                  # Amber (shifted from #ffca28 for wider hue gap)
    5: "#fb8c00",                  # Deep orange (shifted from #ffa726)
    6: "#f4511e",                  # Red-orange (shifted from #ff7043)
    7: "#e53935",                  # Red
    "China": "#880e4f",            # Dark magenta (was #d62728, now distinct from CRC 7)
    "Undefined": "#999999",
}
CRC_LABELS = {
    "United States": "US domestic", "OECD": "OECD",
    1: "CRC 1", 2: "CRC 2", 3: "CRC 3", 4: "CRC 4",
    5: "CRC 5", 6: "CRC 6", 7: "CRC 7",
    "China": "China", "Undefined": "Undefined",
}


def load_data(augment_usgs_reserves=True):
    """Load the full risk data bundle from raw USGS MCS 2025 CSVs +
    Census Bureau International Trade API + OECD CRC 2026.

    The legacy risk_charts_inputs.xlsx was retired 2026-04-19; its six
    hand-compiled sheets are now rebuilt from machine-readable sources
    by `load_risk_data_mcs2025()`, which also extends coverage to the
    thin-film byproducts (Indium, Gallium, Germanium, Tellurium,
    Selenium, Cadmium) that the old xlsx omitted.

    The `augment_usgs_reserves` flag is retained for API compatibility.
    It now layers the Aluminum-via-bauxite and Steel-via-iron-content
    conversions on top of the MCS-built reserves sheet (the other
    augmentations — Indium/Selenium/Tellurium — are already handled
    natively by `_build_reserves_sheet`).
    """
    demand = pd.read_csv(DEMAND_FILE)
    risk = load_risk_data_mcs2025()
    if augment_usgs_reserves:
        augment_reserves_from_usgs_world_data(risk)
    return demand, risk


def _get_crc_map(risk):
    """Return a 2-column country -> CRC lookup (the first two columns of the
    OECD CRC sheet, renamed to ['country', 'crc'])."""
    crc = risk["crc"].iloc[:, :2].copy()
    crc.columns = ["country", "crc"]
    return crc


def _get_import_shares_by_crc(risk):
    """Return DataFrame: material, crc, share (% of imports)."""
    imp = risk["import_shares"].copy()
    crc = _get_crc_map(risk)
    merged = imp.merge(crc, on="country", how="left")
    merged.loc[merged["country"] == "China", "crc"] = "China"
    grouped = merged.groupby(["material", "crc"])["share"].sum().reset_index()
    return grouped


def parse_usgs_nir_cell(cell):
    """Interpret a single USGS import_dependency cell per MCS notation.
    Returns a numeric NIR in percent-points (0-100), or None if unusable.
      "E"    → 0.0   (Net Exporter — official USGS flag)
      "--"   → 0.0   (Not applicable / no imports)
      ""/NaN → None  (no data for this year; skip, don't assume anything)
      "<N"   → N/2   (upper-bounded range midpoint)
      ">N"   → (N+100)/2 (lower-bounded range midpoint)
      numeric → float
    """
    if pd.isna(cell):
        return None
    s = str(cell).strip()
    if s == "":
        return None
    if s in ("E", "e"):
        return 0.0
    if s == "--":
        return 0.0
    if s.startswith("<"):
        try:
            return float(s[1:]) / 2.0
        except ValueError:
            return None
    if s.startswith(">"):
        try:
            return (float(s[1:]) + 100.0) / 2.0
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


def nir_from_aggregate(risk, material):
    """Fallback NIR computed from aggregate trade balance:
        NIR = max(0, import - export) / consumption
    Handles the withheld-production case (e.g., Boron) by deferring to
    the observed trade direction: if exports > imports, the country is a
    net exporter (NIR = 0) regardless of withheld production.
    Returns a fraction in [0, 1] or None if insufficient data.
    """
    if "aggregate" not in risk:
        return None
    agg = risk["aggregate"]
    sub = agg[agg["material"] == material]
    if sub.empty:
        return None

    def _safe_mean(col):
        if col not in sub.columns:
            return None
        vals = pd.to_numeric(sub[col], errors="coerce").dropna()
        return float(vals.mean()) if len(vals) else None

    imp  = _safe_mean("import") or 0.0
    exp  = _safe_mean("export") or 0.0
    prod = _safe_mean("production")

    # Net exporter detectable from trade alone: exports strictly exceed
    # imports → NIR = 0 regardless of withheld / unknown production.
    if exp > imp:
        return 0.0

    if prod is None:
        return None
    consumption = prod + imp - exp
    if consumption <= 0:
        return 0.0
    return max(0.0, imp - exp) / consumption


def _get_import_dependency(risk):
    """Return Series: material → avg NIR in [0, 1].

    Source priority (per project CLAUDE.md NIR priority, revised 2026-04-08):
      1. USGS-published `import_dependency` sheet, correctly interpreting
         "E"/"--"/"<N"/">N" notation (net-exporter and range flags).
      2. Aggregate-sheet trade-balance fallback when the published sheet
         has no usable values (e.g., a full row of blanks).

    A material with a row of all "E" (e.g., Molybdenum, Boron — both
    documented US net exporters) is now correctly reported as NIR = 0.0
    rather than silently dropped and defaulted to 1.0 downstream.
    """
    imp_df = risk["import_dependency"]
    year_cols = [c for c in imp_df.columns if c != "material"]
    deps = {}
    for _, row in imp_df.iterrows():
        mat = row["material"]
        vals = [v for v in (parse_usgs_nir_cell(row[yc]) for yc in year_cols)
                if v is not None]
        if vals:
            deps[mat] = np.mean(vals) / 100.0
            continue
        # No published NIR — try the aggregate trade-balance fallback.
        agg_nir = nir_from_aggregate(risk, mat)
        if agg_nir is not None:
            deps[mat] = agg_nir
    return pd.Series(deps)


def _get_aggregate_stats(risk):
    """Return DataFrames for production, consumption, net_import by material."""
    agg = risk["aggregate"].copy()
    for col in ["production", "import", "export", "consumption", "net_import"]:
        if col in agg.columns:
            agg[col] = pd.to_numeric(agg[col], errors="coerce")
    return agg.groupby("material").mean()


def _get_reserves_by_crc(risk):
    """Return DataFrame: material, crc, reserves_kt."""
    reserves_df = risk["reserves"]
    crc = _get_crc_map(risk)
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]

    skip = {"Global", "Other"}
    country_res = reserves_df[~reserves_df["Unnamed: 0"].isin(skip)].copy()
    country_res = country_res.rename(columns={"Unnamed: 0": "country"})
    country_res = country_res.merge(crc, on="country", how="left")
    country_res.loc[country_res["country"] == "China", "crc"] = "China"
    country_res.loc[country_res["country"] == "United States", "crc"] = "United States"

    records = []
    for mat in mat_cols:
        country_res[mat] = pd.to_numeric(country_res[mat], errors="coerce").fillna(0)
        for crc_val, grp in country_res.groupby("crc"):
            records.append({
                "material": mat, "crc": crc_val,
                "reserves_kt": grp[mat].sum(),
            })
    return pd.DataFrame(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 3: Demand with CRC sourcing breakdown
# ═══════════════════════════════════════════════════════════════════════════════

def fig3_demand_sourcing(demand, risk):
    """
    For each material with risk data, show peak demand across scenarios
    broken down by CRC sourcing category (assuming historical import shares).
    Overlays horizontal lines for US production and consumption baselines.
    """
    import_shares_crc = _get_import_shares_by_crc(risk)
    import_dep = _get_import_dependency(risk)
    agg_stats = _get_aggregate_stats(risk)

    # Peak annual demand per material (across all scenarios and years).
    # Use `_annual` columns for cadence-invariance under PCHIP (default
    # 2026-05-04) and legacy 3-yr-bucket modes.
    peak_col = "mean_annual" if "mean_annual" in demand.columns else "mean"
    peak_demand = demand.groupby("material")[peak_col].max()

    # Only plot materials with risk data
    materials = sorted(set(DEMAND_TO_RISK.keys()) & set(peak_demand.index))
    # Map demand names to risk names for lookup
    risk_names = {m: DEMAND_TO_RISK[m] for m in materials}

    # Compute CRC allocation for each material
    plot_data = []
    for mat in materials:
        rn = risk_names[mat]
        dep = import_dep.get(rn, 1.0)
        peak = peak_demand.get(mat, 0)

        # Domestic share
        domestic_share = 1.0 - dep

        # Import CRC shares for this material
        mat_shares = import_shares_crc[import_shares_crc["material"] == rn]
        share_dict = dict(zip(mat_shares["crc"], mat_shares["share"]))
        total_import_share = sum(share_dict.values())

        for crc_cat in CRC_ORDER:
            if crc_cat == "United States":
                val = peak * domestic_share
            else:
                raw = share_dict.get(crc_cat, 0)
                frac = (raw / total_import_share * dep) if total_import_share > 0 else 0
                val = peak * frac
            plot_data.append({"material": mat, "crc": crc_cat, "demand": val})

    df = pd.DataFrame(plot_data)

    # Sort materials by peak demand
    mat_order = peak_demand.reindex(materials).sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(8, len(mat_order) * 0.4)))

    bottoms = {m: 0 for m in mat_order}
    for crc_cat in CRC_ORDER:
        vals = []
        for m in mat_order:
            row = df[(df["material"] == m) & (df["crc"] == crc_cat)]
            vals.append(row["demand"].values[0] if len(row) > 0 else 0)
        ax.barh(
            mat_order, vals, left=[bottoms[m] for m in mat_order],
            color=CRC_COLORS.get(crc_cat, "#cccccc"),
            label=CRC_LABELS.get(crc_cat, str(crc_cat)),
            edgecolor="white", linewidth=0.3,
        )
        for i, m in enumerate(mat_order):
            bottoms[m] += vals[i]

    # Overlay baseline markers
    for m in mat_order:
        rn = risk_names.get(m, m)
        if rn in agg_stats.index:
            prod = agg_stats.loc[rn, "production"]
            cons = agg_stats.loc[rn, "consumption"]
            if pd.notna(prod) and prod > 0:
                ax.plot(prod, m, "k|", markersize=12, markeredgewidth=2)
            if pd.notna(cons) and cons > 0:
                ax.plot(cons, m, "kx", markersize=8, markeredgewidth=1.5)

    ax.set_xlabel("Material demand (tonnes)")
    ax.set_title("Peak demand by CRC sourcing category\n"
                 "(assuming historical import shares persist)")
    ax.set_xscale("log")

    # Legend
    handles = [mpatches.Patch(color=CRC_COLORS[c], label=CRC_LABELS[c]) for c in CRC_ORDER]
    handles.append(plt.Line2D([], [], color="k", marker="|", linestyle="None",
                              markersize=10, markeredgewidth=2, label="US production"))
    handles.append(plt.Line2D([], [], color="k", marker="x", linestyle="None",
                              markersize=8, markeredgewidth=1.5, label="US consumption"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"fig3_demand_sourcing.{fmt}", dpi=FIGURE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig3_demand_sourcing")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 4: Reserve adequacy by CRC category
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_reserve_adequacy(demand, risk):
    """
    For each material, show reserve coverage ratio (reserves / cumulative demand 2026-2050),
    broken down by CRC category of where reserves are located.
    Uses the MAXIMUM demand scenario for each material independently.

    Rationale: The 61 NREL scenarios are not equally weighted, so the median
    across scenarios is not meaningful.  The maximum-demand framing answers
    "can reserves meet demand even under the most material-intensive pathway?"

    Values > 1 mean reserves exceed projected demand; < 1 means reserves insufficient.
    """
    reserves_crc = _get_reserves_by_crc(risk)

    # Calculate cumulative demand 2026-2050 per scenario
    demand_2026_2050 = demand[(demand["year"] >= 2026) & (demand["year"] <= 2050)]
    scenario_totals = demand_2026_2050.groupby(["scenario", "material"])["mean"].sum().unstack(fill_value=0)

    # Worst case: highest cumulative demand across all scenarios, per material
    max_demand = scenario_totals.max()
    max_scenario = scenario_totals.idxmax()

    materials = sorted(set(DEMAND_TO_RISK.keys()) & set(max_demand.index))
    risk_names = {m: DEMAND_TO_RISK[m] for m in materials}

    plot_data = []
    scenario_labels = {}
    for mat in materials:
        rn = risk_names[mat]
        cum_dem = max_demand.get(mat, 0)
        if cum_dem == 0:
            continue
        # Skip materials where USGS does not report reserves (all zeros).
        # These are either reported under a different commodity (e.g. bauxite
        # for aluminum, iron ore for steel) or are effectively unlimited
        # (cement from limestone, silicon from silica, magnesium from seawater).
        mat_res = reserves_crc[reserves_crc["material"] == rn]
        if mat_res["reserves_kt"].sum() == 0:
            continue
        scenario_labels[mat] = max_scenario.get(mat, "unknown")
        for _, row in mat_res.iterrows():
            coverage = (row["reserves_kt"] * 1000) / cum_dem
            plot_data.append({
                "material": mat, "crc": row["crc"], "coverage": coverage,
            })

    df = pd.DataFrame(plot_data)
    if df.empty:
        print("  No reserve data available for plotting.")
        return

    # Total coverage per material for sorting (ascending = most constrained first)
    total_coverage = df.groupby("material")["coverage"].sum().sort_values(ascending=True)
    mat_order = total_coverage.index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(8, len(mat_order) * 0.4)))

    bottoms = {m: 0 for m in mat_order}
    for crc_cat in CRC_ORDER:
        vals = []
        for m in mat_order:
            row = df[(df["material"] == m) & (df["crc"] == crc_cat)]
            vals.append(row["coverage"].values[0] if len(row) > 0 else 0)
        ax.barh(
            mat_order, vals, left=[bottoms[m] for m in mat_order],
            color=CRC_COLORS.get(crc_cat, "#cccccc"),
            label=CRC_LABELS.get(crc_cat, str(crc_cat)),
            edgecolor="white", linewidth=0.3,
        )
        for i, m in enumerate(mat_order):
            bottoms[m] += vals[i]

    # Add vertical line at coverage = 1 (reserves = cumulative demand)
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="Reserves = Demand")

    ax.set_xlabel("Reserve coverage ratio (reserves / cumulative demand 2026-2050)")
    ax.set_title("Economic reserve adequacy by CRC category — maximum demand scenario\n"
                 "(global reserves ÷ highest-demand scenario per material, n=61 scenarios)")
    ax.set_xscale("log")
    ax.set_xlim(0.1, 1e7)

    handles = [mpatches.Patch(color=CRC_COLORS[c], label=CRC_LABELS[c]) for c in CRC_ORDER]
    handles.append(plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Reserves = Demand"))
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"fig4_reserve_adequacy.{fmt}", dpi=FIGURE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_reserve_adequacy")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 4b: US-only reserve adequacy
# ═══════════════════════════════════════════════════════════════════════════════

def fig4_reserve_adequacy_us(demand, risk):
    """
    For each material, show US reserve coverage ratio (US reserves / cumulative demand 2026-2050).
    Simple horizontal bar chart showing domestic reserve adequacy.
    Uses maximum demand scenario per material.

    Values > 1 mean US reserves exceed projected demand; < 1 means US reserves insufficient.
    """
    # Get US reserves from the reserves sheet
    reserves_df = risk["reserves"]
    us_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
    if us_row.empty:
        print("  No US reserve data found.")
        return

    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]
    us_reserves = {}
    for mat in mat_cols:
        val = pd.to_numeric(us_row[mat].values[0], errors="coerce")
        us_reserves[mat] = val * 1000 if pd.notna(val) else 0  # kt → tonnes

    # Calculate cumulative demand 2026-2050 — worst-case scenario per material
    demand_2026_2050 = demand[(demand["year"] >= 2026) & (demand["year"] <= 2050)]
    scenario_totals = demand_2026_2050.groupby(["scenario", "material"])["mean"].sum().unstack(fill_value=0)
    cumulative_demand = scenario_totals.max()

    materials = sorted(set(DEMAND_TO_RISK.keys()) & set(cumulative_demand.index))
    risk_names = {m: DEMAND_TO_RISK[m] for m in materials}

    # Check which materials have any global reserves reported (same logic as fig4)
    reserves_df_all = risk["reserves"]
    mat_cols_all = [c for c in reserves_df_all.columns if c != "Unnamed: 0"]
    global_row = reserves_df_all[reserves_df_all["Unnamed: 0"] == "Global"]
    has_global_reserves = set()
    for mc in mat_cols_all:
        val = pd.to_numeric(global_row[mc].values[0], errors="coerce") if len(global_row) else 0
        if pd.notna(val) and val > 0:
            has_global_reserves.add(mc)

    plot_data = []
    for mat in materials:
        rn = risk_names[mat]
        cum_dem = cumulative_demand.get(mat, 0)
        if cum_dem == 0:
            continue
        # Skip materials where USGS does not report reserves at all
        if rn not in has_global_reserves:
            continue
        us_res = us_reserves.get(rn, 0)
        coverage = us_res / cum_dem
        plot_data.append({
            "material": mat,
            "coverage": coverage,
            "us_reserves_t": us_res,
            "cumulative_demand_t": cum_dem,
        })

    df = pd.DataFrame(plot_data)
    if df.empty:
        print("  No US reserve data available for plotting.")
        return

    # Sort by coverage (ascending = most constrained first)
    df = df.sort_values("coverage", ascending=True)
    mat_order = df["material"].tolist()

    fig, ax = plt.subplots(figsize=(12, max(8, len(mat_order) * 0.4)))

    # Color bars by adequacy: red if < 1, amber if 1-10, green if > 10
    colors = []
    for cov in df["coverage"]:
        if cov < 0.01:
            colors.append("#d62728")  # Red - essentially no reserves
        elif cov < 1:
            colors.append("#ff7f0e")  # Orange - insufficient
        elif cov < 10:
            colors.append("#ffbb78")  # Light orange - marginal
        else:
            colors.append("#2ca02c")  # Green - adequate

    ax.barh(mat_order, df["coverage"], color=colors, edgecolor="white", linewidth=0.3)

    # Add vertical line at coverage = 1 (reserves = cumulative demand)
    ax.axvline(x=1, color="red", linestyle="--", linewidth=1.5, label="Reserves = Demand")

    ax.set_xlabel("US reserve coverage ratio (US reserves / cumulative demand 2026-2050)")
    ax.set_title("US domestic reserve adequacy — maximum demand scenario\n"
                 "(US reserves ÷ highest-demand scenario per material through 2050)")
    ax.set_xscale("log")

    # Custom legend for colors
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#d62728", label="No reserves (< 1%)"),
        Patch(facecolor="#ff7f0e", label="Insufficient (< 100%)"),
        Patch(facecolor="#ffbb78", label="Marginal (100-1000%)"),
        Patch(facecolor="#2ca02c", label="Adequate (> 1000%)"),
        plt.Line2D([0], [0], color="red", linestyle="--", linewidth=1.5, label="Reserves = Demand"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"fig4_reserve_adequacy_us.{fmt}", dpi=FIGURE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig4_reserve_adequacy_us")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig. SI: Production shares by CRC category
# ═══════════════════════════════════════════════════════════════════════════════

def _get_production_by_crc(risk):
    """Return DataFrame: material, crc, production_kt.

    Migrated 2026-04-29 from the legacy wide-format `risk_charts_inputs.xlsx`
    schema (where the country column was unnamed → 'Unnamed: 0' after
    pandas read) to the long-format USGS MCS 2025 schema returned by
    `usgs_mcs2025_loader.load_risk_data_mcs2025()`. The new schema has
    columns [material, country, production_2023, production_2024]. We use
    production_2024 as the canonical year (most recent reported); 2023
    falls back when 2024 is null. Caller-side dedup of duplicated material
    names (formerly required by the wide-format reader) is no longer
    necessary because the long format has no duplicate columns.
    """
    production_df = risk["production"].copy()
    crc = _get_crc_map(risk)

    # Use 2024 production where available, fall back to 2023 — the long-format
    # CSV reports both years; manuscript convention is "most recent reported".
    if "production_2024" in production_df.columns:
        production_df["production_kt"] = production_df["production_2024"].fillna(
            production_df.get("production_2023")
        )
    elif "production_2023" in production_df.columns:
        production_df["production_kt"] = production_df["production_2023"]
    else:
        # Legacy fallback: assume the 4th column is the production value.
        # Should never trigger with the current loader.
        production_df["production_kt"] = production_df.iloc[:, -1]

    # Drop the global/aggregate rows; keep per-country only.
    skip = {"Global", "World", "Other", "World total"}
    country_prod = production_df[~production_df["country"].isin(skip)].copy()

    # Coerce to numeric; missing → 0.
    country_prod["production_kt"] = pd.to_numeric(
        country_prod["production_kt"], errors="coerce"
    ).fillna(0)

    # Attach CRC classification per-country.
    country_prod = country_prod.merge(crc, on="country", how="left")
    country_prod.loc[country_prod["country"] == "China", "crc"] = "China"
    country_prod.loc[country_prod["country"] == "United States", "crc"] = "United States"

    # Fill OECD bucket where CRC is missing for a known OECD member.
    oecd_countries = [
        "Australia", "Austria", "Belgium", "Canada", "Chile", "Colombia", "Costa Rica",
        "Czech Republic", "Denmark", "Estonia", "Finland", "France", "Germany", "Greece",
        "Hungary", "Iceland", "Ireland", "Israel", "Italy", "Japan", "South Korea",
        "Latvia", "Lithuania", "Luxembourg", "Mexico", "Netherlands", "New Zealand",
        "Norway", "Poland", "Portugal", "Slovakia", "Slovenia", "Spain", "Sweden",
        "Switzerland", "Turkey", "United Kingdom"
    ]
    for oc in oecd_countries:
        mask = country_prod["country"] == oc
        if mask.any() and pd.isna(country_prod.loc[mask, "crc"]).any():
            country_prod.loc[mask, "crc"] = "OECD"

    # Group: long-format makes this a simple groupby.
    return (
        country_prod.groupby(["material", "crc"], dropna=False)["production_kt"]
        .sum()
        .reset_index()
    )


def figSI_production_shares_crc(risk):
    """
    SI Figure: Production shares by CRC category.
    Shows global production distribution by country risk classification.
    """
    production_crc = _get_production_by_crc(risk)

    if production_crc.empty:
        print("  No production data available for plotting.")
        return

    # Deduplicate material names that arise from duplicate Excel columns
    # (e.g., "Cement" and "Cement.1" from pandas reading duplicate headers)
    production_crc["material"] = production_crc["material"].str.replace(
        r"\.\d+$", "", regex=True
    )
    # Re-aggregate after dedup
    production_crc = (
        production_crc.groupby(["material", "crc"])["production_kt"]
        .sum()
        .reset_index()
    )

    # Calculate shares per material
    total_prod = production_crc.groupby("material")["production_kt"].sum()

    # Filter to materials with meaningful production data
    materials_with_data = total_prod[total_prod > 0].index.tolist()
    production_crc = production_crc[production_crc["material"].isin(materials_with_data)]

    if production_crc.empty:
        print("  No production share data after filtering.")
        return

    # Calculate percentage shares
    production_crc = production_crc.merge(
        total_prod.reset_index().rename(columns={"production_kt": "total_kt"}),
        on="material"
    )
    production_crc["share"] = production_crc["production_kt"] / production_crc["total_kt"] * 100

    # Sort materials by total production
    mat_order = total_prod.reindex(materials_with_data).sort_values(ascending=True).index.tolist()

    fig, ax = plt.subplots(figsize=(14, max(8, len(mat_order) * 0.35)))

    bottoms = {m: 0 for m in mat_order}
    for crc_cat in CRC_ORDER:
        vals = []
        for m in mat_order:
            row = production_crc[(production_crc["material"] == m) & (production_crc["crc"] == crc_cat)]
            vals.append(row["share"].values[0] if len(row) > 0 else 0)
        ax.barh(
            mat_order, vals, left=[bottoms[m] for m in mat_order],
            color=CRC_COLORS.get(crc_cat, "#cccccc"),
            label=CRC_LABELS.get(crc_cat, str(crc_cat)),
            edgecolor="white", linewidth=0.3,
        )
        for i, m in enumerate(mat_order):
            bottoms[m] += vals[i]

    ax.set_xlabel("Share of Global Production (%)")
    ax.set_title("Global production distribution by Country Risk Classification\n"
                 "(% of world production by country risk category)")
    ax.set_xlim(0, 100)

    handles = [mpatches.Patch(color=CRC_COLORS[c], label=CRC_LABELS[c]) for c in CRC_ORDER]
    ax.legend(handles=handles, loc="lower right", fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2, axis="x")

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"figSI_production_shares_crc.{fmt}", dpi=FIGURE_DPI,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved figSI_production_shares_crc")


# ═══════════════════════════════════════════════════════════════════════════════
# Fig. 5A: Global supply risk matrix (scatter: reserve adequacy vs production stress)
# ═══════════════════════════════════════════════════════════════════════════════

# Materials whose reserves are NOT separately reported by USGS
# (reported under parent commodity or effectively unlimited)
_NO_SEPARATE_RESERVES = {"Aluminum", "Silicon", "Steel", "Yttrium", "Cement"}

# Materials excluded from risk matrix (no US consumption data)
_EXCLUDED_NO_CONSUMPTION = {"Boron", "Magnesium"}

# Rare earth elements to aggregate
_REE_ELEMENTS = {"Dysprosium", "Neodymium", "Praseodymium", "Terbium"}


def _prepare_risk_matrix_data(demand, risk, use_us_reserves=False):
    """
    Compute (reserve_adequacy, production_stress, import_reliance) per material.

    Parameters
    ----------
    demand : DataFrame
        Material demand data.
    risk : dict
        Risk data sheets.
    use_us_reserves : bool
        If True, use US reserves; otherwise global reserves.

    Returns
    -------
    DataFrame with columns: material, reserve_adequacy, production_stress,
        import_reliance, has_reserves, has_separate_reserves
    """
    import_dep = _get_import_dependency(risk)
    agg_stats = _get_aggregate_stats(risk)
    reserves_df = risk["reserves"]
    mat_cols = [c for c in reserves_df.columns if c != "Unnamed: 0"]

    # ── Get reserves ──
    if use_us_reserves:
        res_row = reserves_df[reserves_df["Unnamed: 0"] == "United States"]
    else:
        res_row = reserves_df[reserves_df["Unnamed: 0"] == "Global"]

    reserves = {}
    for mc in mat_cols:
        val = pd.to_numeric(res_row[mc].values[0], errors="coerce") if len(res_row) else 0
        reserves[mc] = (val * 1000) if pd.notna(val) else 0  # kt → tonnes

    # ── Peak annual demand and cumulative demand per material ──
    # Peak uses `_annual` (annual rate, cadence-invariant). Cumulative
    # uses raw `mean` (period totals — sum is total tonnes under both
    # PCHIP-annual and legacy-3yr-bucket cadences).
    peak_col = "mean_annual" if "mean_annual" in demand.columns else "mean"
    peak_demand = demand.groupby("material")[peak_col].max()
    demand_2026_2050 = demand[(demand["year"] >= 2026) & (demand["year"] <= 2050)]
    cum_demand = demand_2026_2050.groupby("material")["mean"].sum()
    # Use median scenario for cumulative
    scenario_totals = demand_2026_2050.groupby(["scenario", "material"])["mean"].sum().unstack(fill_value=0)
    cum_demand_median = scenario_totals.median()

    # ── Aggregate REEs ──
    ree_peak = sum(peak_demand.get(r, 0) for r in _REE_ELEMENTS)
    ree_cum = sum(cum_demand_median.get(r, 0) for r in _REE_ELEMENTS)

    # ── Build records ──
    materials = sorted(set(DEMAND_TO_RISK.keys()) & set(peak_demand.index))
    # Exclude materials with no consumption data
    materials = [m for m in materials if m not in _EXCLUDED_NO_CONSUMPTION]

    # De-duplicate REEs: skip individual REEs, add aggregate
    materials_dedup = [m for m in materials if m not in _REE_ELEMENTS]

    records = []

    for mat in materials_dedup:
        rn = DEMAND_TO_RISK[mat]
        pk = peak_demand.get(mat, 0)
        cum = cum_demand_median.get(mat, 0)
        if pk == 0 or cum == 0:
            continue

        # US apparent consumption
        consumption = agg_stats.loc[rn, "consumption"] if rn in agg_stats.index else np.nan
        if pd.isna(consumption) or consumption <= 0:
            continue

        prod_stress = pk / (consumption * 1000)  # consumption is in kt, demand in tonnes

        # Reserves
        res = reserves.get(rn, 0)
        has_separate = mat not in _NO_SEPARATE_RESERVES
        has_reserves = res > 0

        if has_separate and has_reserves:
            reserve_adequacy = res / cum
        elif not has_separate:
            # Place at far right (very high adequacy) for display
            reserve_adequacy = None  # will handle in plotting
        else:
            reserve_adequacy = None

        dep = import_dep.get(rn, 1.0)
        records.append({
            "material": mat, "reserve_adequacy": reserve_adequacy,
            "production_stress": prod_stress, "import_reliance": dep,
            "has_reserves": has_reserves, "has_separate_reserves": has_separate,
        })

    # Add aggregated REEs
    ree_risk_name = "Rare Earths"
    ree_consumption = agg_stats.loc[ree_risk_name, "consumption"] if ree_risk_name in agg_stats.index else np.nan
    ree_dep = import_dep.get(ree_risk_name, 1.0)
    ree_res = reserves.get(ree_risk_name, 0)
    if pd.notna(ree_consumption) and ree_consumption > 0 and ree_peak > 0:
        ree_prod_stress = ree_peak / (ree_consumption * 1000)  # kt → tonnes
        ree_adequacy = (ree_res / ree_cum) if ree_res > 0 else None
        records.append({
            "material": "REEs\n(Dy+Nd+Pr+Tb)",
            "reserve_adequacy": ree_adequacy,
            "production_stress": ree_prod_stress,
            "import_reliance": ree_dep,
            "has_reserves": ree_res > 0,
            "has_separate_reserves": True,
        })

    return pd.DataFrame(records)


def fig3_supply_risk_matrix(demand, risk):
    """
    Global supply-risk matrix scatter plot (supporting supply-chain view;
    filename keeps the earlier "fig3" prefix, see module docstring).
    X = reserve adequacy (global reserves / cumulative demand 2026-2050)
    Y = production stress (peak demand / current US consumption)
    Color = net import reliance
    Markers: circles = reserves reported, diamonds = reserves not separately reported
    """
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter

    df = _prepare_risk_matrix_data(demand, risk, use_us_reserves=False)
    if df.empty:
        print("  No data for risk matrix.")
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    # Colormap: green (0% import) → yellow → red (100% import)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "import_risk", ["#2ca02c", "#c0ca33", "#fdd835", "#ff7f0e", "#d62728", "#880e4f"]
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    # Stagger "no separate reserves" materials in the right margin
    _no_res_x = {
        "Silicon": 1500, "Aluminum": 3000, "Cement": 3000,
        "Steel": 2000, "Yttrium": 4000,
    }
    x_no_reserves_default = 5000

    # Short labels for materials
    label_map = {
        "Chromium": "Cr", "Molybdenum": "Mo", "Niobium": "Nb",
        "Aluminum": "Al", "Silicon": "Si", "Vanadium": "V",
        "Manganese": "Mn", "Yttrium": "Y",
    }

    for _, row in df.iterrows():
        if pd.notna(row["reserve_adequacy"]):
            x = row["reserve_adequacy"]
        else:
            x = _no_res_x.get(row["material"], x_no_reserves_default)
        y = row["production_stress"]
        color = cmap(norm(row["import_reliance"]))

        if not row["has_separate_reserves"]:
            marker = "D"
            size = 120
        else:
            marker = "o"
            size = 100

        ax.scatter(x, y, c=[color], s=size, marker=marker, edgecolors="gray",
                   linewidth=0.5, zorder=5)

        display = label_map.get(row["material"], row["material"])
        ax.annotate(display, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, ha="left", va="bottom")

    # Horizontal dashed line at 100% production stress
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(4.5, 1.05, "Peak energy demand = current US consumption",
            fontsize=7, color="gray", va="bottom")

    # Quadrant labels
    ax.text(0.02, 0.97, "Reserve- and\nproduction-\nconstrained",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="top")
    ax.text(0.35, 0.97, "Adequate reserves,\nproduction-constrained",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="top")
    ax.text(0.35, 0.03, "Low risk",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="bottom")
    ax.text(0.02, 0.03, "Reserve-limited",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="bottom")

    # "Reserves not separately reported" label at top right
    ax.text(0.88, 0.97, "Reserves not\nseparately reported",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="center", va="top")

    # Excluded note
    ax.text(0.01, 0.01, "Excluded (no US consumption data): Boron, Magnesium",
            transform=ax.transAxes, fontsize=7, color="gray", fontstyle="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v * 100:.0f}%" if v < 10 else f"{v * 100:,.0f}%"
    ))

    ax.set_xlabel("Reserve adequacy\n(global economic reserves \u00f7 cumulative energy "
                   "transition demand, 2026\u20132050)", fontsize=10)
    ax.set_ylabel("Production stress\n(peak annual energy demand \u00f7 current US "
                   "apparent consumption)", fontsize=10)
    ax.set_title("Material supply risk: geological adequacy vs. production capacity",
                 fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.15, which="both")
    ax.set_axisbelow(True)

    # Legend: marker shapes
    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="gray", label="Both axes reported"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="gray", label="Reserves not reported"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Net import reliance", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"fig3_supply_risk_matrix.{fmt}",
                    dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3_supply_risk_matrix")


def fig3b_us_supply_risk_matrix(demand, risk):
    """
    US domestic supply-risk matrix scatter plot (supporting supply-chain
    view; filename keeps the earlier "fig3b" prefix, see module docstring).
    Same as the global matrix above but X = US reserves / cumulative demand.
    Materials with zero US reserves shown as squares at left margin.
    """
    import matplotlib.colors as mcolors
    from matplotlib.ticker import FuncFormatter

    df = _prepare_risk_matrix_data(demand, risk, use_us_reserves=True)
    if df.empty:
        print("  No data for US risk matrix.")
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "import_risk", ["#2ca02c", "#c0ca33", "#fdd835", "#ff7f0e", "#d62728", "#880e4f"]
    )
    norm = mcolors.Normalize(vmin=0, vmax=1)

    _no_res_x_us = {
        "Silicon": 1500, "Aluminum": 3000, "Cement": 3000,
        "Steel": 2000, "Yttrium": 4000,
    }
    x_no_reserves_default = 5000
    x_zero_us = 0.03               # left margin for zero US reserves

    # Short labels for materials
    label_map = {
        "Chromium": "Cr", "Molybdenum": "Mo", "Niobium": "Nb",
        "Aluminum": "Al", "Silicon": "Si", "Vanadium": "V",
        "Manganese": "Mn", "Yttrium": "Y",
    }

    # Shaded region for zero US reserves
    ax.axvspan(0.015, 0.06, color="#f8d7da", alpha=0.4, zorder=0)
    ax.text(0.025, 0.98, "Zero US\nreserves", transform=ax.transAxes,
            fontsize=7, color="#999", fontstyle="italic", ha="left", va="top")

    # Vertical dashed line at adequacy = 1
    ax.axvline(x=1, color="green", linestyle="--", linewidth=1, alpha=0.6)
    ax.text(1.1, 0.02, "US reserves =\ncum. demand", fontsize=7, color="green",
            alpha=0.7, va="bottom", transform=ax.get_xaxis_transform())

    for _, row in df.iterrows():
        y = row["production_stress"]
        color = cmap(norm(row["import_reliance"]))

        if not row["has_separate_reserves"]:
            x = _no_res_x_us.get(row["material"], x_no_reserves_default)
            marker = "D"
            size = 120
        elif pd.isna(row["reserve_adequacy"]) or not row["has_reserves"]:
            x = x_zero_us
            marker = "s"
            size = 110
        else:
            x = row["reserve_adequacy"]
            marker = "o"
            size = 100

        ax.scatter(x, y, c=[color], s=size, marker=marker, edgecolors="gray",
                   linewidth=0.5, zorder=5)

        display = label_map.get(row["material"], row["material"])
        ax.annotate(display, (x, y), textcoords="offset points",
                    xytext=(8, 4), fontsize=8, ha="left", va="bottom")

    # Horizontal dashed line at 100% production stress
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(0.5, 1.05, "Peak energy demand = current US consumption",
            fontsize=7, color="gray", va="bottom",
            transform=ax.get_yaxis_transform())

    # Quadrant labels
    ax.text(0.12, 0.97, "Reserve- and\nproduction-\nconstrained",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="top")
    ax.text(0.45, 0.97, "Adequate reserves,\nproduction-constrained",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="top")
    ax.text(0.45, 0.03, "Low risk",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="left", va="bottom")

    # "Reserves not separately reported" label
    ax.text(0.88, 0.97, "Reserves not\nseparately reported",
            transform=ax.transAxes, fontsize=7, color="#999",
            fontstyle="italic", ha="center", va="top")

    ax.text(0.01, 0.01, "Excluded (no US consumption data): Boron, Magnesium",
            transform=ax.transAxes, fontsize=7, color="gray", fontstyle="italic")

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v * 100:.0f}%" if v < 10 else f"{v * 100:,.0f}%"
    ))

    ax.set_xlabel("US reserve adequacy\n(US economic reserves \u00f7 cumulative energy "
                   "transition demand, 2026\u20132050)", fontsize=10)
    ax.set_ylabel("Production stress\n(peak annual energy demand \u00f7 current US "
                   "apparent consumption)", fontsize=10)
    ax.set_title("US domestic supply risk: reserve adequacy vs. production capacity",
                 fontsize=13, fontweight="bold")

    ax.grid(True, alpha=0.15, which="both")
    ax.set_axisbelow(True)

    legend_elements = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="gray", label="US reserves reported"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="gray", label="Zero US reserves"),
        plt.Line2D([0], [0], marker="D", color="w", markerfacecolor="gray",
                   markersize=9, markeredgecolor="gray", label="Reserves not reported"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8, framealpha=0.9)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("Net import reliance", fontsize=9)
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
    cbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])

    fig.tight_layout()
    for fmt in FIGURE_FORMAT:
        fig.savefig(FIGURES_MANUSCRIPT_DIR / f"fig3b_us_supply_risk_matrix.{fmt}",
                    dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close(fig)
    print("  Saved fig3b_us_supply_risk_matrix")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # This module is a standalone supporting script: it is run directly to
    # render the alternate supply-chain views and is not invoked by
    # reproduce.py (the canonical supply-tier figures come from
    # fig4_fig5_supply_tiers.py). When run standalone, route output to the
    # supply_chain_charts/current/ subdirectory of the manuscript figure dir
    # (numbering-neutral name; was fig5_supply_chain/ under the old scheme).
    # If another caller imports these functions, it can set
    # FIGURES_MANUSCRIPT_DIR before calling them.
    FIGURES_MANUSCRIPT_DIR = FIGURES_MANUSCRIPT_DIR / "supply_chain_charts" / "current"
    FIGURES_MANUSCRIPT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    demand, risk = load_data()
    print(f"  {demand['material'].nunique()} materials, "
          f"{demand['scenario'].nunique()} scenarios")

    print("\nGenerating Fig. 3: Demand with CRC sourcing breakdown...")
    fig3_demand_sourcing(demand, risk)

    print("\nGenerating Fig. 4: Reserve adequacy by CRC category (maximum demand)...")
    fig4_reserve_adequacy(demand, risk)

    print("\nGenerating Fig. 4b: US-only reserve adequacy (maximum demand)...")
    fig4_reserve_adequacy_us(demand, risk)

    print("\nGenerating Fig. SI: Production shares by CRC category...")
    figSI_production_shares_crc(risk)

    print("\nGenerating Fig. 5A: Global supply risk matrix...")
    fig3_supply_risk_matrix(demand, risk)

    print("\nGenerating Fig. 5B: US domestic supply risk matrix...")
    fig3b_us_supply_risk_matrix(demand, risk)

    print("\nDone.")
