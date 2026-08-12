"""
Shared data library for the "material demand across technologies" figure.

INVENTORY:
  name: demand_by_technology
  output: outputs/data/demand_by_tech_shares_{cumulative,peak2035}.csv
  category: Data Export
  x_axis: n/a (shared data library, not a figure)
  y_axis: n/a
  color: n/a
  data_sources:
    - StdScen24_annual_national.csv (NREL Standard Scenarios 2024)
    - fitted_distributions.csv (mean intensity, t/MW)
    - material_demand_by_scenario.csv (published MC mean)
    - src/technology_mapping.py (TECHNOLOGY_MAPPING weights)
  description: >
    Shared library for the demand-by-technology figure family (opt_a/b/c).
    Decomposes each material's demand into per-technology shares for a
    reference scenario using the fitted-lognormal mean intensities the MC
    samples (validation ratios ~1.00 vs the published MC means; totals pinned
    exactly to published values). Display groups label the CCS/SMR-only
    technologies as such. Covers all 27 demand-bearing materials, including Glass and
    Fiberglass (criticality-ineligible but demand-bearing; added 2026-07-07).
    export_shares() writes the traceability CSV backing manuscript
    Figure S1 claims (registry IDs F4a-F4c) in
    Scientific_Draft/claims_registry.yaml.
    2026-08-07 numbering cleanup: docstrings updated from the old "Figure 4"
    numbering to Figure S1 (registry claim IDs F4a-F4c are unchanged).
    No rendering changes.
END_INVENTORY

"""
from __future__ import annotations

import sys
from pathlib import Path

import math
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "src"))

from technology_mapping import TECHNOLOGY_MAPPING  # noqa: E402

# ───────────────────────────────────────────────────────────────────────────
# PATHS
# ───────────────────────────────────────────────────────────────────────────
DATA_DIR = BASE_DIR / "data"
OUT_DATA = BASE_DIR / "outputs" / "data"
NREL_FILE = DATA_DIR / "StdScen24_annual_national.csv"
INTENSITY_FILE = OUT_DATA / "fitted_distributions.csv"
DEMAND_FILE = OUT_DATA / "material_demand_by_scenario.csv"
OUT_DIR = BASE_DIR / "outputs" / "figures" / "experimental" / "demand_by_tech"

from scenario_config import REFERENCE_SCENARIO  # noqa: E402
SCENARIO_LABEL = "Mid Case (with IRA)"
PEAK_YEAR = 2035

MODE_LABEL = {
    "cumulative": "cumulative 2026–2050",
    # Totals are pinned to published ANNUAL 2035 demand; the share weights
    # come from the 2032-2035 additions interval of the raw 3-yr NREL series
    # (shares are insensitive to the interval-vs-annual scale factor).
    "peak2035": f"annual {PEAK_YEAR} (shares from the 2032–{PEAK_YEAR} additions interval)",
}

# ───────────────────────────────────────────────────────────────────────────
# CATEGORIES (families mirror figS3_sensitivity_tornado.py, plus Glass and
# Fiberglass in Bulk commodities per fig2_demand_projections.py). This figure
# covers all 27 demand-bearing materials: Glass and Fiberglass are excluded
# from the criticality analysis (no critical-minerals listing) but bear
# demand, and a demand decomposition needs no criticality eligibility.
# ───────────────────────────────────────────────────────────────────────────
from material_classes import (  # noqa: E402
    CLASS_MEMBERS as _CLASS_MEMBERS, ZERO_DEMAND_MATERIALS)
ZERO_DEMAND = set(ZERO_DEMAND_MATERIALS)
EXCLUDED: set = set()  # was {Fiberglass, Glass} before 2026-07-07

MATERIAL_GROUPS = {c: list(ms) for c, ms in _CLASS_MEMBERS.items()}
FAMILY_ORDER = list(MATERIAL_GROUPS.keys())
MATERIAL_FAMILY = {m: fam for fam, mats in MATERIAL_GROUPS.items() for m in mats}

# Capacity technology -> display technology group.
DISPLAY_TECH = {
    "upv": "Utility solar PV",
    "distpv": "Distributed solar",
    "csp": "CSP",
    "wind_onshore": "Onshore wind",
    "wind_offshore": "Offshore wind",
    # Unabated coal, conventional nuclear, and unabated biomass have zero
    # additions in every scenario, so their display groups contain only the
    # CCS/SMR variants and are labeled accordingly. H2 combustion turbines
    # are grouped with natural gas (they carry natural-gas-turbine material
    # intensities; Table S2); pumped hydro is grouped with hydro.
    "nuclear": "Nuclear (SMR)", "nuclear_smr": "Nuclear (SMR)",
    "gas_cc": "Natural gas", "gas_ct": "Natural gas",
    "gas_cc_ccs": "Natural gas", "h2-ct": "Natural gas",
    "coal": "Coal CCS", "coal_ccs": "Coal CCS",
    "bio": "Biomass CCS", "bio-ccs": "Biomass CCS",
    "hydro": "Hydro", "pumped-hydro": "Hydro",
    "geo": "Geothermal",
}

# Stacking / legend order (variable renewables, firm low-carbon, fossil).
TECH_ORDER = [
    "Utility solar PV", "Distributed solar", "CSP",
    "Onshore wind", "Offshore wind",
    "Hydro", "Geothermal", "Biomass CCS", "Nuclear (SMR)",
    "Natural gas", "Coal CCS",
]
TECH_COLORS = {
    "Utility solar PV":  "#E8A33D",
    "Distributed solar": "#F6CE84",
    "CSP":               "#B5651D",
    "Onshore wind":      "#4FA3D1",
    "Offshore wind":     "#08519C",
    "Hydro":             "#66C2A5",
    "Geothermal":        "#D55E00",
    "Biomass CCS":       "#4DAF4A",
    "Nuclear (SMR)":     "#7B5AA6",
    "Natural gas":       "#9E9E9E",
    "Coal CCS":          "#4D4D4D",
}


# ───────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────
def fmt_tonnes(t: float) -> str:
    """Compact tonnage label (Mt / kt / t)."""
    if t >= 1e6:
        return f"{t / 1e6:.1f} Mt"
    if t >= 1e3:
        return f"{t / 1e3:.0f} kt"
    return f"{t:.0f} t"


def _mean_intensity() -> dict:
    """Mean intensity per (technology, material) as the MC samples it: the
    mean of the fitted lognormal, loc + scale * exp(s^2 / 2). All 167 pairs
    are parametric lognormals with loc = 0. By linearity of expectation, a
    decomposition built from these means reproduces the published MC-mean
    totals (the sample-mean column does not — the fits are median-preserving
    with borrowed CVs, so their means differ from sample means by up to
    ~2.5x for skew-heavy pairs like Praseodymium / onshore wind).
    """
    di = pd.read_csv(INTENSITY_FILE)
    return {
        (r.technology, r.material):
            float(r.param_loc) + float(r.param_scale) * math.exp(float(r.param_s) ** 2 / 2.0)
        for r in di.itertuples()
    }


def published_demand(scenario: str, mode: str) -> pd.Series:
    """Published MC-mean demand per material (tonnes) for the chosen mode.

    material_demand_by_scenario.csv is on the PCHIP annual grid (one row per
    year, mean == mean_annual). cumulative -> sum of annual values;
    peak2035 -> the annual 2035 value.
    """
    d = pd.read_csv(DEMAND_FILE)
    d = d[d["scenario"] == scenario]
    if mode == "cumulative":
        return d.groupby("material")["mean"].sum()
    if mode == "peak2035":
        return d[d["year"] == PEAK_YEAR].set_index("material")["mean"]
    raise ValueError(f"unknown mode {mode!r}")


def _additions(scenario: str, mode: str) -> dict:
    """Gross capacity additions (MW) per capacity technology for the mode."""
    nrel = pd.read_csv(NREL_FILE, skiprows=3).rename(columns={"t": "year"})
    g = nrel[nrel["scenario"] == scenario].sort_values("year").reset_index(drop=True)
    if g.empty:
        raise SystemExit(f"Scenario {scenario!r} not in {NREL_FILE.name}")
    add = {}
    for cap_tech in TECHNOLOGY_MAPPING:
        col = f"{cap_tech}_MW"
        if col not in g.columns:
            continue
        diffs = g[col].diff().fillna(0.0).clip(lower=0.0)
        if mode == "cumulative":
            total = float(diffs.sum())
        else:  # peak2035 = the single 2032->2035 step
            total = float(diffs[g["year"] == PEAK_YEAR].sum())
        if total > 0:
            add[cap_tech] = total
    return add


def compute_split(scenario: str = REFERENCE_SCENARIO,
                  mode: str = "cumulative",
                  verbose: bool = True) -> pd.DataFrame:
    """Long DataFrame: material, family, tech, demand_t.

    Technology shares come from decomposing with the fitted-lognormal mean
    intensities the MC samples, so by linearity of expectation the
    decomposition reproduces the published MC-mean totals (validation ratios
    ~1.00); each material's total is then pinned exactly to the published
    value to absorb residual sampling noise.
    """
    mean_int = _mean_intensity()
    add_mw = _additions(scenario, mode)
    pub = published_demand(scenario, mode)

    materials = sorted(
        m for m in pub.index
        if m not in ZERO_DEMAND and m not in EXCLUDED and m in MATERIAL_FAMILY
    )
    raw = {m: {t: 0.0 for t in TECH_ORDER} for m in materials}
    for cap_tech, mw in add_mw.items():
        disp = DISPLAY_TECH.get(cap_tech)
        if disp is None:
            continue
        for itech, weight in TECHNOLOGY_MAPPING[cap_tech].items():
            for m in materials:
                inten = mean_int.get((itech, m))
                if inten:
                    raw[m][disp] += mw * weight * inten

    if verbose:
        print(f"\nValidation (scenario={scenario}, mode={mode}): "
              "decomposition vs published MC mean")
        print(f"{'material':<14}{'decomp (t)':>14}{'published (t)':>16}{'ratio':>8}")
    rows = []
    for m in materials:
        raw_tot = sum(raw[m].values())
        pub_tot = float(pub.get(m, 0.0))
        if verbose:
            ratio = (raw_tot / pub_tot) if pub_tot else float("nan")
            print(f"{m:<14}{raw_tot:>14,.0f}{pub_tot:>16,.0f}{ratio:>8.2f}")
        scale = (pub_tot / raw_tot) if raw_tot > 0 else 0.0
        for t in TECH_ORDER:
            rows.append({"material": m, "family": MATERIAL_FAMILY[m],
                         "tech": t, "demand_t": raw[m][t] * scale})
    return pd.DataFrame(rows)


def ordered_materials(df: pd.DataFrame) -> list:
    """Materials grouped by family, sorted within family by total demand desc."""
    tot = df.groupby("material")["demand_t"].sum()
    out = []
    for fam in FAMILY_ORDER:
        fam_mats = [m for m in tot.index if MATERIAL_FAMILY.get(m) == fam]
        out += sorted(fam_mats, key=lambda m: tot[m], reverse=True)
    return out


def pivot(df: pd.DataFrame, materials: list) -> pd.DataFrame:
    p = df.pivot_table(index="material", columns="tech", values="demand_t",
                       aggfunc="sum", fill_value=0.0)
    return p.reindex(index=materials, columns=TECH_ORDER, fill_value=0.0)


def active_techs(p: pd.DataFrame, min_share: float = 0.0) -> list:
    """Technologies contributing anywhere; optionally require a min total share."""
    tot = p.sum().sum()
    keep = []
    for t in TECH_ORDER:
        if p[t].sum() > 0 and (min_share == 0.0 or p[t].sum() / tot >= min_share):
            keep.append(t)
    return keep


def export_shares(mode: str = "cumulative") -> Path:
    """Write the decomposition (tonnes + % share per material-tech) to CSV.

    Traceability export for manuscript Figure S1 claims (claims_registry.yaml
    IDs F4a-F4c): every number in the figure caption/prose traces to this file.
    """
    df = compute_split(mode=mode)
    tot = df.groupby("material")["demand_t"].transform("sum")
    df = df.assign(share_pct=100.0 * df["demand_t"] / tot, mode=mode)
    out = OUT_DATA / f"demand_by_tech_shares_{mode}.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out} ({df['material'].nunique()} materials)")
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Export demand-by-technology shares CSV")
    ap.add_argument("--mode", choices=["cumulative", "peak2035"], default="cumulative")
    args = ap.parse_args()
    export_shares(args.mode)
