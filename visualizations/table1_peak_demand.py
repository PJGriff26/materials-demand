#!/usr/bin/env python3
"""Manuscript Table 1 generator: peak annual demand by scenario.

For each demand-bearing material, pulls peak annual demand (argmax of
mean_annual over years) under the four highlighted scenarios plus the peak year,
straight from outputs/data/material_demand_by_scenario.csv. Writes
outputs/data/table1_peak_demand.{csv,md}.
"""
import argparse
import re
from pathlib import Path

import pandas as pd

# Repo-relative paths (this file lives in visualizations/, so the repository
# root is one level up). Reads the canonical demand CSV and writes the table
# next to the other results, in outputs/data/.
REPO_ROOT = Path(__file__).resolve().parent.parent
DEMAND_CSV = REPO_ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
OUT_DIR = REPO_ROOT / "outputs" / "data"

# Four highlighted scenarios -> display label (order = table column order).
SCENARIOS = [
    ("Mid_Case", "Mid_Case"),
    ("Mid_Case_100by2035", "NZ-2035"),
    ("Mid_Case_95by2050", "NZ-2050"),
    ("Mid_Case_No_IRA", "No-IRA"),
]


def round_sig(x, sig=3):
    """Round x to `sig` significant figures."""
    if x == 0 or pd.isna(x):
        return 0.0
    from math import floor, log10
    return round(x, -int(floor(log10(abs(x)))) + (sig - 1))


def fmt(v):
    """Format a tonnes/yr value: >=1 Mt as 'X.X Mt' (3 s.f.); >=1000 as comma
    integer (3 s.f.); <1000 as a 3-s.f. number."""
    if v is None or pd.isna(v):
        return "n/a"
    if v >= 1e6:
        return f"{round_sig(v / 1e6, 3):g} Mt"
    if v >= 1000:
        return f"{int(round_sig(v, 3)):,}"
    return f"{round_sig(v, 3):g}"


def build_table1(df):
    """Return (markdown_str, dataframe) for Table 1 from the demand CSV."""
    mcol = "mean_annual" if "mean_annual" in df.columns else "mean"
    scen_keys = [s for s, _ in SCENARIOS]

    # Demand-bearing materials = Mid_Case peak > 0 (drops the 4 zeroed).
    mc = df[df.scenario == "Mid_Case"]
    materials = [m for m in mc.material.unique()
                 if mc[mc.material == m][mcol].max() > 0]

    rows = []
    for m in materials:
        sub = df[df.material == m]
        rec = {"Material": m}
        for s in scen_keys:
            ss = sub[sub.scenario == s]
            rec[s] = float(ss.loc[ss[mcol].idxmax(), mcol]) if not ss.empty else None
        mcm = sub[sub.scenario == "Mid_Case"]
        rec["peak_year"] = int(mcm.loc[mcm[mcol].idxmax(), "year"])
        rows.append(rec)

    tab = pd.DataFrame(rows).sort_values("Mid_Case", ascending=False).reset_index(drop=True)

    # Caption: derive the dominant peak year and the exceptions dynamically.
    yr_mode = int(tab["peak_year"].mode().iloc[0])
    exceptions = sorted(tab.loc[tab["peak_year"] != yr_mode, "Material"].tolist())
    exc_years = sorted(tab.loc[tab["peak_year"] != yr_mode, "peak_year"].unique().tolist())
    exc_txt = (f" Peak demand occurs in {yr_mode} for all materials except "
               f"{', '.join(exceptions[:-1])}, and {exceptions[-1]} "
               f"({'/'.join(str(y) for y in exc_years)})." if exceptions
               else f" Peak demand occurs in {yr_mode} for all materials.")
    caption = (
        f"**Table 1.** Peak annual demand (t/yr unless noted) under the four "
        f"highlighted scenarios, and the peak year, for the {len(tab)} demand-bearing "
        f"materials.{exc_txt} NZ-2035 = 100% clean by 2035; NZ-2050 = 95% clean by "
        f"2050; No-IRA = Mid_Case without the Inflation Reduction Act. Generated from "
        f"`material_demand_by_scenario.csv` by `table1_peak_demand.py`."
    )

    # Markdown table.
    header = "| Material | " + " | ".join(lbl for _, lbl in SCENARIOS) + " | Peak yr |"
    sep = "|---|" + "---:|" * len(SCENARIOS) + ":---:|"
    lines = [header, sep]
    for _, r in tab.iterrows():
        cells = [r["Material"]] + [fmt(r[s]) for s, _ in SCENARIOS] + [str(r["peak_year"])]
        lines.append("| " + " | ".join(cells) + " |")
    md = caption + "\n\n" + "\n".join(lines)
    return md, tab


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args()

    df = pd.read_csv(DEMAND_CSV)
    md, tab = build_table1(df)

    # --- verification ---
    assert len(tab) == 27, f"expected 27 demand-bearing materials, got {len(tab)}"
    for s, _ in SCENARIOS:
        assert tab[s].notna().all() and (tab[s] > 0).all(), f"non-positive/missing peak in {s}"
    print("Table 1 verification: OK")
    print(f"  source: {DEMAND_CSV.relative_to(REPO_ROOT)}")
    print(f"  materials: {len(tab)}  scenarios: {[l for _, l in SCENARIOS]}")
    print(tab.to_string(index=False, float_format=lambda v: f"{v:,.1f}"))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "table1_peak_demand.csv").write_text(tab.to_csv(index=False))
    (OUT_DIR / "table1_peak_demand.md").write_text(md + "\n")
    print(f"\nWrote outputs/data/table1_peak_demand.csv and .md")


if __name__ == "__main__":
    main()
