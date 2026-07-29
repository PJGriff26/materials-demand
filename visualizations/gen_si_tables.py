"""
Generate every manuscript table as a code-produced CSV.

INVENTORY:
  name: gen_si_tables
  output: outputs/data/tables/table_s1.csv ... table_s9.csv (docx numbering)
          outputs/data/tables/table_{lifetimes,fit_corrections}.csv
  category: Tables
  data_sources:
    - src/technology_mapping.py (TECHNOLOGY_LIFETIMES)                 [lifetimes, S2]
    - visualizations/supply_tiers_shared.py (WITHHELD_OVERRIDES, RE data)[S5, S6]
    - src/data_quality.py (KNOWN_CORRECTIONS, KNOWN_REMOVALS)          [fit-corrections]
    - outputs/data/material_demand_by_scenario.csv                     [S1, S8]
    - outputs/data/fitted_distributions.csv + cv_borrowing_report.csv  [S4]
    - visualizations/fig4_fig5_supply_tiers.py::build_records        [S9]
    - outputs/data/material_features.csv                               [S3]
  description: >
    Emits the supplementary tables S1-S9 as CSVs so every table in the paper is
    reproducible from the pipeline (Table 1 is produced separately by
    table1_peak_demand.py). Table numbering matches the SI docx (renumbered 2026-07-28 to the
    7-23 revision). Table S9 imports the SAME build_records() the supply
    figures use, so the table and Figures 4/5 cannot diverge. Rare-earth
    reserve ratios use the aggregate TREO reserve base (see S5/S6 notes);
    per-element reserves are not independently published. The lifetimes and
    fit-corrections tables are no longer SI tables but stay reproducible
    (stems table_lifetimes.csv, table_fit_corrections.csv).
END_INVENTORY

Usage:
    python visualizations/gen_si_tables.py
    python visualizations/gen_si_tables.py --only S5 S7 S9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "visualizations"))

DATA = ROOT / "outputs" / "data"
TABLES_DIR = DATA / "tables"
DEMAND_FILE = DATA / "material_demand_by_scenario.csv"
FIT_FILE = DATA / "fitted_distributions.csv"
CV_FILE = DATA / "cv_borrowing_report.csv"
FEATURES_FILE = DATA / "material_features.csv"

from scenario_config import REFERENCE_SCENARIO  # noqa: E402

# Aggregate USGS MCS 2025 TREO reserve anchors (Rare Earths chapter p.145),
# the basis used for every rare-earth reserve ratio in S9 and Figs 4/5.
US_TREO_RESERVE_T = 1_900_000
WORLD_TREO_RESERVE_T = 90_000_000

# Material class map for S3 - canonical taxonomy from src/material_classes.py
# (Option 2 placements 2026-07-28; incl. the 'Gadium' data-spelling alias).
from material_classes import MATERIAL_CLASS as CLASS_MAP  # noqa: E402
from material_classes import CLASS_ORDER  # noqa: E402


def _write(df: pd.DataFrame, stem: str):
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / f"{stem}.csv"
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


# ─────────────────────────────────────────────────────────── S1
def table_lifetimes():
    """Assumed operational lifetimes by modeled technology (yr)."""
    from technology_mapping import TECHNOLOGY_LIFETIMES
    disp = [
        ("Solar PV", "upv"), ("Onshore wind", "wind_onshore"),
        ("Offshore wind", "wind_offshore"), ("Nuclear", "nuclear"),
        ("Hydropower", "hydro"), ("Battery storage", "battery_4"),
        ("Coal", "coal"), ("Gas combined cycle", "gas_cc"),
        ("Gas combustion turbine", "gas_ct"), ("Biomass", "bio"),
        ("Geothermal", "geo"),
    ]
    rows = [{"Technology": name, "Lifetime_yr": TECHNOLOGY_LIFETIMES[key]}
            for name, key in disp]
    _write(pd.DataFrame(rows), "table_lifetimes")


# ─────────────────────────────────────────────────────────── S2
def table_s5():
    """Supply-risk metric edge cases: every material-specific deviation from the
    uniform Section 2.4 rules, with the value the pipeline actually uses."""
    from supply_tiers_shared import WITHHELD_OVERRIDES, RE_ELEMENT_DATA
    te = WITHHELD_OVERRIDES["Tellurium"]["us_prod_kt"] * 1000  # kt -> t
    b = WITHHELD_OVERRIDES["Boron"]["us_prod_kt"]
    mg = WITHHELD_OVERRIDES["Magnesium"]["us_prod_kt"]
    ree_prod = "; ".join(f"{k[:2]} {v['global_t']:,}"
                         for k, v in RE_ELEMENT_DATA.items())
    rows = [
        ("Tellurium", "US production (A)", f"{te:.0f} t/yr",
         "USGS withholds; single-facility floor, so the ratio is an upper bound",
         "SCRREEN2 2023"),
        ("Boron", "US production (A)", f"{b:.0f} kt/yr",
         "USGS withholds (net exporter); export volume as a lower bound", "USGS MCS 2025"),
        ("Magnesium", "US production (A)", f"{mg:.0f} kt/yr",
         "USGS withholds; secondary production (primary smelters closed 2023)",
         "USGS MCS 2025"),
        ("Indium, manganese, niobium, yttrium", "US production (A)", "undefined",
         "no US production; ratios A and B mark the absence of a domestic supply chain",
         "USGS MCS 2025"),
        ("Tellurium", "Global production (C)", "962 t/yr",
         "below the 1,000 t injection threshold, so US is not added; world total excludes the US",
         "USGS MCS 2025"),
        ("Tin, vanadium", "Global production (C)", "US injected",
         "withheld in the per-country world data but exceeds 1,000 t", "USGS MCS 2025"),
        ("Nd, Pr, Dy, Tb, Y", "Global production (C)", ree_prod + " t/yr",
         "USGS reports only a basket; the magnet REE are co-mined",
         "U.S. DOE 2023 (Nd/Pr/Dy/Tb); USGS MCS (Y)"),
        ("Aluminum", "Reserves (B, D)", "bauxite x 0.25",
         "4 t bauxite yields 1 t aluminum", "USGS MCS 2025"),
        ("Steel", "Reserves (B, D)", "iron-ore 'iron content'",
         "USGS reports iron ore, not finished steel", "USGS MCS 2025"),
        ("Cement, silicon, cadmium, gallium, germanium, fiberglass, glass",
         "Reserves (B, D)", "excluded", "no separate USGS reserve concept", "USGS MCS 2025"),
        ("Rare earths (Nd, Pr, Dy, Tb, Y)", "Reserves (B, D)",
         f"aggregate TREO (US {US_TREO_RESERVE_T:,} t / world {WORLD_TREO_RESERVE_T:,} t)",
         "no source publishes per-element REE reserves; each element is measured against "
         "the aggregate rare-earth-oxide reserve base (an optimistic upper bound)",
         "USGS MCS 2025 Rare Earths"),
        ("Boron, molybdenum", "NIR", "0",
         "USGS net exporters", "USGS MCS 2025"),
        ("Nd, Pr, Dy, Tb, Gd", "NIR", "0.945",
         "no per-element trade data; inherit the rare-earth basket", "USGS MCS 2025"),
        ("Rare earths", "NIR", "published value",
         "stage-asymmetry guard: USGS NIR at least 0.5 but trade-balance at most 0.05",
         "USGS MCS 2025"),
        ("Germanium", "HHI", "0.81",
         "country shares unavailable from USGS; EU CRM processing-stage shares "
         "used (China 89.6, Russia 5.4, US 2.1, Japan 1.9, Ukraine 1.0 percent)",
         "EU CRM 2023"),
        ("Rare earths", "HHI", "0.73 light, 1.0 heavy",
         "refining stage governs, replacing the mining basket of about 0.50",
         "EU CRM 2023"),
        ("Gallium, germanium, selenium, gadolinium", "Scope", "~0 demand",
         "tied to the zero-weight CIGS / a-Si chemistries", "Seel et al. 2025"),
    ]
    df = pd.DataFrame(rows, columns=["Material", "Metric", "Value", "Reason", "Source"])
    _write(df, "table_s5")


# ─────────────────────────────────────────────────────────── S3
def table_fit_corrections():
    """Manual corrections and single-point removals applied before fitting."""
    from data_quality import KNOWN_CORRECTIONS, KNOWN_REMOVALS
    rows = []
    for tech, mat, raw, corr, reason in KNOWN_CORRECTIONS:
        typ = "Decimal" if "decimal" in reason.lower() else "Value"
        rows.append(("Decimal" if typ == "Decimal" else "Value",
                     tech, mat, raw, corr, reason))
    for tech, mat, raw, reason in KNOWN_REMOVALS:
        rows.append(("Removal", tech, mat, raw, "removed", reason))
    df = pd.DataFrame(rows, columns=["Type", "Technology", "Material",
                                     "Raw", "Corrected", "Basis"])
    _write(df, "table_fit_corrections")


# ─────────────────────────────────────────────────────────── S4
def table_s1():
    """The 61 NREL Standard Scenarios 2024 analyzed (scenario keys)."""
    d = pd.read_csv(DEMAND_FILE, usecols=["scenario"])
    scen = sorted(d["scenario"].unique())
    df = pd.DataFrame({"n": range(1, len(scen) + 1), "scenario_key": scen})
    _write(df, "table_s1")


# ─────────────────────────────────────────────────────────── S5
def table_s4():
    """All technology x material lognormal fits (n, sigma, CV, source, GOF)."""
    fit = pd.read_csv(FIT_FILE)
    cv = pd.read_csv(CV_FILE)
    m = cv.merge(fit[["technology", "material", "param_s", "ks_pvalue",
                      "ad_statistic"]],
                 on=["technology", "material"], how="left")
    out = pd.DataFrame({
        "Technology": m["technology"],
        # Display alias only: intensity_data.csv preserves the "Gadium" typo
        # to keep joins stable (doc_assertions.yaml::gadium_typo_preserved);
        # published tables must show the correct element name.
        "Material": m["material"].replace({"Gadium": "Gadolinium"}),
        "n": m["n"],
        "sigma": m["param_s"].round(3),
        "CV": m["cv"].round(3),
        "CV_source": m["cv_source"],
        "Median_t_per_MW": m["median_intensity"],
        "KS_p": m["ks_pvalue"].round(3),
        "AD": m["ad_statistic"].round(3),
    }).sort_values(["Technology", "Material"]).reset_index(drop=True)
    _write(out, "table_s4")


# ─────────────────────────────────────────────────────────── S6
def table_s6():
    """Per-element rare-earth production (DOE 2023 App A for Nd/Pr/Dy/Tb,
    USGS MCS for Y) -- the only per-element REE quantities the pipeline
    consumes.

    Reserve columns were REMOVED 2026-07-28 (PJ decision): no source publishes
    per-element REE reserves, and the deposit-composition apportionment that
    once populated them fed nothing -- every reserve ratio in Table S9 and every
    reserve bar in Figures 4/5 divides by the AGGREGATE TREO base. Printing
    apportioned reserves here therefore contradicted the analysis; the aggregate
    basis is now stated in the caption and documented in Table S5.
    """
    from supply_tiers_shared import RE_ELEMENT_DATA
    rows = []
    for el, d in RE_ELEMENT_DATA.items():
        rows.append({
            "Element": el,
            "Global_production_t_per_yr": d["global_t"],
            # USGS reports no U.S. yttrium production (NA in every vintage);
            # the pipeline carries 0, the table prints n/a.
            "US_production_t_per_yr": d["us_t"] if d["us_t"] else "n/a",
        })
    _write(pd.DataFrame(rows), "table_s6")


# ─────────────────────────────────────────────────────────── S7
def table_s9():
    """Supply-adequacy ratios A-D by material (imports the figures' own
    build_records so the table and Figures 4/5 never diverge)."""
    import fig4_fig5_supply_tiers as F5
    # per-element: A/C use per-element REE production; B/D still divide by the
    # aggregate TREO reserve (Option B, PJ decision 2026-07-27), matching
    # Figures 4/5 and Table S8.
    df, _mat_order = F5.build_records(re_mode="per-element")

    def ratio(num, den):
        num = num or 0.0
        den = den or 0.0
        return (num / den) if den > 0 else np.nan

    def sig(x, n=3):
        """Round to n significant figures (keeps precision for tiny ratios like
        2.46e-05 that fixed-decimal rounding would collapse to 0)."""
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return "n/a"
        if x == 0:
            return 0.0
        from math import floor, log10
        return round(x, -int(floor(log10(abs(x)))) + (n - 1))

    rows = []
    for mat, r in df.iterrows():
        if mat == "Rare Earths":
            continue
        peak = r.get("peak_mid_case", np.nan)
        cum = r.get("cum_mid_case", np.nan)
        rows.append({
            "Material": mat,
            "A_peak_over_US_production": sig(ratio(peak, r.get("us_prod_t"))),
            "B_cumulative_over_US_reserves": sig(ratio(cum, r.get("us_reserves_t"))),
            "C_peak_over_world_production": sig(ratio(peak, r.get("global_prod_t"))),
            "D_cumulative_over_world_reserves": sig(ratio(cum, r.get("global_reserves_t"))),
        })
    out = pd.DataFrame(rows).sort_values("Material").reset_index(drop=True)
    _write(out, "table_s9")


# ─────────────────────────────────────────────────────────── S8
def table_s3():
    """The 31 tracked materials, by class and status."""
    feat = pd.read_csv(FEATURES_FILE)
    rows = []
    for _, r in feat.iterrows():
        mat = r["material"]
        demand_bearing = not bool(r["zero_demand"])
        crit_eligible = not bool(r["excluded_from_criticality"])
        if not demand_bearing:
            status = "tracked; ~0 demand (CIGS / a-Si only)"
        elif not crit_eligible:
            status = "demand-bearing; excluded from criticality"
        else:
            status = "demand-bearing; criticality-eligible"
        rows.append({"Material": mat, "Class": CLASS_MAP.get(mat, "?"),
                     "Status": status})
    order = {c: i for i, c in enumerate(CLASS_ORDER)}
    out = pd.DataFrame(rows)
    out = out.sort_values(by=["Class", "Material"],
                          key=lambda s: s.map(order) if s.name == "Class" else s)
    _write(out.reset_index(drop=True), "table_s3")


# ─────────────────────────────────────────────────────────── S9
def table_s8():
    """Baseline-with-IRA peak (t/yr) and cumulative (t) demand with the 95%
    within-scenario intensity interval [p2.5, p97.5] at the peak year."""
    d = pd.read_csv(DEMAND_FILE)
    d = d[d["scenario"] == REFERENCE_SCENARIO]
    # Restrict to demand-bearing materials (drop the ~0-demand CIGS/a-Si-only
    # materials Gallium, Germanium, Selenium, Gadolinium), matching Table S3.
    feat = pd.read_csv(FEATURES_FILE)
    zero_mats = set(feat.loc[feat["zero_demand"] == True, "material"])
    d = d[~d["material"].isin(zero_mats)]
    rows = []
    for mat, g in d.groupby("material"):
        g = g.sort_values("year")
        peak_idx = g["mean"].idxmax()
        peak_row = g.loc[peak_idx]
        rows.append({
            "Material": mat,
            "Peak_t_per_yr": round(float(peak_row["mean"]), 1),
            "Peak_year": int(peak_row["year"]),
            "Interval_p2_5": round(float(peak_row["p2_5"]), 1),
            "Interval_p97_5": round(float(peak_row["p97_5"]), 1),
            "Cumulative_t": round(float(g["mean"].sum()), 1),
        })
    out = pd.DataFrame(rows).sort_values("Peak_t_per_yr", ascending=False)
    _write(out.reset_index(drop=True), "table_s8")



# ─────────────────────────────────────────────────────────── S2 (crosswalk)
def table_s2():
    """The 23 NREL technologies with their material-intensity category crosswalk
    (docx Table S2). Excluded technologies carry an empty mapping; H2-CT maps to
    NGGT intensities (turbine hardware only); OGS has no additions in any
    scenario."""
    from technology_mapping import TECHNOLOGY_MAPPING, TECHNOLOGY_LIFETIMES
    rows = []
    for tech_key in sorted(TECHNOLOGY_MAPPING):
        mapping = TECHNOLOGY_MAPPING[tech_key]
        if mapping:
            xwalk = "; ".join(f"{cat} ({w:.0%})" if w != 1 else cat
                              for cat, w in sorted(mapping.items()))
        else:
            xwalk = "n/a (excluded)"
        rows.append({"NREL_technology": tech_key,
                     "Intensity_category_crosswalk": xwalk,
                     "Lifetime_yr": TECHNOLOGY_LIFETIMES.get(tech_key, "")})
    _write(pd.DataFrame(rows), "table_s2")


# ─────────────────────────────────────────────────────────── S7 (peak, 4 scen)
def table_s7():
    """Peak annual demand (t/yr) under the four representative scenarios
    (docx Table S7); re-emits table1_peak_demand.csv so the SI table stem
    matches the docx numbering."""
    src = DATA / "table1_peak_demand.csv"
    _write(pd.read_csv(src), "table_s7")


# ─────────────────────────────────────────── class deltas (not an SI table)
def table_class_scenario_deltas():
    """Per-class demand deltas vs Baseline with IRA for the two net-zero
    scenarios and their CO2e siblings: cumulative 2026-2050 and peak annual
    class-total demand (peak magnitudes compared regardless of peak year;
    the peak years themselves are reported alongside). Classes from
    src/material_classes.py (demand-side: Glass/Fiberglass in Bulk)."""
    from material_classes import (MATERIAL_CLASS, CLASS_ORDER,
                                  ZERO_DEMAND_MATERIALS)
    from scenario_config import REFERENCE_SCENARIO, SCENARIO_LABELS

    scenarios = [
        ("Mid_Case_CO2e_100by2035", SCENARIO_LABELS["Mid_Case_CO2e_100by2035"]),
        ("Mid_Case_100by2035", "Net-Zero by 2035 (combustion-CO2 variant)"),
        ("Mid_Case_CO2e_95by2050", SCENARIO_LABELS["Mid_Case_CO2e_95by2050"]),
        ("Mid_Case_95by2050", "Net-Zero by 2050 (combustion-CO2 variant)"),
    ]
    d = pd.read_csv(DEMAND_FILE)
    d = d[~d["material"].isin(ZERO_DEMAND_MATERIALS)]
    d["cls"] = d["material"].map(MATERIAL_CLASS)
    annual = (d.groupby(["scenario", "cls", "year"])["mean"].sum()
               .unstack("year"))
    rows = []
    for cls in CLASS_ORDER:
        base = annual.loc[(REFERENCE_SCENARIO, cls)]
        for key, label in scenarios:
            s = annual.loc[(key, cls)]
            rows.append({
                "Material_class": cls,
                "Scenario": label,
                "Scenario_key": key,
                "Cumulative_vs_baseline_pct": round(
                    (s.sum() / base.sum() - 1) * 100, 1),
                "Peak_annual_vs_baseline_pct": round(
                    (s.max() / base.max() - 1) * 100, 1),
                "Peak_year": int(s.idxmax()),
                "Baseline_peak_year": int(base.idxmax()),
            })
    _write(pd.DataFrame(rows), "table_class_scenario_deltas")


ALL = {"S1": table_s1, "S2": table_s2, "S3": table_s3, "S4": table_s4,
       "S5": table_s5, "S6": table_s6, "S7": table_s7, "S8": table_s8,
       "S9": table_s9, "LIFETIMES": table_lifetimes,
       "FITCORR": table_fit_corrections,
       "CLASSDELTAS": table_class_scenario_deltas}


def main():
    ap = argparse.ArgumentParser(description="Generate manuscript tables S1-S9 as CSVs.")
    ap.add_argument("--only", nargs="*", choices=list(ALL),
                    help="generate only these tables (default: all)")
    args = ap.parse_args()
    want = args.only or list(ALL)
    for key in want:
        print(f"[{key}]")
        ALL[key]()


if __name__ == "__main__":
    main()
