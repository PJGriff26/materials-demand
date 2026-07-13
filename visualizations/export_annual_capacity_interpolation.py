"""Export the PCHIP-interpolated annual electricity-supply data behind Fig 1.

Writes a tidy CSV of the NREL Standard Scenarios 2024 capacity data after the
monotone-PCHIP interpolation from the native 3-year reporting cadence to an
annual grid, the exact same interpolation and supply-type aggregation used by
Fig 1 (visualizations/fig1_capacity_additions_facet.py), whose helpers this
script imports so the export can never drift from the figure.

Added 2026-07-13 to support summary statistics quoted in the manuscript
text.

INVENTORY:
  name: export_annual_capacity_interpolation
  output: outputs/data/annual_capacity_interpolated.csv
  category: Data export (no figure)
  columns:
    scenario: NREL Standard Scenarios 2024 scenario name (61 scenarios)
    representative_label: manuscript label for the four highlighted
      scenarios (Net-Zero by 2035 / Net-Zero by 2050 / Baseline with IRA /
      Baseline without IRA); empty for the other 57
    year: 2026-2050, annual after PCHIP interpolation
    supply_type: electricity-supply group (Fig 1 panel definition,
      fig1_capacity_additions_facet.SUPPLY_TYPES)
    installed_capacity_gw: interpolated installed capacity stock (GW)
    gross_additions_gw: annual gross capacity additions (GW/yr);
      per-technology positive year-over-year change summed within the
      supply type (retirements floored at zero per technology); 2026 is
      zero by construction (no prior year)
  data_sources:
    - NREL Standard Scenarios 2024 (StdScen24_annual_national.csv)
    - src/interpolation.py (interpolate_capacity_pchip)
END_INVENTORY

Usage:
    python visualizations/export_annual_capacity_interpolation.py
"""

import sys
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "visualizations"))
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

# Reuse the Fig 1 loader, supply-type mapping, gross-additions computation,
# and the four representative-scenario labels verbatim.
from fig1_capacity_additions_facet import (  # noqa: E402
    load_nrel, compute_gross_additions, SUPPLY_TYPES, SCENARIOS,
)
from src.interpolation import interpolate_capacity_pchip  # noqa: E402

OUT_CSV = BASE_DIR / "outputs" / "data" / "annual_capacity_interpolated.csv"


def compute_installed_capacity(nrel: pd.DataFrame) -> pd.DataFrame:
    """Installed capacity stock (GW) per scenario x supply type x year."""
    recs = []
    for (scen, yr), g in nrel.groupby(["scenario", "year"]):
        row = g.iloc[0]
        for name, cols in SUPPLY_TYPES.items():
            present = [c for c in cols if c in nrel.columns]
            if not present:
                continue
            recs.append({
                "scenario": scen, "year": int(yr), "supply_type": name,
                "installed_capacity_gw": float(sum(row[c] for c in present)) / 1000.0,
            })
    return pd.DataFrame(recs)


def main():
    nrel = load_nrel()
    print(f"Loaded NREL: {nrel.shape[0]:,} rows "
          f"({nrel['scenario'].nunique()} scenarios, "
          f"{nrel['year'].min()}-{nrel['year'].max()})")

    nrel = interpolate_capacity_pchip(nrel)
    print(f"  PCHIP-interpolated to annual grid: "
          f"{nrel['year'].min()}-{nrel['year'].max()} "
          f"({nrel['year'].nunique()} years)")

    stock = compute_installed_capacity(nrel)
    adds = compute_gross_additions(nrel).rename(
        columns={"add_gw": "gross_additions_gw"})

    out = stock.merge(adds, on=["scenario", "year", "supply_type"], how="left")
    out["representative_label"] = out["scenario"].map(SCENARIOS).fillna("")
    out = out[["scenario", "representative_label", "year", "supply_type",
               "installed_capacity_gw", "gross_additions_gw"]]
    out = out.sort_values(["scenario", "supply_type", "year"]).reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"Wrote {OUT_CSV.relative_to(BASE_DIR)} "
          f"({len(out):,} rows: {out['scenario'].nunique()} scenarios x "
          f"{out['supply_type'].nunique()} supply types x "
          f"{out['year'].nunique()} years)")


if __name__ == "__main__":
    main()
