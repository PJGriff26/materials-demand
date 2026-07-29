"""Write outputs/data/material_features.csv: the per-material supply-chain
feature table (net import reliance, production HHI, ...) that the Figure 6
supply-risk scatter reads.

Computes the features with the live feature_engineering functions (no clustering
code); reproduce.py runs it before the figure step.
"""

import sys
from pathlib import Path

# Make the bare sibling imports (config, feature_engineering) resolve when
# this file is run as a script, and put the repo root on the path for src.*.
_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent
for _p in (str(_ROOT), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from config import RESULTS_DIR  # noqa: E402
from feature_engineering import (  # noqa: E402
    load_demand_data,
    load_risk_data,
    load_usgs_2023_thin_film,
    engineer_material_features,
)

OUTPUT_CSV = RESULTS_DIR / "material_features.csv"


def build(output_csv: Path = OUTPUT_CSV):
    """Compute the per-material feature table and write it to ``output_csv``.

    Returns the DataFrame (material-indexed) so callers can use it in memory.
    """
    demand = load_demand_data()                  # Monte Carlo demand output
    risk_data = load_risk_data()                 # USGS MCS 2025 + OECD + Census
    thin_film_data = load_usgs_2023_thin_film()  # thin-film byproduct supply data
    feats = engineer_material_features(demand, risk_data, thin_film_data)
    feats.index.name = "material"                # ensure a 'material' CSV column
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(output_csv)
    return feats


def main():
    """CLI entry point: build the table, then print a one-line summary
    confirming the row count and that the figure-required columns are present.
    """
    feats = build()
    cols = ["import_dependency", "production_hhi"]
    have = [c for c in cols if c in feats.columns]
    print(f"Wrote {OUTPUT_CSV} ({len(feats)} materials)")
    print(f"  feature columns: {len(feats.columns)}; "
          f"figure-required present: {have}")


if __name__ == "__main__":
    main()
