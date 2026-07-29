"""Guards for the canonical material-class taxonomy (src/material_classes.py).

Added 2026-07-28 with the Option 2 consolidation: seven independent class-map
copies had drifted (five-class variant in gen_si_tables, Glass/Fiberglass in
bulk in one copy only, Selenium unmapped -> a "?" class in the shipped
table_s3.csv). These tests pin the Option 2 placements and the partition
property so membership cannot drift silently again.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.material_classes import (  # noqa: E402
    CLASS_COLORS,
    CLASS_LABELS,
    CLASS_MEMBERS,
    CLASS_ORDER,
    MATERIAL_CLASS,
    ZERO_DEMAND_MATERIALS,
)


def test_four_classes_consistent():
    assert CLASS_ORDER == list(CLASS_MEMBERS)
    assert set(CLASS_LABELS) == set(CLASS_ORDER)
    assert set(CLASS_COLORS) == set(CLASS_ORDER)


def test_partition_no_overlaps():
    seen = []
    for ms in CLASS_MEMBERS.values():
        seen.extend(ms)
    assert len(seen) == len(set(seen)), "a material appears in two classes"


def test_final_placements():
    # 2026-07-28 final scheme (3rd revision): Wang-conforming bulk incl.
    # polysilicon; Nb with the alloying metals; Sn base metal.
    assert MATERIAL_CLASS["Niobium"] == "Base & alloying"
    assert MATERIAL_CLASS["Tin"] == "Base & alloying"
    assert MATERIAL_CLASS["Glass"] == "Bulk commodities"
    assert MATERIAL_CLASS["Fiberglass"] == "Bulk commodities"
    # Matches Wang et al. 2023 verbatim (solar-grade polysilicon = bulk)
    assert MATERIAL_CLASS["Silicon"] == "Bulk commodities"
    # Wang-verbatim-backed placements (Joule 7, p.322)
    assert MATERIAL_CLASS["Aluminum"] == "Bulk commodities"
    assert MATERIAL_CLASS["Copper"] == "Bulk commodities"


def test_gadium_alias():
    assert MATERIAL_CLASS["Gadium"] == MATERIAL_CLASS["Gadolinium"]


def test_covers_all_tracked_materials():
    csv = ROOT / "outputs" / "data" / "material_demand_by_scenario.csv"
    mats = set(pd.read_csv(csv, usecols=["material"])["material"].unique())
    unmapped = {m for m in mats if m not in MATERIAL_CLASS}
    assert not unmapped, f"tracked materials without a class: {unmapped}"
    assert ZERO_DEMAND_MATERIALS <= mats


def test_consumers_agree():
    """The re-export in supply_chain/config.py must mirror the canonical map."""
    sys.path.insert(0, str(ROOT / "supply_chain"))
    import config
    assert config.MATERIAL_GROUPS == {c: list(ms)
                                      for c, ms in CLASS_MEMBERS.items()}
