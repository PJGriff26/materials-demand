"""Regression tests pinning the unit-conversion contract in
`supply_chain/usgs_mcs2025_loader`: `_col_unit_factor` maps a USGS salient-CSV
column name to a multiplicative factor that converts raw values to kilotons (kt),
where a missing suffix mapping silently inflates trade magnitudes (e.g. a missing
`_kg` by 10^6x)."""

import sys
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "supply_chain"))

from usgs_mcs2025_loader import (  # noqa: E402
    _col_unit_factor,
    _col_metal_fraction,
    _ATOMIC_WEIGHTS_IUPAC2021,
    _metal_fraction_in_compound,
)


@pytest.mark.parametrize("col,expected", [
    # kt passthrough
    ("USprod_FeSi-Si_kt",    1.0),
    ("USprod_Primary_kt",    1.0),
    ("Imports_FeSi_kt",      1.0),
    ("Consump_Apprnt_kt",    1.0),
    ("Consump_Total_kt",     1.0),
    # Mt → kt
    ("USprod_Steel_mmt",     1000.0),
    # t → kt
    ("USprod_Mine_t",        0.001),
    ("Imports_Metal_t",      0.001),
    # kg → kt  (Germanium, Gallium, others; added 2026-04-19)
    ("USprod_Primary_kg",    1e-6),
    ("USprod_Secondary_kg",  1e-6),
    ("Imports_Metal_kg",     1e-6),
    ("Imports_GaAs_kg",      1e-6),
    ("Imports_GeO2_kg",      1e-6),
    ("Exports_kg",           1e-6),
    ("Consump_kg",           1e-6),
    ("Stocks_kg",            1e-6),
    # USGS header-typo alias: Consump_g  ≡  Consump_kg (Germanium CSV).
    ("Consump_g",            1e-6),
    # Non-unit suffixes must not mis-match
    ("Price_Metal_dkg",      1.0),  # dollars per kg — not a mass unit
    ("NIR_ct",               1.0),  # NIR text column
    ("NIR_pct",              1.0),
    ("NIR_FeSi-Si_pct",      1.0),
])
def test_col_unit_factor(col, expected):
    actual = _col_unit_factor(col)
    assert actual == pytest.approx(expected, rel=1e-9), (
        f"{col!r} → {actual} (expected {expected})"
    )


def test_longest_suffix_wins():
    """`_mmt` must beat `_t`; `_kg` must beat `_g`; `_kt` must beat `_t`."""
    assert _col_unit_factor("USprod_Steel_mmt") == pytest.approx(1000.0)
    assert _col_unit_factor("Imports_Metal_kg") == pytest.approx(1e-6)
    assert _col_unit_factor("Imports_FeSi_kt") == pytest.approx(1.0)


def test_unknown_suffix_falls_back_to_kt():
    """Unknown suffixes should default to 1.0 (kt). Matches historical
    behavior for USGS columns without explicit unit suffixes."""
    assert _col_unit_factor("Something_Weird") == pytest.approx(1.0)
    assert _col_unit_factor("Production") == pytest.approx(1.0)


# ═══════════════════════════════════════════════════════════════════════════
# Compound-weight → metal-content conversion
# ═══════════════════════════════════════════════════════════════════════════

def test_metal_fraction_in_compound_matches_iupac_derivation():
    """Ga-mass-fraction-in-GaAs must be derived from IUPAC 2021 atomic
    weights (Prohaska et al. 2022, Pure Appl. Chem. 94:573). Hardcoding
    a numeric literal would decouple the factor from its physical
    source and silently drift if the atomic weights dict is updated.
    """
    ga = _ATOMIC_WEIGHTS_IUPAC2021["Ga"]
    as_ = _ATOMIC_WEIGHTS_IUPAC2021["As"]
    expected = ga / (ga + as_)
    assert _metal_fraction_in_compound("Ga", ["As"]) == pytest.approx(expected)


def test_atomic_weights_are_plausible():
    """Guard against data-entry corruption of `_ATOMIC_WEIGHTS_IUPAC2021`
    without introducing a hardcoded literal from outside IUPAC. Every
    entry must satisfy basic atomic-physics constraints: the relative
    atomic mass is (a) greater than the element's atomic number Z
    (each proton contributes ~1 AMU and neutrons add on top), and
    (b) less than ~3·Z (no stable isotope exceeds this bound). This
    catches the most common corruption modes (typo, wrong order of
    magnitude, swapped elements) without pinning specific values."""
    _Z = {"Ga": 31, "As": 33}  # atomic numbers — integer count of protons
    for symbol, mass in _ATOMIC_WEIGHTS_IUPAC2021.items():
        z = _Z[symbol]
        assert z < mass < 3 * z, (
            f"Atomic weight for {symbol} ({mass}) violates "
            f"{z} < mass < {3*z} — check _ATOMIC_WEIGHTS_IUPAC2021."
        )


def test_col_metal_fraction_gaas_uses_iupac_fraction():
    """`_col_metal_fraction('Imports_GaAs_kg')` must return the same
    fraction as `_metal_fraction_in_compound('Ga', ['As'])`.
    Regression guard: if someone replaces the computed fraction with
    a literal, the two will drift and this test fails."""
    from_dict = _col_metal_fraction("Imports_GaAs_kg")
    from_derivation = _metal_fraction_in_compound("Ga", ["As"])
    assert from_dict == pytest.approx(from_derivation)


def test_col_metal_fraction_metal_columns_unchanged():
    """Columns that are already metal-content (Imports_Metal_kg,
    Consump_kg, USprod_Primary_kg, ...) must return factor 1.0 —
    applying a compound fraction would silently undercount."""
    for c in ("Imports_Metal_kg", "USprod_Primary_kg", "Consump_kg",
              "Exports_kg", "Stocks_kg", "USprod_FeSi-Si_kt"):
        assert _col_metal_fraction(c) == pytest.approx(1.0), c


def test_col_metal_fraction_pre_converted_ge_columns_unchanged():
    """USGS metadata states the Germanium GeO2 columns are pre-
    converted to Ge content ('Germanium dioxide data were multiplied
    by 69% to calculate the germanium content'). Applying another
    fraction would double-convert. These columns must therefore NOT
    appear in `_COMPOUND_METAL_FRACTION` and must return 1.0."""
    assert _col_metal_fraction("Imports_GeO2_kg") == pytest.approx(1.0)
    assert _col_metal_fraction("Exports_GeO2_kg") == pytest.approx(1.0)


def test_extract_trade_applies_ga_metal_fraction():
    """End-to-end: loading the Gallium salient CSV should yield
    metal-equivalent trade totals that equal
    Imports_Metal_kg + Imports_GaAs_kg × metal_fraction_Ga_in_GaAs.
    This pins the overall contract — any regression in the suffix
    mapping, typo alias, or metal-fraction helper must fail this
    test before it corrupts aggregate output."""
    from usgs_mcs2025_loader import _load_salient, _extract_trade
    df = _load_salient("Gallium", "galli")
    assert df is not None
    results = _extract_trade(df, "Gallium")
    frac = _metal_fraction_in_compound("Ga", ["As"])
    # Rebuild the expected value straight from the raw CSV, column by
    # column, applying the documented rules once.
    for year, imp_kt, exp_kt, _cons_kt in results:
        row = df[df["Year"] == year].iloc[0]
        metal_kg = float(row["Imports_Metal_kg"])
        gaas_kg = float(row["Imports_GaAs_kg"])
        expected_imports_kt = (metal_kg + gaas_kg * frac) * 1e-6
        assert imp_kt == pytest.approx(expected_imports_kt, rel=1e-9), (
            f"Ga imports for {year}: pipeline={imp_kt} kt, expected "
            f"{expected_imports_kt} kt"
        )
