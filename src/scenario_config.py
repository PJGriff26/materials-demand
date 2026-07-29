"""Canonical representative-scenario configuration (single source of truth).

Added 2026-07-28 after the CO2e scenario mix-up: the manuscript's abstract and
Results percentages had been computed from the Mid_Case_CO2e_* ensemble
members while every figure and table used the non-CO2e runs, and nothing in
the code enforced one choice. Per the 2026-07-27 decision (DECISIONS.md), the
four representative scenarios are the NON-CO2e (combustion-CO2 constraint)
runs. REVISED 2026-07-28 (PJ, evening): the net-zero highlights are now the
CO2e-constrained runs - they are the family carrying the ensemble's
20-scenario deep-decarbonization sensitivity suite, they set every
ensemble-max finding (hatched bars, tornado spans), and "Net-Zero" is only
semantically accurate for a CO2e constraint. The two baselines carry no
emissions constraint and are family-neutral. The combustion-CO2 singletons
(Mid_Case_100by2035, Mid_Case_95by2050) remain ordinary ensemble members.

Every consumer of "the representative scenarios" or "the reference scenario"
must import from this module rather than hardcoding keys, so a future change
of family is a one-line edit plus regeneration. Display ORDER is a per-figure
presentation choice and stays local to each figure; this module owns the set,
the labels, and the cross-figure scenario palette (locked 2026-07-13).
"""

# The reference (baseline) scenario used for MC uncertainty intervals,
# demand-by-technology shares, and all "Baseline with IRA" columns.
REFERENCE_SCENARIO = "Mid_Case"

# Scenario key -> manuscript display label. Order = the canonical listing
# order used in material_demand_summary.csv (baseline first).
SCENARIO_LABELS = {
    "Mid_Case":            "Baseline with IRA",
    "Mid_Case_No_IRA":     "Baseline without IRA",
    "Mid_Case_CO2e_100by2035": "Net-Zero by 2035",
    "Mid_Case_CO2e_95by2050":  "Net-Zero by 2050",
}

REPRESENTATIVE_SCENARIO_KEYS = list(SCENARIO_LABELS)

# Cross-figure scenario palette (Figs 1, 2, 3 share it; locked 2026-07-13):
# NZ-2035 red (most aggressive), NZ-2050 purple, baseline blue, No-IRA orange.
SCENARIO_COLORS = {
    "Mid_Case_CO2e_100by2035": "#d62728",
    "Mid_Case_CO2e_95by2050":  "#9467bd",
    "Mid_Case":            "#1f77b4",
    "Mid_Case_No_IRA":     "#ff7f0e",
}
