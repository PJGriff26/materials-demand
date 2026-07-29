"""Canonical material-class taxonomy (single source of truth).

Consolidated 2026-07-28 (with the PJ-approved "Option 2" corrections) from
seven independent copies that had already begun to drift: supply_chain/config
MATERIAL_GROUPS, fig3 FAMILIES, gen_si_tables CLASS_MAP (which carried a fifth
"Bulk (structural)" class and no Selenium entry), figS3 tornado
MATERIAL_GROUPS, si_figures CLASS_MEMBERS, demand_by_technology
MATERIAL_GROUPS (which alone already included Glass/Fiberglass in bulk), and
fig4_fig5_supply_tiers MATERIAL_FAMILIES.

Provenance: the four-class scheme is THIS PROJECT'S OWN construct. It refines
the two-bin split in Wang et al. 2023 (Joule 7, p.322: "bulk materials" =
{aluminum, cement, Cu, fiberglass, glass, solar-grade polysilicon, steel};
everything else "specialty metals") and the informal usage in Liang et al.
2022 ("bulk metals such as steel, copper, and aluminum are the backbone...").
It must never be cited TO Wang or Liang in the manuscript.

Final placements (2026-07-28, PJ decisions, third and final revision):
- Glass + Fiberglass -> Bulk commodities on the DEMAND side (Figs 3/S2/S3,
  Tables S3/S8), matching Wang's bulk bin. They remain excluded from the
  criticality / supply-chain assessment (Figs 4/5/6, Table S9) via
  supply_chain.feature_engineering.EXCLUDED_FROM_CRITICALITY - no
  peer-reviewed criticality source exists for them.
- Silicon -> Bulk commodities: matches Wang verbatim (solar-grade
  polysilicon is bulk); the model's Si is polysilicon, not ferrosilicon.
  Bulk now equals Wang's bin plus nothing unconventional.
- Niobium -> Base & alloying: conventional ferroalloy taxonomy (~90% of Nb
  use is HSLA-steel microalloying; in this model Nb appears only as
  wind-tower steel alloying).
- Tin -> Base & alloying (conventional: Sn is an LME base metal).
- Specialty metals is now a clean thin-film / precious set
  (Te, In, Cd, Ag, Ga, Ge, Se).
The scheme remains the project's own four-class refinement of Wang's two
bins; the manuscript should still carry a one-sentence class definition.

The demand/intensity CSVs spell Gadolinium as "Gadium" (join-preserving typo,
kept intentionally); MATERIAL_CLASS carries both spellings so lookups never
miss. Display layers must render "Gadolinium".
"""

CLASS_ORDER = ["Bulk commodities", "Base & alloying",
               "Specialty metals", "Rare earth elements"]

# Internal keys stay short and stable; CLASS_LABELS carries the display form
# (legends, panel titles - "metals" suffix per the 2026-07-13/28 revisions).
CLASS_LABELS = {
    "Bulk commodities":     "Bulk commodities",
    "Base & alloying":      "Base & alloying metals",
    "Specialty metals":     "Specialty metals",
    "Rare earth elements":  "Rare earth elements",
}

CLASS_MEMBERS = {
    "Bulk commodities":    ["Cement", "Steel", "Aluminum", "Copper",
                            "Glass", "Fiberglass", "Silicon"],
    "Base & alloying":     ["Zinc", "Lead", "Nickel", "Tin", "Manganese",
                            "Chromium", "Molybdenum", "Vanadium",
                            "Magnesium", "Boron", "Niobium"],
    "Specialty metals":    ["Tellurium", "Indium", "Cadmium", "Silver",
                            "Gallium", "Germanium", "Selenium"],
    "Rare earth elements": ["Dysprosium", "Neodymium", "Praseodymium",
                            "Terbium", "Yttrium", "Gadolinium"],
}

# Flat material -> class lookup, including the "Gadium" data-spelling alias.
MATERIAL_CLASS = {m: c for c, ms in CLASS_MEMBERS.items() for m in ms}
MATERIAL_CLASS["Gadium"] = "Rare earth elements"

# Locked manuscript class palette (2026-07-13 figure revision: REE navy
# matches the Figs 4/5 REE labels; Base & alloying takes the freed plum).
CLASS_COLORS = {
    "Bulk commodities":     "#00693E",
    "Base & alloying":      "#8E3A62",
    "Specialty metals":     "#C97B3A",
    "Rare earth elements":  "#0d3b66",
}

# Materials tracked by the pipeline but with exactly zero simulated demand
# (tied to the zero-weight CIGS / a-Si chemistries in the UPV mix). Data
# spelling ("Gadium") on purpose - this set is applied to demand CSVs.
ZERO_DEMAND_MATERIALS = {"Gadium", "Selenium", "Germanium", "Gallium"}
