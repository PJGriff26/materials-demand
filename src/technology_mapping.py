"""Maps NREL Standard Scenarios capacity technologies to material intensity
technologies, with per-technology lifetimes for stock-flow retirements.

Defines TECHNOLOGY_MAPPING (capacity tech to intensity rows and weights),
TECHNOLOGY_LIFETIMES, and TECHNOLOGY_CONSOLIDATION (merges split cell/BOS rows).
"""


# ============================================================================
# INTENSITY DATA TECHNOLOGY CONSOLIDATION
# ============================================================================
# The raw intensity_data.csv splits some technologies into separate cell-specific
# and balance-of-system (BOS) entries. This mapping consolidates them during
# preprocessing so that each technology has complete material coverage.
#
# Background:
#   - 'CDTE' (15 rows): BOS materials (Aluminum, Cement, Copper, Glass, Steel)
#     for CdTe thin-film panels. Separate from 'CdTe' (112 rows) which has
#     cell-specific materials (Tellurium, Cadmium, etc.).
#   - 'ASIGE' (15 rows): BOS materials for a-Si/Germanium panels.
#     Identical values to CDTE — same generic thin-film BOS source.
#   - 'CIGS' already includes both cell and BOS materials (no consolidation needed).
#   - 'utility-scale solar pv' already includes both (no consolidation needed).
#
# Without consolidation, the CdTe fraction of UPV is missing BOS material demand
# (~15% underestimate of structural materials from solar PV).

TECHNOLOGY_CONSOLIDATION = {
    'CDTE': 'CdTe',     # BOS materials → merge into CdTe
    'ASIGE': 'a-Si',    # BOS materials → merge into a-Si
}


TECHNOLOGY_MAPPING = {
    # ============================================================================
    # SOLAR TECHNOLOGIES
    # ============================================================================

    'upv': {
        # Utility-scale PV: mix of crystalline Si and thin film
        #
        # Source: Seel et al. (2025), "Utility-Scale Solar, 2025 Data Update",
        #   Lawrence Berkeley National Laboratory. Plant-level EIA-860 data.
        #   https://emp.lbl.gov/utility-scale-solar (slide 26)
        #
        # 2024 annual capacity additions (1,747 projects, 110 GW_AC sample):
        #   - c-Si: 79% of newly installed utility-scale PV capacity
        #   - Thin-film (CdTe): 21% — record 6 GW_AC but lower share than 2022
        #   - CIGS: negligible — no significant US utility-scale deployment
        # Using annual deployment shares (not cumulative) because the model
        # projects material demand from new capacity additions.
        'utility-scale solar pv': 0.79,  # Crystalline Si (mono/poly)
        'CdTe': 0.21,                    # Thin film — First Solar
    },
    
    'distpv': {
        # Distributed PV: mostly rooftop, primarily crystalline Si
        'Solar Distributed': 1.0
    },
    
    'csp': {
        # Concentrated Solar Power
        'CSP': 1.0
    },
    
    # ============================================================================
    # WIND TECHNOLOGIES
    # ============================================================================
    
    'wind_onshore': {
        'onshore wind': 1.0
    },
    
    'wind_offshore': {
        'offshore wind': 1.0
    },
    
    # ============================================================================
    # FOSSIL FUEL TECHNOLOGIES
    # ============================================================================
    
    'coal': {
        'Coal': 1.0
    },
    
    'coal_ccs': {
        'Coal CCS': 1.0
    },
    
    'gas_cc': {
        # Natural Gas Combined Cycle
        'NGCC': 1.0
    },
    
    'gas_ct': {
        # Natural Gas Combustion Turbine
        'NGGT': 1.0
    },
    
    'gas_cc_ccs': {
        # Natural Gas Combined Cycle with CCS
        'Gas CCS': 1.0
    },
    
    'h2-ct': {
        # Hydrogen Combustion Turbine - use gas turbine as proxy
        'NGGT': 1.0
    },
    
    # ============================================================================
    # BIOMASS TECHNOLOGIES
    # ============================================================================
    
    'bio': {
        'Biomass': 1.0
    },
    
    'bio-ccs': {
        'Bio CCS': 1.0
    },
    
    # ============================================================================
    # NUCLEAR TECHNOLOGIES
    # ============================================================================
    
    'nuclear': {
        'Nuclear New': 1.0
    },
    
    'nuclear_smr': {
        # Small Modular Reactors - use Nuclear New as proxy
        'Nuclear New': 1.0
    },
    
    # ============================================================================
    # HYDRO AND GEOTHERMAL
    # ============================================================================
    
    'hydro': {
        'Hydro': 1.0
    },
    
    'pumped-hydro': {
        # Pumped storage - use regular hydro as proxy
        'Hydro': 1.0
    },
    
    'geo': {
        'Geothermal': 1.0
    },
    
    # ============================================================================
    # STORAGE AND OTHER TECHNOLOGIES
    # ============================================================================
    
    # Note: Battery technologies don't have intensity data in current dataset
    # These will be skipped with warnings
    'battery_4': {},   # 4-hour battery storage
    'battery_8': {},   # 8-hour battery storage
    
    # Note: These technologies don't have intensity data
    'o-g-s': {},       # Oil-gas-steam - deprecated technology; no additions in any scenario
    'dac': {},         # Direct Air Capture - no intensity data yet
    'electrolyzer': {},  # Hydrogen electrolyzer - no intensity data (Table S2 exclusion)
}


# ============================================================================
# TECHNOLOGY LIFETIMES (years)
# ============================================================================
# Used for stock-flow model to calculate retirements.
# Source: NREL Standard Scenarios 2024 (Gagnon et al., 2024),
#   Tables A-5 and A-6, pp. 48-49 (NREL/TP-6A40-92256).
#   These are the maximum lifetimes generators are allowed to remain
#   online in the ReEDS model.

TECHNOLOGY_LIFETIMES = {
    # Solar — Table A-5: 30 yr (SunShot Vision, DOE 2012)
    'upv': 30,
    'distpv': 30,
    'csp': 30,

    # Wind — Table A-5: 30 yr (Wind Vision, DOE 2015)
    'wind_onshore': 30,
    'wind_offshore': 30,

    # Fossil — Table A-6 (>=100 MW column)
    'coal': 75,       # Table A-6: 75 yr (>=100 MW)
    'coal_ccs': 75,   # Table A-6: 75 yr (>=100 MW)
    'gas_cc': 60,     # Table A-6: Natural gas CC: 60 yr
    'gas_ct': 50,     # Table A-6: Natural gas CT: 50 yr
    'gas_cc_ccs': 60, # Table A-6: Natural gas CC CCS: 60 yr
    'h2-ct': 50,      # Table A-5: H2-CT: 50 yr

    # Biomass — Table A-5: 50 yr (2021 NEMS plant database, EIA 2021)
    'bio': 50,
    'bio-ccs': 50,

    # Nuclear — Table A-6: 80 yr
    'nuclear': 80,
    'nuclear_smr': 80,

    # Hydro and Geo
    'hydro': 100,        # Table A-5: 100 yr (Hydropower Vision, DOE 2016)
    'pumped-hydro': 100, # Table A-5: 100 yr
    'geo': 30,           # Table A-5: 30 yr (GeoVision, DOE 2019)

    # Storage and Other
    'battery_4': 15,  # Table A-5: 15 yr (Cole, Frazier, Augustine 2021)
    'battery_8': 15,  # Table A-5: 15 yr
    'o-g-s': 75,      # Table A-6: Oil-gas-steam: 75 yr (>=100 MW)
    'dac': 30,        # No StdScen value; assumed same as geothermal
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def validate_mapping():
    """Validate that all weights sum to 1.0"""
    errors = []
    warnings = []
    
    for cap_tech, intensity_mapping in TECHNOLOGY_MAPPING.items():
        if not intensity_mapping:
            warnings.append(f"  - {cap_tech}: No mapping (will be skipped)")
            continue
            
        total_weight = sum(intensity_mapping.values())
        if abs(total_weight - 1.0) > 0.001:
            errors.append(
                f"  - {cap_tech}: Weights sum to {total_weight:.3f}, not 1.0"
            )
    
    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(w)
    
    if errors:
        print("\nERRORS:")
        for e in errors:
            print(e)
        raise ValueError("Technology mapping validation failed")
    
    print("\n✓ Technology mapping validation passed")
    return True


def get_intensity_technologies(capacity_tech):
    """
    Get the intensity technology mappings for a capacity technology.
    
    Returns:
        dict: {intensity_tech: weight}
    """
    return TECHNOLOGY_MAPPING.get(capacity_tech, {})


def get_lifetime(capacity_tech):
    """Get the lifetime (years) for a capacity technology"""
    return TECHNOLOGY_LIFETIMES.get(capacity_tech, 30)  # Default 30 years


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

if __name__ == '__main__':
    print("="*80)
    print("TECHNOLOGY MAPPING VALIDATION")
    print("="*80)
    
    validate_mapping()
    
    print("\n" + "="*80)
    print("EXAMPLE MAPPINGS")
    print("="*80)
    
    # Example: UPV mapping
    print("\nUtility PV (upv):")
    for intensity_tech, weight in get_intensity_technologies('upv').items():
        print(f"  → {intensity_tech}: {weight*100:.1f}%")
    print(f"  Lifetime: {get_lifetime('upv')} years")
    
    # Example: Wind mapping
    print("\nOnshore Wind (wind_onshore):")
    for intensity_tech, weight in get_intensity_technologies('wind_onshore').items():
        print(f"  → {intensity_tech}: {weight*100:.1f}%")
    print(f"  Lifetime: {get_lifetime('wind_onshore')} years")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    total = len(TECHNOLOGY_MAPPING)
    mapped = sum(1 for v in TECHNOLOGY_MAPPING.values() if v)
    unmapped = total - mapped
    print(f"Total capacity technologies: {total}")
    print(f"  Mapped: {mapped}")
    print(f"  Unmapped: {unmapped}")
