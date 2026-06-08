"""Data quality checks and corrections for the raw material intensity data.

Provides outlier detection (IQR and z-score) plus documented corrections and
removals for known data entry errors, applied during ingestion without
modifying the source files.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

# Path to raw intensity data
DATA_DIR = Path(__file__).parent.parent / 'data'
INTENSITY_FILE = DATA_DIR / 'intensity_data.csv'

# ════════════════════════════════════════════════════════════════════════════════
# KNOWN DATA CORRECTIONS
# ════════════════════════════════════════════════════════════════════════════════
# These are documented corrections for known data entry errors.
# Each entry: (technology, material, wrong_value, corrected_value, reason)

KNOWN_CORRECTIONS = [
    # CIGS Indium: 44155 kg/MW appears twice - likely decimal error (should be 44.155)
    # Other CIGS Indium values range from 5-83 kg/MW
    ('CIGS', 'Indium', 44155, 44.155, 'Decimal placement error: 1000x too high vs other CIGS Indium values (5-83 kg/MW)'),

    # Silicon: 4 kg/MW appears in solar PV - likely should be 4000 kg/MW
    # Other silicon values for solar are 638-9000 kg/MW with median ~3653
    ('utility-scale solar pv', 'Silicon', 4, 4000, 'Decimal placement error: 1000x too low vs other solar Silicon values (median 3653 kg/MW)'),
    ('Solar Distributed', 'Silicon', 4, 4000, 'Decimal placement error: 1000x too low vs other solar Silicon values (median 3653 kg/MW)'),

    # Vanadium offshore wind: 7000 kg/MW appears twice (Fizaine & Court 2015 via liang_2022,
    # Watari 2019 via liang_2022 — Watari likely propagated from Fizaine, not independent).
    # The matching onshore-wind vanadium values are 90.4 and 90.0 kg/MW — a 78x offshore:onshore
    # ratio inconsistent with every other structural/alloy metal from the same Fizaine source
    # (Cr, Cu, Mo, Ni, Nb all scale ~1x between offshore and onshore).
    # Physical check: offshore steel is ~1.8x onshore, not 78x. Vanadium is a microalloying
    # element in structural steel (~0.05-0.5% of steel mass).
    # Pipeline check: uncorrected peak US vanadium demand reaches ~10M tonnes in 2032, which
    # exceeds 100x global annual production (~100 kt/yr per USGS MCS 2025) — physically
    # impossible.
    # Fix: borrow the onshore value (90 kg/MW). Both offshore entries (identical 7000 values)
    # are overwritten.
    # Caught by review (slipped past n<3 outlier detection + identical-value
    # variance test, since both offshore entries held the same 7000 value).
    ('offshore wind', 'Vanadium', 7000.0, 90.0,
     'Offshore:onshore ratio 78x inconsistent with other alloy metals '
     '(Cr/Cu/Mo/Ni/Nb ~1x ratio); uncorrected peak demand ~100x global production. '
     'Corrected to 90 kg/MW (matching onshore-wind vanadium values 90.0/90.4 from '
     'Fizaine 2015 and Watari 2019).'),
]

# Values flagged but NOT corrected (require expert review):
# - CIGS/Copper 7000-7530: 3 clustered values, may represent BOS (balance of system) vs cell-only
# - CIGS/Gallium 124: 2 identical occurrences at 18x median - could be different CIGS formulations
# - Onshore wind/Copper 14233-15350: 3 clustered high values, not isolated single points
# - CdTe/Cadmium 244-244.26: 2 near-identical values, not isolated

# ════════════════════════════════════════════════════════════════════════════════
# KNOWN SINGLE-POINT OUTLIER REMOVALS
# ════════════════════════════════════════════════════════════════════════════════
# These are single isolated extreme values with no supporting cluster.
# Each entry: (technology, material, value, reason)

KNOWN_REMOVALS = [
    # CIGS/Cadmium 265: 204x median in a 3-point group [1.3, 1.3, 265]
    ('CIGS', 'Cadmium', 265.0, 'Single point 204x median; other values are 1.3'),

    # CdTe/Tellurium 500: z=5.22, isolated jump from next-highest 260 (1.92x gap)
    ('CdTe', 'Tellurium', 500.0, 'Single point z=5.22; 1.92x gap above next-highest (260)'),

    # Onshore wind/Aluminum 13200: z=4.93, 2.49x gap above next-highest 5300
    ('onshore wind', 'Aluminum', 13200.0, 'Single point z=4.93; 2.49x gap above next-highest (5300)'),

    # Solar Lead 336: 4.67x gap above next-highest 72; 14.8x median in 6-point groups
    ('Solar Distributed', 'Lead', 336.0, 'Single point 14.8x median; 4.67x gap above next-highest (72)'),
    ('utility-scale solar pv', 'Lead', 336.0, 'Single point 14.8x median; 4.67x gap above next-highest (72)'),

    # Offshore wind/Nickel 376.57: 3.4x in 4-point group [111, 111, 111, 376.57]
    ('offshore wind', 'Nickel', 376.57, 'Single point 3.4x median; other 3 values are all 111'),
]

# ════════════════════════════════════════════════════════════════════════════════
# OUTLIER DETECTION THRESHOLDS
# ════════════════════════════════════════════════════════════════════════════════
# Values exceeding these thresholds (relative to group median) are flagged

IQR_MULTIPLIER = 3.0  # Flag values > Q3 + 3*IQR or < Q1 - 3*IQR
Z_SCORE_THRESHOLD = 4.0  # Flag values with |z-score| > 4
RATIO_THRESHOLD = 100  # Flag values > 100x the group median


def load_raw_intensity_data() -> pd.DataFrame:
    """Load raw intensity data without any corrections."""
    return pd.read_csv(INTENSITY_FILE)


def apply_known_corrections(df: pd.DataFrame) -> pd.DataFrame:
    """Apply documented corrections to known data entry errors."""
    df = df.copy()
    corrections_applied = []

    for tech, mat, wrong_val, correct_val, reason in KNOWN_CORRECTIONS:
        mask = (df['technology'] == tech) & (df['Material'] == mat) & (df['value'] == wrong_val)
        n_matches = mask.sum()
        if n_matches > 0:
            df.loc[mask, 'value'] = correct_val
            corrections_applied.append({
                'technology': tech,
                'material': mat,
                'original': wrong_val,
                'corrected': correct_val,
                'count': n_matches,
                'reason': reason
            })

    if corrections_applied:
        print(f"Applied {len(corrections_applied)} known corrections:")
        for c in corrections_applied:
            print(f"  {c['technology']}/{c['material']}: {c['original']} → {c['corrected']} ({c['count']}x) - {c['reason']}")

    return df


def detect_outliers(df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
    """
    Detect statistical outliers within each technology-material group.

    Args:
        df: Raw intensity data
        method: 'iqr' (interquartile range) or 'zscore'

    Returns:
        DataFrame of flagged outliers with context
    """
    outliers = []

    for (tech, mat), group in df.groupby(['technology', 'Material']):
        if len(group) < 3:  # Need at least 3 values for meaningful statistics
            continue

        values = group['value'].values
        median = np.median(values)
        mean = np.mean(values)
        std = np.std(values)
        q1, q3 = np.percentile(values, [25, 75])
        iqr = q3 - q1

        for idx, row in group.iterrows():
            val = row['value']
            is_outlier = False
            reason = []

            # IQR method
            if method == 'iqr' or method == 'all':
                lower_bound = q1 - IQR_MULTIPLIER * iqr
                upper_bound = q3 + IQR_MULTIPLIER * iqr
                if val < lower_bound or val > upper_bound:
                    is_outlier = True
                    reason.append(f'IQR: outside [{lower_bound:.2f}, {upper_bound:.2f}]')

            # Z-score method
            if (method == 'zscore' or method == 'all') and std > 0:
                z = (val - mean) / std
                if abs(z) > Z_SCORE_THRESHOLD:
                    is_outlier = True
                    reason.append(f'Z-score: {z:.2f}')

            # Ratio to median (catch extreme outliers)
            if median > 0 and val / median > RATIO_THRESHOLD:
                is_outlier = True
                reason.append(f'Ratio: {val/median:.1f}x median')

            if is_outlier:
                outliers.append({
                    'technology': tech,
                    'material': mat,
                    'value': val,
                    'median': median,
                    'mean': mean,
                    'std': std,
                    'n_values': len(group),
                    'ratio_to_median': val / median if median > 0 else np.inf,
                    'reason': '; '.join(reason)
                })

    return pd.DataFrame(outliers)


def detect_suspicious_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect suspicious patterns beyond simple outliers:
    - Duplicate exact values that seem wrong
    - Values that look like unit conversion errors (1000x, 1000000x)
    - Negative values
    - Zero values where non-zero expected
    """
    suspicious = []

    for (tech, mat), group in df.groupby(['technology', 'Material']):
        values = group['value'].values

        if len(values) < 2:
            continue

        median = np.median(values)

        # Check for potential unit errors (values that are 1000x or 1/1000x the median)
        for idx, row in group.iterrows():
            val = row['value']

            if median > 0:
                ratio = val / median

                # Check for 1000x error (could be kg vs g, or tonnes vs kg)
                if 900 < ratio < 1100:
                    suspicious.append({
                        'technology': tech,
                        'material': mat,
                        'value': val,
                        'median': median,
                        'issue': f'Possible unit error: ~1000x median ({ratio:.0f}x)',
                        'suggested_fix': f'{val} → {val/1000:.3f}'
                    })
                elif 0.0009 < ratio < 0.0011:
                    suspicious.append({
                        'technology': tech,
                        'material': mat,
                        'value': val,
                        'median': median,
                        'issue': f'Possible unit error: ~1/1000 of median ({ratio:.6f}x)',
                        'suggested_fix': f'{val} → {val*1000:.3f}'
                    })

        # Check for negative values
        neg_mask = group['value'] < 0
        if neg_mask.any():
            for idx, row in group[neg_mask].iterrows():
                suspicious.append({
                    'technology': tech,
                    'material': mat,
                    'value': row['value'],
                    'median': median,
                    'issue': 'Negative value',
                    'suggested_fix': 'Review data source'
                })

    return pd.DataFrame(suspicious)


def load_clean_intensity_data(apply_corrections: bool = True) -> pd.DataFrame:
    """
    Load intensity data with optional corrections applied.

    This is the main entry point for pipeline use.

    Args:
        apply_corrections: If True, apply known corrections to documented errors

    Returns:
        Cleaned intensity DataFrame
    """
    df = load_raw_intensity_data()

    if apply_corrections:
        df = apply_known_corrections(df)

    return df


def generate_quality_report(output_file: Optional[Path] = None) -> str:
    """
    Generate a comprehensive data quality report.

    Args:
        output_file: Optional path to save the report

    Returns:
        Report as string
    """
    df = load_raw_intensity_data()

    lines = []
    lines.append("=" * 80)
    lines.append("MATERIAL INTENSITY DATA QUALITY REPORT")
    lines.append("=" * 80)
    lines.append("")

    # Basic stats
    lines.append(f"Total records: {len(df)}")
    lines.append(f"Technologies: {df['technology'].nunique()}")
    lines.append(f"Materials: {df['Material'].nunique()}")
    lines.append(f"Tech-Material combinations: {df.groupby(['technology', 'Material']).ngroups}")
    lines.append("")

    # Known corrections
    lines.append("-" * 80)
    lines.append("KNOWN DATA CORRECTIONS (will be applied automatically)")
    lines.append("-" * 80)
    for tech, mat, wrong, correct, reason in KNOWN_CORRECTIONS:
        count = ((df['technology'] == tech) & (df['Material'] == mat) & (df['value'] == wrong)).sum()
        lines.append(f"  {tech} / {mat}: {wrong} → {correct} ({count} occurrences)")
        lines.append(f"    Reason: {reason}")
    lines.append("")

    # Statistical outliers
    lines.append("-" * 80)
    lines.append("STATISTICAL OUTLIERS (flagged for review)")
    lines.append("-" * 80)
    outliers = detect_outliers(df, method='all')
    if len(outliers) > 0:
        # Sort by ratio to median (most extreme first)
        outliers = outliers.sort_values('ratio_to_median', ascending=False)
        for _, row in outliers.head(30).iterrows():
            lines.append(f"  {row['technology']} / {row['material']}: {row['value']:.2f}")
            lines.append(f"    Median: {row['median']:.2f}, Ratio: {row['ratio_to_median']:.1f}x, N={row['n_values']}")
            lines.append(f"    Flagged: {row['reason']}")
    else:
        lines.append("  No statistical outliers detected.")
    lines.append("")

    # Suspicious patterns
    lines.append("-" * 80)
    lines.append("SUSPICIOUS PATTERNS (potential unit errors)")
    lines.append("-" * 80)
    suspicious = detect_suspicious_patterns(df)
    if len(suspicious) > 0:
        for _, row in suspicious.iterrows():
            lines.append(f"  {row['technology']} / {row['material']}: {row['value']}")
            lines.append(f"    Issue: {row['issue']}")
            lines.append(f"    Suggested: {row['suggested_fix']}")
    else:
        lines.append("  No suspicious patterns detected.")
    lines.append("")

    # Summary by technology
    lines.append("-" * 80)
    lines.append("SUMMARY BY TECHNOLOGY")
    lines.append("-" * 80)
    for tech in sorted(df['technology'].unique()):
        tech_df = df[df['technology'] == tech]
        n_outliers = len(outliers[outliers['technology'] == tech]) if len(outliers) > 0 else 0
        lines.append(f"  {tech}: {len(tech_df)} records, {tech_df['Material'].nunique()} materials, {n_outliers} outliers")

    report = "\n".join(lines)

    if output_file:
        with open(output_file, 'w') as f:
            f.write(report)
        print(f"Report saved to {output_file}")

    return report


if __name__ == '__main__':
    # When run directly, generate and print the quality report
    report = generate_quality_report()
    print(report)

    # Also save to outputs
    output_dir = DATA_DIR.parent / 'outputs'
    output_dir.mkdir(exist_ok=True)
    generate_quality_report(output_dir / 'data_quality_report.txt')
