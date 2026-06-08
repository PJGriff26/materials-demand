"""Adds companion `*_annual` columns to the stock-flow demand statistics.

Each demand row is a 3-year bucket total (NREL Standard Scenarios report
capacity at 3-year intervals); this divides every statistical column by the
inferred bucket width so downstream consumers can compare per-year demand
against annual USGS production. Applied at export time; the raw MC array is
unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _bucket_years(
    years: pd.Series,
    group_keys: list[pd.Series] | None,
) -> pd.Series:
    """Bucket width per row = current_year − previous_reporting_year within
    the row's time series.

    `group_keys` identifies the time-series grouping. At minimum this must
    include the scenario column (if present) AND any per-row differentiator
    that shares a year with other rows, e.g. `material`. Without the
    per-row grouping, consecutive rows at the same year but different
    materials produce year-diff 0 and collapse the computation.

    Baseline-year rows (first in sequence per group) get NaN.
    """
    years = years.astype(int)
    if group_keys:
        return years.groupby(group_keys).diff()
    return years.diff()


def add_annualized_columns(
    df: pd.DataFrame,
    stat_cols: list[str],
    *,
    year_col: str = "year",
    scenario_col: str = "scenario",
    suffix: str = "_annual",
) -> pd.DataFrame:
    """Append `{col}{suffix}` for each stat column.

    Annualized value = column / bucket_years. std and every percentile
    scale linearly under division, so the same divisor applies uniformly.

    Also adds a `bucket_years` column for transparency and so downstream
    consumers can decide whether to use the annualized or period-total
    columns explicitly. Baseline-year rows (no previous reporting year)
    keep their original zero demand and receive 0 in the annualized
    columns (not NaN, which keeps downstream arithmetic clean).
    """
    out = df.copy()
    # Build the grouping for per-row year-diff. Any column that varies
    # across rows that share a year must be part of the group, or the
    # diff collapses to 0. Material is the standard per-row stratifier
    # in the stock-flow outputs; scenario distinguishes across-scenario
    # time series.
    group_cols: list[str] = []
    if scenario_col in out.columns:
        group_cols.append(scenario_col)
    if "material" in out.columns:
        group_cols.append("material")

    sort_cols = group_cols + [year_col] if group_cols else [year_col]
    out = out.sort_values(sort_cols).reset_index(drop=True)

    group_keys = [out[c] for c in group_cols] if group_cols else None
    out["bucket_years"] = _bucket_years(out[year_col], group_keys)

    divisor = out["bucket_years"].replace(0, np.nan)
    for col in stat_cols:
        if col not in out.columns:
            continue
        annualized = out[col] / divisor
        # Baseline-year rows have no previous reporting year; demand is zero
        # by construction in the stock-flow model, so the annualized rate is
        # also zero rather than undefined.
        annualized = annualized.where(
            out["bucket_years"].notna(),
            other=0.0,
        )
        out[col + suffix] = annualized

    return out
