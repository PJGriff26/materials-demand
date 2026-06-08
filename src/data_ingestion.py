"""Loaders and validators for the model's input data.

Reads and standardizes material intensity data, electricity generation
capacity projections, and transmission capacity projections, with validation,
unit checking, and logging.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
import warnings

# Import data quality module for outlier detection and corrections
from src.data_quality import (
    KNOWN_CORRECTIONS,
    KNOWN_REMOVALS,
    IQR_MULTIPLIER,
    detect_outliers,
)

# Import technology consolidation mapping
from src.technology_mapping import TECHNOLOGY_CONSOLIDATION

logger = logging.getLogger(__name__)


@dataclass
class DataValidationResult:
    """Container for validation results"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict
    
    def __post_init__(self):
        """Ensure errors and warnings are lists"""
        if self.errors is None:
            self.errors = []
        if self.warnings is None:
            self.warnings = []
        if self.metadata is None:
            self.metadata = {}
    
    def log_results(self):
        """Log validation results"""
        if self.errors:
            logger.error(f"Validation failed with {len(self.errors)} errors:")
            for error in self.errors:
                logger.error(f"  - {error}")
        if self.warnings:
            logger.warning(f"Validation completed with {len(self.warnings)} warnings:")
            for warning in self.warnings:
                logger.warning(f"  - {warning}")
        if self.is_valid:
            logger.info("Validation passed successfully")


class MaterialIntensityLoader:
    """
    Loads and validates material intensity data.
    
    Material intensity represents the mass of material required per unit capacity
    of energy generation technology (typically metric tons per MW).
    """
    
    REQUIRED_COLUMNS = ['technology', 'Material', 'value']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.validation_result: Optional[DataValidationResult] = None
        
    def load(self, filepath: Union[str, Path],
             apply_corrections: bool = True,
             filter_outliers: bool = True) -> pd.DataFrame:
        """
        Load material intensity data from CSV file.

        Parameters
        ----------
        filepath : str or Path
            Path to material intensity CSV file
        apply_corrections : bool, default True
            If True, apply known corrections for documented data entry errors
            (e.g., CIGS Indium 44155 → 44.155)
        filter_outliers : bool, default True
            If True, remove statistical outliers (>100x median) to prevent
            distribution fitting from being skewed by data entry errors

        Returns
        -------
        pd.DataFrame
            Loaded and validated material intensity data

        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If validation fails
        """
        filepath = Path(filepath)

        if not filepath.exists():
            raise FileNotFoundError(f"Material intensity file not found: {filepath}")

        logger.info(f"Loading material intensity data from: {filepath}")
        logger.info(f"  apply_corrections={apply_corrections}, filter_outliers={filter_outliers}")

        # Load data
        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")

        # Validate
        self.data = df
        self.validation_result = self._validate()
        self.validation_result.log_results()

        if not self.validation_result.is_valid:
            raise ValueError("Material intensity data validation failed. See log for details.")

        # Apply cleaning/standardization with data quality filtering
        self.data = self._standardize(
            apply_corrections=apply_corrections,
            filter_outliers=filter_outliers
        )
        
        logger.info(f"Successfully loaded {len(self.data)} material intensity records")
        logger.info(f"  Technologies: {self.data['technology'].nunique()}")
        logger.info(f"  Materials: {self.data['material'].nunique()}")
        
        return self.data
    
    def _validate(self) -> DataValidationResult:
        """
        Validate material intensity data structure and content.
        
        Returns
        -------
        DataValidationResult
            Validation results with errors, warnings, and metadata
        """
        errors = []
        warnings = []
        metadata = {}
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return DataValidationResult(False, errors, warnings, metadata)
        
        # Check for null values
        null_counts = self.data[self.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check value column is numeric
        if not pd.api.types.is_numeric_dtype(self.data['value']):
            errors.append("'value' column must be numeric")
        else:
            # Check for negative values
            negative_count = (self.data['value'] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative values (material intensity must be >= 0)")
            
            # Check for zeros (warning only)
            zero_count = (self.data['value'] == 0).sum()
            if zero_count > 0:
                warnings.append(f"Found {zero_count} zero values (valid but unusual)")
            
            # Statistical summary for metadata
            metadata['value_statistics'] = {
                'min': float(self.data['value'].min()),
                'max': float(self.data['value'].max()),
                'mean': float(self.data['value'].mean()),
                'median': float(self.data['value'].median()),
                'std': float(self.data['value'].std())
            }
        
        # Check for duplicate (technology, material) combinations
        duplicates = self.data.groupby(['technology', 'Material']).size()
        duplicates = duplicates[duplicates > 1]
        if len(duplicates) > 0:
            warnings.append(
                f"Found {len(duplicates)} technology-material combinations with multiple values. "
                f"This is expected for representing uncertainty."
            )
            metadata['duplicate_combinations'] = len(duplicates)
        
        # Record unique technologies and materials (skip if data has nulls to avoid errors)
        try:
            metadata['unique_technologies'] = sorted(self.data['technology'].dropna().unique().tolist())
            metadata['unique_materials'] = sorted(self.data['Material'].dropna().unique().tolist())
            metadata['n_technologies'] = self.data['technology'].nunique()
            metadata['n_materials'] = self.data['Material'].nunique()
        except (TypeError, ValueError) as e:
            # If we can't compute metadata due to data issues, just skip it
            metadata['metadata_error'] = str(e)
        
        metadata['total_records'] = len(self.data)
        
        is_valid = len(errors) == 0
        
        return DataValidationResult(is_valid, errors, warnings, metadata)
    
    def _standardize(self, apply_corrections: bool = True, filter_outliers: bool = True) -> pd.DataFrame:
        """
        Standardize material intensity data.

        CRITICAL: Source data is in tonnes/GW (t/GW), but capacity data is in MW.
        Therefore, we must divide by 1000 to convert to tonnes/MW (t/MW).

        Conversion: 1 GW = 1000 MW
        Example: 7000 t/GW ÷ 1000 = 7 t/MW

        Parameters
        ----------
        apply_corrections : bool
            If True, apply known corrections for documented data entry errors
        filter_outliers : bool
            If True, remove statistical outliers (>100x median) from distribution fitting

        Returns
        -------
        pd.DataFrame
            Standardized data with intensity in tonnes per MW
        """
        df = self.data.copy()

        # ════════════════════════════════════════════════════════════════════════
        # STEP 1: Apply known data corrections BEFORE any processing
        # ════════════════════════════════════════════════════════════════════════
        if apply_corrections:
            corrections_applied = 0
            for tech, mat, wrong_val, correct_val, reason in KNOWN_CORRECTIONS:
                mask = (df['technology'] == tech) & (df['Material'] == mat) & (df['value'] == wrong_val)
                n_matches = mask.sum()
                if n_matches > 0:
                    df.loc[mask, 'value'] = correct_val
                    corrections_applied += n_matches
                    logger.info(f"Data correction: {tech}/{mat}: {wrong_val} → {correct_val} ({n_matches}x)")
            if corrections_applied > 0:
                logger.info(f"Applied {corrections_applied} known data corrections")

        # ════════════════════════════════════════════════════════════════════════
        # STEP 1b: Remove known single-point outliers
        # ════════════════════════════════════════════════════════════════════════
        if apply_corrections:
            removals_applied = 0
            for tech, mat, bad_val, reason in KNOWN_REMOVALS:
                mask = (df['technology'] == tech) & (df['Material'] == mat) & (df['value'] == bad_val)
                n_matches = mask.sum()
                if n_matches > 0:
                    df = df[~mask].copy()
                    removals_applied += n_matches
                    logger.info(f"Outlier removal: {tech}/{mat} = {bad_val} ({reason})")
            if removals_applied > 0:
                logger.info(f"Removed {removals_applied} known single-point outliers")

        # ════════════════════════════════════════════════════════════════════════
        # STEP 1c: Consolidate split technology entries
        # ════════════════════════════════════════════════════════════════════════
        # Some technologies in intensity_data.csv are split into cell-specific
        # and BOS (balance-of-system) entries. For example, 'CDTE' contains BOS
        # materials (Aluminum, Cement, Copper, Glass, Steel) that belong with 'CdTe'.
        # See TECHNOLOGY_CONSOLIDATION in technology_mapping.py for details.
        consolidated = 0
        for old_name, new_name in TECHNOLOGY_CONSOLIDATION.items():
            mask = df['technology'] == old_name
            n_matches = mask.sum()
            if n_matches > 0:
                df.loc[mask, 'technology'] = new_name
                consolidated += n_matches
                logger.info(
                    f"Technology consolidation: {old_name} → {new_name} ({n_matches} rows)"
                )
        if consolidated > 0:
            logger.info(f"Consolidated {consolidated} rows across {len(TECHNOLOGY_CONSOLIDATION)} technology aliases")

        # ════════════════════════════════════════════════════════════════════════
        # STEP 2: Filter statistical outliers (>100x median within tech-material group)
        # ════════════════════════════════════════════════════════════════════════
        if filter_outliers:
            n_before = len(df)
            outlier_mask = pd.Series(False, index=df.index)

            for (tech, mat), group in df.groupby(['technology', 'Material']):
                if len(group) < 3:
                    continue  # Need at least 3 values for meaningful statistics

                values = group['value'].values
                median = np.median(values)

                if median > 0:
                    # Flag values >100x the median as outliers
                    group_outliers = group['value'] / median > 100
                    outlier_mask.loc[group.index] = group_outliers

            n_outliers = outlier_mask.sum()
            if n_outliers > 0:
                # Log which outliers are being removed
                outlier_rows = df[outlier_mask]
                for _, row in outlier_rows.iterrows():
                    logger.warning(f"Filtering outlier: {row['technology']}/{row['Material']} = {row['value']}")

                df = df[~outlier_mask].copy()
                logger.info(f"Filtered {n_outliers} statistical outliers (>100x median)")

            n_after = len(df)
            if n_before != n_after:
                logger.info(f"Records: {n_before} → {n_after} after outlier filtering")

        # ════════════════════════════════════════════════════════════════════════
        # STEP 3: Standardize column names
        # ════════════════════════════════════════════════════════════════════════
        df = df.rename(columns={
            'technology': 'technology',
            'Material': 'material',  # Lowercase for consistency
            'value': 'intensity_raw'  # Temporary name before conversion
        })

        # Strip whitespace from string columns
        df['technology'] = df['technology'].str.strip()
        df['material'] = df['material'].str.strip()

        # ════════════════════════════════════════════════════════════════════════
        # STEP 4: UNIT CONVERSION: t/GW → t/MW
        # ════════════════════════════════════════════════════════════════════════
        # Source data is in tonnes per gigawatt (t/GW)
        # Capacity data is in megawatts (MW)
        # Therefore: divide by 1000 to convert to tonnes per MW (t/MW)
        df['intensity_t_per_mw'] = df['intensity_raw'].astype(float) / 1000.0

        # Drop the temporary raw column
        df = df.drop(columns=['intensity_raw'])

        # ════════════════════════════════════════════════════════════════════════
        # STEP 4b: PV Cu boundary harmonization: drop cell+module-level rows
        # ════════════════════════════════════════════════════════════════════════
        # PV Cu intensity literature mixes three LCA system boundaries (cell ~17–
        # 100 t/GW, module ~200–500, BOS-inclusive ~500–7530), inflating MLE σ
        # and producing unphysical MC tails. The spread is visible across the
        # intensity_data.csv sources (cell-level e.g. Watari 2019, Carrara 2020,
        # Moss 2013; BOS-inclusive e.g. Fizaine & Court 2015, Mason 2006, IRENA
        # 2017); the boundary distinction itself is standard PV-LCA practice
        # (IEA-PVPS Task 12 life-cycle inventory 2020; Sinha & de Wild-Scholten
        # 2012, utility-scale PV balance-of-systems LCA). We model per-MW
        # operational capacity = BOS-inclusive (cradle-to-grave plant scope), so
        # cell- and module-only readings are excluded for the four PV
        # technologies. The 0.5 t/MW (= 500 t/GW) cutoff is a harmonization
        # threshold to that scope, not a single published value.
        PV_CU_PAIRS = [
            ('utility-scale solar pv', 'Copper'),
            ('CdTe', 'Copper'),
            ('CIGS', 'Copper'),
            ('a-Si', 'Copper'),
        ]
        pv_cu_drop_mask = pd.Series(False, index=df.index)
        for tech, mat in PV_CU_PAIRS:
            pair_mask = (df['technology'] == tech) & (df['material'] == mat)
            pv_cu_drop_mask |= pair_mask & (df['intensity_t_per_mw'] < 0.5)
        n_pv_cu_dropped = int(pv_cu_drop_mask.sum())
        if n_pv_cu_dropped > 0:
            for _, row in df[pv_cu_drop_mask].iterrows():
                logger.info(
                    f"PV-Cu boundary filter: dropped {row['technology']}/{row['material']} "
                    f"= {row['intensity_t_per_mw']:.3f} t/MW (sub-BOS)"
                )
            df = df[~pv_cu_drop_mask].copy()
            logger.info(f"PV Cu boundary harmonization: dropped {n_pv_cu_dropped} cell/module-level rows")

        # Sort for reproducibility
        df = df.sort_values(['technology', 'material', 'intensity_t_per_mw']).reset_index(drop=True)

        logger.info("Applied unit conversion: t/GW → t/MW (divided by 1000)")

        return df
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics by technology and material.

        Returns
        -------
        pd.DataFrame
            Summary statistics including mean, std, min, max, count
            Values are in tonnes per MW (t/MW)
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")

        summary = self.data.groupby(['technology', 'material'])['intensity_t_per_mw'].agg([
            ('count', 'count'),
            ('mean', 'mean'),
            ('std', 'std'),
            ('min', 'min'),
            ('25%', lambda x: x.quantile(0.25)),
            ('median', 'median'),
            ('75%', lambda x: x.quantile(0.75)),
            ('max', 'max')
        ]).reset_index()

        return summary


class CapacityProjectionLoader:
    """
    Loads and validates electricity generation capacity projection data.
    
    Handles both national and state-level projections from NREL Standard Scenarios.
    """
    
    # Column name mappings from NREL short codes to standardized names
    CAPACITY_COLUMNS = {
        'battery_4_MW': 'battery_4_MW',
        'battery_8_MW': 'battery_8_MW',
        'bio_MW': 'bio_MW',
        'bio-ccs_MW': 'bio-ccs_MW',
        'coal_ccs_MW': 'coal_ccs_MW',
        'coal_MW': 'coal_MW',
        'csp_MW': 'csp_MW',
        'distpv_MW': 'distpv_MW',
        'gas_cc_ccs_MW': 'gas_cc_ccs_MW',
        'gas_cc_MW': 'gas_cc_MW',
        'gas_ct_MW': 'gas_ct_MW',
        'geo_MW': 'geo_MW',
        'hydro_MW': 'hydro_MW',
        'nuclear_MW': 'nuclear_MW',
        'nuclear_smr_MW': 'nuclear_smr_MW',
        'o-g-s_MW': 'o-g-s_MW',
        'pumped-hydro_MW': 'pumped-hydro_MW',
        'h2-ct_MW': 'h2-ct_MW',
        'upv_MW': 'upv_MW',
        'wind_offshore_MW': 'wind_offshore_MW',
        'wind_onshore_MW': 'wind_onshore_MW',
        'dac_MW': 'dac_MW',
    }
    
    METADATA_COLUMNS = ['scenario', 't']
    STATE_COLUMN = 'state'
    
    def __init__(self, level: str = 'national'):
        """
        Initialize loader.
        
        Parameters
        ----------
        level : str
            Either 'national' or 'state'
        """
        if level not in ['national', 'state']:
            raise ValueError("level must be 'national' or 'state'")
        
        self.level = level
        self.data: Optional[pd.DataFrame] = None
        self.validation_result: Optional[DataValidationResult] = None
        
    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load capacity projection data from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to capacity projection CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded and validated capacity projection data
            
        Raises
        ------
        FileNotFoundError
            If file does not exist
        ValueError
            If validation fails
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Capacity projection file not found: {filepath}")
        
        logger.info(f"Loading {self.level} capacity projections from: {filepath}")
        
        # Load data, skipping header rows
        # NREL format has 3 header rows: description, note, column names (long), column codes (short)
        try:
            df = pd.read_csv(filepath, skiprows=3, low_memory=False)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate
        self.data = df
        self.validation_result = self._validate()
        self.validation_result.log_results()
        
        if not self.validation_result.is_valid:
            raise ValueError("Capacity projection data validation failed. See log for details.")
        
        # Apply cleaning/standardization
        self.data = self._standardize()
        
        logger.info(f"Successfully loaded {len(self.data)} capacity projection records")
        logger.info(f"  Scenarios: {self.data['scenario'].nunique()}")
        logger.info(f"  Years: {sorted(self.data['year'].unique())}")
        if self.level == 'state':
            logger.info(f"  States: {self.data['state'].nunique()}")
        
        return self.data
    
    def _validate(self) -> DataValidationResult:
        """
        Validate capacity projection data.
        
        Returns
        -------
        DataValidationResult
            Validation results
        """
        errors = []
        warnings = []
        metadata = {}
        
        # Check for scenario column
        if 'scenario' not in self.data.columns:
            errors.append("Missing 'scenario' column")
        
        # Check for year column (uses 't' as short code)
        if 't' not in self.data.columns:
            errors.append("Missing 't' (year) column")
        
        # For state-level data, check for State column
        if self.level == 'state' and 'state' not in self.data.columns:
            errors.append("Missing 'state' column for state-level data")
        
        # Check for capacity columns
        missing_capacity_cols = []
        for long_name in self.CAPACITY_COLUMNS.keys():
            if long_name not in self.data.columns:
                missing_capacity_cols.append(long_name)
        
        if missing_capacity_cols:
            warnings.append(
                f"Missing {len(missing_capacity_cols)} capacity columns. "
                f"This may be expected if certain technologies are not modeled."
            )
        
        # If we have year column, validate years
        if 't' in self.data.columns:
            years = self.data['t'].unique()
            metadata['years'] = sorted(years.tolist())
            metadata['n_years'] = len(years)
            
            # Check for reasonable year range
            min_year, max_year = years.min(), years.max()
            if min_year < 2020 or max_year > 2060:
                warnings.append(f"Unusual year range: {min_year} to {max_year}")
        
        # Record scenarios
        if 'scenario' in self.data.columns:
            metadata['scenarios'] = sorted(self.data['scenario'].unique().tolist())
            metadata['n_scenarios'] = self.data['scenario'].nunique()
        
        # For state-level, record states
        if self.level == 'state' and 'state' in self.data.columns:
            metadata['states'] = sorted(self.data['state'].unique().tolist())
            metadata['n_states'] = self.data['state'].nunique()
        
        metadata['total_records'] = len(self.data)
        
        is_valid = len(errors) == 0
        
        return DataValidationResult(is_valid, errors, warnings, metadata)
    
    def _standardize(self) -> pd.DataFrame:
        """
        Standardize capacity projection data.
        
        Returns
        -------
        pd.DataFrame
            Standardized data
        """
        df = self.data.copy()
        
        # Rename 't' to 'year' for clarity
        rename_dict = {
            't': 'year'
        }
        
        # Columns are already in short code format from NREL data
        # Just ensure they match our expected names (should be identity mapping mostly)
        # Only rename columns that exist
        rename_dict = {k: v for k, v in rename_dict.items() if k in df.columns}
        df = df.rename(columns=rename_dict)
        
        # Select metadata columns + capacity columns
        keep_cols = ['scenario', 'year']
        if self.level == 'state':
            keep_cols.append('state')
        
        # Add all capacity columns that exist
        capacity_cols_present = [col for col in self.CAPACITY_COLUMNS.keys() if col in df.columns]
        keep_cols.extend(capacity_cols_present)
        
        df = df[keep_cols]
        
        # Ensure year is integer
        df['year'] = df['year'].astype(int)
        
        # Ensure capacity columns are numeric
        capacity_cols = [col for col in df.columns if col.endswith('_MW')]
        for col in capacity_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sort for reproducibility
        sort_cols = ['scenario', 'year']
        if self.level == 'state':
            sort_cols.insert(1, 'state')
        df = df.sort_values(sort_cols).reset_index(drop=True)
        
        return df
    
    def extract_capacity_additions(
        self, 
        baseline_year: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Calculate annual capacity additions (new capacity built each year).
        
        Parameters
        ----------
        baseline_year : int, optional
            Year to use as baseline. If None, uses first year in data.
            
        Returns
        -------
        pd.DataFrame
            Capacity additions by year
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load() first.")
        
        df = self.data.copy()
        
        # Identify capacity columns
        capacity_cols = [col for col in df.columns if col.endswith('_MW')]
        
        # Group columns
        group_cols = ['scenario']
        if self.level == 'state':
            group_cols.append('state')
        
        # Sort by year within each group
        df = df.sort_values(group_cols + ['year'])
        
        # Calculate differences (additions)
        additions = df.copy()
        
        for col in capacity_cols:
            # Calculate difference from previous year within each group
            additions[col] = df.groupby(group_cols)[col].diff()
        
        # For the first year in each group, use absolute capacity if no baseline specified
        if baseline_year is None:
            # First year is treated as new capacity (diff will be NaN)
            # We'll keep the first year's capacity as-is
            pass
        else:
            # TODO: Implement baseline year logic
            pass
        
        # Remove first year where diff is NaN (or keep if it represents initial buildout)
        # For now, we'll fill NaN with the actual capacity (represents buildout from zero)
        for col in capacity_cols:
            additions[col] = additions[col].fillna(df[col])
        
        return additions


class TransmissionCapacityLoader:
    """
    Loads and validates transmission capacity data.
    
    Transmission data represents capacity of power lines between regions.
    """
    
    REQUIRED_COLUMNS = ['scenario', 'r', 'rr', 'trtype', 't', 'tran_cap_MW']
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.validation_result: Optional[DataValidationResult] = None
        
    def load(self, filepath: Union[str, Path]) -> pd.DataFrame:
        """
        Load transmission capacity data from CSV file.
        
        Parameters
        ----------
        filepath : str or Path
            Path to transmission capacity CSV file
            
        Returns
        -------
        pd.DataFrame
            Loaded and validated transmission capacity data
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Transmission capacity file not found: {filepath}")
        
        logger.info(f"Loading transmission capacity data from: {filepath}")
        
        # Load data, skipping header rows
        try:
            df = pd.read_csv(filepath, skiprows=3)
        except Exception as e:
            raise ValueError(f"Failed to read CSV file: {e}")
        
        # Validate
        self.data = df
        self.validation_result = self._validate()
        self.validation_result.log_results()
        
        if not self.validation_result.is_valid:
            raise ValueError("Transmission capacity data validation failed. See log for details.")
        
        # Apply cleaning/standardization
        self.data = self._standardize()
        
        logger.info(f"Successfully loaded {len(self.data)} transmission capacity records")
        logger.info(f"  Scenarios: {self.data['scenario'].nunique()}")
        logger.info(f"  Years: {sorted(self.data['year'].unique())}")
        logger.info(f"  Region pairs: {len(self.data[['region_from', 'region_to']].drop_duplicates())}")
        
        return self.data
    
    def _validate(self) -> DataValidationResult:
        """
        Validate transmission capacity data.
        
        Returns
        -------
        DataValidationResult
            Validation results
        """
        errors = []
        warnings = []
        metadata = {}
        
        # Check required columns
        missing_cols = set(self.REQUIRED_COLUMNS) - set(self.data.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
            return DataValidationResult(False, errors, warnings, metadata)
        
        # Check for null values
        null_counts = self.data[self.REQUIRED_COLUMNS].isnull().sum()
        if null_counts.any():
            errors.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
        
        # Check capacity is numeric and non-negative
        if not pd.api.types.is_numeric_dtype(self.data['tran_cap_MW']):
            errors.append("'tran_cap_MW' column must be numeric")
        else:
            negative_count = (self.data['tran_cap_MW'] < 0).sum()
            if negative_count > 0:
                errors.append(f"Found {negative_count} negative capacity values")
        
        # Record metadata
        metadata['scenarios'] = sorted(self.data['scenario'].unique().tolist())
        metadata['n_scenarios'] = self.data['scenario'].nunique()
        metadata['years'] = sorted(self.data['t'].unique().tolist())
        metadata['n_years'] = self.data['t'].nunique()
        metadata['transmission_types'] = sorted(self.data['trtype'].unique().tolist())
        metadata['total_records'] = len(self.data)
        
        is_valid = len(errors) == 0
        
        return DataValidationResult(is_valid, errors, warnings, metadata)
    
    def _standardize(self) -> pd.DataFrame:
        """
        Standardize transmission capacity data.
        
        Returns
        -------
        pd.DataFrame
            Standardized data
        """
        df = self.data.copy()
        
        # Rename columns for clarity
        df = df.rename(columns={
            'r': 'region_from',
            'rr': 'region_to',
            'trtype': 'transmission_type',
            't': 'year',
            'tran_cap_MW': 'capacity_mw'
        })
        
        # Ensure year is integer
        df['year'] = df['year'].astype(int)
        
        # Ensure capacity is float
        df['capacity_mw'] = df['capacity_mw'].astype(float)
        
        # Sort for reproducibility
        df = df.sort_values([
            'scenario', 'year', 'region_from', 'region_to', 'transmission_type'
        ]).reset_index(drop=True)
        
        return df


# Convenience function for loading all data
def load_all_data(
    intensity_path: Union[str, Path],
    national_capacity_path: Union[str, Path],
    state_capacity_path: Optional[Union[str, Path]] = None,
    transmission_path: Optional[Union[str, Path]] = None
) -> Dict[str, pd.DataFrame]:
    """
    Load all data files with validation.
    
    Parameters
    ----------
    intensity_path : str or Path
        Path to material intensity CSV
    national_capacity_path : str or Path
        Path to national capacity projections CSV
    state_capacity_path : str or Path, optional
        Path to state capacity projections CSV
    transmission_path : str or Path, optional
        Path to transmission capacity CSV
        
    Returns
    -------
    dict
        Dictionary containing loaded data:
        - 'intensity': Material intensity data
        - 'capacity_national': National capacity projections
        - 'capacity_state': State capacity projections (if provided)
        - 'transmission': Transmission capacity (if provided)
    """
    logger.info("="*80)
    logger.info("Loading all data files")
    logger.info("="*80)
    
    data = {}
    
    # Load material intensity
    mi_loader = MaterialIntensityLoader()
    data['intensity'] = mi_loader.load(intensity_path)
    
    # Load national capacity
    nc_loader = CapacityProjectionLoader(level='national')
    data['capacity_national'] = nc_loader.load(national_capacity_path)
    
    # Load state capacity if provided
    if state_capacity_path is not None:
        sc_loader = CapacityProjectionLoader(level='state')
        data['capacity_state'] = sc_loader.load(state_capacity_path)
    
    # Load transmission if provided
    if transmission_path is not None:
        trans_loader = TransmissionCapacityLoader()
        data['transmission'] = trans_loader.load(transmission_path)
    
    logger.info("="*80)
    logger.info("All data loaded successfully")
    logger.info("="*80)
    
    return data


if __name__ == "__main__":
    # Example usage
    print("Data Ingestion Module for Materials Demand Analysis")
    print("This module should be imported and used in analysis scripts.")
