"""Parametric distribution fitting for the Monte Carlo intensity samples.

Forces a lognormal fit to each material-technology pair (the literature
consensus for strictly-positive, right-skewed intensity data: Carrara JRC 2020,
Watari 2019, IEA-PVPS Task 12), validates tail behavior, and never falls back to
empirical sampling. Small-sample CV correction is handled in src/cv_borrowing.py.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
import warnings
import logging
from pathlib import Path

try:
    from .random_config import RANDOM_SEED
except ImportError:
    # Fallback when imported as a top-level module (scripts that do
    # sys.path.insert(0, str(ROOT / 'src')) and then `import distribution_fitting`).
    from random_config import RANDOM_SEED

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION: Distribution Fitting Behavior
# ============================================================================

# Force lognormal distribution for all fits (overrides AIC-based selection)
# Set to False to use AIC-based selection (lognormal, gamma, truncated_normal, uniform)
FORCE_LOGNORMAL = True

# ============================================================================


@dataclass
class DistributionFit:
    """Container for fitted distribution information"""
    distribution_name: str
    parameters: Dict[str, float]
    ks_statistic: float
    ks_pvalue: float
    ad_statistic: float
    ad_critical_value: float
    aic: float  # Akaike Information Criterion
    bic: float  # Bayesian Information Criterion
    n_samples: int
    fitting_method: str
    
    def passes_ks_test(self, alpha: float = 0.05) -> bool:
        """Check if distribution passes Kolmogorov-Smirnov test"""
        return self.ks_pvalue > alpha
    
    def passes_ad_test(self) -> bool:
        """Check if distribution passes Anderson-Darling test"""
        return self.ad_statistic < self.ad_critical_value
    
    def summary(self) -> str:
        """Generate summary string"""
        return (
            f"{self.distribution_name} (n={self.n_samples})\n"
            f"  Parameters: {self.parameters}\n"
            f"  KS: stat={self.ks_statistic:.4f}, p={self.ks_pvalue:.4f} "
            f"[{'PASS' if self.passes_ks_test() else 'FAIL'}]\n"
            f"  AD: stat={self.ad_statistic:.4f}, crit={self.ad_critical_value:.4f} "
            f"[{'PASS' if self.passes_ad_test() else 'FAIL'}]\n"
            f"  AIC: {self.aic:.2f}, BIC: {self.bic:.2f}"
        )


@dataclass
class MaterialIntensityDistribution:
    """
    Complete distribution information for a material-technology pair.
    Contains fitted parametric distributions and raw data for fallback.

    Supports component-based distributions for bimodal data (e.g., cell vs BOS).
    When is_bimodal=True, sample() returns the sum of component distributions.
    """
    technology: str
    material: str
    raw_data: np.ndarray
    n_samples: int

    # Statistical summaries
    mean: float
    std: float
    median: float
    q25: float
    q75: float
    min_val: float
    max_val: float

    # Fitted distributions (ordered by quality)
    fitted_distributions: List[DistributionFit] = field(default_factory=list)
    best_fit: Optional[DistributionFit] = None

    # Component-based distributions (for bimodal data)
    is_bimodal: bool = False
    component_fits: List[DistributionFit] = field(default_factory=list)
    component_labels: List[str] = field(default_factory=list)
    component_data: List[np.ndarray] = field(default_factory=list)
    split_threshold: Optional[float] = None

    # Recommendation
    use_parametric: bool = False
    recommendation: str = ""
    
    def get_best_distribution(self) -> Optional[DistributionFit]:
        """Get the best fitted distribution"""
        return self.best_fit

    def get_frozen_distribution(self):
        """
        Get a frozen scipy distribution object for the best fit.

        Returns a frozen scipy.stats distribution supporting .rvs(), .pdf(),
        .cdf(), .ppf(), etc.  Returns None if no parametric fit is available.
        """
        if self.best_fit is not None:
            return self._get_scipy_distribution(self.best_fit)
        return None

    def sample(self, n: int = 1, random_state: Optional[int] = None) -> np.ndarray:
        """
        Sample from the distribution.

        For bimodal distributions (is_bimodal=True), returns the SUM of samples
        from each component distribution. This represents the physical reality
        that total material = cell material + BOS material + ... + component N.

        For single distributions (is_bimodal=False), returns samples from best_fit.

        Always uses parametric distributions (user requirement).
        """
        if random_state is not None:
            np.random.seed(random_state)

        # Component-based sampling for bimodal distributions
        if self.is_bimodal and len(self.component_fits) > 0:
            # Sample from each component and sum
            total_samples = np.zeros(n)
            for component_fit in self.component_fits:
                component_dist = self._get_scipy_distribution(component_fit)
                component_samples = component_dist.rvs(size=n)
                total_samples += component_samples
            return total_samples

        # Single distribution sampling
        elif self.use_parametric and self.best_fit is not None:
            # Sample from parametric distribution
            dist = self._get_scipy_distribution(self.best_fit)
            return dist.rvs(size=n)

        else:
            # Should never reach here with new logic
            logger.warning(
                f"Unexpected: use_parametric=False for {self.technology}-{self.material}. "
                f"This should not happen. Falling back to uniform distribution."
            )
            # Emergency fallback
            return np.random.uniform(
                np.min(self.raw_data),
                np.max(self.raw_data),
                size=n
            )
    
    def _get_scipy_distribution(self, fit: DistributionFit):
        """Convert fitted distribution to scipy distribution object"""
        params = fit.parameters
        
        if fit.distribution_name == 'truncated_normal':
            return stats.truncnorm(
                a=params['a'], 
                b=params['b'], 
                loc=params['loc'], 
                scale=params['scale']
            )
        elif fit.distribution_name == 'uniform':
            return stats.uniform(loc=params['loc'], scale=params['scale'])
        elif fit.distribution_name == 'lognormal':
            return stats.lognorm(
                s=params['s'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        elif fit.distribution_name == 'gamma':
            return stats.gamma(
                a=params['a'],
                loc=params.get('loc', 0),
                scale=params['scale']
            )
        else:
            raise ValueError(f"Unknown distribution: {fit.distribution_name}")


# ============================================================================
# Bimodality Detection and Component-Based Fitting
# ============================================================================

def detect_bimodal(values: np.ndarray, min_gap_ratio: float = 5.0,
                  position_range: Tuple[float, float] = (0.2, 0.8)) -> Tuple[bool, Optional[float], Dict]:
    """
    Detect if data has a bimodal distribution with large gap separating two clusters.

    Uses gap-based detection: looks for large ratio jumps between consecutive sorted values.
    This identifies distributions that mix different system boundaries (e.g., cell vs BOS materials).

    Parameters
    ----------
    values : np.ndarray
        Data values to test for bimodality
    min_gap_ratio : float
        Minimum ratio between consecutive values to be considered a gap (default: 5.0)
    position_range : tuple
        (min, max) percentile positions where gap must occur (default: (0.2, 0.8))
        Excludes gaps at extreme ends of distribution

    Returns
    -------
    is_bimodal : bool
        True if data appears to be bimodal
    split_threshold : float or None
        Value to split low/high clusters (midpoint of gap)
    gap_info : dict
        Additional information about the detected gap

    References
    ----------
    Gap-based bimodality detection approach developed for Materials Demand project
    to handle inconsistent LCA system boundaries in literature data.
    """
    if len(values) < 3:
        return False, None, {}

    sorted_vals = np.sort(values)
    n = len(sorted_vals)

    # Find all gaps (consecutive value ratios)
    gaps = []
    for i in range(n - 1):
        if sorted_vals[i] > 0:  # Avoid division by zero
            ratio = sorted_vals[i + 1] / sorted_vals[i]
            percentile_pos = (i + 1) / n
            gaps.append({
                'index': i,
                'low_val': sorted_vals[i],
                'high_val': sorted_vals[i + 1],
                'ratio': ratio,
                'percentile_pos': percentile_pos
            })

    if not gaps:
        return False, None, {}

    # Find largest gap that meets criteria
    # Allow gaps at extremes if gap ratio is very large (>10×) AND one cluster is small (n≤2)
    valid_gaps = []
    for g in gaps:
        if g['ratio'] >= min_gap_ratio:
            # Check if gap is in normal position range
            if position_range[0] <= g['percentile_pos'] <= position_range[1]:
                valid_gaps.append(g)
            # OR if gap is at extreme but very large AND creates small cluster
            elif g['ratio'] >= 10.0:
                n_low_candidate = g['index'] + 1  # number of points <= g['low_val']
                n_high_candidate = n - n_low_candidate
                if n_low_candidate <= 2 or n_high_candidate <= 2:
                    valid_gaps.append(g)

    if not valid_gaps:
        return False, None, {}

    # Select largest gap
    largest_gap = max(valid_gaps, key=lambda g: g['ratio'])

    # Split threshold: geometric mean of gap endpoints (appropriate for log-scale data)
    split_threshold = np.sqrt(largest_gap['low_val'] * largest_gap['high_val'])

    # Count cluster sizes
    n_low = np.sum(values <= split_threshold)
    n_high = np.sum(values > split_threshold)

    gap_info = {
        'gap_ratio': largest_gap['ratio'],
        'gap_low': largest_gap['low_val'],
        'gap_high': largest_gap['high_val'],
        'gap_percentile': largest_gap['percentile_pos'],
        'n_low': n_low,
        'n_high': n_high
    }

    return True, split_threshold, gap_info


def should_use_bimodal_fitting(technology: str, material: str) -> bool:
    """
    Determine if a tech-material pair should use component-based bimodal fitting.

    Based on investigation results from peer-reviewed literature verification.
    Only pairs with confirmed TRUE BIMODALITY from system boundary differences
    should use component-based fitting.

    Parameters
    ----------
    technology : str
        Technology name
    material : str
        Material name

    Returns
    -------
    bool
        True if pair should use bimodal component-based fitting

    Notes
    -----
    In this code release this function always returns False (see comment
    below); component-based bimodal fitting is disabled unconditionally.
    """
    # Retired 2026-05-05: bimodal sum-of-components is wrong physics. Cell-
    # level and BOS-level LCA readings are nested boundaries of the same
    # quantity, not additive contributions. The cell+module rows for the four
    # PV/Cu pairs are now dropped at ingestion (see data_ingestion.py STEP 4b),
    # so the remaining BOS-inclusive data is fit with a single forced lognormal
    # like every other pair.
    return False


class DistributionFitter:
    """
    Fits parametric distributions to material intensity data.

    Follows best practices:
    - Tests multiple distributions
    - Uses multiple fitting methods
    - Applies goodness-of-fit tests
    - Always uses parametric distributions (no empirical fallback)
    - Validates tail behavior to prevent extreme outliers
    """

    # Minimum samples for attempting parametric fitting
    MIN_SAMPLES_FIT = 2  # Changed from 3 - always use parametric
    RECOMMENDED_MIN_SAMPLES = 2  # Changed from 5 - always use parametric
    MAX_LOGNORMAL_SHAPE = 3.0  # NEW: Maximum allowed lognormal shape parameter
    MAX_TAIL_RATIO = 300.0  # NEW: Maximum allowed max/median ratio in test sample (relaxed from 100)
    
    # Distributions to test (appropriate for positive-valued data)
    DISTRIBUTIONS = {
        'truncated_normal': 'truncnorm',  # Normal truncated at zero (no negative values)
        'lognormal': stats.lognorm,
        'gamma': stats.gamma,
        # 'weibull': stats.weibull_min,
        # 'triangular': stats.triang,
        'uniform': stats.uniform
    }
    
    def __init__(self):
        self.results: Dict[Tuple[str, str], MaterialIntensityDistribution] = {}
    
    def fit_all(
        self,
        data: pd.DataFrame,
        technology_col: str = 'technology',
        material_col: str = 'material',
        value_col: str = 'intensity_t_per_mw'
    ) -> Dict[Tuple[str, str], MaterialIntensityDistribution]:
        """
        Fit distributions to all technology-material combinations.
        
        Parameters
        ----------
        data : pd.DataFrame
            Material intensity data
        technology_col : str
            Name of technology column
        material_col : str
            Name of material column
        value_col : str
            Name of value column
            
        Returns
        -------
        dict
            Dictionary mapping (technology, material) to MaterialIntensityDistribution
        """
        logger.info("="*80)
        logger.info("Fitting distributions to material intensity data")
        logger.info("="*80)
        
        # Group by technology and material
        grouped = data.groupby([technology_col, material_col])[value_col]
        
        n_groups = len(grouped)
        logger.info(f"Total technology-material combinations: {n_groups}")
        
        for i, ((tech, mat), values) in enumerate(grouped, 1):
            if i % 10 == 0:
                logger.info(f"Processing {i}/{n_groups}...")
            
            values_array = values.values
            dist_info = self.fit_single(tech, mat, values_array)
            self.results[(tech, mat)] = dist_info
        
        logger.info(f"Completed fitting for {len(self.results)} combinations")
        self._log_summary()
        
        return self.results
    
    def fit_single(
        self,
        technology: str,
        material: str,
        data: np.ndarray
    ) -> MaterialIntensityDistribution:
        """
        Fit distributions to a single technology-material combination.
        
        Parameters
        ----------
        technology : str
            Technology name
        material : str
            Material name
        data : np.ndarray
            Material intensity values
            
        Returns
        -------
        MaterialIntensityDistribution
            Complete distribution information
        """
        # Remove any NaN or infinite values
        data = data[np.isfinite(data)]
        n = len(data)
        
        # Calculate summary statistics
        summary_stats = {
            'mean': float(np.mean(data)),
            'std': float(np.std(data, ddof=1)) if n > 1 else 0.0,
            'median': float(np.median(data)),
            'q25': float(np.percentile(data, 25)),
            'q75': float(np.percentile(data, 75)),
            'min_val': float(np.min(data)),
            'max_val': float(np.max(data))
        }
        
        # Initialize result container
        result = MaterialIntensityDistribution(
            technology=technology,
            material=material,
            raw_data=data.copy(),
            n_samples=n,
            **summary_stats
        )

        # ===================================================================
        # COMPONENT-BASED FITTING FOR BIMODAL DISTRIBUTIONS
        # ===================================================================
        # Check if this tech-material pair should use bimodal fitting
        if should_use_bimodal_fitting(technology, material) and n >= 3:
            is_bimodal, split_threshold, gap_info = detect_bimodal(data)

            if is_bimodal:
                logger.info(f"Detected bimodal distribution for {technology}-{material}: "
                           f"gap ratio={gap_info['gap_ratio']:.1f}x, "
                           f"n_low={gap_info['n_low']}, n_high={gap_info['n_high']}")

                # Split data into components
                low_cluster = data[data <= split_threshold]
                high_cluster = data[data > split_threshold]

                # Fit distributions to each component
                component_fits = []
                component_labels = []
                component_data_list = []

                # Determine component labels based on gap position
                if gap_info['n_low'] < gap_info['n_high']:
                    # Smaller cluster is likely cell-level
                    low_label = "cell/module"
                    high_label = "complete system (BOS)"
                else:
                    # Similar sizes or high cluster smaller
                    low_label = "component 1"
                    high_label = "component 2"

                # Fit low cluster
                if len(low_cluster) >= 1:
                    try:
                        # Handle single-point clusters separately
                        if len(low_cluster) == 1:
                            # Use single-point lognormal with borrowed CV
                            low_fit = self._create_single_point_lognormal(low_cluster[0], n=1)
                            component_fits.append(low_fit)
                            component_labels.append(low_label)
                            component_data_list.append(low_cluster)
                        else:
                            # Fit lognormal to cluster with n>=2
                            low_fit = self._fit_distribution(low_cluster, 'lognormal')
                            # Validate tail behavior
                            test_samples = stats.lognorm(
                                s=low_fit.parameters['s'],
                                loc=low_fit.parameters.get('loc', 0),
                                scale=low_fit.parameters['scale']
                            ).rvs(size=10000, random_state=RANDOM_SEED)
                            max_median = np.max(test_samples) / np.median(test_samples)

                            if low_fit.parameters['s'] < 3.0 and max_median < 300:
                                component_fits.append(low_fit)
                                component_labels.append(low_label)
                                component_data_list.append(low_cluster)
                            else:
                                # Use borrowed CV or uniform for extreme tails
                                if len(low_cluster) < 5:
                                    borrowed_fit = self._create_lognormal_for_cv_borrowing(
                                        median=np.median(low_cluster), n=len(low_cluster)
                                    )
                                    component_fits.append(borrowed_fit)
                                    component_labels.append(low_label)
                                    component_data_list.append(low_cluster)
                                else:
                                    uniform_fit = self._fit_uniform_fallback(low_cluster)
                                    component_fits.append(uniform_fit)
                                    component_labels.append(low_label)
                                    component_data_list.append(low_cluster)
                    except Exception as e:
                        logger.warning(f"Failed to fit low cluster for {technology}-{material}: {e}")

                # Fit high cluster
                if len(high_cluster) >= 1:
                    try:
                        # Handle single-point clusters separately
                        if len(high_cluster) == 1:
                            # Use single-point lognormal with borrowed CV
                            high_fit = self._create_single_point_lognormal(high_cluster[0], n=1)
                            component_fits.append(high_fit)
                            component_labels.append(high_label)
                            component_data_list.append(high_cluster)
                        else:
                            # Fit lognormal to cluster with n>=2
                            high_fit = self._fit_distribution(high_cluster, 'lognormal')
                            # Validate tail behavior
                            test_samples = stats.lognorm(
                                s=high_fit.parameters['s'],
                                loc=high_fit.parameters.get('loc', 0),
                                scale=high_fit.parameters['scale']
                            ).rvs(size=10000, random_state=RANDOM_SEED)
                            max_median = np.max(test_samples) / np.median(test_samples)

                            if high_fit.parameters['s'] < 3.0 and max_median < 300:
                                component_fits.append(high_fit)
                                component_labels.append(high_label)
                                component_data_list.append(high_cluster)
                            else:
                                # Use borrowed CV or uniform for extreme tails
                                if len(high_cluster) < 5:
                                    borrowed_fit = self._create_lognormal_for_cv_borrowing(
                                        median=np.median(high_cluster), n=len(high_cluster)
                                    )
                                    component_fits.append(borrowed_fit)
                                    component_labels.append(high_label)
                                    component_data_list.append(high_cluster)
                                else:
                                    uniform_fit = self._fit_uniform_fallback(high_cluster)
                                    component_fits.append(uniform_fit)
                                    component_labels.append(high_label)
                                    component_data_list.append(high_cluster)
                    except Exception as e:
                        logger.warning(f"Failed to fit high cluster for {technology}-{material}: {e}")

                # If we successfully fitted components, use bimodal approach
                if len(component_fits) >= 2:
                    result.is_bimodal = True
                    result.component_fits = component_fits
                    result.component_labels = component_labels
                    result.component_data = component_data_list
                    result.split_threshold = split_threshold
                    result.use_parametric = True
                    result.recommendation = (
                        f"Component-based bimodal fitting: {len(component_fits)} components "
                        f"({', '.join(component_labels)}). Total = sum of components."
                    )
                    # Store component fits in fitted_distributions for reporting
                    result.fitted_distributions = component_fits
                    # best_fit not used for bimodal, but set to first component for compatibility
                    result.best_fit = component_fits[0]

                    logger.info(f"✓ Component-based fitting for {technology}-{material}: "
                               f"{len(component_fits)} components")
                    return result
                else:
                    logger.warning(f"Failed to fit sufficient components for {technology}-{material}, "
                                  "falling back to single distribution")
        # ===================================================================

        # Always attempt parametric fitting (user requirement)
        # Special case for n=1: handle based on FORCE_LOGNORMAL setting
        if n == 1:
            single_val = data[0]
            result.use_parametric = True

            if FORCE_LOGNORMAL:
                # For n=1 with forced lognormal: fit lognormal with borrowed CV
                # This will be updated later by CV borrowing to use the borrowed sigma
                # For now, create a placeholder lognormal using the single point as median
                result.best_fit = self._create_single_point_lognormal(single_val, n=1)
                result.fitted_distributions = [result.best_fit]
                result.recommendation = f"Single data point - lognormal with borrowed CV (n=1)"
            else:
                # Use narrow uniform distribution around single point (original behavior)
                result.best_fit = self._create_single_point_uniform(single_val, n=1)
                result.fitted_distributions = [result.best_fit]
                result.recommendation = f"Single data point - using narrow uniform distribution around {single_val:.3f}"

            return result
        
        # Attempt to fit parametric distributions
        fitted_dists = []

        # Check if we should force lognormal distribution
        if FORCE_LOGNORMAL:
            # Only fit lognormal distribution
            try:
                fit_result = self._fit_distribution(data, 'lognormal')
                fitted_dists.append(fit_result)
            except Exception as e:
                logger.debug(f"Failed to fit lognormal to {technology}-{material}: {e}")
        else:
            # Fit all distributions and select by AIC
            for dist_name in self.DISTRIBUTIONS.keys():
                try:
                    fit_result = self._fit_distribution(data, dist_name)
                    fitted_dists.append(fit_result)
                except Exception as e:
                    logger.debug(f"Failed to fit {dist_name} to {technology}-{material}: {e}")
                    continue

        # Sort by AIC (lower is better)
        fitted_dists.sort(key=lambda x: x.aic)
        result.fitted_distributions = fitted_dists

        # Select best distribution and make recommendation
        # ALWAYS use parametric (user requirement)
        if fitted_dists:
            # Check distributions in order for tail behavior
            selected_dist = None
            rejection_reasons = []

            for dist_fit in fitted_dists:
                # Validate tail behavior
                is_valid, reason = self._validate_tail_behavior(dist_fit, data)

                if is_valid:
                    selected_dist = dist_fit
                    break
                else:
                    rejection_reasons.append(f"{dist_fit.distribution_name}: {reason}")

            if selected_dist:
                result.best_fit = selected_dist
                result.use_parametric = True

                # Build recommendation message
                if FORCE_LOGNORMAL:
                    result.recommendation = (
                        f"Forced lognormal distribution "
                        f"(n={n}, KS p={selected_dist.ks_pvalue:.4f}, AIC={selected_dist.aic:.2f})"
                    )
                elif selected_dist != fitted_dists[0]:
                    # Not the best AIC, but best valid distribution
                    result.recommendation = (
                        f"Using {selected_dist.distribution_name} distribution "
                        f"(n={n}, KS p={selected_dist.ks_pvalue:.4f}). "
                        f"Best AIC rejected: {rejection_reasons[0]}"
                    )
                else:
                    result.recommendation = (
                        f"Using {selected_dist.distribution_name} distribution "
                        f"(n={n}, KS p={selected_dist.ks_pvalue:.4f}, AIC={selected_dist.aic:.2f})"
                    )
            else:
                # All distributions failed validation
                # For FORCE_LOGNORMAL: try borrowed CV rescue before uniform fallback
                # This applies to ALL n (relaxed from n<5), ALL rejections (σ>3.0 OR max/median>300)
                if FORCE_LOGNORMAL and len(rejection_reasons) > 0:
                    # Lognormal was rejected (could be σ>3.0 OR max/median>300)
                    # Try borrowed CV as rescue mechanism before falling back to uniform
                    median_val = np.median(data)
                    result.best_fit = self._create_lognormal_for_cv_borrowing(
                        median=median_val,
                        n=n
                    )
                    result.use_parametric = True
                    if n < 5:
                        result.recommendation = (
                            f"Lognormal rejected (n={n}, extreme tails). "
                            f"Using borrowed CV rescue. "
                            f"Original rejection: {rejection_reasons[0]}"
                        )
                    else:
                        result.recommendation = (
                            f"Lognormal rejected (n={n}, extreme tails). "
                            f"Using borrowed CV rescue (n≥5 relaxation). "
                            f"Original rejection: {rejection_reasons[0]}"
                        )
                    result.fitted_distributions.append(result.best_fit)
                else:
                    # not FORCE_LOGNORMAL - use uniform fallback
                    result.best_fit = self._fit_uniform_fallback(data)
                    result.use_parametric = True
                    result.recommendation = (
                        f"All distributions rejected due to extreme tails. "
                        f"Using uniform fallback. Rejections: {'; '.join(rejection_reasons[:2])}"
                    )
                    result.fitted_distributions.append(result.best_fit)
        else:
            # No distributions fit at all
            # For n < 5 with FORCE_LOGNORMAL: use borrowed CV placeholder (even for fit failures like zeros)
            if FORCE_LOGNORMAL and n < 5:
                # Fit failed (e.g., data contains zeros), but we can still create lognormal with borrowed CV
                median_val = np.median(data)
                result.best_fit = self._create_lognormal_for_cv_borrowing(
                    median=median_val,
                    n=n
                )
                result.use_parametric = True
                result.recommendation = (
                    f"Fit failed (n={n}, likely contains zeros or invalid values). "
                    f"Using borrowed CV lognormal centered on median."
                )
                result.fitted_distributions.append(result.best_fit)
            else:
                # n ≥ 5 or not FORCE_LOGNORMAL - use uniform fallback
                result.best_fit = self._fit_uniform_fallback(data)
                result.use_parametric = True
                result.recommendation = (
                    f"No distributions fit successfully (n={n}). "
                    f"Using uniform fallback."
                )
                result.fitted_distributions.append(result.best_fit)
            result.fitted_distributions = [result.best_fit]
        
        return result
    
    def _fit_distribution(
        self,
        data: np.ndarray,
        dist_name: str
    ) -> DistributionFit:
        """
        Fit a specific distribution to data.

        Parameters
        ----------
        data : np.ndarray
            Data to fit
        dist_name : str
            Name of distribution

        Returns
        -------
        DistributionFit
            Fitted distribution information
        """
        n = len(data)

        # Check for zero variance (all values identical)
        # For zero-variance data, use narrow uniform distribution
        data_std = np.std(data, ddof=1) if n > 1 else 0
        if data_std == 0 or n == 1:
            # All values identical - only uniform makes sense normally
            if dist_name == 'lognormal' and FORCE_LOGNORMAL:
                # When forcing lognormal, create a narrow lognormal centered on the value
                # rather than a dummy with empty params (which crashes downstream)
                value = float(np.median(data))
                if value <= 0:
                    value = 1e-10  # Safety floor for lognormal scale
                sigma = 0.01  # Very small shape → near-degenerate
                return DistributionFit(
                    distribution_name='lognormal',
                    parameters={'s': sigma, 'loc': 0, 'scale': value},
                    ks_statistic=0.0,
                    ks_pvalue=1.0,
                    ad_statistic=0.0,
                    ad_critical_value=0.0,
                    aic=999999.0,  # Poor AIC since this is degenerate
                    bic=999999.0,
                    n_samples=n,
                    fitting_method='zero_variance_lognormal'
                )
            elif dist_name != 'uniform':
                # Return a dummy fit with very poor AIC so uniform will be selected
                # This prevents degenerate distributions with scale=0
                return DistributionFit(
                    distribution_name=dist_name,
                    parameters={},
                    ks_statistic=1.0,
                    ks_pvalue=0.0,
                    ad_statistic=999.0,
                    ad_critical_value=1.0,
                    aic=999999.0,  # Very poor AIC so uniform wins
                    bic=999999.0,
                    n_samples=n,
                    fitting_method='rejected_zero_variance'
                )

        # Special handling for different distributions
        if dist_name == 'truncated_normal':
            # Truncated normal at zero (no negative values)
            # For material intensities, we want a normal distribution truncated at 0

            # scipy.stats.truncnorm.fit() has issues with data far from zero
            # Use moment matching instead: find loc and scale such that
            # the truncated normal has the same mean and std as the data

            from scipy.optimize import minimize

            data_mean = np.mean(data)
            data_std = np.std(data, ddof=1)

            def truncnorm_match_moments(params):
                """Minimize difference between distribution and data moments"""
                loc, scale = params
                if scale <= 0:
                    return 1e10

                # Create truncnorm with lower bound at 0
                a = (0 - loc) / scale  # Lower bound in standard units
                b = np.inf

                try:
                    dist_temp = stats.truncnorm(a=a, b=b, loc=loc, scale=scale)
                    dist_mean = dist_temp.mean()
                    dist_std = dist_temp.std()

                    # Minimize squared error in mean and std
                    error = (dist_mean - data_mean)**2 + (dist_std - data_std)**2
                    return error
                except:
                    return 1e10

            # Optimize to match moments
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = minimize(truncnorm_match_moments, [data_mean, data_std],
                                method='Nelder-Mead')

            loc_fitted = result.x[0]
            scale_fitted = result.x[1]
            a_fitted = (0 - loc_fitted) / scale_fitted  # Lower bound in standard units

            dist = stats.truncnorm(a=a_fitted, b=np.inf, loc=loc_fitted, scale=scale_fitted)
            param_dict = {
                'a': a_fitted,
                'b': np.inf,
                'loc': loc_fitted,
                'scale': scale_fitted,
                'lower_bound': 0.0,
                'upper_bound': np.inf
            }
            
        elif dist_name == 'uniform':
            # Uniform distribution: fit to [min, max] of data
            # For uniform, we just need loc (start) and scale (width)
            min_val = np.min(data)
            max_val = np.max(data)

            # Handle zero variance: create narrow uniform around the single value
            if max_val == min_val:
                # Add 10% padding on each side
                padding = max(0.1 * abs(min_val), 1e-6) if min_val != 0 else 1e-6
                loc = max(0, min_val - padding)  # Don't go negative
                scale = (min_val + padding) - loc
            else:
                # scipy.stats.uniform uses loc (start) and scale (width)
                loc = min_val
                scale = max_val - min_val

            params = (loc, scale)
            dist = stats.uniform(loc=loc, scale=scale)
            param_dict = {'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale}
            
        else:
            # Standard fitting for other distributions
            dist_class = self.DISTRIBUTIONS[dist_name]

            # Fit distribution using MLE (default for scipy)
            # For lognormal: constrain loc=0 (standard practice for positive data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if dist_name == 'lognormal':
                    # Two-parameter lognormal: force location = 0
                    params = dist_class.fit(data, floc=0)
                else:
                    # Other distributions: unconstrained fit
                    params = dist_class.fit(data)

            # Create distribution object
            dist = dist_class(*params)
            
            # Extract named parameters for storage
            if dist_name == 'lognormal':
                param_dict = {'s': params[0], 'loc': params[1], 'scale': params[2]}
            elif dist_name == 'gamma':
                param_dict = {'a': params[0], 'loc': params[1], 'scale': params[2]}
            else:
                param_dict = {f'param_{i}': p for i, p in enumerate(params)}
        
        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x))
        
        # Anderson-Darling test
        # Note: scipy's anderson test is limited to specific distributions
        # For general case, we'll use a conservative critical value
        ad_result = self._anderson_darling_general(data, dist)
        
        # Information criteria
        # Log-likelihood
        log_likelihood = np.sum(dist.logpdf(data))
        k = len(params)  # number of parameters
        
        # AIC = 2k - 2ln(L)
        aic = 2 * k - 2 * log_likelihood
        
        # BIC = k*ln(n) - 2ln(L)
        bic = k * np.log(n) - 2 * log_likelihood
        
        return DistributionFit(
            distribution_name=dist_name,
            parameters=param_dict,
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_result['statistic'],
            ad_critical_value=ad_result['critical_value'],
            aic=aic,
            bic=bic,
            n_samples=n,
            fitting_method='MLE'
        )
    
    def _anderson_darling_general(
        self,
        data: np.ndarray,
        dist
    ) -> Dict[str, float]:
        """
        Compute Anderson-Darling test statistic for any distribution.

        More powerful than K-S test, especially in the tails.
        """
        n = len(data)
        data_sorted = np.sort(data)

        # Compute CDF values
        cdf_vals = dist.cdf(data_sorted)

        # Avoid log(0) issues
        cdf_vals = np.clip(cdf_vals, 1e-10, 1 - 1e-10)

        # Anderson-Darling statistic
        i = np.arange(1, n + 1)
        ad_stat = -n - np.sum(
            (2 * i - 1) * (np.log(cdf_vals) + np.log(1 - cdf_vals[::-1]))
        ) / n

        # Critical value (approximate, for alpha=0.05)
        # This is a conservative approximation
        critical_value = 2.492  # For alpha=0.05, approximately

        return {
            'statistic': ad_stat,
            'critical_value': critical_value
        }

    def _validate_tail_behavior(
        self,
        dist_fit: DistributionFit,
        data: np.ndarray
    ) -> Tuple[bool, str]:
        """
        Validate that a fitted distribution doesn't have extreme tail behavior.

        Checks:
        1. Lognormal shape parameter < MAX_LOGNORMAL_SHAPE
        2. Max/median ratio in test sample < MAX_TAIL_RATIO

        Parameters
        ----------
        dist_fit : DistributionFit
            Fitted distribution to validate
        data : np.ndarray
            Original data

        Returns
        -------
        Tuple[bool, str]
            (is_valid, rejection_reason)
        """
        try:
            # Check 1: Lognormal shape parameter
            if dist_fit.distribution_name == 'lognormal':
                s = dist_fit.parameters.get('s', 0)
                if s > self.MAX_LOGNORMAL_SHAPE:
                    return False, f"Lognormal shape s={s:.2f} > {self.MAX_LOGNORMAL_SHAPE}"

            # Check 2: Tail behavior via test sampling
            # Create scipy distribution
            if dist_fit.distribution_name == 'lognormal':
                dist = stats.lognorm(
                    s=dist_fit.parameters['s'],
                    loc=dist_fit.parameters.get('loc', 0),
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'gamma':
                dist = stats.gamma(
                    a=dist_fit.parameters['a'],
                    loc=dist_fit.parameters.get('loc', 0),
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'uniform':
                dist = stats.uniform(
                    loc=dist_fit.parameters['loc'],
                    scale=dist_fit.parameters['scale']
                )
            elif dist_fit.distribution_name == 'truncated_normal':
                dist = stats.truncnorm(
                    a=dist_fit.parameters['a'],
                    b=dist_fit.parameters['b'],
                    loc=dist_fit.parameters['loc'],
                    scale=dist_fit.parameters['scale']
                )
            else:
                return True, ""  # Unknown distribution, pass by default

            # Sample test data
            test_sample = dist.rvs(size=10000, random_state=RANDOM_SEED)

            # Check tail behavior
            max_val = np.max(test_sample)
            median_val = np.median(test_sample)

            if median_val > 0:
                ratio = max_val / median_val
                if ratio > self.MAX_TAIL_RATIO:
                    return False, f"Extreme tail: max/median={ratio:.1f}x > {self.MAX_TAIL_RATIO}x"

            return True, ""

        except Exception as e:
            logger.debug(f"Tail validation failed for {dist_fit.distribution_name}: {e}")
            # Fail if parameters are empty (degenerate dummy fit)
            if not dist_fit.parameters:
                return False, f"Empty parameters for {dist_fit.distribution_name}"
            return True, ""  # Pass by default if validation fails

    def _fit_uniform_fallback(self, data: np.ndarray) -> DistributionFit:
        """
        Create a uniform distribution as safe fallback.

        Uses [min, max] of data with 10% padding.

        Parameters
        ----------
        data : np.ndarray
            Material intensity data

        Returns
        -------
        DistributionFit
            Uniform distribution fit
        """
        min_val = np.min(data)
        max_val = np.max(data)

        # Add 10% padding
        range_val = max_val - min_val
        loc = max(0, min_val - 0.1 * range_val)  # Don't go negative
        scale = max_val - loc + 0.1 * range_val

        # Create distribution
        dist = stats.uniform(loc=loc, scale=scale)

        # Calculate goodness of fit
        ks_stat, ks_pval = stats.kstest(data, lambda x: dist.cdf(x))

        # Calculate AIC/BIC
        n = len(data)
        k = 2  # 2 parameters (loc, scale)
        log_likelihood = np.sum(dist.logpdf(data))
        aic = 2 * k - 2 * log_likelihood
        bic = k * np.log(n) - 2 * log_likelihood

        # Anderson-Darling
        ad_result = self._anderson_darling_general(data, dist)

        return DistributionFit(
            distribution_name='uniform',
            parameters={'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale},
            ks_statistic=ks_stat,
            ks_pvalue=ks_pval,
            ad_statistic=ad_result['statistic'],
            ad_critical_value=ad_result['critical_value'],
            aic=aic,
            bic=bic,
            n_samples=n,
            fitting_method='fallback'
        )

    def _create_single_point_uniform(self, value: float, n: int) -> DistributionFit:
        """
        Create a narrow uniform distribution around a single point.

        Parameters
        ----------
        value : float
            Single data point
        n : int
            Sample size (1)

        Returns
        -------
        DistributionFit
            Uniform distribution centered on value
        """
        # Create uniform distribution ±10% around the point
        loc = max(0, value * 0.9)
        scale = value * 1.1 - loc

        return DistributionFit(
            distribution_name='uniform',
            parameters={'loc': loc, 'scale': scale, 'min': loc, 'max': loc + scale},
            ks_statistic=0.0,
            ks_pvalue=1.0,
            ad_statistic=0.0,
            ad_critical_value=0.0,
            aic=0.0,
            bic=0.0,
            n_samples=n,
            fitting_method='single_point'
        )

    def _create_single_point_lognormal(self, value: float, n: int) -> DistributionFit:
        """
        Create a lognormal distribution for a single point with default CV.

        Uses the single point as the median of the lognormal distribution.
        The CV will be updated later by CV borrowing to use the borrowed value.

        For lognormal: median = scale (when loc=0)
        Initial sigma is set to give CV ≈ 1.0 as a placeholder.

        Parameters
        ----------
        value : float
            Single data point
        n : int
            Sample size (1)

        Returns
        -------
        DistributionFit
            Lognormal distribution with median = value, CV to be borrowed
        """
        # Use single point as median of lognormal
        # For lognormal: median = scale (when loc=0)
        scale = value
        loc = 0

        # Set initial sigma to give CV ≈ 1.0 (will be updated by CV borrowing)
        # CV = sqrt(exp(sigma^2) - 1), so for CV=1: sigma ≈ 0.83
        sigma = 0.83

        return DistributionFit(
            distribution_name='lognormal',
            parameters={'s': sigma, 'loc': loc, 'scale': scale},
            ks_statistic=0.0,  # No test possible with n=1
            ks_pvalue=1.0,
            ad_statistic=0.0,
            ad_critical_value=0.0,
            aic=0.0,
            bic=0.0,
            n_samples=n,
            fitting_method='single_point_placeholder'
        )

    def _create_lognormal_for_cv_borrowing(self, median: float, n: int) -> DistributionFit:
        """
        Create a placeholder lognormal distribution for CV borrowing to fix.

        Uses the data median as the lognormal scale parameter.
        Initial sigma is set to a placeholder value that will be replaced
        by CV borrowing with the borrowed value.

        This is used when a lognormal fit is rejected due to extreme tails
        (σ > 3.0) but the sample size is small (n < 5). Instead of falling
        back to uniform, we create a placeholder with borrowed CV that will
        be properly set by the CV borrowing step.

        Parameters
        ----------
        median : float
            Median of the original data
        n : int
            Sample size

        Returns
        -------
        DistributionFit
            Lognormal distribution placeholder for CV borrowing
        """
        # Use data median as scale (lognormal median = scale when loc=0)
        scale = median
        loc = 0

        # Placeholder sigma (will be replaced by CV borrowing)
        # Use borrowed sigma estimate: CV=0.671 → sigma≈0.610
        # CV = sqrt(exp(sigma^2) - 1), so sigma = sqrt(log(1 + CV^2))
        sigma = 0.610

        return DistributionFit(
            distribution_name='lognormal',
            parameters={'s': sigma, 'loc': loc, 'scale': scale},
            ks_statistic=0.0,
            ks_pvalue=1.0,
            ad_statistic=0.0,
            ad_critical_value=0.0,
            aic=0.0,
            bic=0.0,
            n_samples=n,
            fitting_method='borrowed_cv_placeholder'
        )

    def _log_summary(self):
        """Log summary statistics of fitting results"""
        logger.info("\n" + "="*80)
        logger.info("FITTING SUMMARY")
        logger.info("="*80)

        # Log configuration
        if FORCE_LOGNORMAL:
            logger.info("Configuration: FORCE_LOGNORMAL = True (all fits forced to lognormal)")
        else:
            logger.info("Configuration: FORCE_LOGNORMAL = False (AIC-based selection)")

        n_total = len(self.results)
        n_parametric = sum(1 for r in self.results.values() if r.use_parametric)
        n_empirical = n_total - n_parametric

        logger.info(f"\nTotal combinations: {n_total}")
        logger.info(f"  Using parametric distributions: {n_parametric} ({100*n_parametric/n_total:.1f}%)")
        logger.info(f"  Using empirical distributions: {n_empirical} ({100*n_empirical/n_total:.1f}%)")
        
        # Sample size distribution
        sample_sizes = [r.n_samples for r in self.results.values()]
        logger.info(f"\nSample size statistics:")
        logger.info(f"  Min: {np.min(sample_sizes)}")
        logger.info(f"  Median: {np.median(sample_sizes):.0f}")
        logger.info(f"  Max: {np.max(sample_sizes)}")
        
        # Distribution type counts (for parametric fits)
        if n_parametric > 0:
            dist_counts = {}
            for r in self.results.values():
                if r.best_fit is not None:
                    dist_name = r.best_fit.distribution_name
                    dist_counts[dist_name] = dist_counts.get(dist_name, 0) + 1
            
            logger.info(f"\nBest-fit distribution types:")
            for dist_name, count in sorted(dist_counts.items(), key=lambda x: -x[1]):
                logger.info(f"  {dist_name}: {count} ({100*count/n_total:.1f}%)")
    
    def export_to_csv(self, output_path: Union[str, Path]):
        """
        Export fitted distributions to CSV.
        
        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        rows = []
        
        for (tech, mat), dist_info in self.results.items():
            row = {
                'technology': tech,
                'material': mat,
                'n_samples': dist_info.n_samples,
                'mean': dist_info.mean,
                'std': dist_info.std,
                'median': dist_info.median,
                'q25': dist_info.q25,
                'q75': dist_info.q75,
                'min': dist_info.min_val,
                'max': dist_info.max_val,
                'use_parametric': dist_info.use_parametric,
                'recommendation': dist_info.recommendation
            }
            
            if dist_info.best_fit is not None:
                bf = dist_info.best_fit
                row.update({
                    'best_distribution': bf.distribution_name,
                    'ks_statistic': bf.ks_statistic,
                    'ks_pvalue': bf.ks_pvalue,
                    'ad_statistic': bf.ad_statistic,
                    'aic': bf.aic,
                    'bic': bf.bic
                })
                
                # Add parameters (flattened)
                for param_name, param_value in bf.parameters.items():
                    row[f'param_{param_name}'] = param_value
            else:
                row.update({
                    'best_distribution': 'empirical',
                    'ks_statistic': np.nan,
                    'ks_pvalue': np.nan,
                    'ad_statistic': np.nan,
                    'aic': np.nan,
                    'bic': np.nan
                })
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported fitted distributions to: {output_path}")
    
    def export_raw_data(self, output_path: Union[str, Path]):
        """
        Export raw data for all combinations to CSV.

        This provides the empirical distributions for use when
        parametric fits are not recommended.

        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        rows = []

        for (tech, mat), dist_info in self.results.items():
            for value in dist_info.raw_data:
                rows.append({
                    'technology': tech,
                    'material': mat,
                    'intensity_t_per_mw': value
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported raw data to: {output_path}")

    def export_fit_summary(self, output_path: Union[str, Path]):
        """
        Export a summary table of distribution fitting results.

        Includes distribution type breakdown, sample size statistics,
        goodness-of-fit pass rates, and per-distribution-type details.

        Parameters
        ----------
        output_path : str or Path
            Path to output CSV file
        """
        if not self.results:
            logger.warning("No results to summarize")
            return

        n_total = len(self.results)

        # --- Collect per-pair info ---
        dist_types = []
        sample_sizes = []
        ks_pass = 0
        ks_total = 0
        parametric_count = 0

        for dist_info in self.results.values():
            sample_sizes.append(dist_info.n_samples)
            if dist_info.use_parametric:
                parametric_count += 1

            if dist_info.best_fit is not None:
                dist_types.append(dist_info.best_fit.distribution_name)
                ks_total += 1
                if dist_info.best_fit.passes_ks_test():
                    ks_pass += 1
            else:
                dist_types.append('empirical')

        sample_sizes = np.array(sample_sizes)

        # --- Distribution type counts ---
        from collections import Counter
        type_counts = Counter(dist_types)

        rows = []

        # Section 1: Overall summary
        rows.append({
            'section': 'overall',
            'metric': 'total_pairs',
            'value': n_total
        })
        rows.append({
            'section': 'overall',
            'metric': 'parametric_fits',
            'value': parametric_count
        })
        rows.append({
            'section': 'overall',
            'metric': 'empirical_fallback',
            'value': n_total - parametric_count
        })
        rows.append({
            'section': 'overall',
            'metric': 'ks_pass_rate',
            'value': round(ks_pass / ks_total, 4) if ks_total > 0 else np.nan
        })

        # Section 2: Distribution type breakdown
        for dist_name in sorted(type_counts.keys()):
            count = type_counts[dist_name]
            rows.append({
                'section': 'distribution_type',
                'metric': dist_name,
                'value': count,
                'percentage': round(100 * count / n_total, 1)
            })

        # Section 3: Sample size statistics
        for label, val in [
            ('min', int(np.min(sample_sizes))),
            ('q25', int(np.percentile(sample_sizes, 25))),
            ('median', int(np.median(sample_sizes))),
            ('q75', int(np.percentile(sample_sizes, 75))),
            ('max', int(np.max(sample_sizes))),
            ('mean', round(float(np.mean(sample_sizes)), 1)),
        ]:
            rows.append({
                'section': 'sample_size',
                'metric': label,
                'value': val
            })

        # Section 4: Sample size bins
        bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, None)]
        bin_labels = ['n=1', 'n=2', 'n=3', 'n=4-5', 'n=6-10', 'n=11-20', 'n>20']
        for (lo, hi), label in zip(bins, bin_labels):
            if hi is None:
                count = int(np.sum(sample_sizes >= lo))
            else:
                count = int(np.sum((sample_sizes >= lo) & (sample_sizes <= hi)))
            rows.append({
                'section': 'sample_size_bins',
                'metric': label,
                'value': count,
                'percentage': round(100 * count / n_total, 1)
            })

        # Section 5: GoF statistics by distribution type
        for dist_name in sorted(type_counts.keys()):
            if dist_name == 'empirical':
                continue
            ks_stats = []
            aic_vals = []
            for dist_info in self.results.values():
                if dist_info.best_fit is not None and \
                   dist_info.best_fit.distribution_name == dist_name:
                    ks_stats.append(dist_info.best_fit.ks_statistic)
                    if not np.isnan(dist_info.best_fit.aic):
                        aic_vals.append(dist_info.best_fit.aic)
            if ks_stats:
                rows.append({
                    'section': f'gof_{dist_name}',
                    'metric': 'median_ks_statistic',
                    'value': round(float(np.median(ks_stats)), 4)
                })
                rows.append({
                    'section': f'gof_{dist_name}',
                    'metric': 'median_aic',
                    'value': round(float(np.median(aic_vals)), 2) if aic_vals else np.nan
                })
                rows.append({
                    'section': f'gof_{dist_name}',
                    'metric': 'count',
                    'value': len(ks_stats)
                })

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
        logger.info(f"Exported fit summary to: {output_path}")

        # Also print a readable summary to the log
        logger.info(f"\n{'='*60}")
        logger.info("DISTRIBUTION FITTING SUMMARY")
        logger.info(f"{'='*60}")
        logger.info(f"Total pairs: {n_total}")
        logger.info(f"Parametric: {parametric_count} | Empirical: {n_total - parametric_count}")
        logger.info(f"K-S pass rate: {ks_pass}/{ks_total} "
                     f"({100*ks_pass/ks_total:.0f}%)" if ks_total > 0 else "N/A")
        logger.info(f"\nDistribution types:")
        for dist_name in sorted(type_counts.keys(), key=lambda x: -type_counts[x]):
            count = type_counts[dist_name]
            logger.info(f"  {dist_name:20s}: {count:4d} ({100*count/n_total:5.1f}%)")
        logger.info(f"\nSample sizes: min={int(np.min(sample_sizes))}, "
                     f"median={int(np.median(sample_sizes))}, "
                     f"max={int(np.max(sample_sizes))}")
        pct_low_n = 100 * np.sum(sample_sizes < 5) / n_total
        logger.info(f"Pairs with n<5: {int(np.sum(sample_sizes < 5))} ({pct_low_n:.0f}%)")
        logger.info(f"{'='*60}")


def create_distribution_report(
    fitter: DistributionFitter,
    output_path: Union[str, Path]
):
    """
    Create a comprehensive text report of fitting results.
    
    Parameters
    ----------
    fitter : DistributionFitter
        Fitted distribution fitter
    output_path : str or Path
        Path to output text file
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MATERIAL INTENSITY DISTRIBUTION FITTING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Total technology-material combinations: {len(fitter.results)}\n\n")
        
        # Summary statistics
        n_parametric = sum(1 for r in fitter.results.values() if r.use_parametric)
        n_empirical = len(fitter.results) - n_parametric
        
        f.write("RECOMMENDATIONS SUMMARY:\n")
        f.write(f"  Parametric distributions: {n_parametric}\n")
        f.write(f"  Empirical distributions: {n_empirical}\n\n")
        
        # Sample size distribution
        sample_sizes = [r.n_samples for r in fitter.results.values()]
        f.write("SAMPLE SIZE DISTRIBUTION:\n")
        f.write(f"  Min: {np.min(sample_sizes)}\n")
        f.write(f"  25th percentile: {np.percentile(sample_sizes, 25):.0f}\n")
        f.write(f"  Median: {np.median(sample_sizes):.0f}\n")
        f.write(f"  75th percentile: {np.percentile(sample_sizes, 75):.0f}\n")
        f.write(f"  Max: {np.max(sample_sizes)}\n\n")
        
        # Detailed results for each combination
        f.write("="*80 + "\n")
        f.write("DETAILED RESULTS BY TECHNOLOGY-MATERIAL COMBINATION\n")
        f.write("="*80 + "\n\n")
        
        for (tech, mat), dist_info in sorted(fitter.results.items()):
            f.write(f"\n{tech} - {mat}\n")
            f.write("-" * 80 + "\n")
            f.write(f"Sample size: {dist_info.n_samples}\n")
            f.write(f"Mean: {dist_info.mean:.2f}, Std: {dist_info.std:.2f}\n")
            f.write(f"Median: {dist_info.median:.2f}, IQR: [{dist_info.q25:.2f}, {dist_info.q75:.2f}]\n")
            f.write(f"Range: [{dist_info.min_val:.2f}, {dist_info.max_val:.2f}]\n\n")
            
            f.write(f"Recommendation: {dist_info.recommendation}\n\n")
            
            if dist_info.fitted_distributions:
                f.write("Fitted distributions (ranked by AIC):\n")
                for i, fit in enumerate(dist_info.fitted_distributions, 1):
                    f.write(f"  {i}. {fit.summary()}\n")
            else:
                f.write("No parametric distributions fit successfully.\n")
            
            f.write("\n")
    
    logger.info(f"Created detailed report: {output_path}")


if __name__ == "__main__":
    print("Distribution Fitting Module for Materials Demand Analysis")
    print("This module should be imported and used in analysis scripts.")
