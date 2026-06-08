"""Stock-flow Monte Carlo engine: the simulation core.

Tracks installed capacity by technology, infers annual additions and
retirements, and multiplies additions by Monte Carlo-sampled material
intensities to get the per-scenario demand distributions. ``run_full_simulation``
ties together data ingestion, distribution fitting, CV borrowing, and technology
mapping into the full pipeline.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field
import logging
import warnings

# Import our modules
from .data_ingestion import load_all_data, CapacityProjectionLoader
from .distribution_fitting import DistributionFitter
from .cv_borrowing import apply_cv_borrowing, create_cv_borrowing_report
from .technology_mapping import (
    TECHNOLOGY_MAPPING,
    TECHNOLOGY_LIFETIMES,
    get_intensity_technologies,
    get_lifetime,
    validate_mapping
)

logger = logging.getLogger(__name__)


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class StockFlowState:
    """
    Tracks the stock-flow state for a single technology in a single scenario.
    
    Stock = cumulative installed capacity
    Additions = new capacity added each year
    Retirements = capacity retired each year (based on lifetime)
    """
    technology: str
    scenario: str
    
    # Time series data (indexed by year)
    stock: Dict[int, float] = field(default_factory=dict)
    additions: Dict[int, float] = field(default_factory=dict)
    retirements: Dict[int, float] = field(default_factory=dict)
    
    def update(self, year: int, new_capacity: float, lifetime: int, is_first_year: bool = False):
        """
        Update stock-flow for a given year.
        
        Parameters
        ----------
        year : int
            Current year
        new_capacity : float
            Total capacity in this year (MW)
        lifetime : int
            Technology lifetime (years)
        is_first_year : bool
            If True, this is the baseline year (no additions counted)
        """
        # Initialize if first year - this is the baseline stock
        if not self.stock or is_first_year:
            self.stock[year] = new_capacity
            self.additions[year] = 0  # No additions in baseline year
            self.retirements[year] = 0
            return
        
        # Get previous year data
        prev_year = max(self.stock.keys())
        prev_stock = self.stock[prev_year]
        
        # Calculate retirements (capacity installed 'lifetime' years ago)
        retirement_year = year - lifetime
        if retirement_year in self.additions:
            retirement = self.additions[retirement_year]
        else:
            retirement = 0
        
        # Stock equation: Stock(t) = Stock(t-1) + Additions(t) - Retirements(t)
        # Since we have total capacity, we need to infer additions
        # Addition(t) = Stock(t) - Stock(t-1) + Retirements(t)
        
        addition = new_capacity - prev_stock + retirement
        
        # Ensure non-negative (capacity can't decrease faster than retirements allow)
        if addition < 0:
            # This can happen if capacity decreases in the data
            # Treat as zero additions and adjust retirements
            retirement = prev_stock - new_capacity
            addition = 0
        
        self.stock[year] = new_capacity
        self.additions[year] = addition
        self.retirements[year] = retirement


@dataclass
class SimulationResult:
    """Container for Monte Carlo simulation results"""
    # Raw results: [iteration, scenario, year, material] = demand
    iterations_data: np.ndarray

    # Metadata
    n_iterations: int
    scenarios: List[str]
    years: List[int]
    materials: List[str]

    # Aggregation indices
    scenario_idx: Dict[str, int]
    year_idx: Dict[int, int]
    material_idx: Dict[str, int]

    # Convergence tracking
    converged: bool = False
    convergence_iteration: Optional[int] = None
    
    def get_statistics(
        self,
        percentiles: List[float] = [2.5, 5, 25, 50, 75, 95, 97.5]
    ) -> pd.DataFrame:
        """
        Calculate summary statistics across Monte Carlo iterations.
        
        Returns DataFrame with columns:
        - scenario, year, material
        - mean, std
        - p5, p25, p50, p75, p95 (or custom percentiles)
        """
        results = []
        
        for scenario in self.scenarios:
            for year in self.years:
                for material in self.materials:
                    # Extract data for this combination
                    i_scenario = self.scenario_idx[scenario]
                    i_year = self.year_idx[year]
                    i_material = self.material_idx[material]
                    
                    data = self.iterations_data[:, i_scenario, i_year, i_material]
                    
                    # Calculate statistics
                    stats = {
                        'scenario': scenario,
                        'year': year,
                        'material': material,
                        'mean': float(np.mean(data)),
                        'std': float(np.std(data))
                    }
                    
                    # Add percentiles. Canonical naming: integer percentiles
                    # use `p{int}` (e.g. p5, p25, p50, p75, p95); fractional
                    # percentiles use `p{int}_{frac}` (e.g. p2_5, p97_5) per
                    # the project standing rule (CLAUDE.md "95% CI uses
                    # p2.5/p97.5"). Legacy truncated aliases (`p2`, `p97`)
                    # are also written for backward compat with consumers
                    # not yet migrated; see Operation Bulletproof DECISIONS
                    # 2026-04-28 and L2-CI-0001 audit finding.
                    pct_values = np.percentile(data, percentiles)
                    for pct, val in zip(percentiles, pct_values):
                        val = float(val)
                        if pct == int(pct):
                            stats[f'p{int(pct)}'] = val
                        else:
                            # Canonical name (e.g. p2_5, p97_5)
                            stats[f'p{pct}'.replace('.', '_')] = val
                            # Legacy truncated alias (e.g. p2, p97) — same value
                            stats[f'p{int(pct)}'] = val

                    results.append(stats)
        
        return pd.DataFrame(results)
    
    def get_summary_by_material(
        self,
        percentiles: List[float] = [2.5, 5, 25, 50, 75, 95 , 97.5]
    ) -> pd.DataFrame:
        """
        Pool across scenarios, summarize by material and year.

        iterations_data has shape (n_iterations, n_scenarios, n_years, n_materials).
        The 61 NREL Standard Scenarios are mutually-exclusive alternative futures,
        not additive components — summing them would produce a meaningless
        aggregate ~61× larger than any single-scenario demand. Instead we pool
        (iteration × scenario) into a single equally-weighted sample axis of
        size n_iterations * n_scenarios and compute mean/std/percentiles over it.
        """
        # Pool iteration and scenario axes into a single sample axis.
        # Shape: [iteration * scenario, year, material]
        n_iter, n_scen, n_year, n_mat = self.iterations_data.shape
        pooled = self.iterations_data.reshape(n_iter * n_scen, n_year, n_mat)

        results = []
        for year in self.years:
            for material in self.materials:
                i_year = self.year_idx[year]
                i_material = self.material_idx[material]

                data = pooled[:, i_year, i_material]
                
                stats = {
                    'year': year,
                    'material': material,
                    'mean': float(np.mean(data)),
                    'std': float(np.std(data))
                }
                
                # Add percentiles. Canonical naming: integer percentiles
                # use `p{int}` (e.g. p5, p25, p50, p75, p95); fractional
                # percentiles use `p{int}_{frac}` (e.g. p2_5, p97_5) per
                # the project standing rule. Legacy truncated aliases
                # (`p2`, `p97`) are also written for backward compat with
                # consumers not yet migrated. Mirrors get_statistics()
                # above; both sites must stay in sync.
                pct_values = np.percentile(data, percentiles)
                for pct, val in zip(percentiles, pct_values):
                    val = float(val)
                    if pct == int(pct):
                        stats[f'p{int(pct)}'] = val
                    else:
                        # Canonical name (e.g. p2_5, p97_5)
                        stats[f'p{pct}'.replace('.', '_')] = val
                        # Legacy truncated alias (e.g. p2, p97) — same value
                        stats[f'p{int(pct)}'] = val

                results.append(stats)
        
        return pd.DataFrame(results)


# ============================================================================
# MAIN SIMULATION CLASS
# ============================================================================

class MaterialsStockFlowSimulation:
    """
    Monte Carlo simulation of materials demand using stock-flow accounting.
    
    Process:
    1. Build stock-flow model for each scenario and technology
    2. Calculate capacity additions and retirements
    3. Sample material intensities from fitted distributions
    4. Calculate material demand = additions × intensity (additions only)
    5. Aggregate across technologies and repeat for multiple iterations
    """
    
    def __init__(
        self,
        capacity_data: pd.DataFrame,
        intensity_data: pd.DataFrame,
        fitted_distributions: Dict,
        technology_mapping: Dict = TECHNOLOGY_MAPPING,
        technology_lifetimes: Dict = TECHNOLOGY_LIFETIMES,
        random_state: Optional[int] = None
    ):
        """
        Initialize simulation.
        
        Parameters
        ----------
        capacity_data : pd.DataFrame
            Capacity projections (from load_all_data)
        intensity_data : pd.DataFrame
            Material intensity data
        fitted_distributions : Dict
            Fitted distributions from DistributionFitter
        technology_mapping : Dict
            Mapping from capacity to intensity technologies
        technology_lifetimes : Dict
            Technology lifetimes in years
        random_state : int, optional
            Random seed for reproducibility
        """
        self.capacity_data = capacity_data
        self.intensity_data = intensity_data
        self.fitted_distributions = fitted_distributions
        self.technology_mapping = technology_mapping
        self.technology_lifetimes = technology_lifetimes
        self.random_state = random_state
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
        
        # Extract metadata
        self.scenarios = sorted(capacity_data['scenario'].unique())
        self.years = sorted(capacity_data['year'].unique())
        self.materials = sorted(intensity_data['material'].unique())
        
        # Capacity technology columns
        self.capacity_tech_cols = [
            col.replace('_MW', '') for col in capacity_data.columns 
            if col.endswith('_MW')
        ]
        
        logger.info("Simulation initialized")
        logger.info(f"  Scenarios: {len(self.scenarios)}")
        logger.info(f"  Years: {self.years}")
        logger.info(f"  Capacity technologies: {len(self.capacity_tech_cols)}")
        logger.info(f"  Materials: {len(self.materials)}")
    
    def build_stock_flow_model(self) -> Dict[Tuple[str, str], StockFlowState]:
        """
        Build stock-flow model for all scenarios and technologies.
        
        Returns
        -------
        dict
            {(scenario, technology): StockFlowState}
        """
        logger.info("Building stock-flow model...")
        
        stock_flow_states = {}
        total_scenarios = len(self.scenarios)
        
        for i, scenario in enumerate(self.scenarios, 1):
            if i % 10 == 0 or i == 1:
                logger.info(f"  Processing scenario {i}/{total_scenarios}: {scenario}")
            
            scenario_data = self.capacity_data[
                self.capacity_data['scenario'] == scenario
            ].sort_values('year')
            
            for tech in self.capacity_tech_cols:
                # Skip if no mapping
                if tech not in self.technology_mapping:
                    continue
                if not self.technology_mapping[tech]:
                    continue
                
                # Initialize state
                state = StockFlowState(technology=tech, scenario=scenario)
                lifetime = get_lifetime(tech)
                
                # Process each year
                for i, (_, row) in enumerate(scenario_data.iterrows()):
                    year = int(row['year'])
                    capacity = float(row[f'{tech}_MW'])
                    is_first = (i == 0)  # First year is baseline
                    state.update(year, capacity, lifetime, is_first_year=is_first)
                
                stock_flow_states[(scenario, tech)] = state
        
        logger.info(f"Stock-flow model built: {len(stock_flow_states)} states")
        return stock_flow_states
    
    def sample_intensities(self) -> Dict[Tuple[str, str], float]:
        """
        Sample material intensities from fitted distributions (one iteration).
        
        Returns
        -------
        dict
            {(technology, material): intensity_mt_per_MW}
        """
        sampled_intensities = {}
        
        for (tech, mat), dist_info in self.fitted_distributions.items():
            # Sample from distribution (returns array of length 1)
            intensity = dist_info.sample(n=1, random_state=None)[0]
            sampled_intensities[(tech, mat)] = intensity
        
        return sampled_intensities
    
    def calculate_material_demand_single_iteration(
        self,
        stock_flow_states: Dict[Tuple[str, str], StockFlowState],
        sampled_intensities: Dict[Tuple[str, str], float]
    ) -> Dict[Tuple[str, int, str], float]:
        """
        Calculate material demand for a single Monte Carlo iteration.

        Material demand = additions × intensity
        (We count materials in new capacity, not retirements -
         retirements are implicit in maintaining stock levels)

        UNITS:
        - addition_MW: capacity additions in megawatts (MW)
        - intensity: material intensity in tonnes per megawatt (t/MW)
        - material_demand: result in tonnes (t)

        Formula: MW × (t/MW) × weight = t

        Returns
        -------
        dict
            {(scenario, year, material): demand_tonnes}
        """
        demand = {}

        for (scenario, cap_tech), state in stock_flow_states.items():
            # Get intensity technologies for this capacity technology
            intensity_mappings = get_intensity_technologies(cap_tech)

            for intensity_tech, weight in intensity_mappings.items():
                # For each year with additions
                for year, addition_MW in state.additions.items():
                    if addition_MW <= 0:
                        continue

                    # Get materials for this intensity technology
                    for material in self.materials:
                        # Check if we have intensity data
                        if (intensity_tech, material) not in sampled_intensities:
                            continue

                        intensity = sampled_intensities[(intensity_tech, material)]

                        # Calculate demand (weighted by mapping)
                        # Units: MW × (t/MW) × dimensionless = t (tonnes)
                        material_demand = addition_MW * weight * intensity

                        # Aggregate
                        key = (scenario, year, material)
                        demand[key] = demand.get(key, 0) + material_demand

        return demand
    
    def run_monte_carlo(
        self,
        n_iterations: int = 10000,
        scenarios_to_run: Optional[List[str]] = None,
        convergence_rtol: float = 0.01,
        convergence_check_every: int = 500,
        convergence_min_iterations: int = 1000,
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation with optional convergence-based early stopping.

        The simulation checks whether running means have stabilised at regular
        intervals.  When the maximum relative change across all
        (scenario, year, material) cells falls below *convergence_rtol*,
        the simulation stops early.  Set ``convergence_rtol=0`` to disable
        early stopping and always run the full *n_iterations*.

        Parameters
        ----------
        n_iterations : int
            Hard maximum number of Monte Carlo iterations (default 10,000).
        scenarios_to_run : List[str], optional
            Subset of scenarios to run (default: all).
        convergence_rtol : float
            Relative tolerance for convergence (default 0.01 = 1 %).
        convergence_check_every : int
            Check convergence every this many iterations (default 500).
        convergence_min_iterations : int
            Minimum iterations before convergence checks begin (default 1,000).

        Returns
        -------
        SimulationResult
            Results container with statistics and convergence metadata.
        """
        logger.info("="*80)
        logger.info("MONTE CARLO SIMULATION")
        logger.info("="*80)
        logger.info(f"Max iterations: {n_iterations:,}")
        if convergence_rtol > 0:
            logger.info(f"Convergence: rtol={convergence_rtol}, "
                        f"check every {convergence_check_every}, "
                        f"min {convergence_min_iterations}")
        else:
            logger.info("Convergence checking disabled")

        # Filter scenarios if specified
        if scenarios_to_run is not None:
            scenarios = [s for s in self.scenarios if s in scenarios_to_run]
        else:
            scenarios = self.scenarios

        logger.info(f"Scenarios: {len(scenarios)}")

        # Build stock-flow model (same for all iterations)
        stock_flow_states = self.build_stock_flow_model()

        # Filter stock-flow states for selected scenarios
        if scenarios_to_run is not None:
            stock_flow_states = {
                k: v for k, v in stock_flow_states.items()
                if k[0] in scenarios_to_run
            }

        # Create result arrays
        # Shape: [iterations, scenarios, years, materials]
        results_array = np.zeros((
            n_iterations,
            len(scenarios),
            len(self.years),
            len(self.materials)
        ))

        # Create index mappings
        scenario_idx = {s: i for i, s in enumerate(scenarios)}
        year_idx = {y: i for i, y in enumerate(self.years)}
        material_idx = {m: i for i, m in enumerate(self.materials)}

        # Convergence state
        prev_means = None
        converged = False
        final_iteration = n_iterations

        # Run Monte Carlo iterations
        logger.info("\nRunning Monte Carlo iterations...")

        for iteration in range(n_iterations):
            # Progress logging
            if iteration % max(1, n_iterations // 10) == 0:
                logger.info(f"  Iteration {iteration:,}/{n_iterations:,} "
                            f"({100*iteration/n_iterations:.0f}%)")

            # Sample intensities for this iteration
            sampled_intensities = self.sample_intensities()

            # Calculate demand for this iteration
            demand = self.calculate_material_demand_single_iteration(
                stock_flow_states,
                sampled_intensities
            )

            # Store in array
            for (scenario, year, material), demand_value in demand.items():
                i_s = scenario_idx[scenario]
                i_y = year_idx[year]
                i_m = material_idx[material]
                results_array[iteration, i_s, i_y, i_m] = demand_value

            # ── Convergence check ──────────────────────────────────────────
            completed = iteration + 1
            if (convergence_rtol > 0
                    and completed >= convergence_min_iterations
                    and completed % convergence_check_every == 0):

                current_means = results_array[:completed].mean(axis=0)

                if prev_means is not None:
                    mask = np.abs(prev_means) > 1e-10
                    if mask.any():
                        rel_change = (
                            np.abs(current_means[mask] - prev_means[mask])
                            / np.abs(prev_means[mask])
                        )
                        max_rel_change = float(rel_change.max())
                        logger.info(f"  Convergence check at {completed:,}: "
                                    f"max relative change = {max_rel_change:.6f}")

                        if max_rel_change < convergence_rtol:
                            converged = True
                            final_iteration = completed
                            logger.info(
                                f"  *** Converged at iteration {completed:,} "
                                f"(rtol={convergence_rtol})")
                            break

                prev_means = current_means

        # Truncate array if converged early
        if converged:
            results_array = results_array[:final_iteration]
        else:
            final_iteration = n_iterations
            if convergence_rtol > 0:
                logger.info(f"  Reached max iterations ({n_iterations:,}) "
                            "without convergence")

        logger.info(f"Monte Carlo complete: {final_iteration:,} iterations used")

        return SimulationResult(
            iterations_data=results_array,
            n_iterations=final_iteration,
            scenarios=scenarios,
            years=self.years,
            materials=self.materials,
            scenario_idx=scenario_idx,
            year_idx=year_idx,
            material_idx=material_idx,
            converged=converged,
            convergence_iteration=final_iteration if converged else None,
        )


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def run_full_simulation(
    intensity_path: Union[str, Path],
    capacity_path: Union[str, Path],
    n_iterations: int = 10000,
    scenarios_to_run: Optional[List[str]] = None,
    random_state: int = 42,
    convergence_rtol: float = 0.01,
    convergence_check_every: int = 500,
    convergence_min_iterations: int = 1000,
    output_dir: Optional[Union[str, Path]] = None,
) -> Tuple[MaterialsStockFlowSimulation, SimulationResult]:
    """
    Run complete simulation pipeline from data files to results.

    Parameters
    ----------
    intensity_path : str or Path
        Path to material intensity CSV
    capacity_path : str or Path
        Path to capacity projections CSV
    n_iterations : int
        Hard maximum number of Monte Carlo iterations (default 10,000).
    scenarios_to_run : List[str], optional
        Subset of scenarios (default: all)
    random_state : int
        Random seed
    convergence_rtol : float
        Relative tolerance for early stopping (default 0.01 = 1 %).
        Set to 0 to disable convergence checking.
    convergence_check_every : int
        Check convergence every this many iterations (default 500).
    convergence_min_iterations : int
        Minimum iterations before convergence checks begin (default 1,000).
    output_dir : str or Path, optional
        If provided, save fitted distribution CSVs to this directory.
        Creates ``fitted_distributions.csv`` and ``fit_summary.csv``.

    Returns
    -------
    simulation : MaterialsStockFlowSimulation
        Simulation object
    results : SimulationResult
        Results with statistics
    """
    logger.info("="*80)
    logger.info("FULL SIMULATION PIPELINE")
    logger.info("="*80)

    # Step 1: Load data
    logger.info("\nStep 1: Loading data...")
    data = load_all_data(
        intensity_path=intensity_path,
        national_capacity_path=capacity_path
    )

    # Step 2: Fit distributions
    logger.info("\nStep 2: Fitting distributions...")
    fitter = DistributionFitter()
    fitted_dists = fitter.fit_all(data['intensity'])

    # Step 2b: Apply CV borrowing for low-sample-size pairs
    logger.info("\nStep 2b: Applying CV borrowing for low-sample-size pairs...")
    fitted_dists = apply_cv_borrowing(
        fitted_distributions=fitted_dists,
        threshold_n=5,  # Pairs with n < 5 get borrowed CV
        reference_n=5   # Median CV from all n > 5
    )

    # Step 2c: Export fit data (if output_dir provided)
    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        fitter.export_to_csv(out / 'fitted_distributions.csv')
        fitter.export_fit_summary(out / 'fit_summary.csv')
        create_cv_borrowing_report(
            fitted_distributions=fitted_dists,
            output_path=str(out / 'cv_borrowing_report.csv'),
            threshold_n=5,
            reference_n=5
        )

    # Step 3: Validate technology mapping
    logger.info("\nStep 3: Validating technology mapping...")
    validate_mapping()

    # Step 4: Initialize simulation
    logger.info("\nStep 4: Initializing simulation...")
    simulation = MaterialsStockFlowSimulation(
        capacity_data=data['capacity_national'],
        intensity_data=data['intensity'],
        fitted_distributions=fitted_dists,
        random_state=random_state
    )

    # Step 5: Run Monte Carlo
    logger.info("\nStep 5: Running Monte Carlo simulation...")
    results = simulation.run_monte_carlo(
        n_iterations=n_iterations,
        scenarios_to_run=scenarios_to_run,
        convergence_rtol=convergence_rtol,
        convergence_check_every=convergence_check_every,
        convergence_min_iterations=convergence_min_iterations,
    )

    logger.info("\n" + "="*80)
    logger.info("SIMULATION COMPLETE")
    if results.converged:
        logger.info(f"Converged at iteration {results.convergence_iteration:,}")
    else:
        logger.info(f"Ran full {results.n_iterations:,} iterations")
    logger.info("="*80)

    return simulation, results


if __name__ == '__main__':
    # Example usage
    print("This is a module. Import it to run simulations.")
    print("\nExample:")
    print("  from stock_flow_simulation import run_full_simulation")
    print("  simulation, results = run_full_simulation(...)")
