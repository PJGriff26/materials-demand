"""The canonical random seed (24601), shared by every stochastic step.

Imported as `RANDOM_SEED` so the whole pipeline's randomness is controlled by a
single source; overridable via the MATERIALS_DEMAND_SEED environment variable.
"""
import os

# Default seed is 24601; any integer works, this fixed value just makes runs
# reproducible. The MATERIALS_DEMAND_SEED env var overrides it when set.
RANDOM_SEED: int = int(os.environ.get("MATERIALS_DEMAND_SEED", 24601))

__all__ = ["RANDOM_SEED"]
