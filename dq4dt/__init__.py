"""DQ4DT: a benchmark and metric framework for quantifying data-quality
effects on digital-twin prognostic fidelity.

The package is organised so that each stage of the pipeline can be used on
its own. Injection operators and metrics run on CPU and need no data. The
training and sweep code needs the C-MAPSS datasets and a GPU is strongly
recommended for the full benchmark.
"""

__version__ = "1.0.0"

from .config import Config, DEFAULT_CONFIG, QUICK_CONFIG  # noqa: F401
from .injectors import INJECTORS, degrade, mixed_degrade  # noqa: F401
from .metrics import (rmse, nasa_score, ece_regression, coverage,  # noqa: F401
                      fit_sigma_scale)
from .stats import holm, rank_biserial, cliffs_delta  # noqa: F401
