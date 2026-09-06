"""Experiment configuration.

All knobs that control the size of the benchmark live here. The defaults
reproduce the paper. QUICK_CONFIG is a smoke test that runs in about an hour
on a single GPU and is useful for checking that everything is wired up.
"""

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class Config:
    # Which C-MAPSS subsets to run. FD002 and FD003 work out of the box.
    subsets: List[str] = field(default_factory=lambda: ["FD001", "FD004"])

    # Degradation dimensions evaluated in the main sweep (eight in total).
    dims: List[str] = field(default_factory=lambda: [
        "accuracy", "accuracy_bias", "completeness_MCAR", "completeness_MAR",
        "completeness_MNAR", "timeliness", "consistency", "drift"])

    # Ordinal severities. Severity 0 (clean) is always added by the sweep.
    severities: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])

    models: List[str] = field(default_factory=lambda: ["lstm", "transformer", "pinn"])
    seeds: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4])
    interaction_seeds: List[int] = field(default_factory=lambda: list(range(8)))

    # Training
    epochs: int = 80
    patience: int = 15
    warmup_epochs: int = 5
    physics_weight: float = 0.01
    improve_margin: float = 0.05
    batch_size: int = 256
    lr: Dict[str, float] = field(default_factory=lambda: {
        "lstm": 1e-3, "transformer": 5e-4, "pinn": 1e-3})

    # Preprocessing
    window: int = 30
    stride: int = 1
    rul_cap: int = 125
    n_regimes: int = 6
    val_fraction: float = 0.2

    # Analysis
    bootstrap_resamples: int = 500
    breakpoint_min_gain: float = 0.20
    cross_dataset_tolerance: float = 0.10

    global_seed: int = 42

    @property
    def n_cells(self) -> int:
        """Number of evaluation cells in the main sweep."""
        return (len(self.subsets) * len(self.models) * len(self.seeds)
                * len(self.dims) * (len(self.severities) + 1))


DEFAULT_CONFIG = Config()

QUICK_CONFIG = Config(
    subsets=["FD001"],
    severities=[1, 3, 5],
    seeds=[0, 1, 2],
    interaction_seeds=[0, 1, 2],
    epochs=30,
    patience=8,
)
