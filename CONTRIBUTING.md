# Contributing

Thanks for your interest. The most useful contributions are new degradation operators, new datasets, and new surrogate architectures, because each one extends the benchmark without changing the metric.

## Adding a degradation operator

1. Implement `inject_<name>(x, s, gen)` in `dq4dt/injectors.py`. It must accept a batch of shape (B, L, d), an ordinal severity 1 to 5, and a torch Generator, and return `(x_degraded, u)` where `u` has shape (B, d) with values in [0, 1].
2. Register it in `INJECTORS`.
3. Add a test in `tests/test_injectors.py`. The parametrized shape test picks it up automatically; add a behavioural test if the operator has a property worth checking.
4. Add the name to `Config.dims` if it should be part of the default sweep.

## Adding a dataset

Write a `build_<dataset>()` in `dq4dt/data.py` that returns the same dict keys as `build_subset` (Xtr, ytr, Xva, yva, Xte, yte, n_feat, subset). Fit any scalers on the training partition only and split at the unit level.

## Adding a model

Subclass `nn.Module` in `dq4dt/models.py`, accept `(n_feat, use_qcg=False)` in the constructor, return `(mu, log_var)` from `forward(x, u=None)`, and register in `MODEL_REGISTRY`. Add a learning rate to `Config.lr`.

## Style

Run `ruff check .` and `python -m pytest` before opening a pull request. Keep line length at 100. Prefer plain functions over classes unless state is needed.

## Reporting results

If you run the benchmark on new hardware or a new dataset, please open an issue with the sweep CSV attached. Collecting these will help establish how stable the collapse point is across settings.
