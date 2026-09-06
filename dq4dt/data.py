"""C-MAPSS ingestion and leakage-safe preprocessing.

Download is attempted from three public mirrors in turn. Files are cached in
`data_root` so the download happens once. Preprocessing follows the paper:
drop the seven constant sensors, per-regime z-scoring for the multi-regime
subsets with the clustering fit on training data only, sliding windows of
length 30, piecewise-linear RUL capped at 125, and an engine-level split.
"""

import io
import os
import urllib.request
import zipfile
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

ALL_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MULTI_REGIME = {"FD002", "FD004"}
CONST_SENSORS = [1, 5, 6, 10, 16, 18, 19]
SENSOR_COLS = [f"s{i}" for i in range(1, 22) if i not in CONST_SENSORS]
OP_COLS = ["op1", "op2", "op3"]
FEAT_COLS = OP_COLS + SENSOR_COLS
COLS = (["unit", "cycle"] + OP_COLS + [f"s{i}" for i in range(1, 22)])

_SOURCES = [
    ("NASA Open Data Portal",
     "https://data.nasa.gov/docs/legacy/CMAPSSData.zip"),
    ("PHM Society S3 mirror",
     "https://phm-datasets.s3.amazonaws.com/NASA/"
     "6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip"),
]
_GITHUB_RAW = "https://raw.githubusercontent.com/edwardzjl/CMAPSSData/master/"
_UA = {"User-Agent": "Mozilla/5.0 (DQ4DT benchmark)"}


def _expected_files():
    return [f"{k}_{s}.txt" for s in ALL_SUBSETS for k in ("train", "test", "RUL")]


def _have_all(dest: str) -> bool:
    return all(os.path.exists(os.path.join(dest, f)) for f in _expected_files())


def _extract_zip_bytes(raw: bytes, dest: str) -> None:
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        for name in z.namelist():
            low = name.lower()
            if low.endswith(".zip"):
                _extract_zip_bytes(z.read(name), dest)
            elif low.endswith(".txt"):
                base = os.path.basename(name)
                if base:
                    with open(os.path.join(dest, base), "wb") as fh:
                        fh.write(z.read(name))


def download_cmapss(data_root: str, verbose: bool = True) -> str:
    """Fetch all twelve C-MAPSS text files into data_root/CMAPSSData."""
    dest = os.path.join(data_root, "CMAPSSData")
    os.makedirs(dest, exist_ok=True)
    if _have_all(dest):
        if verbose:
            print("C-MAPSS: cache hit, all 12 files present.")
        return dest
    for label, url in _SOURCES:
        try:
            if verbose:
                print(f"[C-MAPSS] trying {label}")
            req = urllib.request.Request(url, headers=_UA)
            raw = urllib.request.urlopen(req, timeout=180).read()
            _extract_zip_bytes(raw, dest)
            if _have_all(dest):
                return dest
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"    failed: {exc}")
    # GitHub raw mirror, file by file
    if verbose:
        print("[C-MAPSS] trying GitHub raw mirror")
    for fname in _expected_files():
        try:
            req = urllib.request.Request(_GITHUB_RAW + fname, headers=_UA)
            with open(os.path.join(dest, fname), "wb") as fh:
                fh.write(urllib.request.urlopen(req, timeout=60).read())
        except Exception as exc:  # noqa: BLE001
            if verbose:
                print(f"    {fname} failed: {exc}")
    if not _have_all(dest):
        raise RuntimeError(
            "Could not download C-MAPSS from any mirror. Download manually "
            "(for example the Kaggle dataset behrad3d/nasa-cmaps) and place "
            f"the twelve .txt files in {dest}")
    return dest


def load_raw(subset: str, cmapss_dir: str):
    def _read(kind):
        path = os.path.join(cmapss_dir, f"{kind}_{subset}.txt")
        df = pd.read_csv(path, sep=r"\s+", header=None).dropna(axis=1, how="all")
        if df.shape[1] != 26:
            raise ValueError(f"{path}: expected 26 columns, got {df.shape[1]}")
        df.columns = COLS
        return df

    train, test = _read("train"), _read("test")
    rul = pd.read_csv(os.path.join(cmapss_dir, f"RUL_{subset}.txt"),
                      sep=r"\s+", header=None).iloc[:, 0].values
    if len(rul) != test["unit"].nunique():
        raise ValueError("RUL file length does not match number of test engines")
    return train, test, rul


def add_train_rul(df: pd.DataFrame, cap: int) -> pd.DataFrame:
    df = df.copy()
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (max_cycle - df["cycle"]).clip(upper=cap).astype(np.float32)
    return df


def add_test_rul(df: pd.DataFrame, rul_last: np.ndarray, cap: int) -> pd.DataFrame:
    df = df.copy()
    units = np.sort(df["unit"].unique())
    rul_map = dict(zip(units, rul_last))
    max_cycle = df.groupby("unit")["cycle"].transform("max")
    df["RUL"] = (df["unit"].map(rul_map) + (max_cycle - df["cycle"]))
    df["RUL"] = df["RUL"].clip(upper=cap).astype(np.float32)
    return df


def fit_scalers(train: pd.DataFrame, multi_regime: bool, n_regimes: int, seed: int):
    """Fit per-regime scalers on TRAINING data only."""
    if multi_regime:
        km = KMeans(n_clusters=n_regimes, n_init=10, random_state=seed).fit(train[OP_COLS])
        labels = km.predict(train[OP_COLS])
        scalers = {r: StandardScaler().fit(train.loc[labels == r, SENSOR_COLS])
                   for r in range(n_regimes)}
        return km, scalers
    return None, {0: StandardScaler().fit(train[SENSOR_COLS])}


def apply_scalers(df: pd.DataFrame, km, scalers) -> pd.DataFrame:
    df = df.copy()
    if km is not None:
        labels = km.predict(df[OP_COLS])
        for r, sc in scalers.items():
            mask = labels == r
            if mask.any():
                df.loc[mask, SENSOR_COLS] = sc.transform(df.loc[mask, SENSOR_COLS])
    else:
        df[SENSOR_COLS] = scalers[0].transform(df[SENSOR_COLS])
    for col in OP_COLS:
        lo, hi = df[col].min(), df[col].max()
        df[col] = (df[col] - lo) / (hi - lo + 1e-8)
    return df


def make_windows(df: pd.DataFrame, window: int, stride: int, is_train: bool):
    xs, ys = [], []
    for _, g in df.groupby("unit"):
        g = g.sort_values("cycle")
        feats = g[FEAT_COLS].values.astype(np.float32)
        rul = g["RUL"].values.astype(np.float32)
        n = len(g)
        if is_train:
            for start in range(0, n - window + 1, stride):
                xs.append(feats[start:start + window])
                ys.append(rul[start + window - 1])
        else:
            if n >= window:
                xs.append(feats[-window:])
            else:
                pad = np.repeat(feats[:1], window - n, axis=0)
                xs.append(np.vstack([pad, feats]))
            ys.append(rul[-1])
    return np.asarray(xs, np.float32), np.asarray(ys, np.float32)


def build_subset(subset: str, cmapss_dir: str, window: int = 30, stride: int = 1,
                 rul_cap: int = 125, n_regimes: int = 6, val_fraction: float = 0.2,
                 seed: int = 42) -> Dict:
    """Return a dict with Xtr, ytr, Xva, yva, Xte, yte, n_feat, subset."""
    train_raw, test_raw, rul_test = load_raw(subset, cmapss_dir)
    multi = subset in MULTI_REGIME
    train_raw = add_train_rul(train_raw, rul_cap)
    test_raw = add_test_rul(test_raw, rul_test, rul_cap)
    km, scalers = fit_scalers(train_raw, multi, n_regimes, seed)
    train_s = apply_scalers(train_raw, km, scalers)
    test_s = apply_scalers(test_raw, km, scalers)

    units = train_s["unit"].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(units)
    n_val = max(1, int(val_fraction * len(units)))
    val_units, tr_units = set(units[:n_val]), set(units[n_val:])

    x_tr, y_tr = make_windows(train_s[train_s["unit"].isin(tr_units)], window, stride, True)
    x_va, y_va = make_windows(train_s[train_s["unit"].isin(val_units)], window, stride, True)
    x_te, y_te = make_windows(test_s, window, stride, False)
    return dict(Xtr=x_tr, ytr=y_tr, Xva=x_va, yva=y_va, Xte=x_te, yte=y_te,
                n_feat=len(FEAT_COLS), subset=subset)
