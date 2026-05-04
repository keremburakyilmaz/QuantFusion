from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.regime_service import (
    FEATURE_NAMES,
    LABEL_BEAR,
    LABEL_BULL,
    LABEL_SIDEWAYS,
    SMOOTH_CONF_THRESHOLD,
    SMOOTH_WINDOW,
    RegimeService,
    _smooth_states,
)


EXPECTED_WINDOWS: dict[str, tuple[str, str, str]] = {
    "Dot-com bust (2000-2002)":              ("2000-03-24", "2002-10-09", "bear"),
    "Mid-2000s expansion (2004-2006)":       ("2004-01-02", "2006-12-29", "bull"),
    "GFC (Sep 2008 - Mar 2009)":             ("2008-09-01", "2009-03-31", "bear"),
    "Post-GFC recovery (2009-2010)":         ("2009-04-01", "2010-04-30", "bull"),
    "Aug 2011 debt-ceiling":                 ("2011-07-25", "2011-10-10", "bear"),
    "QE3 grind-up (2013)":                   ("2013-01-02", "2013-12-31", "bull"),
    "China selloff (Aug-Sep 2015)":          ("2015-08-15", "2015-09-30", "bear"),
    "Late-2018 Fed pivot":                   ("2018-10-01", "2018-12-24", "bear"),
    "COVID crash (Feb-Mar 2020)":            ("2020-02-24", "2020-03-23", "bear"),
    "Post-stimulus rally (2021)":            ("2021-01-01", "2021-12-31", "bull"),
    "2022 bear market":                      ("2022-01-01", "2022-10-15", "bear"),
    "SVB stress (Mar 2023)":                 ("2023-03-08", "2023-03-24", "bear"),
}


def _fetch(ticker: str, start: str = "1995-01-01") -> pd.Series:
    df = yf.download(
        ticker, start=start, interval="1d", auto_adjust=False,
        progress=False, threads=False,
    )
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    s = df["Adj Close"].fillna(df["Close"])
    s.index = pd.to_datetime(s.index)
    return s


def _train_fold(features: pd.DataFrame) -> tuple[GaussianHMM, StandardScaler, dict[int, str]]:
    X_raw = features[FEATURE_NAMES].values
    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)

    log_return_idx = FEATURE_NAMES.index("log_return")
    vol5_idx = FEATURE_NAMES.index("realized_vol_5d")
    vol21_idx = FEATURE_NAMES.index("realized_vol_21d")
    dd_idx = FEATURE_NAMES.index("drawdown_252d")
    log_vix_idx = FEATURE_NAMES.index("log_vix")

    bear_q = np.quantile(X[:, log_return_idx], 0.10)
    side_q = np.quantile(X[:, log_return_idx], 0.50)
    bull_q = np.quantile(X[:, log_return_idx], 0.90)

    init_means = np.zeros((3, X.shape[1]))
    init_means[0, log_return_idx] = bear_q
    init_means[0, vol5_idx] = np.quantile(X[:, vol5_idx], 0.85)
    init_means[0, vol21_idx] = np.quantile(X[:, vol21_idx], 0.85)
    init_means[0, dd_idx] = np.quantile(X[:, dd_idx], 0.10)
    init_means[0, log_vix_idx] = np.quantile(X[:, log_vix_idx], 0.85)
    init_means[1, log_return_idx] = side_q
    init_means[2, log_return_idx] = bull_q
    init_means[2, vol5_idx] = np.quantile(X[:, vol5_idx], 0.20)
    init_means[2, vol21_idx] = np.quantile(X[:, vol21_idx], 0.20)
    init_means[2, dd_idx] = np.quantile(X[:, dd_idx], 0.85)
    init_means[2, log_vix_idx] = np.quantile(X[:, log_vix_idx], 0.20)

    transmat_prior = np.eye(3) * 3.0 + 1.0
    model = GaussianHMM(
        n_components=3,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=42,
        transmat_prior=transmat_prior,
        init_params="stc",
        params="stmc",
    )
    model.means_ = init_means
    model.fit(X)

    means = model.means_[:, log_return_idx]
    order = np.argsort(means)
    label_map = {
        int(order[0]): LABEL_BEAR,
        int(order[1]): LABEL_SIDEWAYS,
        int(order[2]): LABEL_BULL,
    }
    return model, scaler, label_map


def _predict_fold(
    model: GaussianHMM, scaler: StandardScaler, label_map: dict[int, str],
    features: pd.DataFrame,
) -> pd.Series:
    X = scaler.transform(features[FEATURE_NAMES].values)
    raw_states = model.predict(X)
    proba = model.predict_proba(X)
    smoothed = _smooth_states(
        raw_states, window=SMOOTH_WINDOW,
        proba=proba, conf_threshold=SMOOTH_CONF_THRESHOLD,
    )
    return pd.Series(
        [label_map[int(s)] for s in smoothed],
        index=features.index, name="regime",
    )


def _predict_fold_with_proba(
    model: GaussianHMM, scaler: StandardScaler, label_map: dict[int, str],
    features: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    X = scaler.transform(features[FEATURE_NAMES].values)
    raw_states = model.predict(X)
    proba = model.predict_proba(X)
    smoothed = _smooth_states(
        raw_states, window=SMOOTH_WINDOW,
        proba=proba, conf_threshold=SMOOTH_CONF_THRESHOLD,
    )
    labels = pd.Series(
        [label_map[int(s)] for s in smoothed],
        index=features.index, name="regime",
    )
    conf = pd.Series(
        proba[np.arange(len(smoothed)), smoothed],
        index=features.index, name="confidence",
    )
    return labels, conf


def _phase_match_rate(labels: pd.Series) -> tuple[int, int, list[tuple[str, str, str, str]]]:
    matches = 0
    total = 0
    rows: list[tuple[str, str, str, str]] = []
    for name, (start, end, expected) in EXPECTED_WINDOWS.items():
        slice_ = labels.loc[start:end]
        if slice_.empty:
            continue
        total += 1
        counts = Counter(slice_)
        dominant = counts.most_common(1)[0][0]
        status = "PASS" if dominant == expected else "FAIL"
        if status == "PASS":
            matches += 1
        rows.append((name, expected, dominant, status))
    return matches, total, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", type=int, default=10,
                        help="years of in-sample data per fold (default 10)")
    parser.add_argument("--start", type=int, default=2005,
                        help="first OOS year to predict (default 2005)")
    parser.add_argument("--end", type=int, default=2025,
                        help="last OOS year to predict (default 2025)")
    args = parser.parse_args()

    print(f"Walk-forward backtest")
    print(f"  Train window:    {args.train_years} years")
    print(f"  OOS years:       {args.start} → {args.end}")
    print(f"  Features:        {FEATURE_NAMES}")
    print()

    print("Fetching SPY + VIX from yfinance ...")
    spy = _fetch("SPY", start="1993-01-01")
    vix = _fetch("^VIX", start="1993-01-01")
    full_features = RegimeService._build_features(spy, vix)
    print(f"  full feature range: {full_features.index.min().date()} -> {full_features.index.max().date()}\n")

    all_oos_labels: list[pd.Series] = []
    all_oos_conf: list[pd.Series] = []
    fold_summaries: list[dict] = []

    print("=" * 80)
    print(f"{'fold':<8}{'train':<26}{'predict':<26}{'bull':>7}{'side':>7}{'bear':>7}{'mean_conf':>11}")
    print("=" * 80)

    for oos_year in range(args.start, args.end + 1):
        train_start = f"{oos_year - args.train_years}-01-01"
        train_end = f"{oos_year - 1}-12-31"
        oos_start = f"{oos_year}-01-01"
        oos_end = f"{oos_year}-12-31"

        train_features = full_features.loc[train_start:train_end]
        oos_features = full_features.loc[oos_start:oos_end]

        if len(train_features) < 320 or oos_features.empty:
            print(f"  {oos_year}: skipped (insufficient data)")
            continue

        model, scaler, label_map = _train_fold(train_features)
        labels, conf = _predict_fold_with_proba(model, scaler, label_map, oos_features)

        all_oos_labels.append(labels)
        all_oos_conf.append(conf)

        counts = Counter(labels)
        n = len(labels)
        bull_pct = counts.get("bull", 0) / n
        side_pct = counts.get("sideways", 0) / n
        bear_pct = counts.get("bear", 0) / n

        train_label = f"{train_start[:10]}..{train_end[:10]}"
        oos_label = f"{oos_start[:10]}..{oos_end[:10]}"
        print(
            f"  {oos_year:<6}{train_label:<26}{oos_label:<26}"
            f"{bull_pct:>6.0%} {side_pct:>6.0%} {bear_pct:>6.0%} "
            f"{conf.mean():>10.1%}"
        )

        fold_summaries.append({
            "year": oos_year,
            "n": n,
            "bull": bull_pct,
            "sideways": side_pct,
            "bear": bear_pct,
            "mean_conf": float(conf.mean()),
        })

    if not all_oos_labels:
        print("\nNo folds produced labels. Bailing.")
        return 1

    oos_labels = pd.concat(all_oos_labels).sort_index()
    oos_conf = pd.concat(all_oos_conf).sort_index()

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE PHASE MATCHES")
    print("=" * 80)
    matches, total, rows = _phase_match_rate(oos_labels)
    for name, expected, dominant, status in rows:
        print(f"  [{status}] {name:<42} expected={expected:<8} dominant={dominant}")
    if total:
        print(f"\n  Phase match rate (out-of-sample): {matches}/{total} ({matches/total:.0%})")

    print("\n" + "=" * 80)
    print("OUT-OF-SAMPLE REGIME DISTRIBUTION (combined)")
    print("=" * 80)
    counts = Counter(oos_labels)
    n = sum(counts.values())
    for label in ("bull", "sideways", "bear"):
        c = counts.get(label, 0)
        bar = "#" * int(40 * c / n) if n else ""
        print(f"  {label:<10} {c:>5} ({c / n:>5.1%})  {bar}")

    print("\n" + "=" * 80)
    print("CONFIDENCE CALIBRATION (out-of-sample)")
    print("=" * 80)
    bins = [0.33, 0.50, 0.70, 0.85, 0.95, 1.01]
    bin_labels = ["33-50%", "50-70%", "70-85%", "85-95%", "95-100%"]
    counts_arr, _ = np.histogram(oos_conf, bins=bins)
    total_n = counts_arr.sum()
    for label, c in zip(bin_labels, counts_arr):
        bar = "#" * int(40 * c / total_n) if total_n else ""
        print(f"  {label:<8} {c:>5} ({c / total_n:>5.1%})  {bar}")
    print(f"\n  mean conf:     {oos_conf.mean():.1%}")
    print(f"  median conf:   {oos_conf.median():.1%}")
    print(f"  10th pctile:   {oos_conf.quantile(0.10):.1%}")

    return 0 if matches == total else 0


if __name__ == "__main__":
    raise SystemExit(main())
