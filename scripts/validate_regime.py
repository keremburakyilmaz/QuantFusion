from __future__ import annotations

import sys
from collections import Counter
from itertools import groupby
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.regime_service import (
    FEATURE_NAMES,
    MODEL_PATH,
    SMOOTH_CONF_THRESHOLD,
    SMOOTH_WINDOW,
    RegimeService,
    _smooth_states,
)


WINDOWS: dict[str, tuple[str, str, str]] = {
    "Dot-com bust (Mar 2000 - Oct 2002)":          ("2000-03-24", "2002-10-09", "bear"),
    "Mid-2000s expansion (2004 - 2006)":           ("2004-01-02", "2006-12-29", "bull"),
    "GFC (Sep 2008 - Mar 2009)":                   ("2008-09-01", "2009-03-31", "bear"),
    "Post-GFC recovery (Apr 2009 - Apr 2010)":     ("2009-04-01", "2010-04-30", "bull"),
    "Aug 2011 debt-ceiling crisis":                ("2011-07-25", "2011-10-10", "bear"),
    "QE3 grind-up (2013)":                         ("2013-01-02", "2013-12-31", "bull"),
    "China devaluation selloff (Aug 2015)":        ("2015-08-15", "2015-09-30", "bear"),
    "Late-2018 Fed pivot drawdown":                ("2018-10-01", "2018-12-24", "bear"),
    "COVID crash phase (Feb 24 - Mar 23, 2020)":   ("2020-02-24", "2020-03-23", "bear"),
    "Post-stimulus rally (2021)":                  ("2021-01-01", "2021-12-31", "bull"),
    "2022 bear market":                            ("2022-01-01", "2022-10-15", "bear"),
    "SVB / banking stress (Mar 2023)":             ("2023-03-08", "2023-03-24", "bear"),
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


def _section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _windows_section(labels: pd.Series) -> int:
    _section("HISTORICAL REGIME WINDOWS")
    matches = 0
    for name, (start, end, expected) in WINDOWS.items():
        slice_ = labels.loc[start:end]
        if slice_.empty:
            print(f"{name}\n  no data in window\n")
            continue
        counts = Counter(slice_)
        total = sum(counts.values())
        dominant, dominant_n = counts.most_common(1)[0]
        breakdown = ", ".join(
            f"{label}={count} ({count / total:.0%})"
            for label, count in counts.most_common()
        )
        ok = "PASS" if dominant == expected else "FAIL"
        if ok == "PASS":
            matches += 1
        print(f"{name}")
        print(f"  expected: {expected:<8}  dominant: {dominant:<8}  [{ok}]")
        print(f"  {breakdown}\n")
    print(f"Phase match rate: {matches}/{len(WINDOWS)}")
    return matches


def _overall_section(labels: pd.Series) -> None:
    _section("OVERALL REGIME DISTRIBUTION")
    counts = Counter(labels)
    total = sum(counts.values())
    for label, count in counts.most_common():
        bar = "#" * int(40 * count / total)
        print(f"  {label:<10} {count:>5} ({count / total:>5.1%})  {bar}")
    print(f"  total      {total:>5}  trading days observed")


def _learned_centroids_section(model, scaler, label_map) -> None:
    _section("LEARNED STATE CENTROIDS (raw feature units)")
    raw_means = scaler.inverse_transform(model.means_)
    short = {
        "log_return": "log_ret",
        "realized_vol_5d": "rv_5d",
        "realized_vol_21d": "rv_21d",
        "momentum_60d": "mom_60d",
        "drawdown_252d": "dd_252",
        "log_vix": "log_vix",
        "vix_change_5d": "dvix_5d",
    }
    header = f"  {'state':<10}" + "".join(f"{short[f]:>10}" for f in FEATURE_NAMES)
    print(header)
    for state_idx in range(model.n_components):
        label = label_map[state_idx]
        line = f"  {label:<10}" + "".join(f"{v:>10.4f}" for v in raw_means[state_idx])
        print(line)
    print()
    print("  Stationary distribution (long-run share of each regime):")
    try:
        eigvals, eigvecs = np.linalg.eig(model.transmat_.T)
        idx = np.argmin(np.abs(eigvals - 1.0))
        stat = np.real(eigvecs[:, idx])
        stat = stat / stat.sum()
        for i, p in enumerate(stat):
            print(f"    {label_map[i]:<10} {p:>6.1%}")
    except Exception as exc:
        print(f"    (could not compute: {exc})")


def _transition_section(model, label_map) -> None:
    _section("TRANSITION MATRIX  (P[next | current])")
    order = sorted(label_map.keys(), key=lambda i: ["bear", "sideways", "bull"].index(label_map[i]))
    header = "from \\ to     " + "".join(f"{label_map[i]:>10}" for i in order)
    print(header)
    for i in order:
        row = f"  {label_map[i]:<10}    "
        for j in order:
            row += f"{model.transmat_[i, j]:>10.3f}"
        print(row)
    print()
    print("  Implied average regime duration  (1 / (1 - p_stay)):")
    for i in order:
        p_stay = model.transmat_[i, i]
        if p_stay >= 1.0:
            avg = float("inf")
        else:
            avg = 1.0 / (1.0 - p_stay)
        print(f"    {label_map[i]:<10} {avg:>6.1f} trading days")


def _empirical_durations_section(labels: pd.Series) -> None:
    _section("EMPIRICAL REGIME-SEGMENT DURATIONS")
    segments: dict[str, list[int]] = {"bull": [], "sideways": [], "bear": []}
    for label, group in groupby(labels):
        segments[label].append(sum(1 for _ in group))
    print(f"  {'regime':<10}{'segments':>10}{'mean':>10}{'median':>10}{'p90':>10}{'max':>10}")
    for regime in ("bull", "sideways", "bear"):
        runs = segments[regime]
        if not runs:
            print(f"  {regime:<10}        0         -         -         -         -")
            continue
        arr = np.array(runs)
        print(
            f"  {regime:<10}"
            f"{len(runs):>10}"
            f"{arr.mean():>10.1f}"
            f"{int(np.median(arr)):>10}"
            f"{int(np.quantile(arr, 0.9)):>10}"
            f"{arr.max():>10}"
        )


def _per_regime_features_section(features: pd.DataFrame, labels: pd.Series) -> None:
    _section("PER-REGIME FEATURE STATISTICS  (raw units, observed)")
    aligned = features.loc[labels.index].copy()
    aligned["regime"] = labels
    grouped = aligned.groupby("regime")[FEATURE_NAMES].agg(["mean", "std"])
    print(f"  {'feature':<20}{'bear':>22}{'sideways':>22}{'bull':>22}")
    for feat in FEATURE_NAMES:
        line = f"  {feat:<20}"
        for regime in ("bear", "sideways", "bull"):
            if regime in grouped.index:
                m = float(grouped.loc[regime, (feat, "mean")])
                s = float(grouped.loc[regime, (feat, "std")])
                line += f"{m:>10.4f} ± {s:>7.4f}  "
            else:
                line += " " * 22
        print(line)


def _confidence_section(model, scaler, features: pd.DataFrame, labels: pd.Series) -> None:
    _section("PREDICTION CONFIDENCE DISTRIBUTION")
    X = scaler.transform(features[FEATURE_NAMES].values)
    proba = model.predict_proba(X)
    states = model.predict(X)
    smoothed = _smooth_states(
        states, window=SMOOTH_WINDOW, proba=proba, conf_threshold=SMOOTH_CONF_THRESHOLD
    )
    conf = proba[np.arange(len(states)), smoothed]
    bins = [0.33, 0.50, 0.70, 0.85, 0.95, 1.01]
    bin_labels = ["33-50%", "50-70%", "70-85%", "85-95%", "95-100%"]
    counts, _ = np.histogram(conf, bins=bins)
    total = counts.sum()
    print("  Confidence in predicted state on each day:")
    for label, n in zip(bin_labels, counts):
        bar = "#" * int(40 * n / total) if total else ""
        print(f"    {label:<8} {n:>5} ({n / total:>5.1%})  {bar}")
    print(f"\n  mean confidence: {conf.mean():.1%}")
    print(f"  median:          {np.median(conf):.1%}")
    print(f"  10th percentile: {np.quantile(conf, 0.10):.1%}  (lowest-conviction days)")


def _recent_tape_section(model, scaler, features: pd.DataFrame, labels: pd.Series, n: int = 30) -> None:
    _section(f"RECENT {n}-DAY REGIME TAPE")
    X = scaler.transform(features[FEATURE_NAMES].values)
    proba = model.predict_proba(X)
    states = model.predict(X)
    smoothed = _smooth_states(
        states, window=SMOOTH_WINDOW, proba=proba, conf_threshold=SMOOTH_CONF_THRESHOLD
    )
    conf = proba[np.arange(len(states)), smoothed]
    df = pd.DataFrame(
        {
            "regime": labels.values,
            "confidence": conf,
            "log_return": features["log_return"].values,
            "log_vix": features["log_vix"].values,
        },
        index=features.index,
    ).tail(n)
    print(f"  {'date':<14}{'regime':<10}{'conf':>8}{'log_ret':>10}{'log_vix':>10}")
    for ts, row in df.iterrows():
        print(
            f"  {ts.strftime('%Y-%m-%d'):<14}"
            f"{row['regime']:<10}"
            f"{row['confidence']:>8.1%}"
            f"{row['log_return']:>10.4f}"
            f"{row['log_vix']:>10.3f}"
        )


def _recent_transitions_section(labels: pd.Series, n: int = 10) -> None:
    _section(f"MOST RECENT {n} REGIME TRANSITIONS")
    transitions: list[tuple[pd.Timestamp, str, str]] = []
    prev = None
    for ts, label in labels.items():
        if prev is not None and label != prev:
            transitions.append((ts, prev, label))
        prev = label
    if not transitions:
        print("  (no transitions found)")
        return
    print(f"  {'date':<14}{'from':<12}{'to':<12}")
    for ts, frm, to in transitions[-n:]:
        print(f"  {ts.strftime('%Y-%m-%d'):<14}{frm:<12}{to:<12}")


def main() -> int:
    if not MODEL_PATH.exists():
        print(f"Model not found at {MODEL_PATH}.")
        print("Train first: docker compose exec api python scripts/train_regime.py")
        return 1

    loaded = joblib.load(MODEL_PATH)
    if len(loaded) != 4:
        print("Model file missing scaler — retrain first.")
        return 1
    model, scaler, label_map, feature_names = loaded

    print(f"Model loaded:  {MODEL_PATH}")
    print(f"Features:      {feature_names}")
    print(f"Label map:     {label_map}")
    print(f"States:        {model.n_components}, covariance: {model.covariance_type}")

    spy = _fetch("SPY")
    vix = _fetch("^VIX")

    features = RegimeService._build_features(spy, vix)
    if features.empty:
        print("No features built — bailing.")
        return 1

    print(f"Sample range:  {features.index.min().date()} -> {features.index.max().date()}")
    print(f"Total days:    {len(features)}")

    X = scaler.transform(features[FEATURE_NAMES].values)
    raw_states = model.predict(X)
    proba = model.predict_proba(X)
    states = _smooth_states(
        raw_states,
        window=SMOOTH_WINDOW,
        proba=proba,
        conf_threshold=SMOOTH_CONF_THRESHOLD,
    )
    labels = pd.Series(
        [label_map[int(s)] for s in states],
        index=features.index,
        name="regime",
    )

    _windows_section(labels)
    _overall_section(labels)
    _learned_centroids_section(model, scaler, label_map)
    _transition_section(model, label_map)
    _empirical_durations_section(labels)
    _per_regime_features_section(features, labels)
    _confidence_section(model, scaler, features, labels)
    _recent_tape_section(model, scaler, features, labels, n=30)
    _recent_transitions_section(labels, n=10)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
