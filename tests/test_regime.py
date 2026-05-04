from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from app.services.regime_service import (
    FEATURE_NAMES,
    RegimeService,
    _smooth_states,
)


def _synth_market(seed: int = 7):
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=1500)

    bear = rng.normal(-0.001, 0.025, 500)
    bull = rng.normal(0.0015, 0.008, 500)
    side = rng.normal(0.0, 0.012, 500)
    log_returns = np.concatenate([bear, bull, side])
    spy = pd.Series(100 * np.exp(np.cumsum(log_returns)), index=idx, name="SPY")

    vix_levels = np.concatenate([
        rng.normal(35, 5, 500),
        rng.normal(15, 3, 500),
        rng.normal(20, 4, 500),
    ])
    vix = pd.Series(np.clip(vix_levels, 5, 80), index=idx, name="^VIX")

    return spy, vix


def test_train_and_predict_runs(tmp_path, monkeypatch):
    import app.services.regime_service as mod

    monkeypatch.setattr(mod, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)

    spy, vix = _synth_market()
    svc = RegimeService(data_service=None, session_factory=None)
    svc._train_sync(spy, vix)
    assert (tmp_path / "model.pkl").exists()

    features = svc._build_features(spy, vix)
    assert set(features.columns) == set(FEATURE_NAMES)
    assert len(features) > 1000


def test_build_features_drawdown_is_non_positive():
    spy, vix = _synth_market()
    features = RegimeService._build_features(spy, vix)
    assert (features["drawdown_252d"] <= 1e-9).all()


def test_smoothing_suppresses_low_conf_flip():
    states = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    proba = np.full((10, 3), 0.4)
    proba[:, 1] = 0.5
    smoothed = _smooth_states(states, window=5, proba=proba, conf_threshold=0.85)
    assert smoothed[4] == 1


def test_smoothing_passes_high_conf_flip():
    states = np.array([1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
    proba = np.full((10, 3), 0.05)
    proba[:, 1] = 0.95
    proba[6:, 0] = 0.97
    proba[6:, 1] = 0.02
    smoothed = _smooth_states(states, window=5, proba=proba, conf_threshold=0.85)
    assert smoothed[6] == 0


def test_smoothing_without_proba_falls_back_to_mode():
    states = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
    smoothed = _smooth_states(states, window=5)
    assert smoothed[4] == 1


def test_predict_returns_probabilities(tmp_path, monkeypatch):
    import app.services.regime_service as mod

    monkeypatch.setattr(mod, "MODEL_PATH", tmp_path / "model.pkl")
    monkeypatch.setattr(mod, "MODEL_DIR", tmp_path)

    spy, vix = _synth_market()
    svc = RegimeService(data_service=None, session_factory=None)
    svc._train_sync(spy, vix)
    snapshot = svc._predict_sync(spy, vix)

    assert snapshot is not None
    assert snapshot.probabilities is not None
    assert set(snapshot.probabilities.keys()) == {"bull", "sideways", "bear"}
    total = sum(snapshot.probabilities.values())
    assert abs(total - 1.0) < 1e-6
    assert snapshot.regime in {"bull", "sideways", "bear"}


def test_regime_to_strategy_mapping():
    svc = RegimeService(data_service=None, session_factory=None)
    assert svc.regime_to_strategy("bull") == "max_sharpe"
    assert svc.regime_to_strategy("bear") == "min_vol"
    assert svc.regime_to_strategy("sideways") == "risk_parity"
    assert svc.regime_to_strategy("unknown") == "max_sharpe"
