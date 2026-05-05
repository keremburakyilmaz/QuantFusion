

import math

import numpy as np
import pandas as pd
import pytest

from app.services.optimizer import PortfolioOptimizer


def _synth_returns(seed: int = 42, days: int = 252) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=days)
    return pd.DataFrame(
        {
            "A": rng.normal(0.0006, 0.012, days),
            "B": rng.normal(0.0004, 0.010, days),
            "C": rng.normal(0.0008, 0.015, days),
            "D": rng.normal(0.0005, 0.011, days),
        },
        index=idx,
    )


def _cov(returns: pd.DataFrame) -> np.ndarray:
    return returns.cov().values * 252


def test_mvo_max_sharpe_basic():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    res = opt.mvo(returns, _cov(returns), target="max_sharpe", rf=0.04)
    assert math.isfinite(res.sharpe)
    total = sum(res.weights.values())
    assert total == pytest.approx(1.0, abs=1e-6)
    for w in res.weights.values():
        assert 0.01 - 1e-6 <= w <= 0.60 + 1e-6


def test_mvo_min_vol_lower_than_equal_weight():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    eq_w = np.full(len(returns.columns), 1.0 / len(returns.columns))
    eq_vol = float(np.sqrt(eq_w @ cov @ eq_w))
    res = opt.mvo(returns, cov, target="min_vol", rf=0.04)
    assert res.volatility <= eq_vol + 1e-6


def test_risk_parity_marginal_contribs_close_to_equal():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    res = opt.risk_parity(cov, list(returns.columns), returns=returns, rf=0.04)
    w = np.array([res.weights[t] for t in returns.columns])
    port_vol = float(np.sqrt(w @ cov @ w))
    mrc = (cov @ w) / port_vol
    contribs = (w * mrc) / np.sum(w * mrc)
    assert max(contribs) - min(contribs) < 0.05


def test_black_litterman_no_views_uses_equilibrium():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    caps = {"A": 1e9, "B": 1e9, "C": 1e9, "D": 1e9}
    res = opt.black_litterman(
        caps, cov, list(returns.columns), views=None, returns=returns, rf=0.04
    )
    assert sum(res.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_black_litterman_view_shifts_weight():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    caps = {"A": 1e9, "B": 1e9, "C": 1e9, "D": 1e9}
    base = opt.black_litterman(
        caps, cov, list(returns.columns), views=None, returns=returns, rf=0.04
    )
    biased = opt.black_litterman(
        caps,
        cov,
        list(returns.columns),
        views=[{"ticker": "A", "view_return": 0.50, "confidence": 0.99}],
        returns=returns,
        rf=0.04,
    )
    assert biased.weights["A"] > base.weights["A"]


def test_efficient_frontier_returns_grid_and_specials():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    points = opt.efficient_frontier(returns, cov, n=20, rf=0.04)
    assert len(points) >= 20
    kinds = {p.kind for p in points if p.kind}
    assert kinds.issuperset({"max_sharpe", "min_vol"})


def test_regime_blended_pure_bull_matches_max_sharpe():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    max_sharpe = opt.mvo(returns, cov, target="max_sharpe", rf=0.04)
    blended = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 1.0, "sideways": 0.0, "bear": 0.0},
        rf=0.04,
    )
    for ticker in returns.columns:
        assert blended.weights[ticker] == pytest.approx(
            max_sharpe.weights[ticker], abs=1e-6
        )
    assert blended.method == "regime_blended"
    assert blended.regime_weights == {"bull": 1.0, "sideways": 0.0, "bear": 0.0}


def test_regime_blended_pure_bear_matches_min_vol():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    min_vol = opt.mvo(returns, cov, target="min_vol", rf=0.04)
    blended = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 0.0, "sideways": 0.0, "bear": 1.0},
        rf=0.04,
    )
    for ticker in returns.columns:
        assert blended.weights[ticker] == pytest.approx(
            min_vol.weights[ticker], abs=1e-6
        )


def test_regime_blended_mixture_volatility_between_extremes():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    max_sharpe = opt.mvo(returns, cov, target="max_sharpe", rf=0.04)
    min_vol = opt.mvo(returns, cov, target="min_vol", rf=0.04)
    blended = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 0.5, "sideways": 0.0, "bear": 0.5},
        rf=0.04,
    )
    assert min_vol.volatility - 1e-6 <= blended.volatility <= max_sharpe.volatility + 1e-6


def test_regime_blended_renormalizes_probabilities():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    a = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 0.5, "sideways": 0.5},
        rf=0.04,
    )
    b = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 5.0, "sideways": 5.0},
        rf=0.04,
    )
    for ticker in returns.columns:
        assert a.weights[ticker] == pytest.approx(b.weights[ticker], abs=1e-6)


def test_regime_blended_response_includes_components():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    blended = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 0.4, "sideways": 0.4, "bear": 0.2},
        rf=0.04,
    )
    assert blended.components is not None
    assert set(blended.components.keys()) == {"max_sharpe", "risk_parity", "min_vol"}
    for component_weights in blended.components.values():
        assert sum(component_weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_regime_blended_weights_sum_to_one():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    blended = opt.regime_blended(
        returns, cov,
        regime_probabilities={"bull": 0.3, "sideways": 0.5, "bear": 0.2},
        rf=0.04,
    )
    assert sum(blended.weights.values()) == pytest.approx(1.0, abs=1e-6)


def test_regime_blended_rejects_zero_probabilities():
    opt = PortfolioOptimizer()
    returns = _synth_returns()
    cov = _cov(returns)
    with pytest.raises(ValueError):
        opt.regime_blended(
            returns, cov,
            regime_probabilities={"bull": 0.0, "sideways": 0.0, "bear": 0.0},
            rf=0.04,
        )
