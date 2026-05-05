

import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from app.schemas.optimization import FrontierPoint, OptimizationResult


TRADING_DAYS = 252
DEFAULT_MIN_WEIGHT = 0.01
DEFAULT_MAX_WEIGHT = 0.60
TAU = 0.05


class PortfolioOptimizer:
    def mvo(
        self,
        returns: pd.DataFrame,
        cov: np.ndarray,
        target: str = "max_sharpe",
        constraints: dict | None = None,
        rf: float = 0.04,
        expected_returns: np.ndarray | None = None,
    ) -> OptimizationResult:
        constraints = constraints or {}
        min_w = constraints.get("min_weight", DEFAULT_MIN_WEIGHT)
        max_w = constraints.get("max_weight", DEFAULT_MAX_WEIGHT)
        tickers = list(returns.columns)
        n = len(tickers)

        mu = (
            np.asarray(expected_returns)
            if expected_returns is not None
            else returns.mean().values * TRADING_DAYS
        )
        sigma = np.asarray(cov)

        bounds = [(min_w, max_w)] * n
        eq_constraints: list[dict] = [
            {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
        ]

        if target == "target_return":
            tr = constraints.get("target_return")
            if tr is None:
                raise ValueError("target='target_return' requires constraints.target_return")
            eq_constraints.append(
                {"type": "eq", "fun": lambda w, tr=tr: float(w @ mu - tr)}
            )

        x0 = np.full(n, 1.0 / n)
        start = time.perf_counter()

        if target == "max_sharpe":
            def neg_sharpe(w):
                ret = float(w @ mu)
                vol = float(np.sqrt(w @ sigma @ w))
                return -((ret - rf) / vol) if vol > 0 else 0.0

            res = minimize(
                neg_sharpe, x0, method="SLSQP",
                bounds=bounds, constraints=eq_constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )
        else:
            def variance(w):
                return float(w @ sigma @ w)

            res = minimize(
                variance, x0, method="SLSQP",
                bounds=bounds, constraints=eq_constraints,
                options={"maxiter": 200, "ftol": 1e-9},
            )

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        w = res.x / np.sum(res.x)
        return self._make_result(
            "mvo", target, tickers, w, mu, sigma, rf, elapsed_ms
        )

    def risk_parity(
        self,
        cov: np.ndarray,
        tickers: list[str],
        returns: pd.DataFrame | None = None,
        rf: float = 0.04,
    ) -> OptimizationResult:
        n = len(tickers)
        sigma = np.asarray(cov)
        target_contrib = 1.0 / n

        def objective(w):
            port_vol = float(np.sqrt(w @ sigma @ w))
            if port_vol == 0:
                return 1e6
            mrc = (sigma @ w) / port_vol
            contrib = w * mrc
            normalized = contrib / np.sum(contrib)
            return float(np.sum((normalized - target_contrib) ** 2))

        bounds = [(1e-6, 1.0)] * n
        eq = [{"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}]
        x0 = np.full(n, 1.0 / n)

        start = time.perf_counter()
        res = minimize(
            objective, x0, method="SLSQP", bounds=bounds, constraints=eq,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        elapsed_ms = int((time.perf_counter() - start) * 1000)

        w = res.x / np.sum(res.x)
        if returns is not None:
            mu = returns.mean().values * TRADING_DAYS
        else:
            mu = np.zeros(n)
        return self._make_result(
            "risk_parity", None, tickers, w, mu, sigma, rf, elapsed_ms
        )

    _REGIME_STRATEGY = {
        "bull": "max_sharpe",
        "sideways": "risk_parity",
        "bear": "min_vol",
    }

    def regime_blended(
        self,
        returns: pd.DataFrame,
        cov: np.ndarray,
        regime_probabilities: dict[str, float],
        rf: float = 0.04,
        constraints: dict | None = None,
    ) -> OptimizationResult:
        if not regime_probabilities:
            raise ValueError("regime_probabilities must be non-empty")

        total = float(sum(regime_probabilities.values()))
        if total <= 0:
            raise ValueError("regime_probabilities must sum to > 0")
        normalized = {r: p / total for r, p in regime_probabilities.items()}

        tickers = list(returns.columns)
        sigma = np.asarray(cov)

        max_sharpe_res = self.mvo(
            returns, sigma, target="max_sharpe",
            constraints=constraints, rf=rf,
        )
        min_vol_res = self.mvo(
            returns, sigma, target="min_vol",
            constraints=constraints, rf=rf,
        )
        rp_res = self.risk_parity(sigma, tickers, returns=returns, rf=rf)

        sub_results = {
            "bull": max_sharpe_res,
            "sideways": rp_res,
            "bear": min_vol_res,
        }

        start = time.perf_counter()
        w = np.zeros(len(tickers))
        for regime, prob in normalized.items():
            sub = sub_results.get(regime)
            if sub is None or prob == 0:
                continue
            sub_w = np.array([sub.weights[t] for t in tickers])
            w += prob * sub_w

        if w.sum() == 0:
            raise ValueError("regime blend produced zero-weight portfolio")
        w = w / w.sum()

        elapsed_ms = int((time.perf_counter() - start) * 1000)
        elapsed_ms += (
            max_sharpe_res.solve_ms + min_vol_res.solve_ms + rp_res.solve_ms
        )

        mu = returns.mean().values * TRADING_DAYS
        result = self._make_result(
            "regime_blended", None, tickers, w, mu, sigma, rf, elapsed_ms,
        )
        result.regime_weights = normalized
        result.components = {
            "max_sharpe": max_sharpe_res.weights,
            "risk_parity": rp_res.weights,
            "min_vol": min_vol_res.weights,
        }
        return result

    def black_litterman(
        self,
        market_caps: dict[str, float],
        cov: np.ndarray,
        tickers: list[str],
        views: list[dict] | None = None,
        risk_aversion: float = 2.5,
        returns: pd.DataFrame | None = None,
        rf: float = 0.04,
    ) -> OptimizationResult:
        sigma = np.asarray(cov)
        n = len(tickers)
        eps = 1e-10 * np.eye(n)

        caps = np.array([float(market_caps.get(t) or 0.0) for t in tickers])
        if caps.sum() <= 0:
            w_mkt = np.full(n, 1.0 / n)
        else:
            w_mkt = caps / caps.sum()

        pi = risk_aversion * sigma @ w_mkt

        if not views:
            mu_bl = pi
        else:
            P = np.zeros((len(views), n))
            Q = np.zeros(len(views))
            omega_diag = np.zeros(len(views))
            for i, v in enumerate(views):
                ticker = v["ticker"]
                if ticker not in tickers:
                    continue
                j = tickers.index(ticker)
                P[i, j] = 1.0
                Q[i] = float(v["view_return"])
                p_row = P[i]
                omega_diag[i] = max(
                    TAU * float(p_row @ sigma @ p_row) / max(v["confidence"], 1e-6),
                    1e-12,
                )
            omega = np.diag(omega_diag)
            tau_sigma_inv = np.linalg.inv(TAU * sigma + eps)
            omega_inv = np.linalg.inv(omega + 1e-12 * np.eye(len(views)))
            posterior_cov = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
            mu_bl = posterior_cov @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)

        synth = pd.DataFrame(
            np.zeros((1, n)), columns=tickers
        )
        result = self.mvo(
            synth, sigma, target="max_sharpe", rf=rf,
            expected_returns=mu_bl,
        )
        return OptimizationResult(
            method="black_litterman",
            target="max_sharpe",
            weights=result.weights,
            expected_return=result.expected_return,
            volatility=result.volatility,
            sharpe=result.sharpe,
            solve_ms=result.solve_ms,
        )

    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        cov: np.ndarray,
        n: int = 150,
        rf: float = 0.04,
        current_weights: dict[str, float] | None = None,
    ) -> list[FrontierPoint]:
        tickers = list(returns.columns)
        mu = returns.mean().values * TRADING_DAYS
        sigma = np.asarray(cov)

        min_ret = float(np.min(mu))
        max_ret = float(np.max(mu))
        if min_ret >= max_ret:
            return []

        targets = np.linspace(min_ret, max_ret, n)
        points: list[FrontierPoint] = []
        for tr in targets:
            try:
                res = self.mvo(
                    returns, sigma, target="target_return",
                    constraints={"target_return": float(tr)}, rf=rf,
                )
                points.append(
                    FrontierPoint(
                        weights=res.weights,
                        expected_return=res.expected_return,
                        volatility=res.volatility,
                        sharpe=res.sharpe,
                    )
                )
            except Exception:
                continue

        max_sharpe = self.mvo(returns, sigma, target="max_sharpe", rf=rf)
        min_vol = self.mvo(returns, sigma, target="min_vol", rf=rf)
        points.append(
            FrontierPoint(
                weights=max_sharpe.weights,
                expected_return=max_sharpe.expected_return,
                volatility=max_sharpe.volatility,
                sharpe=max_sharpe.sharpe,
                kind="max_sharpe",
            )
        )
        points.append(
            FrontierPoint(
                weights=min_vol.weights,
                expected_return=min_vol.expected_return,
                volatility=min_vol.volatility,
                sharpe=min_vol.sharpe,
                kind="min_vol",
            )
        )
        if current_weights:
            cw = np.array([current_weights.get(t, 0.0) for t in tickers])
            cw = cw / cw.sum() if cw.sum() > 0 else cw
            ret = float(cw @ mu)
            vol = float(np.sqrt(cw @ sigma @ cw))
            sharpe = (ret - rf) / vol if vol > 0 else 0.0
            points.append(
                FrontierPoint(
                    weights={t: float(w) for t, w in zip(tickers, cw)},
                    expected_return=ret,
                    volatility=vol,
                    sharpe=sharpe,
                    kind="current",
                )
            )
        return points

    def _make_result(
        self,
        method: str,
        target: str | None,
        tickers: list[str],
        w: np.ndarray,
        mu: np.ndarray,
        sigma: np.ndarray,
        rf: float,
        solve_ms: int,
    ) -> OptimizationResult:
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ sigma @ w))
        sharpe = (ret - rf) / vol if vol > 0 else 0.0
        return OptimizationResult(
            method=method,
            target=target,
            weights={t: float(wi) for t, wi in zip(tickers, w)},
            expected_return=ret,
            volatility=vol,
            sharpe=sharpe,
            solve_ms=solve_ms,
        )
