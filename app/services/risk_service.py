

from typing import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from app.schemas.risk import RiskMetrics, VaRResult


TRADING_DAYS = 252
Z_95 = 1.6448536269514722
DEFAULT_CONFIDENCE = 0.95
MC_PATHS = 10_000
BENCHMARK = "SPY"


class RiskService:
    def portfolio_returns(
        self, weights: dict[str, float], returns: pd.DataFrame
    ) -> pd.Series:
        cols = [t for t in weights if t in returns.columns]
        if not cols:
            return pd.Series(dtype=float)
        w = np.array([weights[t] for t in cols], dtype=float)
        sub = returns[cols].dropna()
        return pd.Series(sub.values @ w, index=sub.index, name="portfolio")

    def annualized_return(self, port_returns: pd.Series) -> float:
        return float(port_returns.mean() * TRADING_DAYS)

    def annualized_volatility(self, port_returns: pd.Series) -> float:
        return float(port_returns.std(ddof=1) * np.sqrt(TRADING_DAYS))

    def sharpe(self, port_returns: pd.Series, rf: float) -> float:
        vol = self.annualized_volatility(port_returns)
        if vol == 0:
            return 0.0
        return (self.annualized_return(port_returns) - rf) / vol

    def sortino(self, port_returns: pd.Series, rf: float) -> float:
        downside = port_returns[port_returns < 0]
        if downside.empty:
            return 0.0
        downside_vol = float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))
        if downside_vol == 0:
            return 0.0
        return (self.annualized_return(port_returns) - rf) / downside_vol

    def calmar(self, port_returns: pd.Series) -> float:
        mdd = abs(self.max_drawdown(port_returns))
        if mdd == 0:
            return 0.0
        return self._cagr(port_returns) / mdd

    def _cagr(self, port_returns: pd.Series) -> float:
        if port_returns.empty:
            return 0.0
        cumulative = (1.0 + port_returns).prod()
        years = len(port_returns) / TRADING_DAYS
        if years <= 0 or cumulative <= 0:
            return 0.0
        return float(cumulative ** (1.0 / years) - 1.0)

    def max_drawdown(self, port_returns: pd.Series) -> float:
        if port_returns.empty:
            return 0.0
        equity = (1.0 + port_returns).cumprod()
        peak = equity.cummax()
        dd = (equity - peak) / peak
        return float(dd.min())

    def var_historical(
        self, port_returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
    ) -> float:
        if port_returns.empty:
            return 0.0
        return float(np.quantile(port_returns.values, 1.0 - confidence))

    def var_parametric(
        self, port_returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
    ) -> float:
        if port_returns.empty:
            return 0.0
        mu = float(port_returns.mean())
        sigma = float(port_returns.std(ddof=1))
        z = float(stats.norm.ppf(1.0 - confidence))
        return mu + z * sigma

    def var_monte_carlo(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        confidence: float = DEFAULT_CONFIDENCE,
        n_paths: int = MC_PATHS,
        seed: int = 42,
    ) -> float:
        cols = [t for t in weights if t in returns.columns]
        if not cols:
            return 0.0
        sub = returns[cols].dropna()
        if sub.empty:
            return 0.0
        w = np.array([weights[t] for t in cols], dtype=float)
        mu = sub.mean().values
        cov = sub.cov().values

        rng = np.random.default_rng(seed)
        try:
            chol = np.linalg.cholesky(cov + 1e-10 * np.eye(len(cols)))
        except np.linalg.LinAlgError:
            return float(np.quantile(sub.values @ w, 1.0 - confidence))

        z = rng.standard_normal((n_paths, len(cols)))
        sim = mu + z @ chol.T
        port_paths = sim @ w
        return float(np.quantile(port_paths, 1.0 - confidence))

    def cvar(
        self, port_returns: pd.Series, confidence: float = DEFAULT_CONFIDENCE
    ) -> float:
        if port_returns.empty:
            return 0.0
        threshold = self.var_historical(port_returns, confidence)
        tail = port_returns[port_returns <= threshold]
        if tail.empty:
            return float(threshold)
        return float(tail.mean())

    def beta(self, port_returns: pd.Series, bench_returns: pd.Series) -> float | None:
        aligned = pd.concat(
            [port_returns.rename("p"), bench_returns.rename("b")], axis=1
        ).dropna()
        if len(aligned) < 2:
            return None
        cov = aligned.cov().loc["p", "b"]
        var_b = aligned["b"].var(ddof=1)
        if var_b == 0:
            return None
        return float(cov / var_b)

    def tracking_error(
        self, port_returns: pd.Series, bench_returns: pd.Series
    ) -> float | None:
        aligned = pd.concat(
            [port_returns.rename("p"), bench_returns.rename("b")], axis=1
        ).dropna()
        if len(aligned) < 2:
            return None
        excess = aligned["p"] - aligned["b"]
        return float(excess.std(ddof=1) * np.sqrt(TRADING_DAYS))

    def correlation_matrix(
        self, weights: dict[str, float], returns: pd.DataFrame
    ) -> dict[str, dict[str, float]] | None:
        cols = [t for t in weights if t in returns.columns]
        if len(cols) < 2:
            return None
        corr = returns[cols].dropna().corr()
        return {
            row: {col: float(corr.loc[row, col]) for col in cols} for row in cols
        }

    def compute_all(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        rf: float = 0.04,
    ) -> RiskMetrics:
        port = self.portfolio_returns(weights, returns)
        if port.empty:
            return RiskMetrics()

        bench = (
            returns[BENCHMARK].dropna()
            if BENCHMARK in returns.columns
            else pd.Series(dtype=float)
        )

        return RiskMetrics(
            annualized_return=self.annualized_return(port),
            annualized_volatility=self.annualized_volatility(port),
            sharpe=self.sharpe(port, rf),
            sortino=self.sortino(port, rf),
            calmar=self.calmar(port),
            max_drawdown=self.max_drawdown(port),
            var_historical=self.var_historical(port),
            var_parametric=self.var_parametric(port),
            var_monte_carlo=self.var_monte_carlo(weights, returns),
            cvar=self.cvar(port),
            beta=self.beta(port, bench) if not bench.empty else None,
            tracking_error=(
                self.tracking_error(port, bench) if not bench.empty else None
            ),
            correlation_matrix=self.correlation_matrix(weights, returns),
        )

    def compute_var(
        self,
        weights: dict[str, float],
        returns: pd.DataFrame,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> VaRResult:
        port = self.portfolio_returns(weights, returns)
        if port.empty:
            return VaRResult(confidence=confidence)
        return VaRResult(
            confidence=confidence,
            historical=self.var_historical(port, confidence),
            parametric=self.var_parametric(port, confidence),
            monte_carlo=self.var_monte_carlo(weights, returns, confidence),
            cvar=self.cvar(port, confidence),
        )
