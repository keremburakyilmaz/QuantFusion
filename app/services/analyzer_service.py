

import asyncio
import logging
from datetime import datetime, timezone

import pandas as pd

from app.schemas.analyzer import AnalysisReport
from app.schemas.common import HoldingInput
from app.services.agent_service import AgentService
from app.services.backtester import VectorizedBacktester, _run_sync
from app.services.data_service import DataService
from app.services.optimizer import PortfolioOptimizer
from app.services.regime_service import RegimeService
from app.services.risk_service import RiskService


logger = logging.getLogger(__name__)


class AnalyzerService:
    def __init__(
        self,
        data: DataService,
        risk: RiskService,
        optimizer: PortfolioOptimizer,
        backtester: VectorizedBacktester,
        regime: RegimeService | None,
        agent: AgentService | None = None,
    ) -> None:
        self.data = data
        self.risk = risk
        self.optimizer = optimizer
        self.backtester = backtester
        self.regime = regime
        self.agent = agent

    async def validate(self, tickers: list[str]) -> dict[str, list[str]]:
        return await self.data.validate_tickers(tickers)

    async def run(self, holdings: list[HoldingInput]) -> AnalysisReport:
        tickers = [h.ticker for h in holdings]
        weights = {h.ticker: h.weight for h in holdings}

        returns_full, fundamentals_list, rf = await asyncio.gather(
            self.data.get_returns([*tickers, "SPY"], lookback_days=756),
            asyncio.gather(*[self.data.get_fundamentals(t) for t in tickers]),
            self.data.get_risk_free_rate(),
        )
        fundamentals = dict(zip(tickers, fundamentals_list))

        if returns_full.empty:
            raise ValueError("No return data available for the requested tickers")

        user_cols = [t for t in tickers if t in returns_full.columns]
        returns = returns_full[user_cols]
        benchmark = (
            returns_full["SPY"] if "SPY" in returns_full.columns else None
        )
        cov = returns.cov().values * 252

        regime_task = (
            self.regime.predict_current() if self.regime is not None else _none_async()
        )

        regime, risk_metrics, frontier, mvo_res, rp_res, bt_1y, bt_3y = (
            await asyncio.gather(
                regime_task,
                asyncio.to_thread(
                    self.risk.compute_all, weights, returns_full, rf
                ),
                asyncio.to_thread(
                    self.optimizer.efficient_frontier,
                    returns, cov, 150, rf, weights,
                ),
                asyncio.to_thread(
                    self.optimizer.mvo, returns, cov, "max_sharpe", None, rf,
                ),
                asyncio.to_thread(
                    self.optimizer.risk_parity, cov, user_cols, returns, rf,
                ),
                asyncio.to_thread(
                    _run_sync, weights, returns.tail(252), 10_000.0,
                    "monthly", 10.0,
                    benchmark.tail(252) if benchmark is not None else None,
                ),
                asyncio.to_thread(
                    _run_sync, weights, returns, 10_000.0,
                    "monthly", 10.0, benchmark,
                ),
            )
        )

        blended = None
        if regime is not None and regime.probabilities:
            blended = await asyncio.to_thread(
                self.optimizer.regime_blended,
                returns, cov, regime.probabilities, rf, None,
            )

        commentary = ""
        if regime is not None and self.agent is not None and self.agent.enabled:
            commentary = await self.agent.regime_commentary(
                regime, holdings, risk_metrics
            )

        return AnalysisReport(
            holdings=holdings,
            risk=risk_metrics,
            frontier=frontier,
            optimized_mvo=mvo_res,
            optimized_rp=rp_res,
            optimized_blended=blended,
            backtest_1y=bt_1y,
            backtest_3y=bt_3y,
            regime=regime,
            regime_commentary=commentary,
            fundamentals=fundamentals,
            generated_at=datetime.now(tz=timezone.utc),
        )


async def _none_async():
    return None
