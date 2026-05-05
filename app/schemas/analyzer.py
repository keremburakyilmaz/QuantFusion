

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from app.schemas.backtest import BacktestResult
from app.schemas.common import HoldingInput
from app.schemas.market import ValidateRequest, ValidateResponse
from app.schemas.optimization import FrontierPoint, OptimizationResult
from app.schemas.regime import RegimeSnapshotResponse
from app.schemas.risk import RiskMetrics


class AnalysisReport(BaseModel):
    holdings: list[HoldingInput]
    risk: RiskMetrics
    frontier: list[FrontierPoint]
    optimized_mvo: OptimizationResult
    optimized_rp: OptimizationResult
    optimized_blended: OptimizationResult | None = None
    backtest_1y: BacktestResult
    backtest_3y: BacktestResult
    regime: RegimeSnapshotResponse | None = None
    regime_commentary: str = ""
    fundamentals: dict[str, dict[str, Any]]
    generated_at: datetime


class AnalyzerRunRequest(BaseModel):
    holdings: list[HoldingInput] = Field(min_length=1, max_length=20)


AnalyzerValidateRequest = ValidateRequest
AnalyzerValidateResponse = ValidateResponse
