

import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.schemas.common import HoldingInput


Method = Literal["mvo", "risk_parity", "black_litterman", "regime_blended"]
Target = Literal["max_sharpe", "min_vol", "target_return"]


class ViewInput(BaseModel):
    ticker: str = Field(min_length=1, max_length=16)
    view_return: float
    confidence: float = Field(gt=0.0, le=1.0)


class Constraints(BaseModel):
    min_weight: float = 0.01
    max_weight: float = 0.60
    target_return: float | None = None
    sector_limits: dict[str, float] | None = None


class OptimizationResult(BaseModel):
    method: str
    target: str | None = None
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    solve_ms: int
    regime_weights: dict[str, float] | None = None
    components: dict[str, dict[str, float]] | None = None


class OptimizeRunRequest(BaseModel):
    portfolio_id: uuid.UUID
    method: Method
    target: Target | None = "max_sharpe"
    constraints: Constraints | None = None
    views: list[ViewInput] | None = None
    risk_aversion: float = 2.5
    regime_probabilities: dict[str, float] | None = None


class OptimizeStatelessRequest(BaseModel):
    method: Method
    holdings: list[HoldingInput] = Field(min_length=1)
    target: Target | None = "max_sharpe"
    constraints: Constraints | None = None
    views: list[ViewInput] | None = None
    risk_aversion: float = 2.5
    regime_probabilities: dict[str, float] | None = None


class FrontierPoint(BaseModel):
    weights: dict[str, float]
    expected_return: float
    volatility: float
    sharpe: float
    kind: str | None = None


class EfficientFrontierResponse(BaseModel):
    points: list[FrontierPoint]
