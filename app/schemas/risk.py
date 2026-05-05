

from pydantic import BaseModel, Field


class RiskMetrics(BaseModel):
    annualized_return: float | None = None
    annualized_volatility: float | None = None

    sharpe: float | None = None
    sortino: float | None = None
    calmar: float | None = None

    max_drawdown: float | None = None

    var_historical: float | None = None
    var_parametric: float | None = None
    var_monte_carlo: float | None = None
    cvar: float | None = None

    beta: float | None = None
    tracking_error: float | None = None

    correlation_matrix: dict[str, dict[str, float]] | None = None


class VaRResult(BaseModel):
    confidence: float = 0.95
    historical: float | None = None
    parametric: float | None = None
    monte_carlo: float | None = None
    cvar: float | None = None
