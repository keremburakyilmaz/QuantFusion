from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Dict, Optional, Literal

# Import functions from your modular services
from services.risk_analysis.value_at_risk import calculate_var, calculate_cvar
from services.risk_analysis.risk_metrics import calculate_volatility, calculate_drawdown, calculate_beta
from services.risk_analysis.exposure import calculate_sector_exposure
from services.risk_analysis.attribution import calculate_risk_contribution

router = APIRouter()

class RiskAnalysisRequest(BaseModel):
    asset_prices: List[float]
    asset_returns: List[float]
    market_returns: Optional[List[float]] = None
    weights: Optional[List[float]] = None
    sectors: Optional[List[str]] = None
    confidence_level: float = 0.95

@router.post("/risk/analyze")
def full_risk_analysis(request: RiskAnalysisRequest):
    analysis = {}

    # Volatility
    analysis["volatility"] = calculate_volatility(request.asset_returns)

    # Beta
    if request.market_returns:
        analysis["beta"] = calculate_beta(request.asset_returns, request.market_returns)

    # Max drawdown
    analysis["max_drawdown"] = calculate_drawdown(request.asset_prices)

    # VaR and CVaR
    analysis["value_at_risk"] = calculate_var(
        request.asset_prices, request.confidence_level, method="historical"
    )
    analysis["conditional_var"] = calculate_cvar(
        request.asset_prices, request.confidence_level, method="historical"
    )

    # Exposure
    if request.weights and request.sectors:
        analysis["sector_exposure"] = calculate_sector_exposure(request.weights, request.sectors)

    return analysis

# ----------------------------------------

class VaRRequest(BaseModel):
    asset_prices: List[float]
    confidence_level: float
    method: Literal["historical", "parametric"] = "historical"

@router.post("/risk/var")
def value_at_risk(request: VaRRequest):
    return calculate_var(
        prices=request.asset_prices,
        confidence_level=request.confidence_level,
        method=request.method
    )

# ----------------------------------------

class CVaRRequest(BaseModel):
    asset_prices: List[float]
    confidence_level: float
    method: Literal["historical", "parametric"] = "historical"

@router.post("/risk/cvar")
def conditional_var(request: CVaRRequest):
    return calculate_cvar(
        prices=request.asset_prices,
        confidence_level=request.confidence_level,
        method=request.method
    )

# ----------------------------------------

class VolatilityRequest(BaseModel):
    returns: List[float]
    period: Optional[int] = 252  # default: annualized from daily

@router.post("/risk/volatility")
def volatility(request: VolatilityRequest):
    return {"volatility": calculate_volatility(request.returns, request.period)}

# ----------------------------------------

class DrawdownRequest(BaseModel):
    asset_prices: List[float]

@router.post("/risk/drawdown")
def drawdown(request: DrawdownRequest):
    return calculate_drawdown(request.asset_prices)

# ----------------------------------------

class BetaRequest(BaseModel):
    asset_returns: List[float]
    market_returns: List[float]

@router.post("/risk/beta")
def beta(request: BetaRequest):
    return {"beta": calculate_beta(request.asset_returns, request.market_returns)}

# ----------------------------------------

class ExposureRequest(BaseModel):
    weights: List[float]
    sectors: List[str]

@router.post("/risk/exposure")
def sector_exposure(request: ExposureRequest):
    return calculate_sector_exposure(request.weights, request.sectors)

# ----------------------------------------

class AttributionRequest(BaseModel):
    weights: List[float]
    cov_matrix: List[List[float]]

@router.post("/risk/attribution")
def risk_attribution(request: AttributionRequest):
    return calculate_risk_contribution(request.weights, request.cov_matrix)
