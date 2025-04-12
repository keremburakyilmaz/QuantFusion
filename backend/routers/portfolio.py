from fastapi import APIRouter
from pydantic import BaseModel
from typing import Dict, List, Optional

from services.portfolio_optimizers.mean_variance import optimize_mean_variance
from services.portfolio_optimizers.risk_parity import optimize_risk_parity
from services.portfolio_optimizers.black_litterman import optimize_black_litterman

router = APIRouter()

class PortfolioRequest(BaseModel):
    symbols: List[str]
    returns: List[List[float]]
    risk_free_rate: float = 0.01
    min_weight: float = 0.0
    max_weight: float = 1.0
    sectors: Optional[List[str]] = None
    sector_limits: Optional[Dict[str, float]] = None
    benchmark_weights: Optional[List[float]] = None
    tracking_error_limit: Optional[float] = None
    market_weights: Optional[List[float]] = None
    view_matrix: Optional[List[List[float]]] = None
    view_vector: Optional[List[float]] = None
    view_uncertainty: Optional[List[float]] = None
    tau: Optional[float] = 0.05

@router.post("/optimize/mean-variance")
def mean_variance_optimizer(request: PortfolioRequest):
    return optimize_mean_variance(request)

@router.post("/optimize/risk-parity")
def risk_parity_optimizer(request: PortfolioRequest):
    return optimize_risk_parity(request)

@router.post("/optimize/black-litterman")
def black_litterman_optimizer(request: PortfolioRequest):
    return optimize_black_litterman(request)

@router.get("/optimize/summary")
def get_optimizer_summary():
    return {
        "methods": {
            "mean_variance": {
                "title": "Mean-Variance Optimization (Markowitz)",
                "description": (
                    "Minimizes portfolio variance (risk) for a target return. "
                    "This is the classic Modern Portfolio Theory (MPT) method."
                ),
                "objective_function": "Minimize wᵀΣw",
                "constraints": [
                    "∑wᵢ = 1 (full investment)",
                    "wᵢ ≥ min_weight and ≤ max_weight",
                    "Expected return ≥ target_return",
                    "Optional: sector exposure limits",
                    "Optional: tracking error to benchmark"
                ],
                "use_case": (
                    "Use when you have a reliable estimate of expected returns and want "
                    "the most efficient portfolio for a given level of expected performance."
                )
            },
            "risk_parity": {
                "title": "Risk Parity Portfolio",
                "description": (
                    "Allocates weights so each asset contributes equally to total portfolio volatility."
                ),
                "objective_function": (
                    "Minimize sum((RCᵢ - RC_avg)²), where RCᵢ = wᵢ * (Σw)ᵢ / σ_p"
                ),
                "constraints": [
                    "∑wᵢ = 1",
                    "wᵢ ∈ [0, 1]"
                ],
                "use_case": (
                    "Use when you want a diversified portfolio based purely on risk, "
                    "without relying on return forecasts. Common in volatility targeting."
                )
            },
            "black_litterman": {
                "title": "Black-Litterman Model",
                "description": (
                    "Combines market equilibrium returns (implied by market cap weights) "
                    "with subjective investor views to produce a more stable estimate of expected returns."
                ),
                "objective_function": "Same as mean-variance, but using μ₍BL₎ instead of raw mean returns",
                "formulas": {
                    "implied_returns": "π = τΣw_market",
                    "BL_returns": "μ_BL = [ (τΣ)⁻¹ + PᵀΩ⁻¹P ]⁻¹ × [ (τΣ)⁻¹π + PᵀΩ⁻¹Q ]"
                },
                "constraints": [
                    "∑wᵢ = 1",
                    "wᵢ ≥ min_weight and ≤ max_weight",
                    "Expected return ≥ target_return",
                    "Optional: sector limits and tracking error"
                ],
                "use_case": (
                    "Use when you want to incorporate your own views (e.g., 'AAPL will outperform MSFT') "
                    "while grounding the model in realistic market expectations."
                )
            }
        },
        "notes": [
            "All methods support sector constraints (e.g., Tech ≤ 60%)",
            "All methods support tracking error constraints (distance from benchmark)",
            "Plots returned show efficient frontier, optimal point, and Capital Market Line"
        ]
    }

