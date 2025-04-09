from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.portfolio_optimization import optimize_portfolio

router = APIRouter()

class PortfolioRequest(BaseModel):
    symbols: List[str]
    returns: List[List[float]]
    risk_free_rate: float = 0.01
    min_weight: float = 0.0
    max_weight: float = 1.0

@router.post("/optimize")
def optimize_portfolio_router(request: PortfolioRequest):
    result = optimize_portfolio(
        request.symbols,
        request.returns,
        request.risk_free_rate,
        request.min_weight,
        request.max_weight
    )

    return result