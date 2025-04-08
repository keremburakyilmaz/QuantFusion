from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.portfolio_optimization import optimize_portfolio

router = APIRouter()

class PortfolioRequest(BaseModel):
    symbols: List[str]
    returns: List[List[float]]  # each symbol's historical returns
    risk_free_rate: float = 0.01

@router.post("/optimize")
def optimize_portfolio_router(request: PortfolioRequest):
    result = optimize_portfolio(
        request.symbols,
        request.returns,
        request.risk_free_rate
    )