from fastapi import APIRouter
from pydantic import BaseModel
from typing import List

router = APIRouter()

class PortfolioRequest(BaseModel):
    symbols: List[str]
    returns: List[List[float]]  # each symbol's historical returns
    risk_free_rate: float = 0.01

@router.post("/optimize")
def optimize_portfolio(request: PortfolioRequest):
    return {
        "optimal_weights": {},
        "expected_return": None,
        "sharpe_ratio": None
    }
