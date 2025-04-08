from fastapi import APIRouter
from pydantic import BaseModel
from services.trading_engine import run_strategy

router = APIRouter()

class TradingRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    strategy: str

@router.post("/run")
def run_strategy_router(request: TradingRequest):
    result = run_strategy(
        request.symbol,
        request.start_date,
        request.end_date,
        request.strategy
    )
    return result
