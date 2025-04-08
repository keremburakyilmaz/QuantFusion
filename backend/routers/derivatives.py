from fastapi import APIRouter
from pydantic import BaseModel
from services.options_pricing import price_option

router = APIRouter()

class OptionRequest(BaseModel):
    S: float  # underlying price
    K: float  # strike price
    T: float  # time to maturity in years
    r: float  # risk-free rate
    sigma: float  # volatility
    option_type: str  # 'call' or 'put'

@router.post("/price")
def price_option_router(request: OptionRequest):
    result = price_option(
        request.S,
        request.K,
        request.T,
        request.r,
        request.sigma,
        request.option_type
    )

    return result
