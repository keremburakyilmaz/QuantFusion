from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()

class OptionRequest(BaseModel):
    S: float  # underlying price
    K: float  # strike price
    T: float  # time to maturity in years
    r: float  # risk-free rate
    sigma: float  # volatility
    option_type: str  # 'call' or 'put'

@router.post("/price")
def price_option(request: OptionRequest):
    return {
        "option_type": request.option_type,
        "price": None,
        "greeks": {
            "delta": None,
            "vega": None,
            "theta": None,
            "gamma": None
        }
    }
