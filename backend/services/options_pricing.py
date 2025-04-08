def price_option(S: float, K: float, T: float, r: float, sigma: float, option_type: str):
    return {
        "price": 0.0,
        "greeks": {
            "delta": 0.0,
            "vega": 0.0,
            "theta": 0.0,
            "gamma": 0.0
        }
    }
