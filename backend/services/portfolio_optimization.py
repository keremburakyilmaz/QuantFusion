def optimize_portfolio(symbols: list, returns: list[list[float]], risk_free_rate: float = 0.01):
    return {
        "optimal_weights": {symbol: 1/len(symbols) for symbol in symbols},
        "expected_return": 0.0,
        "sharpe_ratio": 0.0
    }
