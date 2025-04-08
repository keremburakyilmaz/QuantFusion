def run_strategy(symbol: str, start_date: str, end_date: str, strategy: str):
    return {
        "buy_signals": [],
        "sell_signals": [],
        "performance": {
            "returns": 0.0,
            "sharpe_ratio": 0.0
        }
    }
