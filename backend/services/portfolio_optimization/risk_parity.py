import numpy as np
from scipy.optimize import minimize
from services.utils.plotting import generate_efficient_frontier

def optimize_risk_parity(request):
    cov_matrix = np.cov(request.returns)
    mean_returns = np.mean(request.returns, axis=1)
    symbols = request.symbols
    n = len(symbols)

    def port_vol(w): return np.sqrt(w.T @ cov_matrix @ w)
    def rc(w): return w * (cov_matrix @ w) / port_vol(w)
    def obj(w): return np.sum((rc(w) - np.mean(rc(w)))**2)

    x0 = np.ones(n) / n
    bounds = [(0, 1)] * n
    cons = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)

    if not result.success:
        return {"error": "Risk parity optimization failed"}

    weights = result.x
    expected_return = float(mean_returns @ weights)
    volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe = (expected_return - request.risk_free_rate) / volatility
    optimal_point = (volatility, expected_return)

    plot = generate_efficient_frontier(mean_returns, cov_matrix, request.risk_free_rate, optimal_point)

    return {
        "optimal_weights": dict(zip(symbols, np.round(weights, 4))),
        "expected_return": round(expected_return, 4),
        "sharpe_ratio": round(sharpe, 4),
        "efficient_frontier_plot": plot
    }
