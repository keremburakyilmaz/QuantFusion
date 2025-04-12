import numpy as np
import cvxpy as cp
from services.utils.plotting import generate_efficient_frontier

def calculate_bl_returns(cov_matrix, market_weights, P, Q, Omega, tau):
    pi = tau * cov_matrix @ market_weights
    inv_tau_cov = np.linalg.pinv(tau * cov_matrix)
    Omega = np.diag(Omega)
    M = np.linalg.pinv(inv_tau_cov + P.T @ np.linalg.inv(Omega) @ P)
    rhs = inv_tau_cov @ pi + P.T @ np.linalg.inv(Omega) @ Q
    return M @ rhs

def optimize_black_litterman(request):
    symbols = request.symbols
    returns = np.array(request.returns)
    cov_matrix = np.cov(returns)
    mean_returns = calculate_bl_returns(
        cov_matrix,
        np.array(request.market_weights),
        np.array(request.view_matrix),
        np.array(request.view_vector),
        request.view_uncertainty,
        request.tau
    )
    n = len(symbols)
    w = cp.Variable(n)
    target_return = float(np.mean(mean_returns))

    constraints = [
        cp.sum(w) == 1,
        w >= request.min_weight,
        w <= request.max_weight,
        w @ mean_returns >= target_return
    ]

    if request.sectors and request.sector_limits:
        sector_map = {}
        for i, sector in enumerate(request.sectors):
            sector_map.setdefault(sector, []).append(i)
        for sector, limit in request.sector_limits.items():
            indices = sector_map.get(sector, [])
            if indices:
                constraints.append(cp.sum(w[indices]) <= limit)

    if request.benchmark_weights and request.tracking_error_limit is not None:
        w_bench = np.array(request.benchmark_weights)
        diff = w - w_bench
        constraints.append(cp.quad_form(diff, cov_matrix) <= request.tracking_error_limit ** 2)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cov_matrix)), constraints)
    prob.solve()

    if w.value is None:
        return {"error": "Black-Litterman optimization failed"}

    weights = w.value
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
