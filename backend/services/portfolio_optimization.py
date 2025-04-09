import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64

import numpy as np
import cvxpy as cp

def optimize_portfolio(symbols, returns, risk_free_rate=0.002, min_weight=0.0, max_weight=1.0):
    returns_array = np.array(returns)
    mean_returns = np.mean(returns_array, axis=1)
    cov_matrix = np.cov(returns_array)

    n = len(symbols)
    w = cp.Variable(n)

    target_return = float(np.mean(mean_returns))  # or max(mean_returns), or custom

    # Objective: minimize variance
    objective = cp.Minimize(cp.quad_form(w, cov_matrix))

    # Constraints: sum to 1, no shorting, target return
    constraints = [
        cp.sum(w) == 1,
        w >= min_weight,
        w <= max_weight,
        w @ mean_returns >= target_return
    ]

    prob = cp.Problem(objective, constraints)
    prob.solve()

    if w.value is None:
        return {"error": "Optimization failed"}

    weights = w.value
    expected_return = float(mean_returns @ weights)
    volatility = float(np.sqrt(weights.T @ cov_matrix @ weights))
    sharpe_ratio = (expected_return - risk_free_rate) / volatility
    optimal_point = (volatility, expected_return)

    frontier_plot = generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_point)

    return {
        "optimal_weights": {symbol: round(float(wt), 4) for symbol, wt in zip(symbols, weights)},
        "expected_return": round(expected_return, 4),
        "sharpe_ratio": round(sharpe_ratio, 4),
        "efficient_frontier_plot": frontier_plot
    }


def generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_point=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    import io
    import base64
    import numpy as np

    num_assets = len(mean_returns)
    num_portfolios = 5000

    results = {
        "returns": [],
        "volatility": [],
        "sharpe": []
    }

    for _ in range(num_portfolios):
        weights = np.random.dirichlet(np.ones(num_assets))
        port_return = np.dot(weights, mean_returns)
        port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        results["returns"].append(port_return)
        results["volatility"].append(port_volatility)
        results["sharpe"].append(sharpe_ratio)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        x=results["volatility"],
        y=results["returns"],
        hue=results["sharpe"],
        palette="viridis",
        ax=ax,
        legend=False
    )

    if optimal_point:
        # Plot optimal point
        opt_vol, opt_ret = optimal_point
        ax.scatter(opt_vol, opt_ret, color="red", marker="*", s=200, label="Optimal Portfolio")

        # Plot CML from (0, rf) to (opt_vol, opt_ret)
        ax.plot(
            [0, opt_vol],
            [risk_free_rate, opt_ret],
            linestyle="--",
            color="green",
            label="Capital Market Line"
        )

    ax.set(title="Efficient Frontier with CML", xlabel="Volatility (Risk)", ylabel="Expected Return")
    ax.legend()
    plt.tight_layout()

    # Encode to base64
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded_image = base64.b64encode(buffer.read()).decode()
    plt.close()

    return encoded_image
