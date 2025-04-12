import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import base64
import io

def generate_efficient_frontier(mean_returns, cov_matrix, risk_free_rate, optimal_point=None):
    num_assets = len(mean_returns)
    results = {"returns": [], "volatility": [], "sharpe": []}

    for _ in range(5000):
        weights = np.random.dirichlet(np.ones(num_assets))
        ret = np.dot(weights, mean_returns)
        vol = np.sqrt(weights.T @ cov_matrix @ weights)
        sharpe = (ret - risk_free_rate) / vol
        results["returns"].append(ret)
        results["volatility"].append(vol)
        results["sharpe"].append(sharpe)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=results["volatility"], y=results["returns"],
                    hue=results["sharpe"], palette="viridis", ax=ax, legend=False)

    if optimal_point:
        opt_vol, opt_ret = optimal_point
        ax.scatter(opt_vol, opt_ret, color="red", marker="*", s=200, label="Optimal Portfolio")
        ax.plot([0, opt_vol], [risk_free_rate, opt_ret], linestyle="--", color="green", label="Capital Market Line")

    ax.set(title="Efficient Frontier with CML", xlabel="Volatility (Risk)", ylabel="Expected Return")
    ax.legend()
    plt.tight_layout()

    buffer = io.BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode()
    plt.close()

    return encoded
