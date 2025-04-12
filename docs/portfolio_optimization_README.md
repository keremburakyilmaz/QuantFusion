# 📈 Portfolio Optimization API

Welcome to the **QuantFusion Optimizers API** — a modular, extensible backend for institutional-grade portfolio optimization powered by **FastAPI**, **cvxpy**, and **numpy**.

This backend supports multiple quantitative portfolio construction methods with real-world constraints, including:
- Mean-Variance Optimization (Markowitz)
- Risk Parity Portfolios
- Black-Litterman Model

---

## API Endpoints

| Endpoint                          | Method | Description                                    |
|----------------------------------|--------|------------------------------------------------|
| `/optimize/mean-variance`        | POST   | Markowitz-style variance minimization          |
| `/optimize/risk-parity`          | POST   | Risk-balanced portfolio allocation             |
| `/optimize/black-litterman`      | POST   | Bayesian blending of views + market data       |
| `/optimize/summary`              | GET    | Returns explanation of all methods             |

---

## Supported Methods & Their Meaning

### 1. Mean-Variance Optimization
- **Goal:** Minimize portfolio variance for a given expected return
- **Mathematics:**
  - Objective: $\min \mathbf{w}^\top \Sigma \mathbf{w}$
  - Constraints: $\sum w_i = 1,\quad w_i \geq 0,\quad E[R] \geq \text{target}$
- **Use Case:** When you have confident return estimates

### 2. Risk Parity
- **Goal:** Equalize each asset's contribution to total portfolio volatility
- **Mathematics:**
  - Risk contribution: $RC_i = \frac{w_i (\Sigma w)_i}{\sigma_p}$
  - Objective: $\min \sum (RC_i - RC_{avg})^2$
- **Use Case:** When you want diversification without return forecasts

### 3. Black-Litterman
- **Goal:** Blend market equilibrium with subjective views
- **Mathematics:**
  - Implied returns: $\pi = \tau \Sigma w_{market}$
  - Posterior returns:
    $$
    \mu_{BL} = \left[ (\tau\Sigma)^{-1} + P^\top \Omega^{-1} P \right]^{-1}
    \left[ (\tau\Sigma)^{-1} \pi + P^\top \Omega^{-1} Q \right]
    $$
- **Use Case:** When you want to encode beliefs like “AAPL will outperform MSFT by 2%”

---

## Advanced Constraints Supported

| Constraint Type       | Description                                      |
|-----------------------|--------------------------------------------------|
| Sector limits         | Cap sector exposure (e.g., Tech ≤ 60%)          |
| Tracking error        | Limit distance from benchmark                   |
| No shorting           | Enforce long-only weights                       |
| Weight bounds         | Set min/max per-asset allocations               |

---

## Sample Payload
```json
{
  "symbols": ["AAPL", "MSFT", "GOOG"],
  "returns": [[0.01, 0.02], [0.015, -0.005], [0.012, 0.003]],
  "risk_free_rate": 0.01,
  "min_weight": 0.0,
  "max_weight": 1.0,
  "sectors": ["Tech", "Tech", "Tech"],
  "sector_limits": {"Tech": 0.8},
  "mode": "mean_variance"
}
```

---

## Author
Kerem Burak Yilmaz
