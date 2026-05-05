# QuantFusion

> A quantitative portfolio intelligence system - built to answer the questions every serious investor should be asking, but rarely has the tools to explore.

---

## What Is This?

QuantFusion is a portfolio analytics engine that applies the same quantitative methods used by professional asset managers and hedge funds - risk decomposition, mathematical optimization, machine learning regime detection, earnings signal extraction, and AI-powered narrative generation - and makes them available through a structured API.

If you have a portfolio of stocks or ETFs, QuantFusion tells you not just what your holdings are worth, but **why** your portfolio behaves the way it does, whether it is constructed efficiently, what the market environment means for your specific exposures, and what the most recent earnings from your companies say about the near-term outlook.

The goal is to show you to think like a quantitative analyst.

---

## What Problem Does This Solve?

Most individual investors - and many professionals - evaluate portfolios intuitively. They look at returns, read news, and make judgments based on narrative. This approach has a fundamental weakness: **intuition cannot separate skill from luck, and it cannot tell you whether the risk you are carrying is being rewarded.**

Quantitative finance exists to answer precisely those questions with numbers. QuantFusion brings that discipline together in one place:

- Is my portfolio taking more risk than the return justifies?
- Am I diversified, or am I just holding multiple assets that all move together?
- Would a different set of weights produce a better risk-adjusted outcome?
- Is the current market environment - bull, bear, or transition - appropriate for how my portfolio is positioned?
- What did the most recent earnings season tell us about the companies I own?
- How has my exact portfolio actually behaved during the past one and three years?

Each of those questions maps to a specific analytical module in this system.

---

## What You Can Learn By Using This System

### 1. The True Cost of Risk

Most people think about risk as "how much can I lose?" QuantFusion teaches a more precise version: **how much risk am I taking per unit of expected return?** This is the foundation of modern portfolio theory. Once you start measuring risk with numbers - volatility, Value at Risk, drawdown, beta - the intuitive sense of "this feels risky" becomes something you can quantify, compare, and optimize.

### 2. Why Diversification Is Mathematical, Not Conceptual

Holding ten stocks does not mean you are diversified. If those stocks all move together - high correlation - you have concentration in a single risk factor dressed up as a portfolio. QuantFusion's covariance analysis and optimization engine expose this. You will learn what a covariance matrix actually represents, why correlations matter more than the number of positions, and how true diversification emerges from the structure of returns, not the count of tickers.

### 3. The Difference Between Return Maximization and Risk-Adjusted Optimization

There is always a portfolio that returns more than yours. There is also always one that takes less risk. The interesting question is which portfolios sit on the **efficient frontier** - the set of portfolios where no reallocation can improve return without increasing risk, or reduce risk without sacrificing return. QuantFusion maps that frontier visually and tells you where your current portfolio sits relative to it.

### 4. Why Markets Have Regimes and Why That Changes Everything

A strategy that works beautifully in a bull market can be catastrophic in a bear market. Professional portfolio managers do not run the same allocation in all environments - they tilt exposures based on the detected market regime. QuantFusion's Hidden Markov Model learns to detect these regime transitions from returns data, giving you a probabilistic read on whether the current environment is more likely bull, bear, or sideways. That probability then directly flows into your optimal allocation.

### 5. What Earnings Actually Tell You - Quantitatively

An earnings beat is not just good news. The magnitude of the beat, the revision to forward guidance, and the overall sentiment of the press release together form a **signal**. QuantFusion automatically fetches the most recent earnings filing from the SEC, runs it through an AI model to extract structured signals (EPS actual vs. estimate, revenue, sentiment), and then uses those signals to tilt your portfolio weights. This is a simplified version of how systematic funds incorporate earnings momentum.

### 6. The Gap Between Theoretical Optimization and Real Performance

Backtesting bridges theory and reality. QuantFusion runs your exact portfolio weights through historical return data, charges you realistic transaction costs, rebalances at a specified frequency, and measures what actually happened - equity curve, monthly returns, Sharpe, drawdowns. When you compare the backtest output against the optimizer's theoretical projections, you develop an intuition for how models interact with real markets: the slippage, the timing effects, the moments where theory and reality diverge.

### 7. How AI Can Synthesize Quantitative Data Into Investment Narrative

Numbers without context are hard to act on. QuantFusion uses a large language model to synthesize all of the quantitative output - regime, risk metrics, earnings signals - into a short, specific commentary. Learning to read that commentary and trace it back to the underlying numbers trains you to connect narrative reasoning with quantitative evidence. This is exactly the skill that separates good analysts from those who can either model or communicate, but not both.

---

## The Methods

### Risk Analytics

**Volatility (Annualized Standard Deviation)**
Volatility measures how much your portfolio's daily returns fluctuate around their mean, scaled to an annual figure. A portfolio with 20% volatility will swing roughly ±20% around its expected return in a typical year. Volatility is not inherently bad - it is the raw material of return - but it must be compensated. If your Sharpe ratio is low, your volatility is higher than your return justifies.

**Sharpe Ratio**
The Sharpe ratio divides a portfolio's excess return (return above the risk-free rate) by its volatility. A Sharpe of 1.0 means you earn one unit of return for every unit of risk. A Sharpe above 1.5 is considered strong. Most broad equity indices run at 0.5–0.8 over long periods. When you optimize your portfolio, maximizing Sharpe means finding the most efficient point on the risk-return tradeoff - not the highest return, and not the lowest risk, but the best ratio of the two.

**Value at Risk (VaR) at 95% Confidence - Historical Method**
VaR answers the question: "In a bad day, how much can I expect to lose?" At 95% confidence, a VaR of 2% means that on 95% of days your loss is no worse than 2% of portfolio value. The historical method uses your actual return distribution rather than assuming returns are normally distributed, which matters because equity returns have fat tails - extreme events happen more often than a normal distribution would predict.

**Maximum Drawdown**
Maximum drawdown is the largest peak-to-trough decline in portfolio value over the observed period. A drawdown of -40% means at some point the portfolio fell 40% from its high before recovering. This metric matters more to many investors than volatility because it captures the lived experience of holding through a loss, which is psychologically and practically very different from the abstract concept of standard deviation.

**Beta**
Beta measures your portfolio's sensitivity to the broad market (proxied by SPY). A beta of 1.2 means your portfolio tends to move 1.2% for every 1% the market moves. Beta above 1 amplifies both gains and losses relative to the market. A low-beta portfolio offers more defensive characteristics; a high-beta portfolio is an amplified market bet. Beta is not the same as volatility - a concentrated sector fund can have high volatility but low beta if that sector's returns are uncorrelated with the index.

---

### Portfolio Optimization

Optimization is the mathematical process of finding the portfolio weights that best achieve a given objective while satisfying your constraints (minimum and maximum position sizes, target return, sector limits). QuantFusion implements five distinct optimization methods, each embodying a different philosophical approach to the problem of capital allocation.

**Mean-Variance Optimization (MVO) - Markowitz 1952**
MVO is the original framework of modern portfolio theory, for which Harry Markowitz won the Nobel Prize in Economics. The idea is deceptively simple: for a given level of expected return, find the portfolio with the lowest possible volatility. Or equivalently, for a given level of volatility, find the portfolio with the highest expected return. The set of all such optimal portfolios traces out the efficient frontier.

MVO requires two inputs: expected returns and a covariance matrix. In practice, both are estimated from historical data, which introduces significant estimation error - small changes in expected return estimates can produce dramatically different optimal portfolios. This fragility is the main criticism of pure MVO and the motivation for the alternative methods below.

The system offers three objective variants: **max Sharpe** (find the tangency portfolio - the efficient point with the best return-per-unit-risk), **minimum volatility** (find the portfolio with the lowest possible variance regardless of return), and **target return** (find the lowest-risk portfolio that achieves a specified return level).

**Risk Parity**
Risk parity reframes the optimization problem entirely. Instead of allocating capital equally across assets, it allocates **risk** equally. Each position contributes the same amount to total portfolio volatility. This typically produces much more stable portfolios than MVO because it does not depend on expected return estimates at all - only on the covariance structure.

Risk parity emerged as a serious institutional framework in the 1990s and became prominent after the 2008 financial crisis, when it outperformed traditional 60/40 portfolios. The intuition is that a traditional equity-heavy portfolio is not balanced by weight - it is almost entirely dominated by equity risk. Risk parity corrects this by sizing positions to equalize their risk contribution.

**Black-Litterman**
Black-Litterman addresses the estimation error problem of MVO by combining two sources of information: the **market equilibrium** (what the current market prices imply about expected returns) and your **personal views** (specific forecasts you have for individual assets, with explicit confidence levels).

Rather than estimating expected returns purely from historical data, Black-Litterman starts from the assumption that current market capitalizations represent a consensus equilibrium. Your views then adjust that equilibrium in proportion to their specified confidence. A high-confidence view produces a large adjustment; a low-confidence view barely moves the needle.

The result is a more stable, more intuitive set of expected returns that blends what the market collectively believes with what you specifically think. This produces optimized portfolios that tilt toward your convictions without overreacting to noisy signals.

**Regime-Blended Optimization**
Market regimes matter for optimization. The efficient portfolio in a bull market looks very different from the efficient portfolio in a bear market. If you optimize over mixed historical data, you are averaging across fundamentally different environments and the resulting weights are appropriate for none of them.

Regime-blended optimization runs a separate optimization for each detected market regime - bull, bear, and sideways - and then combines the resulting portfolios as a weighted average, where the weights equal the current regime probabilities estimated by the HMM. The output is a portfolio that is tilted toward the bull regime allocation when the model is confident the market is in a bull state, and toward the bear allocation when the model detects deteriorating conditions.

This is the most contextually aware of the static optimization methods and directly operationalizes the regime detector described below.

**Earnings Tilt**
Earnings tilt starts with the regime-blended portfolio as a base and then applies a systematic adjustment to each position based on the most recent earnings signal for that company. Positions where the company beat earnings estimates and sentiment was positive are increased by a fixed factor. Positions where earnings missed and sentiment was negative are reduced.

This method captures **earnings momentum** - the well-documented tendency of stocks that beat earnings estimates to continue outperforming in the short term. Rather than reacting to earnings news narratively, this method translates it into a structured, repeatable weight adjustment with defined magnitude and direction.

---

### Market Regime Detection - Hidden Markov Model

A Hidden Markov Model (HMM) is a statistical model for systems that transition between a fixed number of hidden states over time, where you can observe the outputs of the system but not the states themselves directly.

Applied to financial markets: the market is always in some state - call it bull, bear, or sideways - but you cannot observe that state directly. What you can observe is the sequence of daily returns, volatility, and drawdown. The HMM learns, from historical data, what pattern of observable return characteristics is associated with each hidden regime, and then uses that learned pattern to infer which regime the market is most likely in right now.

The system trains the HMM on rolling return, volatility, and momentum features derived from a broad market index. Once trained, it outputs a probability distribution over the three regimes: for example, "70% probability bull, 20% sideways, 10% bear." These probabilities flow directly into the regime-blended optimizer and into the narrative commentary generated by the AI model.

What makes the HMM valuable is that it does not require you to define what a bull or bear market is - it learns the distinction from the data. The regimes that emerge reflect the statistical structure of market returns, not an arbitrary threshold like "up 20% from the low."

---

### Backtesting Engine

Backtesting simulates how a portfolio strategy would have performed over a historical period using actual return data. QuantFusion implements a vectorized backtester that runs two parallel simulations: one covering the trailing one-year period and one covering three years.

For each simulation:
- Returns are loaded from the price database at daily resolution
- The portfolio is rebalanced monthly (or at another specified frequency) back to the target weights
- Transaction costs of 10 basis points per trade are deducted at each rebalance
- Performance is benchmarked against SPY (S&P 500 ETF)

The output includes: ending portfolio value on a $10,000 initial investment, annualized return and volatility, Sharpe ratio, maximum drawdown over the period, benchmark comparison (alpha, beta, information ratio), and a month-by-month return log.

Backtesting is a limited but important tool. It tells you what *would have* happened with your exact weights in the past. It cannot tell you what will happen in the future, and results are always subject to look-ahead bias if the weights themselves were derived from the same historical data used to test them. QuantFusion makes this limitation explicit by testing on two different windows: the one-year result tells you about recent behavior; the three-year result gives you a broader sample that includes different market environments.

---

### SEC EDGAR Earnings Intelligence

SEC EDGAR is the U.S. Securities and Exchange Commission's public filing database. Every publicly traded company is required to file earnings press releases (form 8-K) within four business days of reporting results, as well as quarterly (10-Q) and annual (10-K) reports. These filings are freely available to anyone - they are the same source that institutional analysts use.

QuantFusion's earnings pipeline automatically:

1. Translates a ticker symbol to the company's SEC CIK identifier
2. Retrieves the most recent filing of the requested type from EDGAR
3. Downloads the earnings press release exhibit (the actual document)
4. If the document is a PDF, renders each page to an image and runs it through NVIDIA's NIM vision model to extract the raw text
5. If the document is HTML, strips the markup to extract the text directly
6. Sends the extracted text to a large language model with a structured prompt to extract EPS actual, EPS estimate, revenue, and overall sentiment
7. Cross-references the reported EPS against analyst consensus estimates from public market data to compute the beat/miss signal
8. Persists the structured signals to the database, linked to the ticker and filing date

The result is a structured `EarningsSignal` record for each company: numeric EPS actual and estimate, revenue reported, whether the company beat the estimate (true/false), and a sentiment classification (positive, negative, neutral) derived from the tone of the document.

These signals then flow into three places: the earnings tilt optimizer adjusts portfolio weights based on them, the regime commentary prompt is enriched with company-specific earnings context, and the backtesting engine annotates the equity curve with the earnings events that occurred during the period.

---

### AI-Powered Portfolio Commentary

After computing risk metrics, regime state, and earnings signals, QuantFusion assembles all of that quantitative output into a single prompt and sends it to a large language model (LLaMa 3.3 70B via NVIDIA NIM). The model is instructed to produce a two-to-three sentence commentary that synthesizes:

- The current market regime and its confidence level
- The portfolio's specific Sharpe ratio, VaR, drawdown, and beta
- The most recent earnings signals for the holdings
- The main risk to watch given those inputs

The commentary is grounded strictly in the computed numbers - the model is instructed not to fabricate statistics. The result is a compact narrative that bridges the quantitative output and actionable insight.

This is cached in Redis against a key that includes the regime state, risk metrics bucket, and a hash of the current earnings filing dates. When new earnings are fetched for a holding, the cache is automatically invalidated and the next request generates fresh commentary that incorporates the new signals.

---

### Natural Language Agent

The LangGraph agent wraps the entire system in a conversational interface. You can ask questions in plain English about a portfolio:

- *"What is the risk profile of my portfolio?"* → routes to risk analytics
- *"How should I rebalance to maximize Sharpe?"* → routes to optimizer
- *"What is the current market regime?"* → routes to regime detector
- *"How did my portfolio perform over the last year?"* → routes to backtester
- *"What are my current holdings?"* → queries the portfolio database
- *"Did any of my holdings beat earnings last quarter?"* → routes to earnings signals

The agent uses a three-stage pipeline: first, an LLM classifies the intent of the question and extracts any structured arguments (lookback period, optimization method, etc.). Second, the appropriate data tool is called and returns structured data. Third, a second LLM call formats that data into a specific, number-grounded answer to the original question.

The agent's design enforces that all answers cite actual numbers from the data tools - it cannot hallucinate statistics because the statistics are injected into the prompt at the final stage.

---

## How to Read the Full Analyzer Report

The `POST /api/analyzer/run` endpoint is the entry point to the complete analysis. Given a set of portfolio holdings, it runs all of the above in parallel and returns a single report containing:

| Field | What It Tells You |
|---|---|
| `risk` | Your current portfolio's complete risk profile: Sharpe, VaR, drawdown, beta, volatility. The baseline for everything else. |
| `frontier` | 150 points along the efficient frontier for your ticker set. Every point above and to the left of your current position represents a Pareto improvement. |
| `optimized_mvo` | The max-Sharpe MVO portfolio for your tickers. Compare its weights to yours - large differences indicate significant inefficiency. |
| `optimized_rp` | The risk parity portfolio for your tickers. If dramatically different from MVO, your tickers have very unequal risk contributions. |
| `optimized_blended` | The regime-aware optimal allocation given current market conditions. This is the most contextually relevant recommendation. |
| `backtest_1y` | Your actual weights simulated over the trailing year with transaction costs. |
| `backtest_3y` | Same over three years. Compare Sharpe and drawdown across windows to understand consistency. |
| `regime` | Current regime state, confidence, and probability distribution across all states. |
| `regime_commentary` | AI narrative synthesizing regime, risk, and earnings into two to three sentences. |
| `fundamentals` | P/E ratio, market cap, dividend yield, and sector for each holding, sourced from live market data. |

The most educational exercise is to compute the gap between your current portfolio's Sharpe ratio (in `risk`) and the MVO optimal's Sharpe ratio (in `optimized_mvo`). That gap is the cost - measured in return-per-unit-risk - of the current allocation relative to the mathematical optimum for the same asset universe.

---

## The Analytical Workflow

A structured way to use this system:

**Step 1 - Establish Baseline**
Submit your holdings to `/api/analyzer/run`. Read your current Sharpe ratio, VaR, and maximum drawdown. These are your benchmarks.

**Step 2 - Understand Your Frontier**
Examine the efficient frontier output. Where does your current allocation sit? Is it inside the frontier (inefficient) or approximately on it? The distance between your current volatility and the minimum-variance portfolio's volatility at the same expected return quantifies the waste.

**Step 3 - Explore Optimization Methods**
Run `/api/optimize/run` with each method. Compare the resulting weights across MVO, risk parity, and Black-Litterman. Where they agree, the signal is strong. Where they diverge, you are in a region of high estimation uncertainty.

**Step 4 - Incorporate the Regime**
Run regime-blended and compare against the static MVO. Does the regime adjustment increase or decrease your equity exposure? What is the model's confidence? Low confidence (near 33/33/33 probability split) means the model sees an ambiguous environment - this is not a signal to take strong positions.

**Step 5 - Fetch and Apply Earnings Signals**
For each individual stock holding, post to `/api/documents/earnings` to retrieve the latest EDGAR filing and extract signals. Then run earnings_tilt. The weight changes show you exactly how the earnings results translate into an allocation adjustment.

**Step 6 - Stress Test With Backtesting**
Apply your optimized weights to the backtester. Does the theoretical Sharpe improvement hold up historically? If the optimizer produces a high-Sharpe projection but the backtest produces a lower Sharpe, you are likely seeing estimation error in the optimizer's expected return inputs. That gap is a calibration signal.

**Step 7 - Ask the Agent**
Query the agent with natural language to synthesize across all the above. The agent is most useful not for generating new analysis but for forcing clarity: if you cannot articulate what the numbers mean as a plain sentence, you do not fully understand them yet.

---

## What This System Is Not

QuantFusion is an analytical and educational tool. It is not:

- **A trading system.** It computes optimal weights; it does not execute trades, manage orders, or integrate with any broker.
- **A prediction engine.** The optimizer produces weights that would have been optimal in history, subject to the model's assumptions. Past optimality does not guarantee future performance.
- **A substitute for professional advice.** The methods here are real and widely used, but applying them to actual investment decisions requires judgment, risk tolerance assessment, and context that this system cannot provide.
- **A complete picture.** The system works on price returns and does not model liquidity, tax implications, leverage constraints, or the practical limits of rebalancing at the computed precision.

The value of the system is in developing the analytical vocabulary and quantitative intuition to think more rigorously about portfolio construction - not in following its outputs mechanically.

---

## Quick Start

```bash
# Clone and configure
git clone <repo>
cp .env.example .env
# Edit .env: set NVIDIA_API_KEY for LLM features (optional but recommended)

# Start the stack
docker compose up --build -d
docker compose exec api alembic upgrade head

# Run a full analysis
curl -X POST http://localhost:8000/api/analyzer/run \
  -H "Content-Type: application/json" \
  -d '{"holdings": [{"ticker": "AAPL", "weight": 0.4}, {"ticker": "MSFT", "weight": 0.4}, {"ticker": "BND", "weight": 0.2}]}'

# Fetch earnings for a holding
curl -X POST http://localhost:8000/api/documents/earnings \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "form_type": "8-K"}'
```

---

## API Reference

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | System status |
| GET | `/api/market/quote/{ticker}` | Live price and fundamentals |
| GET | `/api/market/returns/{ticker}` | Historical returns |
| POST | `/api/risk/compute` | Risk metrics for a set of holdings |
| POST | `/api/optimize/run` | Portfolio optimization (portfolio-based) |
| POST | `/api/optimize/stateless` | Portfolio optimization (holdings-based, no DB) |
| GET | `/api/optimize/history/{portfolio_id}` | Past optimization runs |
| GET | `/api/regime/current` | Current HMM regime state |
| POST | `/api/backtest/run` | Historical backtest for a portfolio |
| POST | `/api/analyzer/run` | Full analysis report (all modules) |
| POST | `/api/analyzer/snapshot` | Create a shareable public snapshot |
| GET | `/api/analyzer/snapshot/{token}` | Retrieve a snapshot by token |
| POST | `/api/agent/query` | Natural language portfolio query |
| POST | `/api/documents/earnings` | Fetch and process latest earnings from EDGAR |
| GET | `/api/documents/signals/{ticker}` | Retrieve stored earnings signals |

---

## Environment Variables

| Variable | Purpose | Required |
|---|---|---|
| `DATABASE_URL` | PostgreSQL connection string | Yes |
| `REDIS_URL` | Redis connection string | Yes |
| `NVIDIA_API_KEY` | NVIDIA NIM API key for LLM and OCR features | No (degrades gracefully) |
| `NIM_MODEL` | LLM model ID (default: `meta/llama-3.3-70b-instruct`) | No |
| `NIM_BASE_URL` | NIM gateway URL | No |
| `NIM_OCR_MODEL` | Vision model for PDF OCR (default: `nvidia/nemotron-ocr-v1`) | No |
| `CORS_ORIGINS` | Allowed CORS origins, comma-separated | No |

---

## Methods Reference Summary

| Method | Family | What It Optimizes For | Key Assumption |
|---|---|---|---|
| MVO max Sharpe | Mean-Variance | Best return-per-unit-risk | Returns are stationary and normally distributed |
| MVO min vol | Mean-Variance | Lowest possible portfolio variance | Same as above |
| Risk Parity | Variance-based | Equal risk contribution per asset | No expected return estimate needed |
| Black-Litterman | Bayesian | Sharpe, adjusted for personal views | Market prices reflect equilibrium |
| Regime-Blended | Regime-aware | Sharpe weighted by market state probabilities | HMM regime estimates are informative |
| Earnings Tilt | Signal-augmented | Regime-blended + earnings momentum | EPS beat/miss is a short-term price signal |
