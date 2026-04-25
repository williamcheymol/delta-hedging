# Delta & Gamma Hedging Simulation — European Options 

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![Tests](https://img.shields.io/badge/tests-37%20passed-brightgreen)
![Framework](https://img.shields.io/badge/framework-Black--Scholes-indigo)
![Status](https://img.shields.io/badge/status-Phase%201%20complete-success)

A self-contained Python project simulating the delta and gamma hedging of European options under Black-Scholes assumptions.

---

## Background

### Black-Scholes pricing

Under the Black-Scholes model, the stock price follows a Geometric Brownian Motion (GBM):

$$dS = \mu S \, dt + \sigma S \, dW$$

Under the risk-neutral measure ($\mu \to r$), the price of a European call option with strike $K$ and maturity $T$ is:

$$C = S \cdot N(d_1) - K e^{-rT} \cdot N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r + \sigma^2/2) \cdot T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}$$

where $N(\cdot)$ is the cumulative standard normal distribution. The put price follows from put-call parity: $P = C - S + K e^{-rT}$.

### The Greeks

The Greeks measure the sensitivity of the option price to its inputs:

| Greek | Formula | Interpretation |
|-------|---------|----------------|
| **Delta** $\Delta$ | $N(d_1)$ | Price change per €1 move in $S$ |
| **Gamma** $\Gamma$ | $\frac{N'(d_1)}{S \sigma \sqrt{T}}$ | Rate of change of delta |
| **Vega** $\nu$ | $S \cdot N'(d_1) \cdot \sqrt{T}$ | Price change per 1% move in $\sigma$ |

Gamma is highest when the option is at-the-money (S ≈ K) and near expiry — this is when the delta changes most rapidly and hedging is hardest.

### Delta hedging

When a trader sells an option, they take on directional risk. **Delta hedging** neutralises this by holding Δ shares per option sold, making the portfolio locally immune to small price changes.

In continuous time, perfect delta hedging replicates the option exactly — the portfolio P&L is zero. In practice, hedging is discrete (e.g. daily), which introduces a residual **tracking error** that scales as:

$$\text{Hedging error} \propto \sigma \sqrt{\frac{T}{n_{\text{steps}}}}$$

The P&L at each rebalancing step decomposes into:
- **Gamma P&L** $= \frac{1}{2} \Gamma (\Delta S)^2$ — profit from realised price moves (convexity)
- **Theta cost** $= -\frac{1}{2} \Gamma \sigma^2 S^2 \, dt$ — cost of holding gamma over time

These two terms cancel exactly in continuous time. The residual is the discretisation error.

### Gamma hedging

Delta hedging leaves the portfolio exposed to **gamma risk** — large moves in S cause the delta to shift and the hedge to break down. **Gamma hedging** adds a second option position to neutralise gamma simultaneously:

$$h_2 = \frac{\Gamma_{\text{target}}}{\Gamma_{\text{hedge}}} \qquad \text{(gamma neutrality)}$$

$$h_1 = \Delta_{\text{target}} - h_2 \cdot \Delta_{\text{hedge}} \qquad \text{(delta neutrality)}$$

With both delta and gamma neutralised, the portfolio is protected against larger moves and the hedging error is significantly reduced, especially at low rebalancing frequencies.

---

## Key results

| Metric | Value |
|--------|-------|
| Call price (ATM, T=1, σ=20%) | 10.45 |
| Delta at-the-money | 0.637 |
| Hedging error (daily, 1000 paths) | ~4.1% |
| Hedging error (weekly, 1000 paths) | ~9.1% |
| Gamma hedge error (weekly) | ~4.8% |

Hedging error follows the theoretical **1/√n** relationship — halving the rebalancing interval reduces error by ~30%.

---

## Features

**Pricing**
- Black-Scholes call and put pricing
- Full Greeks: delta, gamma, vega
- Put-call parity verification

**Simulation**
- Geometric Brownian Motion (GBM) — vectorised across paths
- Antithetic variates for variance reduction
- Reproducible seeded runs

**Hedging**
- Discrete delta hedging with configurable rebalancing frequency
- Delta-gamma hedging (second option to neutralise gamma)
- Transaction costs
- Volatility mismatch analysis (sigma_hedge ≠ sigma_real)
- P&L attribution: gamma P&L vs theta cost

**Analysis & Visualisation**
- Hedging error vs rebalancing frequency
- Vol mismatch impact on P&L
- Interactive Plotly dashboard with ipywidgets sliders
- Quant Dark design system (centralised colour palette + layout template)

**Testing**
- 37 unit tests across 3 modules (pytest)
- Algebraic tests (exact tolerances) and statistical tests (Monte Carlo tolerances)

---

## Project structure

```
delta-hedging/
├── config.py                   # Market & simulation parameters
├── main.py                     # Pipeline entry point
│
├── pricing/
│   └── black_scholes.py        # BS pricing + Greeks (delta, gamma, vega)
│
├── simulation/
│   └── monte_carlo.py          # GBM path generation, antithetic variates
│
├── hedging/
│   ├── delta_hedge.py          # Delta hedging loop + P&L attribution
│   └── gamma_hedge.py          # Delta-gamma hedging
│
├── analysis/
│   └── metrics.py              # Hedging metrics, frequency & vol mismatch analysis
│
├── style/
│   └── theme.py                # Quant Dark palette + Plotly layout template
│
├── notebooks/
│   └── visualization.ipynb     # Interactive dashboard (8 slides)
│
├── tests/
│   ├── test_pricing.py         # 18 tests — BS pricing & Greeks
│   ├── test_simulation.py      # 10 tests — GBM properties & antithetic variates
│   └── test_hedging.py         # 9 tests  — hedging P&L & gamma neutrality
│
└── results/
    ├── data/                   # Generated CSVs
    └── figures/                # Saved charts
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full simulation
python main.py

# 3. Open the interactive notebook
jupyter notebook notebooks/visualization.ipynb

# 4. Run the test suite
pytest tests/ -v
```

---

## Interactive notebook

The notebook contains 8 interactive slides powered by Plotly and ipywidgets:

1. Black-Scholes pricing (call & put) — live slider on S, K, σ
2. Greeks — delta, gamma, vega vs spot price
3. GBM paths — simulated trajectories with σ slider
4. P&L distribution — hedging error with n_steps, transaction cost, option type sliders
5. Hedging error vs frequency — with 1/√n theoretical fit
6. Volatility mismatch — P&L mean and std as a function of σ_hedge
7. Delta vs gamma hedging comparison
8. P&L attribution — gamma P&L vs theta cost with σ_real / σ_hedge sliders

---

## Technical highlights

**Vectorisation** — the hedging loop iterates over time steps (sequential by nature) but operates on all Monte Carlo paths simultaneously as numpy arrays. This removes the outer path loop and gives a ~100x speedup over a naive double loop.

**Antithetic variates** — for each random draw $Z$, the simulation also generates a path from $-Z$. Since log-returns satisfy:

$$\log r_1 + \log r_2 = 2\left(r - \frac{\sigma^2}{2}\right)dt$$

the noise cancels algebraically, reducing Monte Carlo variance with no additional computation.

**P&L attribution** — at each time step, the hedging error decomposes into:

$$\text{P\&L} = \underbrace{\frac{1}{2}\Gamma(\Delta S)^2}_{\text{gamma P\&L}} - \underbrace{\frac{1}{2}\Gamma\sigma^2 S^2 \, dt}_{\text{theta cost}}$$

The net P&L tracks the difference between realised and implied volatility.

---

## Parameters (`config.py`)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `S0` | 100 | Initial stock price |
| `K` | 100 | Strike (at-the-money) |
| `T` | 1.0 | Maturity (1 year) |
| `r` | 0.05 | Risk-free rate (5%) |
| `SIGMA` | 0.20 | Volatility (20%) |
| `N_STEPS` | 252 | Daily rebalancing steps |
| `N_PATHS` | 1000 | Monte Carlo paths |

---

## Coming soon — Phase 2: Real market data

Phase 2 will replace the synthetic GBM paths with real historical data extracted via **yfinance**:

- Historical price extraction with dividend and split adjustment
- Realised volatility estimation (rolling window)
- Implied vs realised volatility comparison
- Hedging simulation on real equity paths (S&P 500 constituents)

---

## Gallery

| Pricing (call & put) | Greeks vs spot |
|:---:|:---:|
| ![](Graphiques/Pricing.png) | ![](Graphiques/Greeks.png) |

| GBM paths | P&L distribution |
|:---:|:---:|
| ![](Graphiques/Paths.png) | ![](Graphiques/Distribution.png) |

| Hedging error vs frequency | Call hedge — portfolio vs option |
|:---:|:---:|
| ![](Graphiques/Errors.png) | ![](Graphiques/Call%20Hedge.png) |

| Final P&L | P&L attribution — gamma vs theta |
|:---:|:---:|
| ![](Graphiques/P%26L.png) | ![](Graphiques/P%26L%20Attribution.png) |

---

*Black-Scholes framework — Phase 1 requires no external market data.*