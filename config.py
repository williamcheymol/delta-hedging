# =============================================================================
# config.py — Global simulation parameters
# =============================================================================
# All parameters are centralised here so you only need to change one file
# to re-run the entire project with different market conditions.

# --- Option parameters ---
S0    = 100.0   # Initial stock price
K     = 100.0   # Strike price (at-the-money by default)
T     = 1.0     # Time to maturity in years (1 year)
r     = 0.05    # Annual risk-free rate (5%)
SIGMA = 0.20    # Annual volatility (20%)

# --- Simulation parameters ---
N_STEPS = 252       # Number of hedging steps (1 per trading day)
N_PATHS = 1000      # Number of Monte Carlo paths

# --- Output paths ---
RESULTS_DIR  = "results/"
FIGURES_DIR  = "results/figures/"
DATA_DIR     = "results/data/"
