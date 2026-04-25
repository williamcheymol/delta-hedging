# =============================================================================
# simulation/monte_carlo.py — GBM path simulation (vectorised)
# =============================================================================
# Under Black-Scholes, stock prices follow a Geometric Brownian Motion:
#
#   S(t+dt) = S(t) · exp[(r − σ²/2)·dt + σ·√dt·Z],   Z ~ N(0,1)
#
# Phase 2 upgrades:
#   - Fully vectorised: no Python loop over paths (numpy broadcasting)
#   - Antithetic variates: simulate Z and −Z together to halve variance
# =============================================================================

import numpy as np


def simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=42,
                       antithetic=False):
    """
    Simulate stock price paths under Geometric Brownian Motion.

    Parameters
    ----------
    S0         : float — initial stock price
    r          : float — annual risk-free rate
    sigma      : float — annual volatility
    T          : float — total time horizon (in years)
    n_steps    : int   — number of time steps
    n_paths    : int   — number of Monte Carlo paths
    seed       : int   — random seed for reproducibility
    antithetic : bool  — if True, use antithetic variates (n_paths must be even)

    Returns
    -------
    paths     : np.ndarray (n_steps+1, n_paths)
    time_grid : np.ndarray (n_steps+1,)
    """
    np.random.seed(seed)

    dt        = T / n_steps
    time_grid = np.linspace(0, T, n_steps + 1)

    if antithetic:
        # Antithetic variates
        # Generate only n_paths//2 base draws, then mirror them with −Z.
        # This gives n_paths total paths with lower variance at no extra cost.
        #
        # Step-by-step:
        #   a) Generate Z of shape (n_steps, n_paths//2)  ← base draws
        #   b) Build Z_full of shape (n_steps, n_paths) by concatenating Z and −Z
        #      Hint: np.concatenate([Z, -Z], axis=1)
        #   c) Proceed with Z_full as normal

        Z      = np.random.standard_normal((n_steps, n_paths // 2))
        Z_full = np.concatenate([Z, -Z], axis=1)

    else:
        # Standard draws — shape (n_steps, n_paths)
        Z_full = np.random.standard_normal((n_steps, n_paths))

    # Vectorised GBM (replace the old Python loop)
    # Compute all increments at once: shape (n_steps, n_paths)
    # Each increment = exp[(r − σ²/2)·dt + σ·√dt·Z]
    # Then use np.cumprod to build cumulative path from S0.
    #
    # Hint:
    #   increments = np.exp(...)               # shape (n_steps, n_paths)
    #   paths[1:]  = S0 * np.cumprod(...)      # cumulative product along axis=0

    paths      = np.empty((n_steps + 1, n_paths))
    paths[0]   = S0

    increments = np.exp((r-sigma**2/2)*dt + sigma * np.sqrt(dt)*Z_full)
    paths[1:]  = S0 * np.cumprod(increments, axis=0)

    return paths, time_grid
