# =============================================================================
# simulation/monte_carlo.py — GBM path simulation (vectorised)
# =============================================================================
# Under Black-Scholes, stock prices follow a Geometric Brownian Motion:
#
#   S(t+dt) = S(t) · exp[(r − σ²/2)·dt + σ·√dt·Z],   Z ~ N(0,1)
#
# Fully vectorised across paths via numpy broadcasting.
# Supports antithetic variates to reduce variance at no extra simulation cost.
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
        # Simulate Z and −Z together: same variance reduction, no extra cost.
        Z      = np.random.standard_normal((n_steps, n_paths // 2))
        Z_full = np.concatenate([Z, -Z], axis=1)
    else:
        Z_full = np.random.standard_normal((n_steps, n_paths))

    paths      = np.empty((n_steps + 1, n_paths))
    paths[0]   = S0

    increments = np.exp((r - sigma**2 / 2) * dt + sigma * np.sqrt(dt) * Z_full)
    paths[1:]  = S0 * np.cumprod(increments, axis=0)

    return paths, time_grid
