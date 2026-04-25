# =============================================================================
# hedging/delta_hedge.py — Delta hedging simulation (vectorised)
# =============================================================================
# Phase 2 upgrades vs Phase 1:
#   - Vectorised across paths: all n_paths run simultaneously (no outer loop)
#   - Transaction costs: a small cost is paid on each share traded
#   - Vol mismatch: hedge with sigma_hedge ≠ sigma_real (vega risk analysis)
#
# The inner loop over time steps remains sequential (cash[t] depends on cash[t-1]).
# But all paths are updated simultaneously as numpy vectors at each step.
# =============================================================================

import numpy as np
from pricing.black_scholes import (black_scholes_call, black_scholes_delta,
                                   black_scholes_gamma)


def run_delta_hedge(paths, time_grid, K, r, sigma,
                   sigma_hedge=None, transaction_cost=0.0,
                   option_type="call", return_attribution=False):
    """
    Run a vectorised delta hedging simulation over all Monte Carlo paths.

    Parameters
    ----------
    paths              : np.ndarray (n_steps+1, n_paths)
    time_grid          : np.ndarray (n_steps+1,)
    K                  : float — strike price
    r                  : float — risk-free rate
    sigma              : float — TRUE volatility used to simulate paths
    sigma_hedge        : float or None — volatility used to compute delta.
                         If None, uses sigma (perfect vol knowledge).
    transaction_cost   : float — proportional cost per unit of shares traded
    option_type        : str — "call" or "put"
    return_attribution : bool — if True, also return gamma and theta P&L arrays

    Returns
    -------
    pnl : np.ndarray (n_paths,) — total final P&L per path

    If return_attribution=True, returns (pnl, gamma_pnl, theta_pnl) where:
      gamma_pnl : np.ndarray (n_paths,) — cumulative gamma P&L  = Σ ½Γ(ΔS)²
      theta_pnl : np.ndarray (n_paths,) — cumulative theta P&L  = Σ -½Γσ²S²dt
      total P&L ≈ gamma_pnl + theta_pnl  (up to discrete hedging error)

    Economic interpretation:
      - gamma_pnl > 0 when large price moves occur (you profit from convexity)
      - theta_pnl < 0 always (you pay for time decay, the cost of owning gamma)
      - When sigma_real > sigma_hedge: gamma_pnl > |theta_pnl| → net gain
      - When sigma_real < sigma_hedge: gamma_pnl < |theta_pnl| → net loss
    """
    if sigma_hedge is None:
        sigma_hedge = sigma

    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt      = time_grid[1] - time_grid[0]

    # --- t = 0: open the hedge ---
    T_total = time_grid[-1]
    S0      = paths[0]

    option_premium = black_scholes_call(S0, K, T_total, r, sigma)
    delta          = black_scholes_delta(S0, K, T_total, r, sigma_hedge)

    cash   = option_premium.copy()
    shares = delta.copy()
    cash  -= shares * S0
    cash  -= transaction_cost * np.abs(delta) * S0

    # P&L attribution accumulators — shape (n_paths,)
    gamma_pnl = np.zeros(n_paths)
    theta_pnl = np.zeros(n_paths)

    # --- Hedging loop ---
    for t in range(1, n_steps):
        S_t         = paths[t]
        S_prev      = paths[t - 1]
        T_remaining = time_grid[-1] - time_grid[t]

        cash *= np.exp(r * dt)

        # P&L attribution: decompose hedging error at each step
        # Gamma P&L = ½ · Γ · (ΔS)²   — profit from realized price moves
        # Theta P&L = -½ · Γ · σ² · S² · dt  — cost of time decay
        gamma_t    = black_scholes_gamma(S_prev, K, T_remaining + dt, r, sigma_hedge)
        dS         = S_t - S_prev
        gamma_pnl += 0.5 * gamma_t * dS**2
        theta_pnl -= 0.5 * gamma_t * sigma_hedge**2 * S_prev**2 * dt

        new_delta    = black_scholes_delta(S_t, K, T_remaining, r, sigma_hedge)
        delta_change = new_delta - shares
        shares      += delta_change
        cash        -= delta_change * S_t
        cash        -= transaction_cost * np.abs(delta_change) * S_t

    # --- t = T: close the hedge ---
    S_T = paths[-1]
    cash *= np.exp(r * dt)
    cash += shares * S_T

    if option_type == "call":
        payoff = np.maximum(S_T - K, 0)
    else:
        payoff = np.maximum(K - S_T, 0)

    cash -= payoff

    if return_attribution:
        return cash, gamma_pnl, theta_pnl
    return cash
