# =============================================================================
# hedging/gamma_hedge.py — Delta-Gamma hedging
# =============================================================================
# Delta hedging neutralises dC/dS (first-order risk).
# But gamma (d²C/dS²) causes the delta to drift between rebalancings —
# that is the main source of hedging error.
#
# Delta-Gamma hedging adds a SECOND option to the portfolio to also
# neutralise gamma. The resulting portfolio needs fewer rebalancings
# to achieve the same accuracy.
#
# Portfolio: short 1 option (to hedge), long h1 shares, long h2 hedge options
#
# Conditions at each step:
#   Δ_portfolio = 0  →  h1 + h2·Δ_hedge = Δ_target
#   Γ_portfolio = 0  →  h2·Γ_hedge = Γ_target
#
# Solving:
#   h2 = Γ_target / Γ_hedge           (units of hedge option needed)
#   h1 = Δ_target − h2·Δ_hedge        (shares needed after option hedge)
# =============================================================================

import numpy as np
from pricing.black_scholes import (black_scholes_call, black_scholes_delta,
                                   black_scholes_gamma)


def run_gamma_hedge(paths, time_grid, K, r, sigma,
                    K_hedge, transaction_cost=0.0):
    """
    Run a delta-gamma hedging simulation.

    Uses a second European call (with strike K_hedge) to neutralise gamma,
    then adjusts the stock position to neutralise delta.

    Parameters
    ----------
    paths            : np.ndarray (n_steps+1, n_paths)
    time_grid        : np.ndarray (n_steps+1,)
    K                : float — strike of the option being hedged (short)
    K_hedge          : float — strike of the hedging option (long)
    r                : float — risk-free rate
    sigma            : float — volatility (same for both options)
    transaction_cost : float — proportional cost per unit traded

    Returns
    -------
    pnl : np.ndarray (n_paths,) — final P&L per path
    """
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt      = time_grid[1] - time_grid[0]
    T_total = time_grid[-1]
    S0      = paths[0]

    # --- t = 0: open the hedge ---
    T_rem = T_total

    # Sell the target option, receive premium
    option_premium = black_scholes_call(S0, K, T_rem, r, sigma)
    cash = option_premium.copy()

    # Compute gamma of target and hedge options at t=0
    gamma_target = black_scholes_gamma(S0, K, T_rem, r, sigma)   # gamma of the option we sold (strike K)
    gamma_hedge  = black_scholes_gamma(S0, K_hedge, T_rem, r, sigma)   # gamma of the hedge option (strike K_hedge)

    # Compute h2: number of hedge options to buy
    h2 = gamma_target / gamma_hedge

    # Pay for hedge options
    price_hedge = black_scholes_call(S0, K_hedge, T_rem, r, sigma)
    cash -= h2 * price_hedge

    # Compute h1: shares to hold for delta neutrality
    # delta_target = delta of the option we sold
    # delta_hedge  = delta of the hedge option
    # h1 = delta_target - h2 * delta_hedge
    delta_target = black_scholes_delta(S0, K, T_rem, r, sigma)
    delta_hedge  = black_scholes_delta(S0, K_hedge, T_rem, r, sigma)
    h1 = delta_target - h2 * delta_hedge

    shares = h1.copy()
    cash  -= shares * S0

    # --- Hedging loop ---
    for t in range(1, n_steps):
        S_t   = paths[t]
        T_rem = time_grid[-1] - time_grid[t]

        cash *= np.exp(r * dt)

        # Recompute gammas and rebalance h2 (hedge options)
        gamma_target  = black_scholes_gamma(S_t, K, T_rem, r, sigma)
        gamma_hedge   = black_scholes_gamma(S_t, K_hedge, T_rem, r, sigma)
        new_h2        = gamma_target / gamma_hedge
        dh2           = new_h2 - h2

        price_hedge   = black_scholes_call(S_t, K_hedge, T_rem, r, sigma)
        cash         -= dh2 * price_hedge
        cash         -= transaction_cost * np.abs(dh2) * price_hedge
        h2            = new_h2

        # Rebalance shares for delta neutrality
        delta_target  = black_scholes_delta(S_t, K, T_rem, r, sigma)
        delta_hedge   = black_scholes_delta(S_t, K_hedge, T_rem, r, sigma)
        new_h1        = delta_target - h2 * delta_hedge
        dh1           = new_h1 - shares

        cash   -= dh1 * S_t
        cash   -= transaction_cost * np.abs(dh1) * S_t
        shares  = new_h1

    # --- t = T: close ---
    S_T = paths[-1]
    cash *= np.exp(r * dt)

    # Liquidate shares
    cash += shares * S_T

    # Hedge options expire — collect their payoff
    cash += h2 * np.maximum(S_T - K_hedge, 0)

    # Pay target option payoff
    cash -= np.maximum(S_T - K, 0)

    return cash
