# =============================================================================
# hedging/delta_hedge.py — Delta hedging simulation (vectorised)
# =============================================================================
# Vectorised across all n_paths simultaneously at each time step.
# Supports transaction costs, vol mismatch, and market vol surface hedging.
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


def run_delta_hedge_chunked(S0, K, T_total, r, sigma, n_steps, n_paths,
                             sigma_hedge=None, transaction_cost=0.0,
                             option_type="call", return_attribution=False,
                             chunk_size=50_000, seed=42, verbose=True):
    """
    Memory-efficient version of run_delta_hedge for large n_paths (e.g. 1 million).

    Processes paths in batches of `chunk_size` so that RAM usage is bounded at
    ~chunk_size × n_steps × 8 bytes regardless of total n_paths.
    At chunk_size=50 000 and n_steps=252 that is ~100 MB per chunk.

    Parameters
    ----------
    S0, K, T_total, r, sigma  : same as run_delta_hedge
    n_steps                   : int   — number of hedging steps
    n_paths                   : int   — total number of paths (e.g. 1_000_000)
    chunk_size                : int   — paths per chunk (default 50 000)
    seed                      : int   — base random seed; chunk i uses seed+i
    verbose                   : bool  — print progress to stdout

    All other keyword arguments are forwarded to run_delta_hedge unchanged.

    Returns
    -------
    Same as run_delta_hedge (pnl array, or tuple with attribution arrays).
    """
    from simulation.monte_carlo import simulate_gbm_paths

    pnl_chunks   = []
    gamma_chunks = []
    theta_chunks = []
    done         = 0
    n_chunks     = (n_paths + chunk_size - 1) // chunk_size

    for i in range(n_chunks):
        size      = min(chunk_size, n_paths - done)
        paths, tg = simulate_gbm_paths(S0, r, sigma, T_total, n_steps, size,
                                       seed=seed + i)

        result = run_delta_hedge(
            paths, tg, K, r, sigma,
            sigma_hedge=sigma_hedge,
            transaction_cost=transaction_cost,
            option_type=option_type,
            return_attribution=return_attribution,
        )

        if return_attribution:
            pnl, gp, tp = result
            gamma_chunks.append(gp)
            theta_chunks.append(tp)
        else:
            pnl = result

        pnl_chunks.append(pnl)
        done += size

        if verbose:
            pct = 100 * done / n_paths
            print(f"\r  {done:>9,} / {n_paths:,} paths  ({pct:.0f}%)",
                  end="", flush=True)

    if verbose:
        print()

    pnl_all = np.concatenate(pnl_chunks)
    if return_attribution:
        return pnl_all, np.concatenate(gamma_chunks), np.concatenate(theta_chunks)
    return pnl_all


def run_surface_hedge(paths, time_grid, K, r, sigma_price, fast_grid):
    """
    Delta hedge using an implied volatility surface for delta computation.

    The option premium at t=0 is priced with sigma_price (flat vol), keeping
    the comparison with run_delta_hedge fair. Deltas at each step are computed
    using the local IV queried from fast_grid at the current (T_rem, S_t).

    Parameters
    ----------
    paths        : np.ndarray (n_steps+1, n_paths)
    time_grid    : np.ndarray (n_steps+1,)
    K            : float — strike of the option being hedged
    r            : float — risk-free rate
    sigma_price  : float — flat vol used to price the option at t=0
    fast_grid    : RegularGridInterpolator — as returned by VolSurface.build_fast_grid()
                   called as fast_grid([[T_rem, S], ...]) → implied vol array

    Returns
    -------
    pnl : np.ndarray (n_paths,)
    """
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    dt      = time_grid[1] - time_grid[0]
    T_total = time_grid[-1]
    S0      = paths[0]

    option_premium = black_scholes_call(S0, K, T_total, r, sigma_price)

    sigma0 = fast_grid(np.column_stack([np.full(n_paths, T_total), S0]))
    delta  = black_scholes_delta(S0, K, T_total, r, sigma0)

    cash   = option_premium.copy()
    shares = delta.copy()
    cash  -= shares * S0

    for t in range(1, n_steps):
        S_t   = paths[t]
        T_rem = T_total - time_grid[t]
        cash *= np.exp(r * dt)

        sigma_t      = fast_grid(np.column_stack([np.full(n_paths, T_rem), S_t]))
        new_delta    = black_scholes_delta(S_t, K, T_rem, r, sigma_t)
        delta_change = new_delta - shares
        shares      += delta_change
        cash        -= delta_change * S_t

    S_T  = paths[-1]
    cash *= np.exp(r * dt)
    cash += shares * S_T
    cash -= np.maximum(S_T - K, 0)

    return cash
