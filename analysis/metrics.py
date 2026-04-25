# =============================================================================
# analysis/metrics.py — Hedging performance metrics and analyses
# =============================================================================

import numpy as np
import pandas as pd


# =============================================================================
# Core metrics
# =============================================================================

def compute_hedging_metrics(pnl, option_price):
    """
    Compute summary statistics for the hedging P&L distribution.

    Parameters
    ----------
    pnl          : np.ndarray (n_paths,)
    option_price : float — used to normalise the hedging error

    Returns
    -------
    dict — summary statistics
    """
    return {
        "mean_pnl":      np.mean(pnl),
        "std_pnl":       np.std(pnl),
        "min_pnl":       np.min(pnl),
        "max_pnl":       np.max(pnl),
        "hedging_error": np.std(pnl) / option_price,
        "n_paths":       len(pnl),
    }


def summarise_metrics(metrics):
    """Print a formatted summary table."""
    print("=" * 40)
    print("  Delta Hedging — Performance Summary")
    print("=" * 40)
    print(f"  Paths simulated : {metrics['n_paths']}")
    print(f"  Mean P&L        : {metrics['mean_pnl']:.4f}")
    print(f"  Std  P&L        : {metrics['std_pnl']:.4f}")
    print(f"  Min  P&L        : {metrics['min_pnl']:.4f}")
    print(f"  Max  P&L        : {metrics['max_pnl']:.4f}")
    print(f"  Hedging error   : {metrics['hedging_error']:.2%}")
    print("=" * 40)


def save_pnl_to_csv(pnl, filepath):
    """Save P&L array to CSV."""
    pd.DataFrame({"pnl": pnl}).to_csv(filepath, index=False)
    print(f"P&L saved to {filepath}")


# =============================================================================
# Volatility mismatch analysis
# =============================================================================

def vol_mismatch_analysis(run_hedge_fn, paths, time_grid, K, r,
                          sigma_real, sigma_hedge_range):
    """
    Study P&L bias when hedging with the wrong volatility.

    The hedger uses sigma_hedge to compute deltas, but paths are simulated
    with sigma_real. If sigma_hedge > sigma_real, the hedger over-hedges
    and pockets a positive P&L on average (and vice versa).

    Parameters
    ----------
    run_hedge_fn      : callable — run_delta_hedge function
    paths             : np.ndarray — simulated paths (fixed, with sigma_real)
    time_grid         : np.ndarray
    K, r              : floats
    sigma_real        : float — true volatility of the simulation
    sigma_hedge_range : list of floats — vol values to test for hedging

    Returns
    -------
    results : pd.DataFrame — columns: sigma_hedge, mean_pnl, std_pnl
    """

    rows = []
    for sigma_h in sigma_hedge_range:
        pnl = run_hedge_fn(paths, time_grid, K, r, sigma_real, sigma_hedge=sigma_h)
        rows.append({
            "sigma_hedge": sigma_h,
            "mean_pnl":    np.mean(pnl),
            "std_pnl":     np.std(pnl),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Monte Carlo convergence analysis
# =============================================================================

def convergence_analysis(run_simulation_fn, n_paths_range):
    """
    Study how the estimate of mean P&L converges as n_paths increases.

    Demonstrates the law of large numbers: std(mean) ∝ 1/√n_paths.

    Parameters
    ----------
    run_simulation_fn : callable — returns pnl array for a given n_paths
                        signature: fn(n_paths) -> np.ndarray
    n_paths_range     : list of int — e.g. [50, 100, 500, 1000, 5000]

    Returns
    -------
    results : pd.DataFrame — columns: n_paths, mean_pnl, std_of_mean
    """

    rows = []
    for n in n_paths_range:
        pnl = run_simulation_fn(n)
        rows.append({
            "n_paths":      n,
            "mean_pnl":     np.mean(pnl),
            "std_of_mean":  np.std(pnl) / np.sqrt(n),
        })

    return pd.DataFrame(rows)


# =============================================================================
# Hedging error vs rebalancing frequency
# =============================================================================

def frequency_analysis(simulate_fn, hedge_fn, S0, K, r, sigma, T,
                       n_steps_range, n_paths=500, option_price=None):
    """
    Compute hedging error for different rebalancing frequencies.

    Parameters
    ----------
    simulate_fn    : callable — simulate_gbm_paths
    hedge_fn       : callable — run_delta_hedge
    n_steps_range  : list of int — rebalancing steps to test
    option_price   : float or None — if None, recomputed each time

    Returns
    -------
    pd.DataFrame — columns: n_steps, hedging_error
    """
    from pricing.black_scholes import black_scholes_call

    rows = []
    for n_steps in n_steps_range:
        paths, time_grid = simulate_fn(
            S0=S0, r=r, sigma=sigma, T=T,
            n_steps=n_steps, n_paths=n_paths, seed=42
        )
        pnl   = hedge_fn(paths, time_grid, K, r, sigma)
        price = option_price or black_scholes_call(S0, K, T, r, sigma)
        rows.append({
            "n_steps":       n_steps,
            "hedging_error": np.std(pnl) / price,
        })

    return pd.DataFrame(rows)
