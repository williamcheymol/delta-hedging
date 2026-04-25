# =============================================================================
# main.py — Project entry point
# =============================================================================
# Run this file to execute the full delta hedging simulation pipeline:
#
#   1. Load parameters from config.py
#   2. Simulate stock price paths (Monte Carlo / GBM)
#   3. Run the delta hedging loop
#   4. Compute and display performance metrics
#   5. Save results to disk
#
# Usage:
#   python main.py
# =============================================================================

import config
from pricing      import black_scholes_call
from simulation   import simulate_gbm_paths
from hedging      import run_delta_hedge
from analysis     import compute_hedging_metrics
from analysis.metrics import summarise_metrics, save_pnl_to_csv


def main():
    print("=" * 50)
    print("  Delta Hedging Simulation — Black-Scholes")
    print("=" * 50)

    # ------------------------------------------------------------------
    # Step 1: Simulate stock price paths
    # ------------------------------------------------------------------
    print("\n[1/4] Simulating GBM paths...")
    paths, time_grid = simulate_gbm_paths(
        S0      = config.S0,
        r       = config.r,
        sigma   = config.SIGMA,
        T       = config.T,
        n_steps = config.N_STEPS,
        n_paths = config.N_PATHS,
    )
    print(f"      {config.N_PATHS} paths × {config.N_STEPS} steps generated.")

    # ------------------------------------------------------------------
    # Step 2: Price the option at t=0
    # ------------------------------------------------------------------
    option_price = black_scholes_call(config.S0, config.K, config.T,
                                      config.r, config.SIGMA)
    print(f"\n[2/4] Option price at t=0 : {option_price:.4f}")

    # ------------------------------------------------------------------
    # Step 3: Run delta hedging loop
    # ------------------------------------------------------------------
    print("\n[3/4] Running delta hedging...")
    pnl = run_delta_hedge(
        paths     = paths,
        time_grid = time_grid,
        K         = config.K,
        r         = config.r,
        sigma     = config.SIGMA,
    )

    # ------------------------------------------------------------------
    # Step 4: Compute and display metrics
    # ------------------------------------------------------------------
    print("\n[4/4] Computing performance metrics...")
    metrics = compute_hedging_metrics(pnl, option_price)
    summarise_metrics(metrics)

    # ------------------------------------------------------------------
    # Step 5: Save results
    # ------------------------------------------------------------------
    save_pnl_to_csv(pnl, config.DATA_DIR + "pnl.csv")
    print("\nDone. Run the notebook for visualisations.")


if __name__ == "__main__":
    main()
