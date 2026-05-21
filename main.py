# =============================================================================
# main.py — Project entry point
# =============================================================================
# Run the full delta hedging simulation pipeline.
#
# Usage:
#   python main.py
#       → flat-vol simulation only (no external data required)
#
#   python main.py --csv path/to/SPY_options.csv --spot 741.41
#       → flat-vol simulation + market vol surface comparison
#         (CSV produced by the Market Fetcher project)
# =============================================================================

import argparse
import os
import numpy as np

import config
from pricing                import black_scholes_call
from simulation             import simulate_gbm_paths
from hedging.delta_hedge    import run_delta_hedge_chunked, run_surface_hedge
from analysis               import compute_hedging_metrics
from analysis.metrics       import summarise_metrics, save_pnl_to_csv


def main():
    parser = argparse.ArgumentParser(
        description="Delta Hedging Simulation — Black-Scholes"
    )
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to options CSV produced by the Market Fetcher project",
    )
    parser.add_argument(
        "--spot", type=float, default=None,
        help="Spot price at the time the CSV was fetched (required with --csv)",
    )
    args = parser.parse_args()

    print("=" * 52)
    print("  Delta Hedging Simulation — Black-Scholes")
    print("=" * 52)

    option_price = black_scholes_call(
        config.S0, config.K, config.T, config.r, config.SIGMA
    )
    print(f"\n  Option price at t=0 : {option_price:.4f}"
          f"  (ATM call, σ={config.SIGMA:.0%})")

    # ------------------------------------------------------------------
    # [1] Flat-vol delta hedge
    # ------------------------------------------------------------------
    print(f"\n[1/{'2' if args.csv else '1'}] Flat-vol delta hedge"
          f"  ({config.N_PATHS:,} paths, {config.N_STEPS} steps)...")

    pnl_flat = run_delta_hedge_chunked(
        config.S0, config.K, config.T, config.r, config.SIGMA,
        n_steps=config.N_STEPS, n_paths=config.N_PATHS,
        chunk_size=config.CHUNK_SIZE, verbose=True,
    )

    metrics_flat = compute_hedging_metrics(pnl_flat, option_price)
    print(f"\n  Hedging error (flat vol) : {metrics_flat['hedging_error']:.2%}")
    summarise_metrics(metrics_flat)
    save_pnl_to_csv(pnl_flat, config.DATA_DIR + "pnl_flat.csv")

    if args.csv is None:
        print("\nTip: pass --csv <path> --spot <price> to add the market surface comparison.")
        print("\nDone. Run the notebook for visualisations.")
        return

    # ------------------------------------------------------------------
    # [2] Market vol surface delta hedge (optional)
    # ------------------------------------------------------------------
    if not os.path.exists(args.csv):
        print(f"\nError: CSV not found at '{args.csv}'")
        return

    if args.spot is None:
        print("\nError: --spot is required when --csv is provided.")
        return

    print(f"\n[2/2] Market vol surface hedge  ({config.N_PATHS:,} paths)...")
    print(f"      Loading surface from '{args.csv}'  (spot={args.spot})...")

    from analysis.vol_surface import VolSurface
    surface   = VolSurface.from_csv(args.csv, spot=args.spot)
    fast_grid = surface.build_fast_grid()
    print(f"      {len(surface.df)} contracts · fast grid ready.")

    pnl_chunks = []
    done       = 0
    n_chunks   = (config.N_PATHS + config.CHUNK_SIZE - 1) // config.CHUNK_SIZE

    for i in range(n_chunks):
        size = min(config.CHUNK_SIZE, config.N_PATHS - done)
        paths, tg = simulate_gbm_paths(
            args.spot, config.r, config.SIGMA,
            config.T, config.N_STEPS, size, seed=42 + i,
        )
        pnl_chunks.append(
            run_surface_hedge(paths, tg, K=args.spot, r=config.r,
                              sigma_price=config.SIGMA, fast_grid=fast_grid)
        )
        done += size
        print(f"\r  {done:>9,} / {config.N_PATHS:,} paths"
              f"  ({100 * done / config.N_PATHS:.0f}%)", end="", flush=True)

    print()
    pnl_surface = np.concatenate(pnl_chunks)

    price_surface = black_scholes_call(
        args.spot, args.spot, config.T, config.r, config.SIGMA
    )
    metrics_surf = compute_hedging_metrics(pnl_surface, price_surface)
    print(f"\n  Hedging error (market surface) : {metrics_surf['hedging_error']:.2%}")
    summarise_metrics(metrics_surf)
    save_pnl_to_csv(pnl_surface, config.DATA_DIR + "pnl_surface.csv")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 52)
    print("  Results summary")
    print("=" * 52)
    print(f"  Flat vol (BS)   : {metrics_flat['hedging_error']:.2%}")
    print(f"  Market surface  : {metrics_surf['hedging_error']:.2%}")
    print("=" * 52)
    print("\nDone. Run the notebook for visualisations.")


if __name__ == "__main__":
    main()
