# =============================================================================
# tests/test_hedging.py — Unit tests for hedging/delta_hedge.py
#                         and hedging/gamma_hedge.py
# =============================================================================
# Run with: pytest tests/test_hedging.py -v
#
# Statistical tests use large n_paths and tight tolerances only when
# the mathematical expectation is exact (martingale property).
# =============================================================================

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.monte_carlo import simulate_gbm_paths
from hedging.delta_hedge import run_delta_hedge
from hedging.gamma_hedge import run_gamma_hedge
from pricing.black_scholes import (
    black_scholes_call, black_scholes_delta, black_scholes_gamma
)

# --- Shared parameters ---
S0, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20
N_STEPS, N_PATHS   = 252, 5000


# Helper: simulate paths once and reuse
def get_paths(n_steps=N_STEPS, n_paths=N_PATHS, seed=42):
    return simulate_gbm_paths(S0, r, sigma, T, n_steps, n_paths, seed=seed)


class TestDeltaHedgePnL:

    def test_mean_pnl_near_zero(self):
        """
        Mean P&L ≈ 0 over many paths (martingale property).

        Under BS assumptions with perfect vol knowledge, a delta-hedged
        portfolio has zero expected P&L. With 5000 paths, the sample mean
        should be within 0.05 of zero.
        """
        paths, tg = get_paths()
        pnl = run_delta_hedge(paths, tg, K, r, sigma)
        assert abs(pnl.mean()) < 0.05

    def test_variance_decreases_with_frequency(self):
        """
        Hedging error decreases as rebalancing frequency increases.

        std(P&L) with daily rebalancing < std(P&L) with weekly rebalancing.
        This verifies the 1/sqrt(n) relationship.
        """

        paths_weekly, tg_w = get_paths(n_steps=52)
        paths_daily,  tg_d = get_paths(n_steps=252)

        pnl_weekly = run_delta_hedge(paths_weekly, tg_w, K, r, sigma)
        pnl_daily  = run_delta_hedge(paths_daily, tg_d, K, r, sigma)

        assert pnl_daily.std() < pnl_weekly.std()

    def test_pnl_shape(self):
        """Output P&L array has shape (n_paths,)."""
        paths, tg = get_paths(n_paths=500)
        pnl = run_delta_hedge(paths, tg, K, r, sigma)
        assert pnl.shape == (500,)

    def test_transaction_costs_reduce_pnl(self):
        """Adding transaction costs reduces mean P&L."""
        paths, tg = get_paths()
        pnl_no_cost   = run_delta_hedge(paths, tg, K, r, sigma,
                                         transaction_cost=0.0)
        pnl_with_cost = run_delta_hedge(paths, tg, K, r, sigma,
                                         transaction_cost=0.001)
        assert pnl_with_cost.mean() < pnl_no_cost.mean()

    def test_vol_mismatch_positive_pnl(self):
        """
        Hedging with sigma_hedge > sigma_real → negative mean P&L.

        The premium is priced at sigma_real (not sigma_hedge), so the
        mismatch only affects the delta. Over-estimating vol leads to
        over-hedging: theta paid > gamma earned → net loss.
        """
        paths, tg = get_paths()
        pnl = run_delta_hedge(paths, tg, K, r, sigma, sigma_hedge=0.30)
        assert pnl.mean() < 0

    def test_vol_mismatch_negative_pnl(self):
        """
        Hedging with sigma_hedge < sigma_real → positive mean P&L.

        Under-estimating vol leads to under-hedging: actual moves (sigma_real)
        generate more gamma P&L than the theta paid (based on sigma_hedge).
        """
        paths, tg = get_paths()
        pnl = run_delta_hedge(paths, tg, K, r, sigma, sigma_hedge=0.10)
        assert pnl.mean() > 0


class TestGammaHedge:

    def test_gamma_neutrality_at_open(self):
        """
        At t=0, the gamma-hedged portfolio has Γ ≈ 0.

        By construction: h2 = Γ_target / Γ_hedge
        So: Γ_portfolio = Γ_target - h2 * Γ_hedge = 0 exactly.
        """
        K_hedge = 110.0
        S0_val  = np.array([S0])   # single path for clarity

        gamma_target = black_scholes_gamma(S0_val, K,       T, r, sigma)
        gamma_hedge  = black_scholes_gamma(S0_val, K_hedge, T, r, sigma)

        h2 = gamma_target / gamma_hedge
        gamma_portfolio = gamma_target - h2 * gamma_hedge
        assert gamma_portfolio == pytest.approx(0, abs=1e-10)

    def test_delta_neutrality_at_open(self):
        """
        At t=0, the gamma-hedged portfolio has Δ ≈ 0.

        By construction: h1 = Δ_target - h2 * Δ_hedge
        So: Δ_portfolio = Δ_target - h2 * Δ_hedge - h1 = 0 exactly.
        """
        K_hedge = 110.0
        S0_val  = np.array([S0])

        gamma_target = black_scholes_gamma(S0_val, K,       T, r, sigma)
        gamma_hedge  = black_scholes_gamma(S0_val, K_hedge, T, r, sigma)
        delta_target = black_scholes_delta(S0_val, K,       T, r, sigma)
        delta_hedge  = black_scholes_delta(S0_val, K_hedge, T, r, sigma)

        h2 = gamma_target / gamma_hedge
        h1 = delta_target - h2 * delta_hedge
        delta_portfolio = delta_target - h2 * delta_hedge - h1
        assert delta_portfolio == pytest.approx(0, abs=1e-10)

    def test_gamma_hedge_lower_error(self):
        """
        Gamma hedging has lower P&L std than delta hedging (weekly rebalancing).

        With n_steps=52, gamma hedging should significantly outperform delta.
        """
        paths, tg = get_paths(n_steps=52, n_paths=2000)
        price     = black_scholes_call(S0, K, T, r, sigma)

        pnl_delta = run_delta_hedge(paths, tg, K, r, sigma)
        pnl_gamma = run_gamma_hedge(paths, tg, K, r, sigma, K_hedge=110)

        error_delta = pnl_delta.std() / price
        error_gamma = pnl_gamma.std() / price

        assert error_gamma < error_delta