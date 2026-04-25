# =============================================================================
# tests/test_simulation.py — Unit tests for simulation/monte_carlo.py
# =============================================================================
# Run with: pytest tests/test_simulation.py -v
#
# Tests here verify structural properties (shape, dtype) and
# statistical properties (law of large numbers). Statistical tests
# use large n_paths so convergence is reliable.
# =============================================================================

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from simulation.monte_carlo import simulate_gbm_paths

# --- Shared parameters ---
S0, r, sigma, T = 100.0, 0.05, 0.20, 1.0
N_STEPS, N_PATHS = 252, 5000   # large n_paths for statistical tests


class TestOutputShape:

    def test_paths_shape(self):
        """Output paths array has shape (n_steps+1, n_paths)."""
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=42,
                       antithetic=False) 
        assert paths.shape == (N_STEPS+1, N_PATHS)

    def test_time_grid_shape(self):
        """Time grid has shape (n_steps+1,) and spans [0, T]."""
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=42,
                       antithetic=False)
        assert tg.shape == (N_STEPS+1,)
        assert tg[0] == pytest.approx(0.0)
        assert tg[-1] == pytest.approx(T)

    def test_initial_price(self):
        """First row of paths equals S0 for all paths."""
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS)
        assert np.all(paths[0] == pytest.approx(S0))

    def test_prices_positive(self):
        """All simulated prices are strictly positive (GBM property)."""
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS)
        assert np.all(paths > 0)


class TestStatisticalProperties:

    def test_mean_log_return(self):
        """
        Mean log-return per step ≈ (r - sigma²/2) * dt.

        This verifies the GBM drift is correctly implemented.
        Under risk-neutral measure: E[log(S(t+dt)/S(t))] = (r - σ²/2)*dt
        """
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=0)
        dt = T / N_STEPS

        log_returns = np.log(paths[1:] / paths[:-1])
        expected    = (r - sigma**2 / 2) * dt
        assert log_returns.mean() == pytest.approx(expected, abs=1e-3)

    def test_terminal_mean(self):
        """
        Mean terminal price ≈ S0 * exp(r * T).

        Under risk-neutral measure, E[S(T)] = S0 * e^(rT).
        """
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, N_PATHS, seed=0)

        expected = S0 * np.exp(r * T)
        assert paths[-1].mean() == pytest.approx(expected, rel=0.05)

    def test_reproducibility(self):
        """Same seed produces identical paths."""
        paths_1, _ = simulate_gbm_paths(S0, r, sigma, T, 52, 100, seed=42)
        paths_2, _ = simulate_gbm_paths(S0, r, sigma, T, 52, 100, seed=42)
        assert np.array_equal(paths_1, paths_2)

    def test_different_seeds_differ(self):
        """Different seeds produce different paths."""
        paths_1, _ = simulate_gbm_paths(S0, r, sigma, T, 52, 100, seed=0)
        paths_2, _ = simulate_gbm_paths(S0, r, sigma, T, 52, 100, seed=1)
        assert not np.array_equal(paths_1, paths_2)


class TestAntitheticVariates:

    def test_antithetic_shape(self):
        """Antithetic output has same shape as standard."""
        paths, tg = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, 1000,
                                       antithetic=True)
        assert paths.shape == (N_STEPS+1, 1000)

    def test_antithetic_pairs(self):
        """
        Antithetic paths come in symmetric pairs.

        For antithetic variates, log-returns of path i and path i+n_paths//2
        should sum to exactly 2*(r - sigma²/2)*dt (the two noise terms cancel).
        """
        n = 500
        paths, _ = simulate_gbm_paths(S0, r, sigma, T, N_STEPS, n,
                                      antithetic=True, seed=42)
        dt = T / N_STEPS

        log_ret_1 = np.log(paths[1:, :n//2] / paths[:-1, :n//2])
        log_ret_2 = np.log(paths[1:, n//2:] / paths[:-1, n//2:])
        expected  = 2 * (r - sigma**2 / 2) * dt
        np.testing.assert_allclose(log_ret_1 + log_ret_2, expected, atol=1e-10)
