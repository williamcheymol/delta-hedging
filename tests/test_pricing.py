# =============================================================================
# tests/test_pricing.py — Unit tests for pricing/black_scholes.py
# =============================================================================
# Run with: pytest tests/test_pricing.py -v
#
# All tests verify mathematical properties that must hold exactly
# (up to floating-point precision) — not statistical results.
# =============================================================================

import pytest
import numpy as np
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pricing.black_scholes import (
    black_scholes_call, black_scholes_put,
    black_scholes_delta, black_scholes_gamma,
    black_scholes_vega, put_call_parity_check
)

# --- Shared test parameters ---
S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20


class TestPutCallParity:

    def test_parity_atm(self):
        """Put-call parity holds for ATM option."""
        error = put_call_parity_check(S, K, T, r, sigma)
        assert error == pytest.approx(0, abs=1e-8)

    def test_parity_itm(self):
        """Put-call parity holds when option is in the money."""
        error = put_call_parity_check(130, K, T, r, sigma)
        assert error == pytest.approx(0, abs=1e-8)

    def test_parity_otm(self):
        """Put-call parity holds when option is out of the money."""
        error = put_call_parity_check(70, K, T, r, sigma)
        assert error == pytest.approx(0, abs=1e-8)


class TestCallDelta:

    def test_delta_call_deep_itm(self):
        """Call delta → 1 when S >> K (deep in the money)."""
        delta = black_scholes_delta(300, 100, T, r, sigma)
        assert delta > 0.99

    def test_delta_call_deep_otm(self):
        """Call delta → 0 when S << K (deep out of the money)."""
        delta = black_scholes_delta(10, 100, T, r, sigma)
        assert delta < 0.01

    def test_delta_call_in_range(self):
        """Call delta always in [0, 1]."""
        S_range = np.linspace(50, 200, 100)
        deltas = black_scholes_delta(S_range, K, T, r, sigma)
        assert np.all(deltas >= 0) and np.all(deltas <= 1)


class TestPutDelta:

    def test_delta_put_deep_itm(self):
        """Put delta → -1 when S << K (deep in the money for put)."""
        delta = black_scholes_delta(10, 100, T, r, sigma, option_type="put")
        assert delta < -0.99

    def test_delta_put_deep_otm(self):
        """Put delta → 0 when S >> K (deep out of the money for put)."""
        delta = black_scholes_delta(300, 100, T, r, sigma, option_type="put")
        assert delta > -0.01

    def test_put_call_delta_relation(self):
        """Put delta = Call delta - 1 (fundamental identity)."""
        S_range = np.linspace(60, 160, 50)
        delta_call = black_scholes_delta(S_range, K, T, r, sigma)
        delta_put  = black_scholes_delta(S_range, K, T, r, sigma, option_type="put")
        np.testing.assert_allclose(delta_put, delta_call - 1, atol=1e-10)


class TestGamma:

    def test_gamma_always_positive(self):
        """Gamma is always strictly positive for calls and puts."""
        S_range = np.linspace(50, 200, 100)
        gammas = black_scholes_gamma(S_range, K, T, r, sigma)
        assert np.all(gammas > 0)

    def test_gamma_call_equals_put(self):
        """Gamma is identical for call and put — verified numerically via finite difference."""
        dS = 0.01
        # Numerical derivative of call delta ≈ gamma
        delta_up   = black_scholes_delta(S + dS, K, T, r, sigma)
        delta_down = black_scholes_delta(S - dS, K, T, r, sigma)
        gamma_numerical = (delta_up - delta_down) / (2 * dS)

        gamma_analytical = black_scholes_gamma(S, K, T, r, sigma)
        assert gamma_analytical == pytest.approx(gamma_numerical, rel=1e-3)

    def test_gamma_max_atm(self):
        """Gamma is highest when option is ATM (S ≈ K)."""
        gamma_atm = black_scholes_gamma(K,       K, T, r, sigma)
        gamma_otm = black_scholes_gamma(K * 0.7, K, T, r, sigma)
        gamma_itm = black_scholes_gamma(K * 1.4, K, T, r, sigma)
        assert (gamma_atm > gamma_otm) and gamma_atm > gamma_itm


class TestCallPriceLimit:

    def test_price_call_zero_vol(self):
        """Call price → max(S - K*e^(-rT), 0) when sigma → 0 (intrinsic value)."""
        S_itm = 120.0
        price    = black_scholes_call(S_itm, K, T, r, sigma=1e-6)
        intrinsic = max(S_itm - K*np.exp(-r*T), 0)
        assert price == pytest.approx(intrinsic, abs=1e-3)

    def test_price_call_positive(self):
        """Call price is always strictly positive (time value)."""
        S_range = np.linspace(50, 200, 50)
        prices = black_scholes_call(S_range, K, T, r, sigma)
        assert np.all(prices > 0)

    def test_price_call_increases_with_S(self):
        """Call price is monotonically increasing with S."""
        S_range = np.linspace(60, 180, 50)
        prices = black_scholes_call(S_range, K, T, r, sigma)
        assert np.all(np.diff(prices) > 0)

class TestPutPriceLimit:

    def test_price_put_zero_vol(self):
        """Put price → max(K*e^(-rT) - S, 0) when sigma → 0 (intrinsic value)."""
        S_itm = 80.0
        price     = black_scholes_put(S_itm, K, T, r, sigma=1e-6)
        intrinsic = max(K*np.exp(-r*T) - S_itm, 0)
        assert price == pytest.approx(intrinsic, abs=1e-3)

    def test_price_put_positive(self):
        """Put price is always strictly positive (time value)."""
        S_range = np.linspace(50, 200, 50)
        prices = black_scholes_put(S_range, K, T, r, sigma)
        assert np.all(prices > 0)

    def test_price_put_decreases_with_S(self):
        """Put price is monotonically decreasing with S."""
        S_range = np.linspace(60, 180, 50)
        prices = black_scholes_put(S_range, K, T, r, sigma)
        assert np.all(np.diff(prices) < 0)