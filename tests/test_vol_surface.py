# =============================================================================
# tests/test_vol_surface.py — Unit tests for analysis/vol_surface.py
# =============================================================================
# Run with: pytest tests/test_vol_surface.py -v
#
# All tests use a synthetic DataFrame so no market data file is needed.
# The fixture builds a flat vol surface (σ = 0.20 everywhere), which gives
# a known ground truth for every assertion.
# =============================================================================

import pytest
import numpy as np
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from analysis.vol_surface import VolSurface


# =============================================================================
# Shared fixture — flat vol surface (σ = 0.20 at all strikes and maturities)
# =============================================================================

def make_surface(iv: float = 0.20, spot: float = 100.0,
                 n_strikes: int = 20, n_maturities: int = 5,
                 smoothing: float = 0.5) -> VolSurface:
    """
    Build a VolSurface from a synthetic grid with constant implied vol.

    A flat surface means get_iv(K, T) ≈ iv for any (K, T) inside the grid,
    which gives a clean ground truth for tolerance-based assertions.
    """
    strikes    = np.linspace(80, 120, n_strikes)
    maturities = np.linspace(0.05, 1.0, n_maturities)

    rows = []
    for T_val in maturities:
        for K_val in strikes:
            opt_type = "call" if K_val >= spot else "put"
            rows.append({"strike": K_val, "T": T_val,
                         "implied_vol": iv, "option_type": opt_type})

    df = pd.DataFrame(rows)
    return VolSurface(df, spot=spot, smoothing=smoothing)


@pytest.fixture(scope="module")
def flat_surface():
    return make_surface(iv=0.20, spot=100.0)


# =============================================================================
# Construction
# =============================================================================

class TestConstruction:

    def test_df_stored(self, flat_surface):
        """VolSurface stores the filtered DataFrame."""
        assert flat_surface.df is not None
        assert len(flat_surface.df) > 0

    def test_interpolator_fitted(self, flat_surface):
        """RBF interpolator is fitted after construction."""
        assert flat_surface._interp is not None

    def test_spot_stored(self, flat_surface):
        """Spot price is stored correctly."""
        assert flat_surface.spot == pytest.approx(100.0)


# =============================================================================
# get_iv — single point query
# =============================================================================

class TestGetIV:

    def test_returns_positive(self, flat_surface):
        """get_iv always returns a strictly positive value."""
        iv = flat_surface.get_iv(strike=100.0, T=0.5)
        assert iv > 0

    def test_flat_surface_approx(self, flat_surface):
        """On a flat surface, get_iv ≈ 0.20 at any interior point."""
        iv = flat_surface.get_iv(strike=100.0, T=0.5)
        assert iv == pytest.approx(0.20, abs=0.02)

    def test_clamps_short_maturity(self, flat_surface):
        """T below the observed range is clamped — no extrapolation error."""
        iv = flat_surface.get_iv(strike=100.0, T=1e-6)
        assert iv > 0

    def test_clamps_long_maturity(self, flat_surface):
        """T above the observed range is clamped — no extrapolation error."""
        iv = flat_surface.get_iv(strike=100.0, T=100.0)
        assert iv > 0

    def test_floor_at_1e4(self):
        """get_iv never returns below 1e-4 even with a noisy surface."""
        # Build a near-zero vol surface to trigger the floor
        surface = make_surface(iv=1e-5, spot=100.0)
        iv = surface.get_iv(strike=100.0, T=0.5)
        assert iv >= 1e-4


# =============================================================================
# get_iv_batch — vectorised single-maturity query
# =============================================================================

class TestGetIVBatch:

    def test_output_shape(self, flat_surface):
        """get_iv_batch returns an array with the same length as strikes."""
        strikes = np.linspace(90, 110, 50)
        ivs = flat_surface.get_iv_batch(strikes, T=0.5)
        assert ivs.shape == (50,)

    def test_all_positive(self, flat_surface):
        """All returned IVs are strictly positive."""
        strikes = np.linspace(85, 115, 30)
        ivs = flat_surface.get_iv_batch(strikes, T=0.3)
        assert np.all(ivs > 0)

    def test_consistent_with_get_iv(self, flat_surface):
        """Batch query matches scalar query within interpolation tolerance."""
        strikes = np.array([95.0, 100.0, 105.0])
        ivs_batch = flat_surface.get_iv_batch(strikes, T=0.5)
        ivs_scalar = np.array([flat_surface.get_iv(k, 0.5) for k in strikes])
        np.testing.assert_allclose(ivs_batch, ivs_scalar, atol=1e-6)


# =============================================================================
# build_fast_grid — RegularGridInterpolator
# =============================================================================

class TestBuildFastGrid:

    def test_returns_callable(self, flat_surface):
        """build_fast_grid returns a callable interpolator."""
        grid = flat_surface.build_fast_grid()
        assert callable(grid)

    def test_grid_output_close_to_rbf(self, flat_surface):
        """
        Fast grid values are close to the underlying RBF at the same points.

        The grid uses bilinear interpolation on a 200×100 precomputed mesh,
        so small deviations from the RBF are expected — but not large ones.
        """
        grid = flat_surface.build_fast_grid(n_strikes=200, n_maturities=100)

        test_points = [(0.3, 95.0), (0.5, 100.0), (0.7, 105.0)]
        for T_val, K_val in test_points:
            iv_rbf  = flat_surface.get_iv(K_val, T_val)
            iv_grid = float(grid([[T_val, K_val]]).item())
            assert iv_grid == pytest.approx(iv_rbf, abs=0.01), (
                f"Grid vs RBF mismatch at K={K_val}, T={T_val}: "
                f"{iv_grid:.4f} vs {iv_rbf:.4f}"
            )

    def test_grid_all_positive(self, flat_surface):
        """All grid values are strictly positive."""
        grid = flat_surface.build_fast_grid()
        T_vals = np.linspace(0.1, 0.9, 10)
        K_vals = np.linspace(85, 115, 10)
        pts = np.array([[t, k] for t in T_vals for k in K_vals])
        ivs = grid(pts)
        assert np.all(ivs > 0)


# =============================================================================
# get_iv_surface — full paths matrix query
# =============================================================================

class TestGetIVSurface:

    def test_output_shape(self, flat_surface):
        """get_iv_surface returns an array of shape (n_steps+1, n_paths)."""
        n_steps, n_paths = 10, 50
        rng   = np.random.default_rng(0)
        paths = 100.0 * np.exp(rng.normal(0, 0.01, (n_steps + 1, n_paths)).cumsum(axis=0))
        tg    = np.linspace(0, 1.0, n_steps + 1)

        ivs = flat_surface.get_iv_surface(paths, tg)
        assert ivs.shape == (n_steps + 1, n_paths)

    def test_all_positive(self, flat_surface):
        """All returned IVs are strictly positive."""
        n_steps, n_paths = 10, 30
        rng   = np.random.default_rng(1)
        paths = 100.0 * np.exp(rng.normal(0, 0.01, (n_steps + 1, n_paths)).cumsum(axis=0))
        tg    = np.linspace(0, 1.0, n_steps + 1)

        ivs = flat_surface.get_iv_surface(paths, tg)
        assert np.all(ivs > 0)
