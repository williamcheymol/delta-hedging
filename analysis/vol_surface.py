# =============================================================================
# analysis/vol_surface.py — Implied volatility surface from market data
# =============================================================================
#
# What is the vol surface?
#   The vol surface σ(K, T) is the 3D object that maps each (strike, maturity)
#   pair to its market-implied volatility. It is extracted from live option
#   prices via Black-Scholes inversion (done upstream in the Market Fetcher).
#
# Why do we need interpolation?
#   Market data is discrete — only certain strikes and maturities are quoted.
#   To hedge at any arbitrary (K, T), we need to interpolate between observed
#   points. We use scipy's RBF interpolator which handles scattered data well.
#
# Usage:
#   surface = VolSurface.from_csv("path/to/AAPL_options.csv", spot=287.51)
#   iv = surface.get_iv(strike=290, T=0.08)
#   fig = surface.plot_3d()
# =============================================================================

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import RBFInterpolator, RegularGridInterpolator
from datetime import date


class VolSurface:
    """
    Implied volatility surface σ(K, T) built from a market option chain.

    Attributes
    ----------
    df      : raw filtered DataFrame (strike, T, implied_vol)
    spot    : current spot price S (used to compute moneyness)
    _interp : fitted RBF interpolator — callable as _interp([[K, T]])
    """

    def __init__(self, df: pd.DataFrame, spot: float, smoothing: float = 2.0):
        self.spot      = spot
        self.df        = df
        self.smoothing = smoothing
        self._interp   = None
        self._fit()

    # -------------------------------------------------------------------------
    #  Construction
    # -------------------------------------------------------------------------

    @classmethod
    def from_csv(cls, path: str, spot: float,
                 iv_min: float = 0.05,
                 iv_max: float = 0.80,
                 smoothing: float = 2.0) -> "VolSurface":
        """
        Load an option chain CSV produced by the Market Fetcher and build
        the vol surface.

        Parameters
        ----------
        path    : path to {ticker}_options.csv
        spot    : current spot price S
        iv_min  : discard contracts with IV below this (noise / illiquid)
        iv_max  : discard contracts with IV above this (deep OTM artefacts)

        Returns
        -------
        VolSurface instance ready for interpolation and plotting
        """
        df = pd.read_csv(path)

        # Keep only the columns we need
        df = df[["strike", "T", "implied_vol", "option_type"]].dropna()

        # Filter out noisy contracts
        df = df[(df["implied_vol"] >= iv_min) & (df["implied_vol"] <= iv_max)]

        # Decide whether to use calls, puts, or both.
        # Using both gives more data points but can introduce put-call skew
        # artefacts near ATM. A common choice is OTM-only:
        # calls where strike > spot, puts where strike < spot.
        calls = df[(df["option_type"] == "call") & (df["strike"] >= spot)]
        puts  = df[(df["option_type"] == "put")  & (df["strike"] <= spot)]
        df    = pd.concat([calls, puts])

        if df.empty:
            raise ValueError(f"No valid data after filtering in '{path}'.")

        return cls(df, spot, smoothing=smoothing)

    # -------------------------------------------------------------------------
    #  Interpolation
    # -------------------------------------------------------------------------

    def _fit(self) -> None:
        """Fit a Radial Basis Function interpolator on (strike, T) → IV."""
        X = self.df[["strike", "T"]].values
        y = self.df["implied_vol"].values

        # Tune the smoothing parameter if the surface looks bumpy.
        # smoothing=0 → exact interpolation (can overfit noisy data)
        # smoothing=1 → smoother surface, may miss some market features
        self._interp = RBFInterpolator(X, y, smoothing=self.smoothing, kernel="thin_plate_spline")

    def get_iv(self, strike: float, T: float) -> float:
        """
        Query the surface at a single (strike, T) point.

        Parameters
        ----------
        strike : option strike price
        T      : time to maturity in years

        Returns
        -------
        float : interpolated implied volatility
        """
        T_min = self.df["T"].min()
        T_max = self.df["T"].max()
        T     = float(np.clip(T, T_min, T_max))  # clamp to observed range
        iv    = float(self._interp([[strike, T]]).item())
        return max(iv, 1e-4)  # clamp to avoid negative vol artefacts

    def build_fast_grid(self,
                        n_strikes: int = 200,
                        n_maturities: int = 100) -> RegularGridInterpolator:
        """
        Precompute the RBF surface on a regular (K, T) grid and return a
        RegularGridInterpolator for ultra-fast IV lookup.

        Why this matters
        ----------------
        RBFInterpolator evaluates IV at a point in O(n_train) time — fine for
        a handful of points, but too slow when called millions of times during
        a Monte Carlo hedging loop. RegularGridInterpolator uses precomputed
        bilinear interpolation: O(1) per point regardless of training set size.

        Typical speedup over direct RBF: 100-1000×.

        Parameters
        ----------
        n_strikes    : grid resolution along the strike axis (default 200)
        n_maturities : grid resolution along the maturity axis (default 100)

        Returns
        -------
        RegularGridInterpolator  callable as  grid([[T, K], ...])
        """
        k_min = self.df["strike"].quantile(0.02)
        k_max = self.df["strike"].quantile(0.98)
        t_min = self.df["T"].min()
        t_max = self.df["T"].max()

        k_grid = np.linspace(k_min, k_max, n_strikes)
        t_grid = np.linspace(t_min, t_max, n_maturities)

        KK, TT = np.meshgrid(k_grid, t_grid)           # (n_mat, n_str)
        pts    = np.column_stack([KK.ravel(), TT.ravel()])
        iv_grid = np.maximum(
            self._interp(pts).reshape(n_maturities, n_strikes),
            1e-4
        )

        return RegularGridInterpolator(
            (t_grid, k_grid), iv_grid,
            method="linear",
            bounds_error=False,
            fill_value=None,   # extrapolate at boundaries
        )

    def get_iv_surface(self, paths: np.ndarray,
                       time_grid: np.ndarray) -> np.ndarray:
        """
        Query IV for a full (n_steps+1, n_paths) paths matrix in ONE RBF call.

        Evaluating all (strike, T_rem) pairs at once is 100–300× faster than
        calling get_iv_batch() in a loop over time steps, because numpy/BLAS
        handles one large matrix multiplication far more efficiently than many
        small ones.

        Parameters
        ----------
        paths     : np.ndarray (n_steps+1, n_paths) — stock price paths
        time_grid : np.ndarray (n_steps+1,)         — time_grid[-1] = T_total

        Returns
        -------
        np.ndarray (n_steps+1, n_paths) — implied vol at every (step, path)
        """
        n_rows, n_paths = paths.shape
        T_total = time_grid[-1]
        T_rem   = T_total - time_grid                  # (n_steps+1,) remaining time

        T_min = self.df["T"].min()
        T_max = self.df["T"].max()

        # Flatten: (n_steps+1) × n_paths pairs
        all_S = paths.ravel()                          # each path's strike at each step
        all_T = np.repeat(np.clip(T_rem, T_min, T_max), n_paths)  # matching T_rem

        ivs = self._interp(np.column_stack([all_S, all_T]))
        return np.maximum(ivs.reshape(n_rows, n_paths), 1e-4)

    def get_iv_batch(self, strikes: np.ndarray, T: float) -> np.ndarray:
        """
        Query the surface for a vector of strikes at the same maturity T.

        Much faster than calling get_iv() in a loop because the RBF evaluates
        all points in a single vectorised call.

        Parameters
        ----------
        strikes : 1-D array of strike prices  (shape: n_paths,)
        T       : scalar time to maturity in years (same for all strikes)

        Returns
        -------
        np.ndarray (n_paths,) : interpolated implied volatilities, clamped ≥ 1e-4
        """
        T_min   = self.df["T"].min()
        T_max   = self.df["T"].max()
        T_clamp = np.clip(T, T_min, T_max)  # clamp to observed range — no extrapolation
        points  = np.column_stack([strikes, np.full(len(strikes), T_clamp)])
        ivs     = self._interp(points)
        return np.maximum(ivs, 1e-4)

    # -------------------------------------------------------------------------
    #  Visualisation
    # -------------------------------------------------------------------------

    def plot_3d(self,
                n_strikes: int = 40,
                n_maturities: int = 30,
                ticker: str = "") -> go.Figure:
        """
        Plot the implied volatility surface as an interactive 3D surface.

        Parameters
        ----------
        n_strikes    : grid resolution along the strike axis
        n_maturities : grid resolution along the maturity axis
        ticker       : used in the chart title

        Returns
        -------
        plotly Figure
        """
        # Build interpolation grid
        k_min = self.df["strike"].quantile(0.05)
        k_max = self.df["strike"].quantile(0.95)
        t_min = self.df["T"].min()
        t_max = self.df["T"].max()

        strikes    = np.linspace(k_min, k_max, n_strikes)
        maturities = np.linspace(t_min, t_max, n_maturities)

        KK, TT = np.meshgrid(strikes, maturities)
        points  = np.column_stack([KK.ravel(), TT.ravel()])
        IV      = self._interp(points).reshape(KK.shape) * 100  # → %
        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=maturities * 365,   # convert years → days for readability
            z=IV,
            colorscale="Viridis",
            colorbar=dict(title="IV (%)", tickfont=dict(color="white")),
            contours=dict(
                z=dict(show=True, usecolormap=True, highlightcolor="white",
                       project_z=True)
            ),
        )])

        fig.add_scatter3d(
            x=self.df["strike"],
            y=self.df["T"] * 365,
            z=self.df["implied_vol"] * 100,
            mode="markers",
            marker=dict(size=2, color="white", opacity=0.6),
            name="Market quotes",
        )

        fig.update_layout(
            title=f"{ticker}  |  Implied Volatility Surface",
            scene=dict(
                xaxis=dict(title="Strike (K)", backgroundcolor="#111111",
                           gridcolor="#2a2a2a", color="white"),
                yaxis=dict(title="Days to expiry", backgroundcolor="#111111",
                           gridcolor="#2a2a2a", color="white"),
                zaxis=dict(title="Implied Vol (%)", backgroundcolor="#111111",
                           gridcolor="#2a2a2a", color="white"),
                bgcolor="#111111",
            ),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color="white", family="monospace"),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        return fig

    def plot_smile(self, ticker: str = "") -> go.Figure:
        """
        Plot IV vs moneyness (K/S) for each available maturity — the smile.

        One line per maturity, x-axis = K/S (moneyness), y-axis = IV (%).
        ATM = moneyness 1.0, OTM puts on the left, OTM calls on the right.
        """
        palette = [
            "#00bfff", "#ffa500", "#ff4444", "#00e676",
            "#ff69b4", "#a259ff", "#ffff00",
        ]

        # Round T values to group contracts into clean maturity buckets
        self.df["T_round"] = self.df["T"].round(3)
        maturities = sorted(self.df["T_round"].unique())

        fig = go.Figure()

        for i, T_val in enumerate(maturities):
            slice_df = self.df[self.df["T_round"] == T_val].copy()
            slice_df["moneyness"] = slice_df["strike"] / self.spot
            slice_df = slice_df.sort_values("moneyness")

            days = round(T_val * 365)
            color = palette[i % len(palette)]

            fig.add_trace(go.Scatter(
                x=slice_df["moneyness"],
                y=slice_df["implied_vol"] * 100,
                mode="lines+markers",
                name=f"T = {days}d",
                line=dict(color=color, width=2),
                marker=dict(size=5, color=color),
            ))

        fig.add_vline(
            x=1.0, line_dash="dash", line_color="white",
            annotation_text="ATM", annotation_font_color="white",
        )

        fig.update_layout(
            title=f"{ticker}  |  Volatility Smile by Maturity",
            xaxis_title="Moneyness (K / S)",
            yaxis_title="Implied Vol (%)",
            xaxis=dict(tickformat=".2f", gridcolor="#2a2a2a", color="white"),
            yaxis=dict(gridcolor="#2a2a2a", color="white"),
            paper_bgcolor="#111111",
            plot_bgcolor="#111111",
            font=dict(color="white", family="monospace"),
            legend=dict(bgcolor="#1a1a1a", bordercolor="#333333", borderwidth=1),
            margin=dict(l=60, r=20, t=50, b=60),
        )

        return fig
