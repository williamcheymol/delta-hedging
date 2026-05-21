# =============================================================================
# pricing/black_scholes.py — Black-Scholes pricing and Greeks
# =============================================================================
# European call and put pricing under Black-Scholes assumptions.
#
# Call formula:   C = S·N(d1) − K·e^(−rT)·N(d2)
# Put formula:    P = K·e^(−rT)·N(−d2) − S·N(−d1)
#
# where:
#   d1 = [ln(S/K) + (r + σ²/2)·T] / (σ·√T)
#   d2 = d1 − σ·√T
#   N  = cumulative standard normal CDF
#   N' = standard normal PDF
# =============================================================================

import numpy as np
from scipy.stats import norm


# =============================================================================
# Helper — compute d1 and d2 (shared by all functions)
# =============================================================================

def _d1_d2(S, K, T, r, sigma):
    """Compute d1 and d2 from Black-Scholes parameters."""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


# =============================================================================
# Pricing
# =============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """
    Compute the Black-Scholes price of a European call option.

    Parameters
    ----------
    S, K, T, r, sigma : float or np.ndarray

    Returns
    -------
    float — call price C
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def black_scholes_put(S, K, T, r, sigma):
    """
    Compute the Black-Scholes price of a European put option.

    Put formula: P = K·e^(−rT)·N(−d2) − S·N(−d1)

    Parameters
    ----------
    S, K, T, r, sigma : float or np.ndarray

    Returns
    -------
    float — put price P
    """
    d1, d2 = _d1_d2(S, K, T, r, sigma)
    return K * np.exp(-r * T) * (1 - norm.cdf(d2)) - S * (1 - norm.cdf(d1))


# =============================================================================
# Delta — dC/dS and dP/dS
# =============================================================================

def black_scholes_delta(S, K, T, r, sigma, option_type="call"):
    """
    Compute the delta of a European option.

    Call delta = N(d1)       ∈ [0, 1]
    Put  delta = N(d1) − 1   ∈ [−1, 0]

    Parameters
    ----------
    option_type : str — "call" or "put"
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    if option_type == "call":
        return norm.cdf(d1)
    elif option_type == "put":
        return norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")


# =============================================================================
# Gamma — d²C/dS² (same for call and put)
# =============================================================================

def black_scholes_gamma(S, K, T, r, sigma):
    """
    Compute the gamma of a European option (identical for call and put).

    Gamma = N'(d1) / (S·σ·√T)

    Gamma measures the rate of change of delta. It is highest when the option
    is at-the-money and close to expiry — exactly when hedging is hardest.

    Parameters
    ----------
    S, K, T, r, sigma : float or np.ndarray

    Returns
    -------
    float — gamma (always positive)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


# =============================================================================
# Vega — dC/dσ (same for call and put)
# =============================================================================

def black_scholes_vega(S, K, T, r, sigma):
    """
    Compute the vega of a European option (identical for call and put).

    Vega = S·N'(d1)·√T

    Vega measures sensitivity to volatility. A vega of 0.20 means the option
    gains 0.20€ in value for each +1% increase in implied volatility.

    Parameters
    ----------
    S, K, T, r, sigma : float or np.ndarray

    Returns
    -------
    float — vega (always positive)
    """
    d1, _ = _d1_d2(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T)


# =============================================================================
# Put-call parity verification
# =============================================================================

def put_call_parity_check(S, K, T, r, sigma):
    """
    Verify put-call parity: C − P = S − K·e^(−rT)

    This should hold exactly (up to floating-point precision).
    Returns the absolute difference — should be < 1e-10.

    Parameters
    ----------
    S, K, T, r, sigma : float

    Returns
    -------
    float — |C − P − (S − K·e^(−rT))|
    """
    C = black_scholes_call(S, K, T, r, sigma)
    P = black_scholes_put(S, K, T, r, sigma)
    lhs = C - P
    rhs = S - K * np.exp(-r * T)
    return abs(lhs - rhs)
