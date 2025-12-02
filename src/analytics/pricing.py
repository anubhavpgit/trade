"""
Option pricing models: Black-Scholes and extensions.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

from src.data.models import OptionType


@dataclass
class PricingResult:
    """Result from option pricing."""

    price: float
    delta: float
    gamma: float
    theta: float
    vega: float
    rho: float


class BlackScholes:
    """Black-Scholes option pricing model."""

    @staticmethod
    def d1(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d1 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate d2 parameter."""
        if T <= 0 or sigma <= 0:
            return 0.0
        return BlackScholes.d1(S, K, T, r, sigma) - sigma * np.sqrt(T)

    @staticmethod
    def price(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> float:
        """
        Calculate Black-Scholes option price.

        Args:
            S: Spot price
            K: Strike price
            T: Time to expiry (years)
            r: Risk-free rate
            sigma: Implied volatility
            option_type: Call or Put

        Returns:
            Option price
        """
        if T <= 0:
            # At expiry
            if option_type == OptionType.CALL:
                return max(0, S - K)
            else:
                return max(0, K - S)

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(0, price)

    @staticmethod
    def delta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> float:
        """Calculate option delta."""
        if T <= 0:
            if option_type == OptionType.CALL:
                return 1.0 if S > K else 0.0
            else:
                return -1.0 if S < K else 0.0

        d1 = BlackScholes.d1(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            return norm.cdf(d1)
        else:
            return norm.cdf(d1) - 1

    @staticmethod
    def gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option gamma (same for calls and puts)."""
        if T <= 0 or sigma <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return norm.pdf(d1) / (S * sigma * np.sqrt(T))

    @staticmethod
    def theta(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> float:
        """Calculate option theta (per day)."""
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        d2 = BlackScholes.d2(S, K, T, r, sigma)

        term1 = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

        if option_type == OptionType.CALL:
            term2 = -r * K * np.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * np.exp(-r * T) * norm.cdf(-d2)

        # Convert to daily theta
        return (term1 + term2) / 365

    @staticmethod
    def vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
        """Calculate option vega (per 1% change in volatility)."""
        if T <= 0:
            return 0.0

        d1 = BlackScholes.d1(S, K, T, r, sigma)
        return S * norm.pdf(d1) * np.sqrt(T) / 100

    @staticmethod
    def rho(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> float:
        """Calculate option rho (per 1% change in rate)."""
        if T <= 0:
            return 0.0

        d2 = BlackScholes.d2(S, K, T, r, sigma)

        if option_type == OptionType.CALL:
            return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

    @staticmethod
    def implied_volatility(
        price: float,
        S: float,
        K: float,
        T: float,
        r: float,
        option_type: OptionType,
        max_iterations: int = 100,
        precision: float = 1e-6,
    ) -> Optional[float]:
        """
        Calculate implied volatility using Brent's method.

        Returns None if IV cannot be found.
        """
        if T <= 0 or price <= 0:
            return None

        # Check for intrinsic value violations
        if option_type == OptionType.CALL:
            intrinsic = max(0, S - K * np.exp(-r * T))
        else:
            intrinsic = max(0, K * np.exp(-r * T) - S)

        if price < intrinsic:
            return None

        def objective(sigma):
            return BlackScholes.price(S, K, T, r, sigma, option_type) - price

        try:
            iv = brentq(objective, 0.001, 5.0, maxiter=max_iterations, xtol=precision)
            return iv
        except (ValueError, RuntimeError):
            return None

    @staticmethod
    def full_greeks(
        S: float,
        K: float,
        T: float,
        r: float,
        sigma: float,
        option_type: OptionType,
    ) -> PricingResult:
        """Calculate all Greeks at once."""
        return PricingResult(
            price=BlackScholes.price(S, K, T, r, sigma, option_type),
            delta=BlackScholes.delta(S, K, T, r, sigma, option_type),
            gamma=BlackScholes.gamma(S, K, T, r, sigma),
            theta=BlackScholes.theta(S, K, T, r, sigma, option_type),
            vega=BlackScholes.vega(S, K, T, r, sigma),
            rho=BlackScholes.rho(S, K, T, r, sigma, option_type),
        )


class OptionPricer:
    """High-level option pricer with multiple model support."""

    def __init__(self, model: str = "black_scholes"):
        self.model = model
        self._pricer = BlackScholes()

    def price_option(
        self,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float,
        option_type: OptionType,
    ) -> PricingResult:
        """Price an option and compute all Greeks."""
        return self._pricer.full_greeks(
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            sigma=volatility,
            option_type=option_type,
        )

    def calculate_iv(
        self,
        market_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        risk_free_rate: float,
        option_type: OptionType,
    ) -> Optional[float]:
        """Calculate implied volatility from market price."""
        return self._pricer.implied_volatility(
            price=market_price,
            S=spot,
            K=strike,
            T=time_to_expiry,
            r=risk_free_rate,
            option_type=option_type,
        )
