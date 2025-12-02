"""
Tests for option pricing module.
"""

import pytest
import numpy as np
from src.analytics.pricing import BlackScholes, OptionPricer
from src.data.models import OptionType


class TestBlackScholes:
    """Tests for Black-Scholes pricing."""

    def test_call_price_atm(self):
        """Test ATM call option price."""
        price = BlackScholes.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )
        # ATM call with 20% vol, 1 year should be around 10
        assert 8 < price < 12

    def test_put_price_atm(self):
        """Test ATM put option price."""
        price = BlackScholes.price(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.PUT
        )
        # Should be less than call due to cost of carry
        assert 5 < price < 10

    def test_put_call_parity(self):
        """Test put-call parity holds."""
        S, K, T, r, sigma = 100, 100, 1.0, 0.05, 0.20

        call = BlackScholes.price(S, K, T, r, sigma, OptionType.CALL)
        put = BlackScholes.price(S, K, T, r, sigma, OptionType.PUT)

        # C - P = S - K*exp(-rT)
        lhs = call - put
        rhs = S - K * np.exp(-r * T)

        assert abs(lhs - rhs) < 0.01

    def test_call_delta_range(self):
        """Test call delta is in [0, 1]."""
        delta = BlackScholes.delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )
        assert 0 <= delta <= 1

    def test_put_delta_range(self):
        """Test put delta is in [-1, 0]."""
        delta = BlackScholes.delta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.PUT
        )
        assert -1 <= delta <= 0

    def test_gamma_positive(self):
        """Test gamma is always positive."""
        gamma = BlackScholes.gamma(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert gamma > 0

    def test_vega_positive(self):
        """Test vega is always positive."""
        vega = BlackScholes.vega(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
        assert vega > 0

    def test_theta_call_negative(self):
        """Test call theta is typically negative (time decay)."""
        theta = BlackScholes.theta(
            S=100, K=100, T=1.0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )
        assert theta < 0

    def test_implied_volatility(self):
        """Test IV recovery."""
        true_iv = 0.25
        price = BlackScholes.price(
            S=100, K=100, T=1.0, r=0.05, sigma=true_iv,
            option_type=OptionType.CALL
        )

        recovered_iv = BlackScholes.implied_volatility(
            price=price, S=100, K=100, T=1.0, r=0.05,
            option_type=OptionType.CALL
        )

        assert abs(recovered_iv - true_iv) < 0.001

    def test_expired_option(self):
        """Test pricing at expiry."""
        # ITM call at expiry
        call_itm = BlackScholes.price(
            S=105, K=100, T=0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )
        assert call_itm == 5

        # OTM call at expiry
        call_otm = BlackScholes.price(
            S=95, K=100, T=0, r=0.05, sigma=0.20,
            option_type=OptionType.CALL
        )
        assert call_otm == 0


class TestOptionPricer:
    """Tests for high-level OptionPricer."""

    def test_full_greeks(self):
        """Test all Greeks are computed."""
        pricer = OptionPricer()
        result = pricer.price_option(
            spot=100,
            strike=100,
            time_to_expiry=1.0,
            risk_free_rate=0.05,
            volatility=0.20,
            option_type=OptionType.CALL,
        )

        assert result.price > 0
        assert 0 <= result.delta <= 1
        assert result.gamma > 0
        assert result.theta < 0
        assert result.vega > 0
