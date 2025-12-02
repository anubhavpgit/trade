"""
Tests for Greeks calculation module.
"""

import pytest
from datetime import date, timedelta

from src.analytics.greeks import GreeksCalculator, PortfolioGreeks
from src.data.models import OptionContract, OptionType, Position


class TestGreeksCalculator:
    """Tests for GreeksCalculator."""

    @pytest.fixture
    def calculator(self):
        return GreeksCalculator(risk_free_rate=0.05)

    @pytest.fixture
    def sample_call(self):
        return OptionContract(
            symbol="SPY240101C450",
            underlying="SPY",
            option_type=OptionType.CALL,
            strike=450.0,
            expiration=date.today() + timedelta(days=30),
            bid=5.0,
            ask=5.20,
            implied_volatility=0.20,
        )

    @pytest.fixture
    def sample_put(self):
        return OptionContract(
            symbol="SPY240101P450",
            underlying="SPY",
            option_type=OptionType.PUT,
            strike=450.0,
            expiration=date.today() + timedelta(days=30),
            bid=4.5,
            ask=4.70,
            implied_volatility=0.20,
        )

    def test_calculate_option_greeks(self, calculator, sample_call):
        """Test Greeks calculation for single option."""
        result = calculator.calculate_option_greeks(sample_call, spot_price=450.0)

        assert result.delta is not None
        assert result.gamma is not None
        assert result.theta is not None
        assert result.vega is not None
        assert result.rho is not None

        # ATM call delta should be around 0.5
        assert 0.4 < result.delta < 0.6

    def test_portfolio_greeks_single_position(self, calculator, sample_call):
        """Test portfolio Greeks with single position."""
        position = Position(
            contract=sample_call,
            quantity=10,
            avg_cost=5.10,
        )

        greeks = calculator.calculate_portfolio_greeks(
            positions=[position],
            spot_price=450.0,
        )

        assert isinstance(greeks, PortfolioGreeks)
        assert greeks.num_positions == 1
        # 10 contracts * 100 multiplier * ~0.5 delta
        assert 400 < greeks.portfolio_delta < 600

    def test_portfolio_greeks_hedge(self, calculator, sample_call, sample_put):
        """Test delta-neutral portfolio."""
        call_pos = Position(contract=sample_call, quantity=10, avg_cost=5.10)
        put_pos = Position(contract=sample_put, quantity=10, avg_cost=4.60)

        greeks = calculator.calculate_portfolio_greeks(
            positions=[call_pos, put_pos],
            spot_price=450.0,
        )

        # ATM straddle should have near-zero delta
        assert abs(greeks.portfolio_delta) < 100

    def test_hedge_shares_calculation(self, calculator):
        """Test hedge shares calculation."""
        shares = calculator.calculate_hedge_shares(current_delta=500, target_delta=0)
        assert shares == -500

        shares = calculator.calculate_hedge_shares(current_delta=-300, target_delta=0)
        assert shares == 300

    def test_hedge_cost(self, calculator):
        """Test hedge cost calculation."""
        cost = calculator.delta_hedge_cost(
            shares=100,
            spot_price=450.0,
            transaction_cost_bps=5.0,
        )
        # 100 * 450 * 5/10000 = 22.50
        assert abs(cost - 22.50) < 0.01
