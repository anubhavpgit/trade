"""
Tests for risk metrics module.
"""

import pytest
import numpy as np

from src.analytics.risk_metrics import RiskMetricsEngine, RiskMetrics
from src.analytics.greeks import PortfolioGreeks
from src.data.models import MarketSnapshot, Position
from datetime import datetime


class TestRiskMetricsEngine:
    """Tests for RiskMetricsEngine."""

    @pytest.fixture
    def engine(self):
        return RiskMetricsEngine()

    @pytest.fixture
    def sample_greeks(self):
        return PortfolioGreeks(
            portfolio_delta=100.0,
            portfolio_gamma=0.05,
            portfolio_theta=-50.0,
            portfolio_vega=200.0,
            portfolio_rho=10.0,
            dollar_delta=45000.0,
            dollar_gamma=1000.0,
            num_positions=5,
            total_notional=100000.0,
        )

    @pytest.fixture
    def sample_snapshot(self):
        return MarketSnapshot(
            underlying="SPY",
            spot_price=450.0,
            timestamp=datetime.now(),
            positions=[],
            stock_position=None,
            option_chain=None,
            iv_surface=None,
        )

    def test_calculate_var(self, engine, sample_greeks):
        """Test VaR calculation."""
        var_95, var_99 = engine._calculate_var(sample_greeks, spot_price=450.0)

        assert var_95 > 0
        assert var_99 > var_95  # 99% VaR should be higher

    def test_calculate_expected_shortfall(self, engine, sample_greeks):
        """Test Expected Shortfall calculation."""
        es_95, es_99 = engine._calculate_expected_shortfall(
            sample_greeks, spot_price=450.0
        )

        assert es_95 > 0
        assert es_99 > es_95

    def test_monte_carlo_var(self, engine, sample_greeks):
        """Test Monte Carlo VaR."""
        var_95, var_99 = engine.monte_carlo_var(
            sample_greeks,
            spot_price=450.0,
            num_simulations=1000,
        )

        assert var_95 > 0
        assert var_99 > var_95

    def test_cornish_fisher_var(self, engine, sample_greeks):
        """Test Cornish-Fisher VaR with skewness."""
        # Normal distribution
        var_normal = engine.cornish_fisher_var(
            sample_greeks,
            spot_price=450.0,
            skewness=0.0,
            kurtosis=3.0,
        )

        # Left-skewed distribution (higher VaR)
        var_skewed = engine.cornish_fisher_var(
            sample_greeks,
            spot_price=450.0,
            skewness=-0.5,
            kurtosis=4.0,
        )

        # Skewed should generally have higher VaR
        assert var_skewed[0] != var_normal[0]

    def test_historical_returns(self, engine):
        """Test with historical returns."""
        returns = np.random.normal(0, 0.02, 252)
        engine.set_historical_returns(returns)

        vol = engine._calculate_realized_volatility()
        assert 0.1 < vol < 0.5  # Reasonable annualized vol

    def test_full_metrics(self, engine, sample_snapshot, sample_greeks):
        """Test complete metrics calculation."""
        # Need to set some portfolio value
        sample_snapshot.positions = []

        metrics = engine.calculate_metrics(sample_snapshot, sample_greeks)

        assert isinstance(metrics, RiskMetrics)
        assert metrics.var_95 >= 0
        assert metrics.es_95 >= 0
        assert 0 <= metrics.realized_volatility <= 1
