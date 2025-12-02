"""
Risk metrics calculation: VaR, Expected Shortfall, and related measures.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from loguru import logger

from src.config import settings
from src.data.models import MarketSnapshot
from src.analytics.greeks import PortfolioGreeks


@dataclass
class RiskMetrics:
    """Complete risk metrics for a portfolio."""

    # Value at Risk
    var_95: float
    var_99: float

    # Expected Shortfall (CVaR)
    es_95: float
    es_99: float

    # As percentages of portfolio
    var_percent: float
    es_percent: float

    # Volatility metrics
    realized_volatility: float
    iv_percentile: float

    # Drawdown
    current_drawdown: float
    max_drawdown: float
    drawdown_percent: float

    # Position metrics
    portfolio_value: float
    pnl_today: float

    def __str__(self) -> str:
        return (
            f"Risk Metrics:\n"
            f"  VaR 95%: ${self.var_95:,.0f} ({self.var_percent*100:.2f}%)\n"
            f"  ES 95%:  ${self.es_95:,.0f} ({self.es_percent*100:.2f}%)\n"
            f"  Realized Vol: {self.realized_volatility*100:.1f}%\n"
            f"  IV Percentile: {self.iv_percentile:.0f}th\n"
            f"  Drawdown: {self.drawdown_percent*100:.2f}%\n"
            f"  Portfolio Value: ${self.portfolio_value:,.0f}"
        )


class RiskMetricsEngine:
    """Engine for calculating portfolio risk metrics."""

    def __init__(self):
        self.config = settings.risk
        self._historical_returns: Optional[np.ndarray] = None
        self._peak_value: float = 0.0
        self._historical_iv: list[float] = []

    def set_historical_returns(self, returns: np.ndarray):
        """Set historical returns for VaR calculation."""
        self._historical_returns = returns

    def calculate_metrics(
        self,
        snapshot: MarketSnapshot,
        greeks: PortfolioGreeks,
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.

        Args:
            snapshot: Current market snapshot
            greeks: Portfolio Greeks

        Returns:
            RiskMetrics with all calculated values
        """
        portfolio_value = snapshot.total_market_value
        if portfolio_value <= 0:
            portfolio_value = 1.0  # Avoid division by zero

        # Update peak for drawdown
        self._peak_value = max(self._peak_value, portfolio_value)

        # Calculate VaR
        var_95, var_99 = self._calculate_var(greeks, snapshot.spot_price)

        # Calculate Expected Shortfall
        es_95, es_99 = self._calculate_expected_shortfall(greeks, snapshot.spot_price)

        # Calculate volatility metrics
        realized_vol = self._calculate_realized_volatility()
        iv_percentile = self._calculate_iv_percentile(snapshot)

        # Calculate drawdown
        current_drawdown = self._peak_value - portfolio_value
        drawdown_percent = current_drawdown / self._peak_value if self._peak_value > 0 else 0

        return RiskMetrics(
            var_95=var_95,
            var_99=var_99,
            es_95=es_95,
            es_99=es_99,
            var_percent=var_95 / portfolio_value,
            es_percent=es_95 / portfolio_value,
            realized_volatility=realized_vol,
            iv_percentile=iv_percentile,
            current_drawdown=current_drawdown,
            max_drawdown=current_drawdown,  # Track max separately
            drawdown_percent=drawdown_percent,
            portfolio_value=portfolio_value,
            pnl_today=0.0,  # TODO: Track daily P&L
        )

    def _calculate_var(
        self,
        greeks: PortfolioGreeks,
        spot_price: float,
    ) -> tuple[float, float]:
        """
        Calculate Value at Risk using delta-gamma approximation.

        Returns:
            (VaR 95%, VaR 99%)
        """
        if self._historical_returns is None or len(self._historical_returns) == 0:
            # Use parametric VaR with assumed volatility
            sigma = 0.20  # 20% annualized vol assumption
            daily_vol = sigma / np.sqrt(252)
        else:
            daily_vol = np.std(self._historical_returns)

        # Delta-Gamma VaR
        # P&L ≈ Delta * ΔS + 0.5 * Gamma * ΔS²
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        delta = greeks.portfolio_delta
        gamma = greeks.portfolio_gamma

        # Expected 1-day move
        expected_move_95 = spot_price * daily_vol * z_95
        expected_move_99 = spot_price * daily_vol * z_99

        # Delta-Gamma P&L impact
        var_95 = abs(delta * expected_move_95 + 0.5 * gamma * expected_move_95**2)
        var_99 = abs(delta * expected_move_99 + 0.5 * gamma * expected_move_99**2)

        return var_95, var_99

    def _calculate_expected_shortfall(
        self,
        greeks: PortfolioGreeks,
        spot_price: float,
    ) -> tuple[float, float]:
        """
        Calculate Expected Shortfall (Conditional VaR).

        Returns:
            (ES 95%, ES 99%)
        """
        if self._historical_returns is None or len(self._historical_returns) == 0:
            # Use parametric ES
            sigma = 0.20
            daily_vol = sigma / np.sqrt(252)
        else:
            daily_vol = np.std(self._historical_returns)

        # For normal distribution: ES = sigma * phi(z) / (1 - alpha)
        # where phi is the standard normal PDF
        z_95 = stats.norm.ppf(0.95)
        z_99 = stats.norm.ppf(0.99)

        es_factor_95 = stats.norm.pdf(z_95) / 0.05
        es_factor_99 = stats.norm.pdf(z_99) / 0.01

        delta = greeks.portfolio_delta

        es_95 = abs(delta * spot_price * daily_vol * es_factor_95)
        es_99 = abs(delta * spot_price * daily_vol * es_factor_99)

        return es_95, es_99

    def _calculate_realized_volatility(self) -> float:
        """Calculate realized volatility from historical returns."""
        if self._historical_returns is None or len(self._historical_returns) < 20:
            return 0.20  # Default 20%

        # Annualized volatility
        return float(np.std(self._historical_returns) * np.sqrt(252))

    def _calculate_iv_percentile(self, snapshot: MarketSnapshot) -> float:
        """Calculate current IV percentile vs historical."""
        if snapshot.iv_surface is None:
            return 50.0

        current_iv = snapshot.iv_surface.atm_iv
        self._historical_iv.append(current_iv)

        # Keep last 252 trading days
        if len(self._historical_iv) > 252:
            self._historical_iv = self._historical_iv[-252:]

        if len(self._historical_iv) < 20:
            return 50.0

        percentile = stats.percentileofscore(self._historical_iv, current_iv)
        return float(percentile)

    def monte_carlo_var(
        self,
        greeks: PortfolioGreeks,
        spot_price: float,
        num_simulations: int = 10000,
        horizon_days: int = 1,
        volatility: float = 0.20,
    ) -> tuple[float, float]:
        """
        Calculate VaR using Monte Carlo simulation.

        Returns:
            (VaR 95%, VaR 99%)
        """
        daily_vol = volatility / np.sqrt(252)

        # Simulate price paths
        returns = np.random.normal(0, daily_vol, (num_simulations, horizon_days))
        cumulative_returns = np.sum(returns, axis=1)
        price_changes = spot_price * cumulative_returns

        # Calculate P&L using delta-gamma
        delta = greeks.portfolio_delta
        gamma = greeks.portfolio_gamma

        pnl = delta * price_changes + 0.5 * gamma * price_changes**2

        var_95 = -np.percentile(pnl, 5)
        var_99 = -np.percentile(pnl, 1)

        return float(var_95), float(var_99)

    def cornish_fisher_var(
        self,
        greeks: PortfolioGreeks,
        spot_price: float,
        volatility: float = 0.20,
        skewness: float = 0.0,
        kurtosis: float = 3.0,
    ) -> tuple[float, float]:
        """
        Calculate VaR using Cornish-Fisher expansion for non-normal returns.

        Returns:
            (VaR 95%, VaR 99%)
        """
        daily_vol = volatility / np.sqrt(252)

        def cf_quantile(alpha: float) -> float:
            z = stats.norm.ppf(alpha)
            # Cornish-Fisher adjustment
            cf_z = (
                z
                + (z**2 - 1) * skewness / 6
                + (z**3 - 3 * z) * (kurtosis - 3) / 24
                - (2 * z**3 - 5 * z) * skewness**2 / 36
            )
            return cf_z

        z_95 = cf_quantile(0.95)
        z_99 = cf_quantile(0.99)

        delta = greeks.portfolio_delta
        var_95 = abs(delta * spot_price * daily_vol * z_95)
        var_99 = abs(delta * spot_price * daily_vol * z_99)

        return var_95, var_99
