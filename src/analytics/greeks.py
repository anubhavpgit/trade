"""
Greeks calculation and portfolio aggregation.
"""

from dataclasses import dataclass
from typing import Optional

from loguru import logger

from src.data.models import Position, OptionContract, IVSurface, StockPosition
from src.analytics.pricing import BlackScholes, OptionPricer


@dataclass
class PortfolioGreeks:
    """Aggregated portfolio Greeks."""

    portfolio_delta: float
    portfolio_gamma: float
    portfolio_theta: float
    portfolio_vega: float
    portfolio_rho: float

    # Dollar-weighted metrics
    dollar_delta: float  # Delta in dollar terms
    dollar_gamma: float  # Gamma in dollar terms

    # Position details
    num_positions: int
    total_notional: float

    def __str__(self) -> str:
        return (
            f"Portfolio Greeks:\n"
            f"  Delta: {self.portfolio_delta:+.2f} ({self.dollar_delta:+,.0f} USD)\n"
            f"  Gamma: {self.portfolio_gamma:+.4f} ({self.dollar_gamma:+,.0f} USD)\n"
            f"  Theta: {self.portfolio_theta:+.2f}/day\n"
            f"  Vega:  {self.portfolio_vega:+.2f}\n"
            f"  Rho:   {self.portfolio_rho:+.2f}\n"
            f"  Positions: {self.num_positions}\n"
            f"  Notional: ${self.total_notional:,.0f}"
        )


class GreeksCalculator:
    """Calculate and aggregate portfolio Greeks."""

    def __init__(self, risk_free_rate: float = 0.05):
        self.risk_free_rate = risk_free_rate
        self.pricer = OptionPricer()

    def calculate_option_greeks(
        self,
        contract: OptionContract,
        spot_price: float,
        iv_surface: Optional[IVSurface] = None,
    ) -> OptionContract:
        """
        Calculate Greeks for a single option contract.
        Returns the contract with Greeks populated.
        """
        # Get IV from surface or use contract's IV
        if iv_surface and contract.time_to_expiry > 0:
            iv = iv_surface.get_iv(contract.strike, contract.time_to_expiry)
        elif contract.implied_volatility:
            iv = contract.implied_volatility
        else:
            iv = 0.25  # Default IV

        if contract.time_to_expiry <= 0:
            # Handle expired options
            contract.delta = 0.0
            contract.gamma = 0.0
            contract.theta = 0.0
            contract.vega = 0.0
            contract.rho = 0.0
            return contract

        result = self.pricer.price_option(
            spot=spot_price,
            strike=contract.strike,
            time_to_expiry=contract.time_to_expiry,
            risk_free_rate=self.risk_free_rate,
            volatility=iv,
            option_type=contract.option_type,
        )

        contract.delta = result.delta
        contract.gamma = result.gamma
        contract.theta = result.theta
        contract.vega = result.vega
        contract.rho = result.rho

        return contract

    def calculate_portfolio_greeks(
        self,
        positions: list[Position],
        spot_price: float,
        iv_surface: Optional[IVSurface] = None,
        stock_position: Optional[StockPosition] = None,
    ) -> PortfolioGreeks:
        """
        Calculate aggregated portfolio Greeks.

        Args:
            positions: List of option positions
            spot_price: Current underlying price
            iv_surface: Implied volatility surface
            stock_position: Stock/underlying hedge position

        Returns:
            PortfolioGreeks with aggregated values
        """
        total_delta = 0.0
        total_gamma = 0.0
        total_theta = 0.0
        total_vega = 0.0
        total_rho = 0.0
        total_notional = 0.0

        for position in positions:
            # Calculate Greeks for the contract
            contract = self.calculate_option_greeks(
                position.contract, spot_price, iv_surface
            )

            # Aggregate (quantity already includes sign)
            multiplier = position.contract.multiplier
            qty = position.quantity

            total_delta += (contract.delta or 0) * qty * multiplier
            total_gamma += (contract.gamma or 0) * qty * multiplier
            total_theta += (contract.theta or 0) * qty * multiplier
            total_vega += (contract.vega or 0) * qty * multiplier
            total_rho += (contract.rho or 0) * qty * multiplier

            total_notional += abs(qty) * spot_price * multiplier

        # Add stock position delta (stock delta = 1 per share)
        if stock_position:
            total_delta += stock_position.quantity
            total_notional += abs(stock_position.quantity) * spot_price

        # Dollar-weighted metrics
        dollar_delta = total_delta * spot_price
        dollar_gamma = total_gamma * spot_price * spot_price / 100  # Per 1% move

        return PortfolioGreeks(
            portfolio_delta=total_delta,
            portfolio_gamma=total_gamma,
            portfolio_theta=total_theta,
            portfolio_vega=total_vega,
            portfolio_rho=total_rho,
            dollar_delta=dollar_delta,
            dollar_gamma=dollar_gamma,
            num_positions=len(positions) + (1 if stock_position else 0),
            total_notional=total_notional,
        )

    def calculate_hedge_shares(
        self,
        current_delta: float,
        target_delta: float = 0.0,
    ) -> int:
        """
        Calculate number of shares needed to hedge to target delta.

        Args:
            current_delta: Current portfolio delta
            target_delta: Target delta (default 0 for delta-neutral)

        Returns:
            Number of shares to buy (positive) or sell (negative)
        """
        delta_to_hedge = target_delta - current_delta
        # Round to nearest share
        return round(delta_to_hedge)

    def delta_hedge_cost(
        self,
        shares: int,
        spot_price: float,
        transaction_cost_bps: float = 5.0,
    ) -> float:
        """
        Calculate the cost of a delta hedge trade.

        Args:
            shares: Number of shares to trade
            spot_price: Current spot price
            transaction_cost_bps: Transaction cost in basis points

        Returns:
            Total cost including transaction costs
        """
        notional = abs(shares) * spot_price
        tc = notional * (transaction_cost_bps / 10000)
        return tc
