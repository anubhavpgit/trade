"""
Data models for the trading system.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class OptionType(str, Enum):
    """Option type enumeration."""
    CALL = "call"
    PUT = "put"


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    """Order type enumeration."""
    MARKET = "market"
    LIMIT = "limit"


class OptionContract(BaseModel):
    """Represents an option contract."""

    symbol: str
    underlying: str
    option_type: OptionType
    strike: float
    expiration: date
    multiplier: int = 100

    # Market data
    bid: Optional[float] = None
    ask: Optional[float] = None
    last: Optional[float] = None
    volume: Optional[int] = None
    open_interest: Optional[int] = None
    implied_volatility: Optional[float] = None

    # Greeks (populated by analytics)
    delta: Optional[float] = None
    gamma: Optional[float] = None
    theta: Optional[float] = None
    vega: Optional[float] = None
    rho: Optional[float] = None

    @property
    def mid_price(self) -> Optional[float]:
        if self.bid is not None and self.ask is not None:
            return (self.bid + self.ask) / 2
        return self.last

    @property
    def days_to_expiry(self) -> int:
        return (self.expiration - date.today()).days

    @property
    def time_to_expiry(self) -> float:
        """Time to expiry in years."""
        return self.days_to_expiry / 365.0


class Position(BaseModel):
    """Represents a portfolio position."""

    contract: OptionContract
    quantity: int  # Positive for long, negative for short
    avg_cost: float
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        price = self.contract.mid_price or self.avg_cost
        return self.quantity * price * self.contract.multiplier

    @property
    def pnl(self) -> float:
        price = self.contract.mid_price or self.avg_cost
        return self.quantity * (price - self.avg_cost) * self.contract.multiplier

    @property
    def position_delta(self) -> float:
        if self.contract.delta is None:
            return 0.0
        return self.quantity * self.contract.delta * self.contract.multiplier

    @property
    def position_gamma(self) -> float:
        if self.contract.gamma is None:
            return 0.0
        return self.quantity * self.contract.gamma * self.contract.multiplier

    @property
    def position_theta(self) -> float:
        if self.contract.theta is None:
            return 0.0
        return self.quantity * self.contract.theta * self.contract.multiplier

    @property
    def position_vega(self) -> float:
        if self.contract.vega is None:
            return 0.0
        return self.quantity * self.contract.vega * self.contract.multiplier


class StockPosition(BaseModel):
    """Represents a stock/underlying position for hedging."""

    symbol: str
    quantity: int  # Positive for long, negative for short
    avg_cost: float
    current_price: float
    timestamp: datetime = Field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_cost)


@dataclass
class OptionChain:
    """Represents an option chain for an underlying."""

    underlying: str
    spot_price: float
    expirations: list[date]
    strikes: list[float]
    calls: dict[tuple[date, float], OptionContract] = field(default_factory=dict)
    puts: dict[tuple[date, float], OptionContract] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def get_atm_options(self, expiration: date) -> tuple[Optional[OptionContract], Optional[OptionContract]]:
        """Get at-the-money call and put for given expiration."""
        atm_strike = min(self.strikes, key=lambda x: abs(x - self.spot_price))
        call = self.calls.get((expiration, atm_strike))
        put = self.puts.get((expiration, atm_strike))
        return call, put

    def to_dataframe(self) -> pd.DataFrame:
        """Convert chain to DataFrame."""
        records = []
        for (exp, strike), contract in {**self.calls, **self.puts}.items():
            records.append({
                "expiration": exp,
                "strike": strike,
                "type": contract.option_type.value,
                "bid": contract.bid,
                "ask": contract.ask,
                "iv": contract.implied_volatility,
                "delta": contract.delta,
                "gamma": contract.gamma,
                "theta": contract.theta,
                "vega": contract.vega,
            })
        return pd.DataFrame(records)


@dataclass
class IVSurface:
    """Implied volatility surface."""

    underlying: str
    spot_price: float
    strikes: np.ndarray
    expirations: np.ndarray  # In years
    iv_matrix: np.ndarray  # Shape: (len(expirations), len(strikes))
    timestamp: datetime = field(default_factory=datetime.now)

    def get_iv(self, strike: float, time_to_expiry: float) -> float:
        """Interpolate IV for given strike and time to expiry."""
        from scipy.interpolate import RectBivariateSpline

        # Clamp to surface bounds
        strike = np.clip(strike, self.strikes.min(), self.strikes.max())
        time_to_expiry = np.clip(time_to_expiry, self.expirations.min(), self.expirations.max())

        spline = RectBivariateSpline(self.expirations, self.strikes, self.iv_matrix)
        return float(spline(time_to_expiry, strike)[0, 0])

    @property
    def atm_iv(self) -> float:
        """Get ATM implied volatility for front month."""
        return self.get_iv(self.spot_price, self.expirations[0])


@dataclass
class MarketSnapshot:
    """Complete market state snapshot."""

    underlying: str
    spot_price: float
    timestamp: datetime
    positions: list[Position]
    stock_position: Optional[StockPosition]
    option_chain: Optional[OptionChain]
    iv_surface: Optional[IVSurface]
    risk_free_rate: float = 0.05

    @property
    def total_market_value(self) -> float:
        option_value = sum(p.market_value for p in self.positions)
        stock_value = self.stock_position.market_value if self.stock_position else 0
        return option_value + stock_value


@dataclass
class Trade:
    """Represents an executed trade."""

    trade_id: str
    symbol: str
    side: OrderSide
    quantity: int
    price: float
    timestamp: datetime
    commission: float = 0.0
    slippage: float = 0.0

    @property
    def total_cost(self) -> float:
        sign = 1 if self.side == OrderSide.BUY else -1
        return sign * self.quantity * self.price + self.commission


@dataclass
class Order:
    """Represents an order to be executed."""

    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None

    def validate(self) -> bool:
        if self.quantity <= 0:
            return False
        if self.order_type == OrderType.LIMIT and self.limit_price is None:
            return False
        return True
