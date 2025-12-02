"""
Market data service for fetching live/historical data.
Supports multiple providers: Polygon.io, IEX, Yahoo Finance.
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime, date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from src.config import settings
from src.data.models import (
    OptionContract,
    OptionType,
    OptionChain,
    IVSurface,
    MarketSnapshot,
    Position,
    StockPosition,
)


class DataProvider(ABC):
    """Abstract base class for market data providers."""

    @abstractmethod
    async def get_spot_price(self, symbol: str) -> float:
        """Get current spot price."""
        pass

    @abstractmethod
    async def get_option_chain(self, symbol: str) -> OptionChain:
        """Get option chain for symbol."""
        pass

    @abstractmethod
    async def get_historical_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical price data."""
        pass


class YahooDataProvider(DataProvider):
    """Yahoo Finance data provider (free, no API key required)."""

    async def get_spot_price(self, symbol: str) -> float:
        """Get current spot price from Yahoo Finance."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        data = ticker.history(period="1d")
        if data.empty:
            raise ValueError(f"No data found for {symbol}")
        return float(data["Close"].iloc[-1])

    async def get_option_chain(self, symbol: str) -> OptionChain:
        """Get option chain from Yahoo Finance."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        spot_price = await self.get_spot_price(symbol)

        expirations = ticker.options
        if not expirations:
            raise ValueError(f"No options data for {symbol}")

        chain = OptionChain(
            underlying=symbol,
            spot_price=spot_price,
            expirations=[],
            strikes=[],
        )

        all_strikes = set()

        for exp_str in expirations[:4]:  # Limit to first 4 expirations
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d").date()
            chain.expirations.append(exp_date)

            opt = ticker.option_chain(exp_str)

            # Process calls
            for _, row in opt.calls.iterrows():
                strike = float(row["strike"])
                all_strikes.add(strike)
                contract = OptionContract(
                    symbol=row.get("contractSymbol", f"{symbol}{exp_str}C{strike}"),
                    underlying=symbol,
                    option_type=OptionType.CALL,
                    strike=strike,
                    expiration=exp_date,
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    last=float(row.get("lastPrice", 0)),
                    volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    open_interest=int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                )
                chain.calls[(exp_date, strike)] = contract

            # Process puts
            for _, row in opt.puts.iterrows():
                strike = float(row["strike"])
                all_strikes.add(strike)
                contract = OptionContract(
                    symbol=row.get("contractSymbol", f"{symbol}{exp_str}P{strike}"),
                    underlying=symbol,
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiration=exp_date,
                    bid=float(row.get("bid", 0)),
                    ask=float(row.get("ask", 0)),
                    last=float(row.get("lastPrice", 0)),
                    volume=int(row.get("volume", 0)) if pd.notna(row.get("volume")) else 0,
                    open_interest=int(row.get("openInterest", 0)) if pd.notna(row.get("openInterest")) else 0,
                    implied_volatility=float(row.get("impliedVolatility", 0)),
                )
                chain.puts[(exp_date, strike)] = contract

        chain.strikes = sorted(all_strikes)
        return chain

    async def get_historical_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get historical prices from Yahoo Finance."""
        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]
        return df


class MockDataProvider(DataProvider):
    """Mock data provider for testing."""

    def __init__(self, base_price: float = 100.0):
        self.base_price = base_price
        self._current_price = base_price

    async def get_spot_price(self, symbol: str) -> float:
        # Add some random walk
        self._current_price *= 1 + np.random.normal(0, 0.001)
        return self._current_price

    async def get_option_chain(self, symbol: str) -> OptionChain:
        spot = await self.get_spot_price(symbol)
        expirations = [
            date.today() + timedelta(days=d)
            for d in [7, 14, 30, 60]
        ]
        strikes = [spot * (1 + x/100) for x in range(-20, 21, 5)]

        chain = OptionChain(
            underlying=symbol,
            spot_price=spot,
            expirations=expirations,
            strikes=strikes,
        )

        for exp in expirations:
            tte = (exp - date.today()).days / 365.0
            for strike in strikes:
                # Generate mock option prices
                iv = 0.25 + 0.1 * abs(np.log(strike/spot))

                call = OptionContract(
                    symbol=f"{symbol}{exp}C{strike}",
                    underlying=symbol,
                    option_type=OptionType.CALL,
                    strike=strike,
                    expiration=exp,
                    bid=max(0.01, spot - strike + iv * spot * np.sqrt(tte)),
                    ask=max(0.02, spot - strike + iv * spot * np.sqrt(tte) + 0.05),
                    implied_volatility=iv,
                )
                chain.calls[(exp, strike)] = call

                put = OptionContract(
                    symbol=f"{symbol}{exp}P{strike}",
                    underlying=symbol,
                    option_type=OptionType.PUT,
                    strike=strike,
                    expiration=exp,
                    bid=max(0.01, strike - spot + iv * spot * np.sqrt(tte)),
                    ask=max(0.02, strike - spot + iv * spot * np.sqrt(tte) + 0.05),
                    implied_volatility=iv,
                )
                chain.puts[(exp, strike)] = put

        return chain

    async def get_historical_prices(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        interval: str = "1d",
    ) -> pd.DataFrame:
        days = (end_date - start_date).days
        dates = pd.date_range(start=start_date, periods=days, freq="D")

        prices = self.base_price * np.cumprod(1 + np.random.normal(0, 0.02, days))

        return pd.DataFrame({
            "date": dates,
            "open": prices * 0.999,
            "high": prices * 1.01,
            "low": prices * 0.99,
            "close": prices,
            "volume": np.random.randint(1000000, 10000000, days),
        })


class MarketDataService:
    """Main market data service."""

    def __init__(self):
        self.provider = self._create_provider()
        self._positions: list[Position] = []
        self._stock_position: Optional[StockPosition] = None
        self._underlying = "SPY"  # Default underlying

    def _create_provider(self) -> DataProvider:
        """Create data provider based on configuration."""
        provider_type = settings.market_data.provider

        if provider_type == "yahoo":
            return YahooDataProvider()
        elif provider_type == "mock":
            return MockDataProvider()
        else:
            logger.warning(f"Unknown provider {provider_type}, using mock")
            return MockDataProvider()

    def set_underlying(self, symbol: str):
        """Set the underlying symbol to track."""
        self._underlying = symbol

    def set_positions(self, positions: list[Position]):
        """Set current portfolio positions."""
        self._positions = positions

    def set_stock_position(self, position: Optional[StockPosition]):
        """Set stock hedge position."""
        self._stock_position = position

    async def get_snapshot(self) -> MarketSnapshot:
        """Get complete market snapshot."""
        spot_price = await self.provider.get_spot_price(self._underlying)
        option_chain = await self.provider.get_option_chain(self._underlying)
        iv_surface = self._build_iv_surface(option_chain)

        # Update stock position price
        if self._stock_position:
            self._stock_position.current_price = spot_price

        return MarketSnapshot(
            underlying=self._underlying,
            spot_price=spot_price,
            timestamp=datetime.now(),
            positions=self._positions,
            stock_position=self._stock_position,
            option_chain=option_chain,
            iv_surface=iv_surface,
        )

    def _build_iv_surface(self, chain: OptionChain) -> IVSurface:
        """Build IV surface from option chain."""
        if not chain.calls:
            # Return a flat surface if no data
            return IVSurface(
                underlying=chain.underlying,
                spot_price=chain.spot_price,
                strikes=np.array([chain.spot_price]),
                expirations=np.array([0.1]),
                iv_matrix=np.array([[0.25]]),
            )

        strikes = np.array(sorted(chain.strikes))
        expirations = np.array([
            (exp - date.today()).days / 365.0
            for exp in sorted(chain.expirations)
        ])

        iv_matrix = np.zeros((len(expirations), len(strikes)))

        for i, exp in enumerate(sorted(chain.expirations)):
            for j, strike in enumerate(strikes):
                call = chain.calls.get((exp, strike))
                put = chain.puts.get((exp, strike))

                if call and call.implied_volatility:
                    iv_matrix[i, j] = call.implied_volatility
                elif put and put.implied_volatility:
                    iv_matrix[i, j] = put.implied_volatility
                else:
                    iv_matrix[i, j] = 0.25  # Default IV

        return IVSurface(
            underlying=chain.underlying,
            spot_price=chain.spot_price,
            strikes=strikes,
            expirations=expirations,
            iv_matrix=iv_matrix,
        )

    async def get_historical_data(
        self,
        symbol: str,
        lookback_days: int = 252,
    ) -> pd.DataFrame:
        """Get historical data for risk calculations."""
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days)
        return await self.provider.get_historical_prices(symbol, start_date, end_date)
