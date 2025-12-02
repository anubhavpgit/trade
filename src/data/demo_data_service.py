"""
Demo data service that provides data from the local SQLite database.
Used for dashboard and backtesting without live market connections.
"""

from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
from loguru import logger

from src.data.database import (
    DatabaseManager,
    DBPriceHistory,
    DBVolatilityHistory,
    DBOptionSnapshot,
    DBPosition,
    DBTrade,
    DBRiskSnapshot,
    DBAlert,
    get_db_manager,
)
from src.data.models import (
    OptionContract,
    OptionType,
    Position,
    StockPosition,
    MarketSnapshot,
    OptionChain,
    IVSurface,
)
from src.analytics.greeks import GreeksCalculator, PortfolioGreeks
from src.analytics.risk_metrics import RiskMetricsEngine, RiskMetrics


@dataclass
class DashboardData:
    """Complete data package for dashboard display."""

    # Portfolio metrics
    portfolio_value: float
    pnl_today: float
    pnl_cumulative: float
    num_positions: int

    # Greeks
    delta: float
    gamma: float
    theta: float
    vega: float
    dollar_delta: float
    dollar_gamma: float

    # Risk metrics
    var_95: float
    var_99: float
    es_95: float
    drawdown: float

    # Volatility
    iv_atm: float
    iv_percentile: float
    realized_vol: float
    vix_current: float

    # Underlying
    spot_price: float
    spot_change_pct: float

    # Time series data
    greeks_history: pd.DataFrame
    pnl_history: pd.DataFrame
    price_history: pd.DataFrame

    # Tables
    positions: List[Dict]
    recent_trades: List[Dict]
    recent_alerts: List[Dict]

    # Agent status
    agent_mode: str
    last_hedge_time: Optional[datetime]
    trades_today: int
    next_action: str

    # Timestamp
    timestamp: datetime


class DemoDataService:
    """
    Service to provide demo data from local database.

    This service simulates a live trading environment using historical
    and synthetic data stored in SQLite.
    """

    def __init__(self):
        self.db = get_db_manager()
        self.greeks_calc = GreeksCalculator()
        self.risk_engine = RiskMetricsEngine()
        self._simulation_offset = 0  # For simulating time progression

    def get_dashboard_data(self) -> DashboardData:
        """
        Get complete data package for dashboard.

        Returns:
            DashboardData with all metrics and time series
        """
        with self.db.get_session() as session:
            # Get latest risk snapshot
            risk_snapshot = self.db.get_latest_risk_snapshot(session)

            # Get latest price
            latest_price = self.db.get_latest_price(session, "SPY")
            spot_price = latest_price.close if latest_price else 450.0

            # Get previous day price for change calculation
            prev_price = session.query(DBPriceHistory).filter(
                DBPriceHistory.symbol == "SPY",
                DBPriceHistory.date < latest_price.date if latest_price else date.today()
            ).order_by(DBPriceHistory.date.desc()).first()
            prev_close = prev_price.close if prev_price else spot_price
            spot_change_pct = (spot_price - prev_close) / prev_close * 100

            # Get VIX
            latest_vix = session.query(DBVolatilityHistory).order_by(
                DBVolatilityHistory.date.desc()
            ).first()
            vix_current = latest_vix.vix_close if latest_vix else 18.0

            # Get positions
            positions = self._get_positions_data(session)
            num_positions = len(positions)

            # Get trades
            recent_trades = self._get_recent_trades(session)
            trades_today = sum(
                1 for t in recent_trades
                if t["timestamp"].date() == date.today()
            )

            # Get alerts
            recent_alerts = self._get_recent_alerts(session)

            # Get time series
            greeks_history = self._get_greeks_history(session)
            pnl_history = self._get_pnl_history(session)
            price_history = self._get_price_history(session, "SPY", days=60)

            # Use risk snapshot data or defaults
            if risk_snapshot:
                return DashboardData(
                    portfolio_value=risk_snapshot.portfolio_value,
                    pnl_today=risk_snapshot.pnl_daily,
                    pnl_cumulative=risk_snapshot.pnl_cumulative,
                    num_positions=num_positions,
                    delta=risk_snapshot.delta,
                    gamma=risk_snapshot.gamma,
                    theta=risk_snapshot.theta,
                    vega=risk_snapshot.vega,
                    dollar_delta=risk_snapshot.delta * spot_price,
                    dollar_gamma=risk_snapshot.gamma * spot_price**2 / 100,
                    var_95=risk_snapshot.var_95,
                    var_99=risk_snapshot.var_99,
                    es_95=risk_snapshot.es_95,
                    drawdown=risk_snapshot.drawdown,
                    iv_atm=risk_snapshot.iv_atm,
                    iv_percentile=risk_snapshot.iv_percentile,
                    realized_vol=risk_snapshot.realized_vol,
                    vix_current=vix_current,
                    spot_price=spot_price,
                    spot_change_pct=spot_change_pct,
                    greeks_history=greeks_history,
                    pnl_history=pnl_history,
                    price_history=price_history,
                    positions=positions,
                    recent_trades=recent_trades,
                    recent_alerts=recent_alerts,
                    agent_mode="PAPER",
                    last_hedge_time=recent_trades[0]["timestamp"] if recent_trades else None,
                    trades_today=trades_today,
                    next_action=self._determine_next_action(risk_snapshot),
                    timestamp=datetime.now(),
                )
            else:
                # Return defaults if no data
                return self._get_default_dashboard_data(
                    spot_price, vix_current, positions, recent_trades, recent_alerts
                )

    def _get_positions_data(self, session) -> List[Dict]:
        """Get formatted position data."""
        positions = self.db.get_active_positions(session)
        result = []

        for pos in positions:
            result.append({
                "id": pos.id,
                "symbol": pos.symbol,
                "type": pos.position_type,
                "option_type": pos.option_type,
                "strike": pos.strike,
                "expiration": pos.expiration.isoformat() if pos.expiration else None,
                "quantity": pos.quantity,
                "avg_cost": pos.avg_cost,
                "market_value": abs(pos.quantity) * pos.avg_cost * (100 if pos.position_type == "option" else 1),
                "opened_at": pos.opened_at,
            })

        return result

    def _get_recent_trades(self, session, limit: int = 20) -> List[Dict]:
        """Get recent trades."""
        trades = self.db.get_trades(session, limit=limit)
        result = []

        for trade in trades:
            result.append({
                "trade_id": trade.trade_id,
                "timestamp": trade.timestamp,
                "symbol": trade.symbol,
                "side": trade.side,
                "quantity": trade.quantity,
                "price": trade.price,
                "commission": trade.commission,
                "pnl": trade.pnl,
            })

        return result

    def _get_recent_alerts(self, session, limit: int = 10) -> List[Dict]:
        """Get recent alerts."""
        alerts = self.db.get_recent_alerts(session, limit=limit)
        result = []

        for alert in alerts:
            result.append({
                "timestamp": alert.timestamp,
                "type": alert.alert_type,
                "severity": alert.severity,
                "message": alert.message,
                "acknowledged": alert.acknowledged,
            })

        return result

    def _get_greeks_history(self, session, hours: int = 48) -> pd.DataFrame:
        """Get Greeks history for charting."""
        snapshots = self.db.get_risk_history(session, hours=hours)

        if not snapshots:
            # Return empty DataFrame with correct columns
            return pd.DataFrame(columns=["timestamp", "delta", "gamma", "theta", "vega"])

        data = [{
            "timestamp": s.timestamp,
            "delta": s.delta,
            "gamma": s.gamma,
            "theta": s.theta,
            "vega": s.vega,
        } for s in snapshots]

        return pd.DataFrame(data)

    def _get_pnl_history(self, session, hours: int = 48) -> pd.DataFrame:
        """Get P&L history for charting."""
        snapshots = self.db.get_risk_history(session, hours=hours)

        if not snapshots:
            return pd.DataFrame(columns=["timestamp", "pnl_daily", "pnl_cumulative", "portfolio_value"])

        data = [{
            "timestamp": s.timestamp,
            "pnl_daily": s.pnl_daily,
            "pnl_cumulative": s.pnl_cumulative,
            "portfolio_value": s.portfolio_value,
        } for s in snapshots]

        return pd.DataFrame(data)

    def _get_price_history(self, session, symbol: str, days: int = 60) -> pd.DataFrame:
        """Get price history for charting."""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)

        prices = self.db.get_price_history(session, symbol, start_date, end_date)

        if not prices:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])

        data = [{
            "date": p.date,
            "open": p.open,
            "high": p.high,
            "low": p.low,
            "close": p.close,
            "volume": p.volume,
        } for p in prices]

        return pd.DataFrame(data)

    def _determine_next_action(self, risk_snapshot: Optional[DBRiskSnapshot]) -> str:
        """Determine what the agent should do next."""
        if risk_snapshot is None:
            return "INITIALIZING"

        delta = abs(risk_snapshot.delta) if risk_snapshot.delta else 0
        threshold = 10  # Delta band threshold

        if delta > threshold * 2:
            return f"URGENT HEDGE - Delta: {risk_snapshot.delta:+.1f}"
        elif delta > threshold:
            return f"HEDGE NEEDED - Delta: {risk_snapshot.delta:+.1f}"
        else:
            return f"HOLD - Delta in band ({risk_snapshot.delta:+.1f})"

    def _get_default_dashboard_data(
        self,
        spot_price: float,
        vix_current: float,
        positions: List[Dict],
        trades: List[Dict],
        alerts: List[Dict],
    ) -> DashboardData:
        """Return default dashboard data when no snapshots exist."""
        return DashboardData(
            portfolio_value=100000.0,
            pnl_today=0.0,
            pnl_cumulative=0.0,
            num_positions=len(positions),
            delta=0.0,
            gamma=0.0,
            theta=0.0,
            vega=0.0,
            dollar_delta=0.0,
            dollar_gamma=0.0,
            var_95=0.0,
            var_99=0.0,
            es_95=0.0,
            drawdown=0.0,
            iv_atm=vix_current / 100,
            iv_percentile=50.0,
            realized_vol=0.18,
            vix_current=vix_current,
            spot_price=spot_price,
            spot_change_pct=0.0,
            greeks_history=pd.DataFrame(),
            pnl_history=pd.DataFrame(),
            price_history=pd.DataFrame(),
            positions=positions,
            recent_trades=trades,
            recent_alerts=alerts,
            agent_mode="PAPER",
            last_hedge_time=None,
            trades_today=0,
            next_action="NO DATA - Run seed script",
            timestamp=datetime.now(),
        )

    def get_option_chain(self, underlying: str = "SPY", snapshot_date: Optional[date] = None) -> List[Dict]:
        """Get option chain for a specific date."""
        if snapshot_date is None:
            snapshot_date = date.today()

        with self.db.get_session() as session:
            options = self.db.get_option_chain_snapshot(session, underlying, snapshot_date)

            return [{
                "symbol": opt.symbol,
                "type": opt.option_type,
                "strike": opt.strike,
                "expiration": opt.expiration.isoformat(),
                "bid": opt.bid,
                "ask": opt.ask,
                "iv": opt.implied_volatility,
                "delta": opt.delta,
                "gamma": opt.gamma,
                "theta": opt.theta,
                "vega": opt.vega,
                "volume": opt.volume,
                "oi": opt.open_interest,
            } for opt in options]

    def get_iv_surface_data(self, underlying: str = "SPY") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get IV surface data for 3D visualization."""
        with self.db.get_session() as session:
            # Get most recent option snapshot
            latest_date = session.query(DBOptionSnapshot.snapshot_date).order_by(
                DBOptionSnapshot.snapshot_date.desc()
            ).first()

            if not latest_date:
                # Return mock surface
                strikes = np.linspace(420, 480, 13)
                expiries = np.array([7, 14, 30, 60, 90])
                iv_matrix = np.ones((len(expiries), len(strikes))) * 0.20
                return strikes, expiries, iv_matrix

            options = self.db.get_option_chain_snapshot(session, underlying, latest_date[0])

            # Build surface from call options
            calls = [o for o in options if o.option_type == "call" and o.implied_volatility]

            if not calls:
                strikes = np.linspace(420, 480, 13)
                expiries = np.array([7, 14, 30, 60, 90])
                iv_matrix = np.ones((len(expiries), len(strikes))) * 0.20
                return strikes, expiries, iv_matrix

            # Get unique strikes and expirations
            strikes = sorted(set(o.strike for o in calls))
            expirations = sorted(set(o.expiration for o in calls))

            # Build IV matrix
            iv_dict = {(o.expiration, o.strike): o.implied_volatility for o in calls}

            strikes_arr = np.array(strikes)
            expiries_arr = np.array([(e - latest_date[0]).days for e in expirations])

            iv_matrix = np.zeros((len(expirations), len(strikes)))
            for i, exp in enumerate(expirations):
                for j, strike in enumerate(strikes):
                    iv_matrix[i, j] = iv_dict.get((exp, strike), 0.20)

            return strikes_arr, expiries_arr, iv_matrix

    def get_pnl_surface_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate P&L surface for stress test visualization.

        Returns spot changes, IV changes, and P&L matrix.
        """
        with self.db.get_session() as session:
            risk_snapshot = self.db.get_latest_risk_snapshot(session)

            if risk_snapshot:
                delta = risk_snapshot.delta or 0
                gamma = risk_snapshot.gamma or 0
                vega = risk_snapshot.vega or 0
            else:
                delta, gamma, vega = 10, 0.05, 100

        spot_changes = np.linspace(-10, 10, 21)
        iv_changes = np.linspace(-5, 5, 11)

        X, Y = np.meshgrid(spot_changes, iv_changes)

        # P&L = delta * spot_change + 0.5 * gamma * spot_change^2 + vega * iv_change
        # Assuming spot_change in % and iv_change in percentage points
        spot_dollar = X * 4.5  # ~1% move = $4.5 for SPY at 450
        Z = delta * spot_dollar + 0.5 * gamma * spot_dollar**2 + vega * Y

        return spot_changes, iv_changes, Z


# Global service instance
_demo_service: Optional[DemoDataService] = None


def get_demo_service() -> DemoDataService:
    """Get or create the global demo data service."""
    global _demo_service
    if _demo_service is None:
        _demo_service = DemoDataService()
    return _demo_service
