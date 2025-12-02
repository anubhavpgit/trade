"""
Seed demo data for the trading bot dashboard.

Data sources:
- SPY historical prices: Yahoo Finance (via yfinance)
- VIX historical data: Yahoo Finance (^VIX)
- Options data: Synthetically generated based on historical prices

Usage:
    python -m src.data.seed_demo_data
"""

import asyncio
from datetime import datetime, date, timedelta
from typing import List, Dict
import uuid

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import norm

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
from src.analytics.pricing import BlackScholes
from src.data.models import OptionType


def download_spy_history(start_date: str = "2023-01-01", end_date: str = None) -> pd.DataFrame:
    """Download SPY historical data from Yahoo Finance."""
    try:
        import yfinance as yf

        if end_date is None:
            end_date = date.today().isoformat()

        logger.info(f"Downloading SPY data from {start_date} to {end_date}")

        spy = yf.Ticker("SPY")
        df = spy.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError("No data returned from Yahoo Finance")

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        logger.info(f"Downloaded {len(df)} days of SPY data")
        return df

    except ImportError:
        logger.warning("yfinance not installed, generating synthetic data")
        return generate_synthetic_prices("SPY", start_date, end_date)


def download_vix_history(start_date: str = "2023-01-01", end_date: str = None) -> pd.DataFrame:
    """Download VIX historical data from Yahoo Finance."""
    try:
        import yfinance as yf

        if end_date is None:
            end_date = date.today().isoformat()

        logger.info(f"Downloading VIX data from {start_date} to {end_date}")

        vix = yf.Ticker("^VIX")
        df = vix.history(start=start_date, end=end_date)

        if df.empty:
            raise ValueError("No VIX data returned")

        df = df.reset_index()
        df.columns = [c.lower() for c in df.columns]

        logger.info(f"Downloaded {len(df)} days of VIX data")
        return df

    except Exception as e:
        logger.warning(f"Could not download VIX data: {e}, generating synthetic")
        return generate_synthetic_vix(start_date, end_date)


def generate_synthetic_prices(
    symbol: str,
    start_date: str,
    end_date: str,
    initial_price: float = 450.0,
) -> pd.DataFrame:
    """Generate synthetic price data."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    # Geometric Brownian Motion
    mu = 0.0002  # Daily drift
    sigma = 0.012  # Daily volatility

    returns = np.random.normal(mu, sigma, n_days)
    prices = initial_price * np.exp(np.cumsum(returns))

    # Generate OHLC
    daily_vol = 0.008
    opens = prices * (1 + np.random.normal(0, daily_vol / 2, n_days))
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.normal(0, daily_vol, n_days)))
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.normal(0, daily_vol, n_days)))
    volumes = np.random.randint(50_000_000, 150_000_000, n_days)

    df = pd.DataFrame({
        "date": dates,
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    })

    return df


def generate_synthetic_vix(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate synthetic VIX data."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date) if end_date else pd.Timestamp.now()

    dates = pd.bdate_range(start=start, end=end)
    n_days = len(dates)

    # Mean-reverting VIX around 18-20
    vix_mean = 18.0
    vix_vol = 4.0
    mean_reversion = 0.05

    vix_values = [vix_mean]
    for _ in range(n_days - 1):
        shock = np.random.normal(0, vix_vol / np.sqrt(252))
        new_vix = vix_values[-1] + mean_reversion * (vix_mean - vix_values[-1]) + shock
        vix_values.append(max(10, min(80, new_vix)))  # Clip to reasonable range

    vix_values = np.array(vix_values)
    daily_range = 0.03

    df = pd.DataFrame({
        "date": dates,
        "open": vix_values * (1 + np.random.uniform(-daily_range, daily_range, n_days)),
        "high": vix_values * (1 + np.random.uniform(0, daily_range * 2, n_days)),
        "low": vix_values * (1 - np.random.uniform(0, daily_range * 2, n_days)),
        "close": vix_values,
    })

    return df


def generate_option_chain(
    underlying_price: float,
    snapshot_date: date,
    base_iv: float = 0.20,
) -> List[Dict]:
    """Generate synthetic option chain for a given date."""
    options = []

    # Generate expirations (weekly and monthly)
    expirations = []
    for days_out in [7, 14, 21, 30, 45, 60, 90]:
        exp_date = snapshot_date + timedelta(days=days_out)
        # Skip weekends
        while exp_date.weekday() >= 5:
            exp_date += timedelta(days=1)
        expirations.append(exp_date)

    # Generate strikes around ATM
    atm_strike = round(underlying_price / 5) * 5  # Round to nearest $5
    strikes = [atm_strike + i * 5 for i in range(-10, 11)]  # +/- $50

    for expiration in expirations:
        tte = (expiration - snapshot_date).days / 365.0
        if tte <= 0:
            continue

        for strike in strikes:
            moneyness = np.log(strike / underlying_price)

            # IV smile: higher IV for OTM options
            smile_adjustment = 0.05 * moneyness**2
            term_adjustment = 0.02 * np.sqrt(tte)
            iv = base_iv + smile_adjustment + term_adjustment
            iv = max(0.05, min(1.0, iv))  # Clip IV

            for opt_type in [OptionType.CALL, OptionType.PUT]:
                # Calculate theoretical price and Greeks
                price = BlackScholes.price(
                    underlying_price, strike, tte, 0.05, iv, opt_type
                )
                delta = BlackScholes.delta(
                    underlying_price, strike, tte, 0.05, iv, opt_type
                )
                gamma = BlackScholes.gamma(underlying_price, strike, tte, 0.05, iv)
                theta = BlackScholes.theta(
                    underlying_price, strike, tte, 0.05, iv, opt_type
                )
                vega = BlackScholes.vega(underlying_price, strike, tte, 0.05, iv)

                # Add bid/ask spread
                spread = max(0.05, price * 0.02)
                bid = max(0.01, price - spread / 2)
                ask = price + spread / 2

                # Volume and OI (higher for ATM)
                atm_factor = np.exp(-2 * moneyness**2)
                volume = int(np.random.poisson(500 * atm_factor))
                oi = int(np.random.poisson(5000 * atm_factor))

                symbol = f"SPY{expiration.strftime('%y%m%d')}{opt_type.value[0].upper()}{int(strike)}"

                options.append({
                    "snapshot_date": snapshot_date,
                    "symbol": symbol,
                    "underlying": "SPY",
                    "option_type": opt_type.value,
                    "strike": strike,
                    "expiration": expiration,
                    "bid": round(bid, 2),
                    "ask": round(ask, 2),
                    "last": round(price, 2),
                    "volume": volume,
                    "open_interest": oi,
                    "implied_volatility": round(iv, 4),
                    "delta": round(delta, 4),
                    "gamma": round(gamma, 6),
                    "theta": round(theta, 4),
                    "vega": round(vega, 4),
                    "underlying_price": underlying_price,
                })

    return options


def generate_sample_positions(underlying_price: float) -> List[Dict]:
    """Generate sample portfolio positions for demo."""
    positions = []
    today = date.today()

    # Position 1: Short straddle (classic delta-neutral income strategy)
    exp1 = today + timedelta(days=30)
    while exp1.weekday() >= 5:
        exp1 += timedelta(days=1)

    atm_strike = round(underlying_price / 5) * 5

    positions.append({
        "symbol": f"SPY{exp1.strftime('%y%m%d')}C{int(atm_strike)}",
        "underlying": "SPY",
        "position_type": "option",
        "option_type": "call",
        "strike": atm_strike,
        "expiration": exp1,
        "quantity": -10,  # Short 10 calls
        "avg_cost": 5.50,
        "opened_at": datetime.now() - timedelta(days=5),
        "is_active": True,
    })

    positions.append({
        "symbol": f"SPY{exp1.strftime('%y%m%d')}P{int(atm_strike)}",
        "underlying": "SPY",
        "position_type": "option",
        "option_type": "put",
        "strike": atm_strike,
        "expiration": exp1,
        "quantity": -10,  # Short 10 puts
        "avg_cost": 5.20,
        "opened_at": datetime.now() - timedelta(days=5),
        "is_active": True,
    })

    # Position 2: Long protective put
    positions.append({
        "symbol": f"SPY{exp1.strftime('%y%m%d')}P{int(atm_strike - 20)}",
        "underlying": "SPY",
        "position_type": "option",
        "option_type": "put",
        "strike": atm_strike - 20,
        "expiration": exp1,
        "quantity": 5,  # Long 5 puts
        "avg_cost": 1.20,
        "opened_at": datetime.now() - timedelta(days=3),
        "is_active": True,
    })

    # Position 3: Stock hedge position
    positions.append({
        "symbol": "SPY",
        "underlying": "SPY",
        "position_type": "stock",
        "option_type": None,
        "strike": None,
        "expiration": None,
        "quantity": 150,  # Long 150 shares
        "avg_cost": underlying_price - 2,
        "opened_at": datetime.now() - timedelta(days=10),
        "is_active": True,
    })

    return positions


def generate_sample_trades(positions: List[Dict]) -> List[Dict]:
    """Generate sample trade history."""
    trades = []

    for i, pos in enumerate(positions):
        trade = {
            "trade_id": f"TRD-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": pos["opened_at"],
            "symbol": pos["symbol"],
            "side": "sell" if pos["quantity"] < 0 else "buy",
            "quantity": abs(pos["quantity"]),
            "price": pos["avg_cost"],
            "commission": round(abs(pos["quantity"]) * 0.65, 2),
            "slippage": round(abs(pos["quantity"]) * 0.02, 2),
            "pnl": None,
            "position_id": i + 1,
        }
        trades.append(trade)

    # Add some hedge trades
    for days_ago in [4, 2, 1]:
        trades.append({
            "trade_id": f"TRD-{uuid.uuid4().hex[:8].upper()}",
            "timestamp": datetime.now() - timedelta(days=days_ago, hours=np.random.randint(1, 6)),
            "symbol": "SPY",
            "side": np.random.choice(["buy", "sell"]),
            "quantity": np.random.randint(10, 50),
            "price": round(450 + np.random.uniform(-5, 5), 2),
            "commission": round(np.random.uniform(0.5, 2), 2),
            "slippage": round(np.random.uniform(0.01, 0.1), 2),
            "pnl": round(np.random.uniform(-50, 100), 2),
            "position_id": None,
        })

    return trades


def generate_risk_snapshots(num_snapshots: int = 200) -> List[Dict]:
    """Generate historical risk snapshots for charts."""
    snapshots = []
    base_time = datetime.now() - timedelta(hours=num_snapshots * 0.5)

    portfolio_value = 100000
    cumulative_pnl = 0

    for i in range(num_snapshots):
        timestamp = base_time + timedelta(minutes=i * 30)

        # Simulate P&L changes
        daily_pnl = np.random.normal(50, 200)
        cumulative_pnl += daily_pnl
        portfolio_value += daily_pnl

        # Simulate Greeks (with some autocorrelation)
        if i == 0:
            delta = np.random.uniform(-20, 20)
            gamma = np.random.uniform(0.01, 0.05)
            vega = np.random.uniform(50, 150)
        else:
            delta = snapshots[-1]["delta"] + np.random.normal(0, 5)
            gamma = max(0.001, snapshots[-1]["gamma"] + np.random.normal(0, 0.005))
            vega = max(10, snapshots[-1]["vega"] + np.random.normal(0, 10))

        # Clip values
        delta = np.clip(delta, -100, 100)

        theta = -abs(delta) * 0.5 + np.random.normal(-30, 10)

        # Risk metrics
        var_95 = abs(delta) * 5 + np.random.uniform(500, 1500)
        var_99 = var_95 * 1.3

        # IV metrics
        iv_atm = 0.18 + np.random.uniform(-0.03, 0.03)
        realized_vol = 0.15 + np.random.uniform(-0.02, 0.02)
        iv_percentile = np.random.uniform(30, 70)

        # Drawdown
        peak = max(portfolio_value, 100000)
        drawdown = (peak - portfolio_value) / peak

        snapshots.append({
            "timestamp": timestamp,
            "portfolio_value": round(portfolio_value, 2),
            "pnl_daily": round(daily_pnl, 2),
            "pnl_cumulative": round(cumulative_pnl, 2),
            "delta": round(delta, 2),
            "gamma": round(gamma, 4),
            "theta": round(theta, 2),
            "vega": round(vega, 2),
            "var_95": round(var_95, 2),
            "var_99": round(var_99, 2),
            "es_95": round(var_95 * 1.2, 2),
            "drawdown": round(drawdown, 4),
            "iv_atm": round(iv_atm, 4),
            "realized_vol": round(realized_vol, 4),
            "iv_percentile": round(iv_percentile, 1),
        })

    return snapshots


def generate_sample_alerts() -> List[Dict]:
    """Generate sample alerts."""
    alerts = []
    alert_types = [
        ("HEDGE", "info", "Delta hedge executed: Sold 25 shares SPY @ $451.23"),
        ("HEDGE", "info", "Delta hedge executed: Bought 15 shares SPY @ $449.87"),
        ("WARNING", "warning", "Delta (45.2) approaching limit (50.0)"),
        ("WARNING", "warning", "VaR (1.8%) approaching limit (2.0%)"),
        ("INFO", "info", "Market open - Starting daily monitoring"),
        ("INFO", "info", "Morning risk report generated"),
        ("GAMMA", "warning", "Gamma exposure elevated due to near-term expiration"),
    ]

    for i, (alert_type, severity, message) in enumerate(alert_types):
        alerts.append({
            "timestamp": datetime.now() - timedelta(hours=i * 2, minutes=np.random.randint(0, 60)),
            "alert_type": alert_type,
            "severity": severity,
            "message": message,
            "acknowledged": i > 3,
            "acknowledged_at": datetime.now() - timedelta(hours=i) if i > 3 else None,
        })

    return alerts


def seed_database():
    """Main function to seed the database with demo data."""
    logger.info("Starting database seeding...")

    db = get_db_manager()

    # Download or generate historical data
    logger.info("Fetching SPY historical data...")
    spy_df = download_spy_history("2023-06-01")

    logger.info("Fetching VIX historical data...")
    vix_df = download_vix_history("2023-06-01")

    with db.get_session() as session:
        # Seed price history
        logger.info("Seeding price history...")
        price_records = []
        for _, row in spy_df.iterrows():
            record_date = row["date"]
            if hasattr(record_date, "date"):
                record_date = record_date.date()

            price_records.append({
                "symbol": "SPY",
                "date": record_date,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
                "volume": int(row["volume"]) if pd.notna(row.get("volume")) else 0,
                "adjusted_close": float(row["close"]),
            })
        db.insert_price_history(session, price_records)
        logger.info(f"Inserted {len(price_records)} price records")

        # Seed volatility history
        logger.info("Seeding volatility history...")
        vol_records = []
        for _, row in vix_df.iterrows():
            record_date = row["date"]
            if hasattr(record_date, "date"):
                record_date = record_date.date()

            vol_records.append({
                "date": record_date,
                "vix_open": float(row["open"]),
                "vix_high": float(row["high"]),
                "vix_low": float(row["low"]),
                "vix_close": float(row["close"]),
            })
        db.insert_volatility_history(session, vol_records)
        logger.info(f"Inserted {len(vol_records)} volatility records")

        # Get latest price for option generation
        latest_price = db.get_latest_price(session, "SPY")
        underlying_price = latest_price.close if latest_price else 450.0

        # Seed option snapshots for recent dates
        logger.info("Generating option chain snapshots...")
        for days_ago in [0, 1, 2, 3, 5]:
            snapshot_date = date.today() - timedelta(days=days_ago)
            if snapshot_date.weekday() >= 5:
                continue

            # Get price for that date
            price_record = session.query(DBPriceHistory).filter_by(
                symbol="SPY", date=snapshot_date
            ).first()
            price = price_record.close if price_record else underlying_price

            # Get VIX for that date
            vix_record = session.query(DBVolatilityHistory).filter_by(
                date=snapshot_date
            ).first()
            base_iv = vix_record.vix_close / 100 if vix_record else 0.18

            options = generate_option_chain(price, snapshot_date, base_iv)
            db.insert_option_snapshots(session, options)
            logger.info(f"Inserted {len(options)} options for {snapshot_date}")

        # Seed positions
        logger.info("Seeding sample positions...")
        positions = generate_sample_positions(underlying_price)
        for pos in positions:
            db.insert_position(session, pos)
        logger.info(f"Inserted {len(positions)} positions")

        # Seed trades
        logger.info("Seeding sample trades...")
        trades = generate_sample_trades(positions)
        for trade in trades:
            db.insert_trade(session, trade)
        logger.info(f"Inserted {len(trades)} trades")

        # Seed risk snapshots
        logger.info("Seeding risk snapshots...")
        risk_snapshots = generate_risk_snapshots(200)
        for snapshot in risk_snapshots:
            db.insert_risk_snapshot(session, snapshot)
        logger.info(f"Inserted {len(risk_snapshots)} risk snapshots")

        # Seed alerts
        logger.info("Seeding sample alerts...")
        alerts = generate_sample_alerts()
        for alert in alerts:
            db.insert_alert(session, alert)
        logger.info(f"Inserted {len(alerts)} alerts")

    logger.info("Database seeding completed successfully!")
    logger.info(f"Database location: {db.db_url}")


if __name__ == "__main__":
    seed_database()
