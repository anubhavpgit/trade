"""
SQLite database layer with SQLAlchemy ORM.
Local database for demo and backtesting purposes.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Optional, List

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    Float,
    String,
    DateTime,
    Date,
    Boolean,
    Enum as SQLEnum,
    ForeignKey,
    Index,
    func,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from loguru import logger

from src.config import settings

Base = declarative_base()


# =============================================================================
# Database Models
# =============================================================================

class DBPriceHistory(Base):
    """Historical price data for underlying assets."""

    __tablename__ = "price_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    date = Column(Date, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Integer)
    adjusted_close = Column(Float)

    __table_args__ = (
        Index("ix_price_symbol_date", "symbol", "date", unique=True),
    )


class DBVolatilityHistory(Base):
    """Historical volatility data (VIX, realized vol, etc.)."""

    __tablename__ = "volatility_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(Date, nullable=False, index=True, unique=True)
    vix_open = Column(Float)
    vix_high = Column(Float)
    vix_low = Column(Float)
    vix_close = Column(Float, nullable=False)
    realized_vol_20d = Column(Float)
    realized_vol_60d = Column(Float)


class DBOptionSnapshot(Base):
    """Option chain snapshots for historical analysis."""

    __tablename__ = "option_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_date = Column(Date, nullable=False, index=True)
    symbol = Column(String(50), nullable=False)
    underlying = Column(String(20), nullable=False, index=True)
    option_type = Column(String(4), nullable=False)  # 'call' or 'put'
    strike = Column(Float, nullable=False)
    expiration = Column(Date, nullable=False)
    bid = Column(Float)
    ask = Column(Float)
    last = Column(Float)
    volume = Column(Integer)
    open_interest = Column(Integer)
    implied_volatility = Column(Float)
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)
    underlying_price = Column(Float)

    __table_args__ = (
        Index("ix_option_date_underlying", "snapshot_date", "underlying"),
        Index("ix_option_expiry", "expiration"),
    )


class DBPosition(Base):
    """Portfolio positions."""

    __tablename__ = "positions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(50), nullable=False)
    underlying = Column(String(20), nullable=False)
    position_type = Column(String(10), nullable=False)  # 'option' or 'stock'
    option_type = Column(String(4))  # 'call' or 'put' if option
    strike = Column(Float)
    expiration = Column(Date)
    quantity = Column(Integer, nullable=False)
    avg_cost = Column(Float, nullable=False)
    opened_at = Column(DateTime, default=datetime.now)
    closed_at = Column(DateTime)
    is_active = Column(Boolean, default=True)


class DBTrade(Base):
    """Trade history."""

    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    trade_id = Column(String(20), unique=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    symbol = Column(String(50), nullable=False)
    side = Column(String(4), nullable=False)  # 'buy' or 'sell'
    quantity = Column(Integer, nullable=False)
    price = Column(Float, nullable=False)
    commission = Column(Float, default=0)
    slippage = Column(Float, default=0)
    pnl = Column(Float)
    position_id = Column(Integer, ForeignKey("positions.id"))


class DBRiskSnapshot(Base):
    """Risk metrics snapshots over time."""

    __tablename__ = "risk_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    portfolio_value = Column(Float, nullable=False)
    pnl_daily = Column(Float)
    pnl_cumulative = Column(Float)

    # Greeks
    delta = Column(Float)
    gamma = Column(Float)
    theta = Column(Float)
    vega = Column(Float)

    # Risk metrics
    var_95 = Column(Float)
    var_99 = Column(Float)
    es_95 = Column(Float)
    drawdown = Column(Float)

    # Volatility
    iv_atm = Column(Float)
    realized_vol = Column(Float)
    iv_percentile = Column(Float)


class DBAlert(Base):
    """Alert history."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    alert_type = Column(String(30), nullable=False)
    severity = Column(String(15), nullable=False)
    message = Column(String(500), nullable=False)
    acknowledged = Column(Boolean, default=False)
    acknowledged_at = Column(DateTime)


# =============================================================================
# Database Manager
# =============================================================================

class DatabaseManager:
    """Manage database connections and operations."""

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            db_path = settings.data_dir / "trading_bot.db"

        db_path.parent.mkdir(parents=True, exist_ok=True)
        self.db_url = f"sqlite:///{db_path}"
        self.engine = create_engine(self.db_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine)

        logger.info(f"Database initialized at: {db_path}")

    def create_tables(self):
        """Create all database tables."""
        Base.metadata.create_all(self.engine)
        logger.info("Database tables created")

    def drop_tables(self):
        """Drop all database tables."""
        Base.metadata.drop_all(self.engine)
        logger.warning("Database tables dropped")

    def get_session(self) -> Session:
        """Get a new database session."""
        return self.SessionLocal()

    # =========================================================================
    # Price History Operations
    # =========================================================================

    def insert_price_history(self, session: Session, records: List[dict]):
        """Insert price history records."""
        for record in records:
            existing = session.query(DBPriceHistory).filter_by(
                symbol=record["symbol"],
                date=record["date"]
            ).first()

            if not existing:
                session.add(DBPriceHistory(**record))

        session.commit()

    def get_price_history(
        self,
        session: Session,
        symbol: str,
        start_date: date,
        end_date: date,
    ) -> List[DBPriceHistory]:
        """Get price history for a symbol."""
        return session.query(DBPriceHistory).filter(
            DBPriceHistory.symbol == symbol,
            DBPriceHistory.date >= start_date,
            DBPriceHistory.date <= end_date,
        ).order_by(DBPriceHistory.date).all()

    def get_latest_price(self, session: Session, symbol: str) -> Optional[DBPriceHistory]:
        """Get the most recent price for a symbol."""
        return session.query(DBPriceHistory).filter_by(
            symbol=symbol
        ).order_by(DBPriceHistory.date.desc()).first()

    # =========================================================================
    # Volatility History Operations
    # =========================================================================

    def insert_volatility_history(self, session: Session, records: List[dict]):
        """Insert volatility history records."""
        for record in records:
            existing = session.query(DBVolatilityHistory).filter_by(
                date=record["date"]
            ).first()

            if not existing:
                session.add(DBVolatilityHistory(**record))

        session.commit()

    def get_vix_history(
        self,
        session: Session,
        start_date: date,
        end_date: date,
    ) -> List[DBVolatilityHistory]:
        """Get VIX history."""
        return session.query(DBVolatilityHistory).filter(
            DBVolatilityHistory.date >= start_date,
            DBVolatilityHistory.date <= end_date,
        ).order_by(DBVolatilityHistory.date).all()

    # =========================================================================
    # Option Snapshot Operations
    # =========================================================================

    def insert_option_snapshots(self, session: Session, records: List[dict]):
        """Insert option snapshots."""
        for record in records:
            session.add(DBOptionSnapshot(**record))
        session.commit()

    def get_option_chain_snapshot(
        self,
        session: Session,
        underlying: str,
        snapshot_date: date,
    ) -> List[DBOptionSnapshot]:
        """Get option chain for a specific date."""
        return session.query(DBOptionSnapshot).filter(
            DBOptionSnapshot.underlying == underlying,
            DBOptionSnapshot.snapshot_date == snapshot_date,
        ).all()

    # =========================================================================
    # Position Operations
    # =========================================================================

    def get_active_positions(self, session: Session) -> List[DBPosition]:
        """Get all active positions."""
        return session.query(DBPosition).filter_by(is_active=True).all()

    def insert_position(self, session: Session, position: dict) -> DBPosition:
        """Insert a new position."""
        db_position = DBPosition(**position)
        session.add(db_position)
        session.commit()
        return db_position

    def close_position(self, session: Session, position_id: int):
        """Close a position."""
        position = session.query(DBPosition).filter_by(id=position_id).first()
        if position:
            position.is_active = False
            position.closed_at = datetime.now()
            session.commit()

    # =========================================================================
    # Trade Operations
    # =========================================================================

    def insert_trade(self, session: Session, trade: dict) -> DBTrade:
        """Insert a trade record."""
        db_trade = DBTrade(**trade)
        session.add(db_trade)
        session.commit()
        return db_trade

    def get_trades(
        self,
        session: Session,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[DBTrade]:
        """Get trade history."""
        query = session.query(DBTrade)

        if start_date:
            query = query.filter(DBTrade.timestamp >= start_date)
        if end_date:
            query = query.filter(DBTrade.timestamp <= end_date)

        return query.order_by(DBTrade.timestamp.desc()).limit(limit).all()

    # =========================================================================
    # Risk Snapshot Operations
    # =========================================================================

    def insert_risk_snapshot(self, session: Session, snapshot: dict) -> DBRiskSnapshot:
        """Insert a risk snapshot."""
        db_snapshot = DBRiskSnapshot(**snapshot)
        session.add(db_snapshot)
        session.commit()
        return db_snapshot

    def get_risk_history(
        self,
        session: Session,
        hours: int = 24,
    ) -> List[DBRiskSnapshot]:
        """Get recent risk snapshots."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return session.query(DBRiskSnapshot).filter(
            DBRiskSnapshot.timestamp >= cutoff
        ).order_by(DBRiskSnapshot.timestamp).all()

    def get_latest_risk_snapshot(self, session: Session) -> Optional[DBRiskSnapshot]:
        """Get the most recent risk snapshot."""
        return session.query(DBRiskSnapshot).order_by(
            DBRiskSnapshot.timestamp.desc()
        ).first()

    # =========================================================================
    # Alert Operations
    # =========================================================================

    def insert_alert(self, session: Session, alert: dict) -> DBAlert:
        """Insert an alert."""
        db_alert = DBAlert(**alert)
        session.add(db_alert)
        session.commit()
        return db_alert

    def get_recent_alerts(
        self,
        session: Session,
        limit: int = 50,
        unacknowledged_only: bool = False,
    ) -> List[DBAlert]:
        """Get recent alerts."""
        query = session.query(DBAlert)

        if unacknowledged_only:
            query = query.filter_by(acknowledged=False)

        return query.order_by(DBAlert.timestamp.desc()).limit(limit).all()


# Import for timedelta
from datetime import timedelta


# Global database manager instance
_db_manager: Optional[DatabaseManager] = None


def get_db_manager() -> DatabaseManager:
    """Get or create the global database manager."""
    global _db_manager
    if _db_manager is None:
        _db_manager = DatabaseManager()
        _db_manager.create_tables()
    return _db_manager
