"""
Configuration management for the trading bot.
Uses Pydantic for validation and environment variable loading.
"""

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    model_config = SettingsConfigDict(env_prefix="DB_")

    host: str = "localhost"
    port: int = 5432
    name: str = "trading_bot"
    user: str = "postgres"
    password: str = ""

    @property
    def url(self) -> str:
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"


class MarketDataConfig(BaseSettings):
    """Market data provider configuration."""

    model_config = SettingsConfigDict(env_prefix="MARKET_")

    provider: Literal["polygon", "iex", "yahoo", "mock"] = "yahoo"
    polygon_api_key: str = ""
    iex_api_key: str = ""
    update_interval_seconds: int = 60


class RiskConfig(BaseSettings):
    """Risk management parameters."""

    model_config = SettingsConfigDict(env_prefix="RISK_")

    # VaR/ES parameters
    var_confidence: float = 0.95
    es_confidence: float = 0.975
    var_lookback_days: int = 252
    monte_carlo_simulations: int = 10000

    # Guardrails
    max_delta_exposure: float = 100.0  # In terms of underlying shares
    max_gamma_exposure: float = 50.0
    max_vega_exposure: float = 1000.0
    max_var_percent: float = 0.02  # 2% of portfolio
    max_drawdown_percent: float = 0.05  # 5% kill switch

    # Stress scenarios
    spot_shock_std: float = 1.0  # +/- 1 standard deviation
    iv_shock_points: float = 5.0  # +/- 5 IV points


class AgentConfig(BaseSettings):
    """Trading agent configuration."""

    model_config = SettingsConfigDict(env_prefix="AGENT_")

    mode: Literal["paper", "live", "backtest"] = "paper"
    hedge_frequency_minutes: int = 5
    delta_band_threshold: float = 10.0  # Rehedge if |delta| > threshold
    transaction_cost_bps: float = 5.0  # 5 basis points
    enable_rl_overlay: bool = False


class DashboardConfig(BaseSettings):
    """Dashboard configuration."""

    model_config = SettingsConfigDict(env_prefix="DASH_")

    host: str = "127.0.0.1"
    port: int = 8050
    debug: bool = True
    refresh_interval_ms: int = 5000


class Settings(BaseSettings):
    """Main settings container."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Sub-configurations
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    market_data: MarketDataConfig = Field(default_factory=MarketDataConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)
    agent: AgentConfig = Field(default_factory=AgentConfig)
    dashboard: DashboardConfig = Field(default_factory=DashboardConfig)

    # Paths
    project_root: Path = Path(__file__).parent.parent
    data_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "data")
    reports_dir: Path = Field(default_factory=lambda: Path(__file__).parent.parent / "reports")

    # Logging
    log_level: str = "INFO"


# Global settings instance
settings = Settings()
