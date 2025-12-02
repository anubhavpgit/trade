"""
General utility functions.
"""

import numpy as np
import pandas as pd
from typing import Union


def round_to_tick(price: float, tick_size: float = 0.01) -> float:
    """Round price to nearest tick size."""
    return round(price / tick_size) * tick_size


def calculate_returns(
    prices: Union[pd.Series, np.ndarray],
    method: str = "log",
) -> np.ndarray:
    """
    Calculate returns from price series.

    Args:
        prices: Price series
        method: "log" for log returns, "simple" for simple returns

    Returns:
        Array of returns
    """
    prices = np.asarray(prices)

    if method == "log":
        returns = np.diff(np.log(prices))
    else:  # simple
        returns = np.diff(prices) / prices[:-1]

    return returns


def annualize_volatility(
    daily_vol: float,
    trading_days: int = 252,
) -> float:
    """Annualize daily volatility."""
    return daily_vol * np.sqrt(trading_days)


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sharpe ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sharpe ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    if np.std(excess_returns) == 0:
        return 0.0

    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(periods_per_year)


def sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate Sortino ratio.

    Args:
        returns: Array of returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Number of periods per year

    Returns:
        Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return 0.0

    downside_std = np.std(downside_returns)
    return np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year)


def max_drawdown(prices: np.ndarray) -> tuple[float, int, int]:
    """
    Calculate maximum drawdown.

    Returns:
        (max_drawdown, peak_idx, trough_idx)
    """
    prices = np.asarray(prices)
    cummax = np.maximum.accumulate(prices)
    drawdowns = (prices - cummax) / cummax

    trough_idx = np.argmin(drawdowns)
    peak_idx = np.argmax(prices[:trough_idx + 1]) if trough_idx > 0 else 0

    return float(drawdowns[trough_idx]), int(peak_idx), int(trough_idx)


def format_currency(value: float) -> str:
    """Format value as currency string."""
    if value >= 0:
        return f"${value:,.2f}"
    else:
        return f"-${abs(value):,.2f}"


def format_percent(value: float, decimals: int = 2) -> str:
    """Format value as percentage string."""
    return f"{value * 100:.{decimals}f}%"


def format_greek(value: float, greek: str) -> str:
    """Format Greek value appropriately."""
    if greek.lower() == "gamma":
        return f"{value:+.6f}"
    elif greek.lower() == "theta":
        return f"${value:+.2f}"
    elif greek.lower() in ("delta", "vega", "rho"):
        return f"{value:+.4f}"
    else:
        return f"{value:+.4f}"
