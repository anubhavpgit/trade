"""
Logging configuration.
"""

import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def setup_logging():
    """Configure application logging."""
    # Remove default handler
    logger.remove()

    # Console handler
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format=(
            "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
            "<level>{level: <8}</level> | "
            "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
            "<level>{message}</level>"
        ),
        colorize=True,
    )

    # File handler
    log_dir = settings.project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    logger.add(
        log_dir / "trading_bot_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    # Error-only file
    logger.add(
        log_dir / "errors_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="30 days",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
    )

    logger.info("Logging configured successfully")
