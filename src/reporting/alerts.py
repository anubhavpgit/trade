"""
Alert management for risk notifications.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Callable
import asyncio

from loguru import logger

from src.config import settings
from src.analytics.risk_metrics import RiskMetrics
from src.analytics.greeks import PortfolioGreeks


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertType(str, Enum):
    """Types of alerts."""
    DELTA_BREACH = "delta_breach"
    GAMMA_BREACH = "gamma_breach"
    VAR_BREACH = "var_breach"
    DRAWDOWN = "drawdown"
    KILL_SWITCH = "kill_switch"
    HEDGE_EXECUTED = "hedge_executed"
    SYSTEM_ERROR = "system_error"


@dataclass
class Alert:
    """Represents an alert."""

    alert_id: str
    alert_type: AlertType
    severity: AlertSeverity
    message: str
    details: dict
    timestamp: datetime
    acknowledged: bool = False

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.message}"


class AlertManager:
    """
    Manage and dispatch alerts.

    Features:
    - Alert generation based on thresholds
    - Multiple notification channels (log, slack, email)
    - Alert history tracking
    - Deduplication
    """

    def __init__(self):
        self._alerts: list[Alert] = []
        self._handlers: list[Callable[[Alert], None]] = []
        self._alert_count = 0

        # Register default log handler
        self.register_handler(self._log_handler)

    def register_handler(self, handler: Callable[[Alert], None]):
        """Register an alert handler."""
        self._handlers.append(handler)

    def check_and_alert(
        self,
        greeks: PortfolioGreeks,
        risk_metrics: RiskMetrics,
    ):
        """Check conditions and generate alerts."""
        risk_config = settings.risk

        # Delta breach
        if abs(greeks.portfolio_delta) > risk_config.max_delta_exposure:
            self._create_alert(
                AlertType.DELTA_BREACH,
                AlertSeverity.WARNING,
                f"Delta ({greeks.portfolio_delta:.1f}) exceeds limit ({risk_config.max_delta_exposure})",
                {"current_delta": greeks.portfolio_delta, "limit": risk_config.max_delta_exposure},
            )

        # Gamma breach
        if abs(greeks.portfolio_gamma) > risk_config.max_gamma_exposure:
            self._create_alert(
                AlertType.GAMMA_BREACH,
                AlertSeverity.WARNING,
                f"Gamma ({greeks.portfolio_gamma:.4f}) exceeds limit ({risk_config.max_gamma_exposure})",
                {"current_gamma": greeks.portfolio_gamma, "limit": risk_config.max_gamma_exposure},
            )

        # VaR breach
        if risk_metrics.var_percent > risk_config.max_var_percent:
            severity = (
                AlertSeverity.CRITICAL
                if risk_metrics.var_percent > risk_config.max_var_percent * 1.5
                else AlertSeverity.WARNING
            )
            self._create_alert(
                AlertType.VAR_BREACH,
                severity,
                f"VaR ({risk_metrics.var_percent*100:.1f}%) exceeds limit ({risk_config.max_var_percent*100:.1f}%)",
                {"current_var": risk_metrics.var_percent, "limit": risk_config.max_var_percent},
            )

        # Drawdown breach
        if risk_metrics.drawdown_percent > risk_config.max_drawdown_percent:
            self._create_alert(
                AlertType.KILL_SWITCH,
                AlertSeverity.EMERGENCY,
                f"KILL SWITCH: Drawdown ({risk_metrics.drawdown_percent*100:.1f}%) exceeds limit",
                {"current_drawdown": risk_metrics.drawdown_percent, "limit": risk_config.max_drawdown_percent},
            )

    def _create_alert(
        self,
        alert_type: AlertType,
        severity: AlertSeverity,
        message: str,
        details: dict,
    ):
        """Create and dispatch an alert."""
        self._alert_count += 1
        alert = Alert(
            alert_id=f"ALT-{self._alert_count:06d}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            details=details,
            timestamp=datetime.now(),
        )

        self._alerts.append(alert)

        # Dispatch to handlers
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                logger.error(f"Alert handler failed: {e}")

    def _log_handler(self, alert: Alert):
        """Default handler: log the alert."""
        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(str(alert))
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.error(str(alert))
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(str(alert))
        else:
            logger.info(str(alert))

    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        limit: int = 100,
    ) -> list[Alert]:
        """Get recent alerts, optionally filtered by severity."""
        alerts = self._alerts.copy()
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        return alerts[-limit:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert."""
        for alert in self._alerts:
            if alert.alert_id == alert_id:
                alert.acknowledged = True
                return True
        return False

    def get_unacknowledged_count(self) -> int:
        """Get count of unacknowledged alerts."""
        return sum(1 for a in self._alerts if not a.acknowledged)

    def clear_old_alerts(self, keep_last: int = 1000):
        """Clear old alerts, keeping the most recent."""
        if len(self._alerts) > keep_last:
            self._alerts = self._alerts[-keep_last:]


class SlackAlertHandler:
    """Slack webhook alert handler (placeholder)."""

    def __init__(self, webhook_url: str):
        self.webhook_url = webhook_url

    async def send(self, alert: Alert):
        """Send alert to Slack."""
        # TODO: Implement Slack webhook
        pass


class EmailAlertHandler:
    """Email alert handler (placeholder)."""

    def __init__(self, smtp_config: dict):
        self.smtp_config = smtp_config

    async def send(self, alert: Alert):
        """Send alert via email."""
        # TODO: Implement email sending
        pass
