"""
Risk guardrails for the trading agent.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from loguru import logger

from src.config import settings
from src.data.models import OrderSide
from src.analytics.greeks import PortfolioGreeks
from src.analytics.risk_metrics import RiskMetrics


@dataclass
class GuardrailBreach:
    """Represents a guardrail breach."""

    rule_name: str
    current_value: float
    limit_value: float
    severity: str  # "warning", "critical", "kill_switch"
    message: str


class RiskGuardrails:
    """
    Risk guardrails that enforce strict limits.

    Rules:
    1. Delta bands
    2. Gamma caps
    3. Vega limits
    4. VaR limits
    5. Drawdown kill switch
    """

    def __init__(self):
        self.config = settings.risk
        self._breaches: list[GuardrailBreach] = []

    def check(
        self,
        greeks: PortfolioGreeks,
        risk_metrics: RiskMetrics,
    ) -> Optional["HedgeAction"]:
        """
        Check all guardrails.

        Returns HedgeAction if immediate action required, None otherwise.
        """
        from src.agent.hedging_agent import HedgeAction, ActionType

        self._breaches = []

        # Check each guardrail
        self._check_delta(greeks)
        self._check_gamma(greeks)
        self._check_vega(greeks)
        self._check_var(risk_metrics)
        self._check_drawdown(risk_metrics)

        # Log breaches
        for breach in self._breaches:
            if breach.severity == "kill_switch":
                logger.critical(f"KILL SWITCH: {breach.message}")
            elif breach.severity == "critical":
                logger.error(f"CRITICAL: {breach.message}")
            else:
                logger.warning(f"WARNING: {breach.message}")

        # Return action for kill switch conditions
        kill_switch_breaches = [
            b for b in self._breaches if b.severity == "kill_switch"
        ]

        if kill_switch_breaches:
            return HedgeAction(
                action_type=ActionType.CLOSE_ALL,
                should_hedge=True,
                shares_to_trade=0,  # Will be determined by executor
                side=None,
                reason=f"Kill switch triggered: {kill_switch_breaches[0].message}",
                urgency="critical",
                timestamp=datetime.now(),
            )

        return None

    def _check_delta(self, greeks: PortfolioGreeks):
        """Check delta limits."""
        max_delta = self.config.max_delta_exposure

        if abs(greeks.portfolio_delta) > max_delta * 2:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="delta_limit",
                    current_value=abs(greeks.portfolio_delta),
                    limit_value=max_delta,
                    severity="critical",
                    message=f"Delta ({greeks.portfolio_delta:.1f}) far exceeds limit ({max_delta})",
                )
            )
        elif abs(greeks.portfolio_delta) > max_delta:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="delta_limit",
                    current_value=abs(greeks.portfolio_delta),
                    limit_value=max_delta,
                    severity="warning",
                    message=f"Delta ({greeks.portfolio_delta:.1f}) exceeds limit ({max_delta})",
                )
            )

    def _check_gamma(self, greeks: PortfolioGreeks):
        """Check gamma limits."""
        max_gamma = self.config.max_gamma_exposure

        if abs(greeks.portfolio_gamma) > max_gamma:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="gamma_limit",
                    current_value=abs(greeks.portfolio_gamma),
                    limit_value=max_gamma,
                    severity="warning",
                    message=f"Gamma ({greeks.portfolio_gamma:.4f}) exceeds limit ({max_gamma})",
                )
            )

    def _check_vega(self, greeks: PortfolioGreeks):
        """Check vega limits."""
        max_vega = self.config.max_vega_exposure

        if abs(greeks.portfolio_vega) > max_vega:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="vega_limit",
                    current_value=abs(greeks.portfolio_vega),
                    limit_value=max_vega,
                    severity="warning",
                    message=f"Vega ({greeks.portfolio_vega:.1f}) exceeds limit ({max_vega})",
                )
            )

    def _check_var(self, risk_metrics: RiskMetrics):
        """Check VaR limits."""
        max_var = self.config.max_var_percent

        if risk_metrics.var_percent > max_var * 1.5:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="var_limit",
                    current_value=risk_metrics.var_percent,
                    limit_value=max_var,
                    severity="critical",
                    message=f"VaR ({risk_metrics.var_percent*100:.1f}%) far exceeds limit ({max_var*100:.1f}%)",
                )
            )
        elif risk_metrics.var_percent > max_var:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="var_limit",
                    current_value=risk_metrics.var_percent,
                    limit_value=max_var,
                    severity="warning",
                    message=f"VaR ({risk_metrics.var_percent*100:.1f}%) exceeds limit ({max_var*100:.1f}%)",
                )
            )

    def _check_drawdown(self, risk_metrics: RiskMetrics):
        """Check drawdown kill switch."""
        max_drawdown = self.config.max_drawdown_percent

        if risk_metrics.drawdown_percent > max_drawdown:
            self._breaches.append(
                GuardrailBreach(
                    rule_name="drawdown_kill_switch",
                    current_value=risk_metrics.drawdown_percent,
                    limit_value=max_drawdown,
                    severity="kill_switch",
                    message=f"Drawdown ({risk_metrics.drawdown_percent*100:.1f}%) exceeds kill switch ({max_drawdown*100:.1f}%)",
                )
            )

    def get_breaches(self) -> list[GuardrailBreach]:
        """Get current breaches."""
        return self._breaches.copy()

    def get_status_summary(self) -> dict:
        """Get guardrail status summary."""
        return {
            "total_breaches": len(self._breaches),
            "critical_breaches": sum(
                1 for b in self._breaches if b.severity in ("critical", "kill_switch")
            ),
            "warnings": sum(1 for b in self._breaches if b.severity == "warning"),
            "breaches": [
                {
                    "rule": b.rule_name,
                    "severity": b.severity,
                    "message": b.message,
                }
                for b in self._breaches
            ],
        }
