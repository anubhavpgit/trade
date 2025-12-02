"""
Delta-neutral hedging agent with rule-based core.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional

from loguru import logger

from src.config import settings
from src.data.models import OrderSide
from src.analytics.greeks import PortfolioGreeks
from src.analytics.risk_metrics import RiskMetrics
from src.agent.guardrails import RiskGuardrails


class ActionType(str, Enum):
    """Types of hedging actions."""
    HOLD = "hold"
    HEDGE_DELTA = "hedge_delta"
    REDUCE_GAMMA = "reduce_gamma"
    REDUCE_VEGA = "reduce_vega"
    CLOSE_ALL = "close_all"


@dataclass
class HedgeAction:
    """Represents a hedging action to be executed."""

    action_type: ActionType
    should_hedge: bool
    shares_to_trade: int
    side: Optional[OrderSide]
    reason: str
    urgency: str  # "low", "medium", "high", "critical"
    timestamp: datetime

    def __str__(self) -> str:
        if not self.should_hedge:
            return f"HOLD: {self.reason}"
        direction = self.side.value if self.side else "N/A"
        return (
            f"{self.action_type.value.upper()}: "
            f"{direction} {abs(self.shares_to_trade)} shares | "
            f"Urgency: {self.urgency} | {self.reason}"
        )


class DeltaHedgingAgent:
    """
    Rule-based delta-neutral hedging agent.

    Decision flow:
    1. Check guardrails (kill switch conditions)
    2. Check if delta exceeds threshold
    3. Calculate hedge size
    4. Validate against position limits
    5. Return action
    """

    def __init__(self):
        self.config = settings.agent
        self.risk_config = settings.risk
        self.guardrails = RiskGuardrails()
        self._last_hedge_time: Optional[datetime] = None
        self._hedge_count_today: int = 0

    def decide(
        self,
        greeks: PortfolioGreeks,
        risk_metrics: RiskMetrics,
    ) -> HedgeAction:
        """
        Make hedging decision based on current state.

        Args:
            greeks: Current portfolio Greeks
            risk_metrics: Current risk metrics

        Returns:
            HedgeAction with the decision
        """
        timestamp = datetime.now()

        # Check guardrails first
        guardrail_action = self.guardrails.check(greeks, risk_metrics)
        if guardrail_action:
            return guardrail_action

        # Check if we should hedge delta
        current_delta = greeks.portfolio_delta
        delta_threshold = self.config.delta_band_threshold

        if abs(current_delta) <= delta_threshold:
            return HedgeAction(
                action_type=ActionType.HOLD,
                should_hedge=False,
                shares_to_trade=0,
                side=None,
                reason=f"Delta ({current_delta:.1f}) within band (+/- {delta_threshold})",
                urgency="low",
                timestamp=timestamp,
            )

        # Calculate shares needed to hedge
        target_delta = 0.0  # Delta-neutral
        shares_to_trade = round(target_delta - current_delta)

        if shares_to_trade == 0:
            return HedgeAction(
                action_type=ActionType.HOLD,
                should_hedge=False,
                shares_to_trade=0,
                side=None,
                reason="No hedge needed (rounded to 0 shares)",
                urgency="low",
                timestamp=timestamp,
            )

        # Determine side
        side = OrderSide.BUY if shares_to_trade > 0 else OrderSide.SELL

        # Check cost-effectiveness
        estimated_cost = self._estimate_hedge_cost(abs(shares_to_trade), risk_metrics)
        if not self._is_cost_effective(estimated_cost, greeks, risk_metrics):
            return HedgeAction(
                action_type=ActionType.HOLD,
                should_hedge=False,
                shares_to_trade=0,
                side=None,
                reason=f"Hedge not cost-effective (cost: ${estimated_cost:.2f})",
                urgency="low",
                timestamp=timestamp,
            )

        # Determine urgency
        urgency = self._determine_urgency(current_delta, risk_metrics)

        self._last_hedge_time = timestamp
        self._hedge_count_today += 1

        return HedgeAction(
            action_type=ActionType.HEDGE_DELTA,
            should_hedge=True,
            shares_to_trade=abs(shares_to_trade),
            side=side,
            reason=f"Delta ({current_delta:.1f}) outside band, hedging to neutral",
            urgency=urgency,
            timestamp=timestamp,
        )

    def _estimate_hedge_cost(
        self,
        shares: int,
        risk_metrics: RiskMetrics,
    ) -> float:
        """Estimate transaction cost of hedge."""
        # Assume we have an estimate of current price
        # Using portfolio value as proxy
        estimated_price = risk_metrics.portfolio_value / 100  # Rough estimate
        notional = shares * estimated_price
        cost_bps = self.config.transaction_cost_bps
        return notional * cost_bps / 10000

    def _is_cost_effective(
        self,
        estimated_cost: float,
        greeks: PortfolioGreeks,
        risk_metrics: RiskMetrics,
    ) -> bool:
        """
        Check if hedge is cost-effective.

        Compare transaction cost vs potential benefit from
        reducing delta exposure.
        """
        # Simple heuristic: hedge if cost is less than 0.1% of VaR reduction
        var_reduction = abs(greeks.portfolio_delta) * risk_metrics.realized_volatility
        return estimated_cost < var_reduction * 0.1

    def _determine_urgency(
        self,
        current_delta: float,
        risk_metrics: RiskMetrics,
    ) -> str:
        """Determine urgency level of hedge."""
        delta_threshold = self.config.delta_band_threshold

        if abs(current_delta) > delta_threshold * 3:
            return "critical"
        elif abs(current_delta) > delta_threshold * 2:
            return "high"
        elif abs(current_delta) > delta_threshold * 1.5:
            return "medium"
        else:
            return "low"

    def get_action_plan(
        self,
        greeks: PortfolioGreeks,
        risk_metrics: RiskMetrics,
    ) -> list[str]:
        """
        Generate action plan summary for daily report.

        Returns list of planned actions/considerations.
        """
        plan = []

        # Delta assessment
        delta = greeks.portfolio_delta
        threshold = self.config.delta_band_threshold

        if abs(delta) > threshold:
            plan.append(
                f"- HEDGE: Delta ({delta:.1f}) outside band. "
                f"Will hedge {round(-delta)} shares to neutralize."
            )
        else:
            plan.append(
                f"- HOLD: Delta ({delta:.1f}) within acceptable band (+/- {threshold})."
            )

        # Gamma assessment
        gamma = greeks.portfolio_gamma
        if abs(gamma) > self.risk_config.max_gamma_exposure:
            plan.append(
                f"- WARNING: Gamma ({gamma:.4f}) exceeds limit. "
                f"Consider adjusting option positions."
            )

        # Vega assessment
        vega = greeks.portfolio_vega
        if abs(vega) > self.risk_config.max_vega_exposure:
            plan.append(
                f"- WARNING: Vega ({vega:.1f}) exceeds limit. "
                f"Volatility exposure is high."
            )

        # Theta
        theta = greeks.portfolio_theta
        plan.append(f"- THETA: Daily time decay is ${theta:.2f}.")

        # VaR check
        if risk_metrics.var_percent > self.risk_config.max_var_percent * 0.8:
            plan.append(
                f"- CAUTION: VaR ({risk_metrics.var_percent*100:.1f}%) "
                f"approaching limit ({self.risk_config.max_var_percent*100:.1f}%)."
            )

        return plan

    def reset_daily_counters(self):
        """Reset daily counters (call at start of trading day)."""
        self._hedge_count_today = 0
