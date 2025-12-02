"""
Scenario analysis and stress testing engine.
"""

from dataclasses import dataclass
from typing import Optional
from enum import Enum

import numpy as np
from loguru import logger

from src.config import settings
from src.data.models import Position, MarketSnapshot, IVSurface
from src.analytics.pricing import BlackScholes
from src.analytics.greeks import GreeksCalculator


class ScenarioType(str, Enum):
    """Types of stress scenarios."""
    SPOT_UP = "spot_up"
    SPOT_DOWN = "spot_down"
    VOL_UP = "vol_up"
    VOL_DOWN = "vol_down"
    SPOT_UP_VOL_UP = "spot_up_vol_up"
    SPOT_DOWN_VOL_UP = "spot_down_vol_up"
    CUSTOM = "custom"


@dataclass
class ScenarioResult:
    """Result of a stress scenario."""

    name: str
    scenario_type: ScenarioType
    spot_change_pct: float
    iv_change_points: float
    pnl: float
    new_delta: float
    new_gamma: float
    new_vega: float
    breaches_var: bool
    breaches_es: bool

    def __str__(self) -> str:
        direction = "+" if self.pnl >= 0 else ""
        return (
            f"{self.name}: Spot {self.spot_change_pct:+.1f}%, "
            f"IV {self.iv_change_points:+.1f}pts -> "
            f"P&L {direction}${self.pnl:,.0f}"
        )


@dataclass
class StressTestReport:
    """Complete stress test report."""

    scenarios: list[ScenarioResult]
    worst_case_pnl: float
    best_case_pnl: float
    average_pnl: float
    max_delta_exposure: float
    var_breaches: int
    timestamp: str

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "STRESS TEST REPORT",
            "=" * 50,
        ]
        for scenario in self.scenarios:
            lines.append(str(scenario))
        lines.append("-" * 50)
        lines.append(f"Worst case P&L: ${self.worst_case_pnl:,.0f}")
        lines.append(f"Best case P&L: ${self.best_case_pnl:,.0f}")
        lines.append(f"VaR breaches: {self.var_breaches}/{len(self.scenarios)}")
        return "\n".join(lines)


class ScenarioEngine:
    """Engine for running stress scenarios."""

    def __init__(self):
        self.config = settings.risk
        self.greeks_calc = GreeksCalculator()
        self._standard_scenarios = self._build_standard_scenarios()

    def _build_standard_scenarios(self) -> list[dict]:
        """Build standard stress scenarios."""
        spot_shock = self.config.spot_shock_std
        iv_shock = self.config.iv_shock_points

        return [
            {
                "name": "Spot +1σ",
                "type": ScenarioType.SPOT_UP,
                "spot_pct": spot_shock * 1.5,  # ~1.5% for typical equity
                "iv_pts": 0,
            },
            {
                "name": "Spot -1σ",
                "type": ScenarioType.SPOT_DOWN,
                "spot_pct": -spot_shock * 1.5,
                "iv_pts": 0,
            },
            {
                "name": "Spot +2σ",
                "type": ScenarioType.SPOT_UP,
                "spot_pct": spot_shock * 3,
                "iv_pts": 0,
            },
            {
                "name": "Spot -2σ",
                "type": ScenarioType.SPOT_DOWN,
                "spot_pct": -spot_shock * 3,
                "iv_pts": 0,
            },
            {
                "name": "Vol +5pts",
                "type": ScenarioType.VOL_UP,
                "spot_pct": 0,
                "iv_pts": iv_shock,
            },
            {
                "name": "Vol -5pts",
                "type": ScenarioType.VOL_DOWN,
                "spot_pct": 0,
                "iv_pts": -iv_shock,
            },
            {
                "name": "Crash: Spot -3σ, Vol +10pts",
                "type": ScenarioType.SPOT_DOWN_VOL_UP,
                "spot_pct": -spot_shock * 4.5,
                "iv_pts": iv_shock * 2,
            },
            {
                "name": "Rally: Spot +3σ, Vol -5pts",
                "type": ScenarioType.SPOT_UP_VOL_UP,
                "spot_pct": spot_shock * 4.5,
                "iv_pts": -iv_shock,
            },
        ]

    def run_scenario(
        self,
        positions: list[Position],
        snapshot: MarketSnapshot,
        spot_change_pct: float,
        iv_change_points: float,
        scenario_name: str = "Custom",
        scenario_type: ScenarioType = ScenarioType.CUSTOM,
    ) -> ScenarioResult:
        """
        Run a single stress scenario.

        Args:
            positions: Current positions
            snapshot: Market snapshot
            spot_change_pct: Percentage change in spot price
            iv_change_points: Absolute change in IV (in percentage points)
            scenario_name: Name of the scenario
            scenario_type: Type of scenario

        Returns:
            ScenarioResult with P&L and new Greeks
        """
        # Calculate new spot price
        new_spot = snapshot.spot_price * (1 + spot_change_pct / 100)

        # Create shocked IV surface
        shocked_iv_surface = self._shock_iv_surface(
            snapshot.iv_surface, iv_change_points / 100
        )

        # Calculate current value
        current_value = sum(
            self._price_position(pos, snapshot.spot_price, snapshot.iv_surface)
            for pos in positions
        )

        # Calculate new value under scenario
        new_value = sum(
            self._price_position(pos, new_spot, shocked_iv_surface)
            for pos in positions
        )

        pnl = new_value - current_value

        # Calculate new Greeks
        new_greeks = self.greeks_calc.calculate_portfolio_greeks(
            positions, new_spot, shocked_iv_surface
        )

        # Check for VaR/ES breaches
        var_limit = snapshot.total_market_value * self.config.max_var_percent
        breaches_var = abs(pnl) > var_limit

        return ScenarioResult(
            name=scenario_name,
            scenario_type=scenario_type,
            spot_change_pct=spot_change_pct,
            iv_change_points=iv_change_points,
            pnl=pnl,
            new_delta=new_greeks.portfolio_delta,
            new_gamma=new_greeks.portfolio_gamma,
            new_vega=new_greeks.portfolio_vega,
            breaches_var=breaches_var,
            breaches_es=breaches_var,  # Simplified
        )

    def run_stress_test(
        self,
        positions: list[Position],
        snapshot: MarketSnapshot,
        custom_scenarios: Optional[list[dict]] = None,
    ) -> StressTestReport:
        """
        Run complete stress test with all scenarios.

        Args:
            positions: Current positions
            snapshot: Market snapshot
            custom_scenarios: Optional additional scenarios

        Returns:
            StressTestReport with all results
        """
        scenarios_to_run = self._standard_scenarios.copy()
        if custom_scenarios:
            scenarios_to_run.extend(custom_scenarios)

        results = []
        for scenario in scenarios_to_run:
            result = self.run_scenario(
                positions=positions,
                snapshot=snapshot,
                spot_change_pct=scenario["spot_pct"],
                iv_change_points=scenario["iv_pts"],
                scenario_name=scenario["name"],
                scenario_type=scenario["type"],
            )
            results.append(result)

        pnls = [r.pnl for r in results]
        deltas = [abs(r.new_delta) for r in results]

        from datetime import datetime

        return StressTestReport(
            scenarios=results,
            worst_case_pnl=min(pnls),
            best_case_pnl=max(pnls),
            average_pnl=np.mean(pnls),
            max_delta_exposure=max(deltas),
            var_breaches=sum(1 for r in results if r.breaches_var),
            timestamp=datetime.now().isoformat(),
        )

    def _price_position(
        self,
        position: Position,
        spot: float,
        iv_surface: Optional[IVSurface],
    ) -> float:
        """Price a single position."""
        contract = position.contract

        if iv_surface and contract.time_to_expiry > 0:
            iv = iv_surface.get_iv(contract.strike, contract.time_to_expiry)
        elif contract.implied_volatility:
            iv = contract.implied_volatility
        else:
            iv = 0.25

        if contract.time_to_expiry <= 0:
            # At expiry
            price = BlackScholes.price(
                spot, contract.strike, 0.001, 0.05, iv, contract.option_type
            )
        else:
            price = BlackScholes.price(
                spot,
                contract.strike,
                contract.time_to_expiry,
                0.05,
                iv,
                contract.option_type,
            )

        return position.quantity * price * contract.multiplier

    def _shock_iv_surface(
        self,
        surface: Optional[IVSurface],
        iv_change: float,
    ) -> Optional[IVSurface]:
        """Create a shocked IV surface."""
        if surface is None:
            return None

        # Simple parallel shift
        shocked_matrix = surface.iv_matrix + iv_change
        shocked_matrix = np.clip(shocked_matrix, 0.01, 2.0)  # Reasonable bounds

        return IVSurface(
            underlying=surface.underlying,
            spot_price=surface.spot_price,
            strikes=surface.strikes,
            expirations=surface.expirations,
            iv_matrix=shocked_matrix,
        )

    def generate_pnl_surface(
        self,
        positions: list[Position],
        snapshot: MarketSnapshot,
        spot_range: tuple[float, float] = (-10, 10),
        iv_range: tuple[float, float] = (-5, 5),
        steps: int = 21,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate P&L surface for visualization.

        Returns:
            (spot_changes, iv_changes, pnl_matrix)
        """
        spot_changes = np.linspace(spot_range[0], spot_range[1], steps)
        iv_changes = np.linspace(iv_range[0], iv_range[1], steps)
        pnl_matrix = np.zeros((steps, steps))

        for i, spot_pct in enumerate(spot_changes):
            for j, iv_pts in enumerate(iv_changes):
                result = self.run_scenario(
                    positions, snapshot, spot_pct, iv_pts
                )
                pnl_matrix[i, j] = result.pnl

        return spot_changes, iv_changes, pnl_matrix
