"""
Main entry point for the Delta-Hedging Agent.
"""

import asyncio
import signal
import sys
from datetime import datetime

from loguru import logger

from src.config import settings
from src.data.market_data import MarketDataService
from src.analytics.greeks import GreeksCalculator
from src.analytics.risk_metrics import RiskMetricsEngine
from src.agent.hedging_agent import DeltaHedgingAgent
from src.execution.paper_trader import PaperTrader
from src.reporting.generator import ReportGenerator


class TradingSystem:
    """Main trading system orchestrator."""

    def __init__(self):
        self.running = False
        self.market_data = MarketDataService()
        self.greeks_calc = GreeksCalculator()
        self.risk_engine = RiskMetricsEngine()
        self.agent = DeltaHedgingAgent()
        self.executor = PaperTrader()
        self.reporter = ReportGenerator()

    async def start(self):
        """Start the trading system."""
        logger.info("Starting Delta-Hedging Agent...")
        logger.info(f"Mode: {settings.agent.mode}")
        logger.info(f"Hedge frequency: {settings.agent.hedge_frequency_minutes} minutes")

        self.running = True

        # Generate morning report
        await self._generate_morning_report()

        # Start main loop
        await self._main_loop()

    async def stop(self):
        """Gracefully stop the trading system."""
        logger.info("Stopping trading system...")
        self.running = False
        await self.executor.close_all_positions()

    async def _main_loop(self):
        """Main trading loop."""
        hedge_interval = settings.agent.hedge_frequency_minutes * 60

        while self.running:
            try:
                # Fetch latest market data
                market_snapshot = await self.market_data.get_snapshot()

                # Calculate Greeks
                greeks = self.greeks_calc.calculate_portfolio_greeks(
                    market_snapshot.positions,
                    market_snapshot.spot_price,
                    market_snapshot.iv_surface,
                )

                # Calculate risk metrics
                risk_metrics = self.risk_engine.calculate_metrics(
                    market_snapshot, greeks
                )

                # Check guardrails
                if self._check_kill_switch(risk_metrics):
                    logger.warning("Kill switch triggered!")
                    await self.executor.close_all_positions()
                    break

                # Agent decision
                action = self.agent.decide(greeks, risk_metrics)

                if action.should_hedge:
                    await self.executor.execute(action)

                # Log status
                logger.info(
                    f"Delta: {greeks.portfolio_delta:.2f} | "
                    f"Gamma: {greeks.portfolio_gamma:.4f} | "
                    f"VaR: ${risk_metrics.var_95:.2f}"
                )

                await asyncio.sleep(hedge_interval)

            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(10)

    def _check_kill_switch(self, risk_metrics) -> bool:
        """Check if risk limits are breached."""
        if risk_metrics.drawdown_percent > settings.risk.max_drawdown_percent:
            return True
        if risk_metrics.var_percent > settings.risk.max_var_percent:
            return True
        return False

    async def _generate_morning_report(self):
        """Generate the daily morning risk digest."""
        logger.info("Generating morning risk report...")
        try:
            report_path = await self.reporter.generate_daily_digest()
            logger.info(f"Report saved to: {report_path}")
        except Exception as e:
            logger.error(f"Failed to generate report: {e}")


def handle_shutdown(system: TradingSystem):
    """Handle shutdown signals."""
    def signal_handler(sig, frame):
        logger.info(f"Received signal {sig}")
        asyncio.create_task(system.stop())
        sys.exit(0)
    return signal_handler


def main():
    """Main entry point."""
    # Configure logging
    logger.remove()
    logger.add(
        sys.stderr,
        level=settings.log_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
    )
    logger.add(
        settings.project_root / "logs" / "trading_{time}.log",
        rotation="1 day",
        retention="30 days",
        level="DEBUG",
    )

    # Create and run system
    system = TradingSystem()

    # Register signal handlers
    signal.signal(signal.SIGINT, handle_shutdown(system))
    signal.signal(signal.SIGTERM, handle_shutdown(system))

    # Run
    asyncio.run(system.start())


if __name__ == "__main__":
    main()
