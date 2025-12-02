"""
Daily risk digest and report generation.
"""

from datetime import datetime, date
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from loguru import logger

from src.config import settings
from src.data.models import MarketSnapshot, Position
from src.analytics.greeks import PortfolioGreeks
from src.analytics.risk_metrics import RiskMetrics
from src.analytics.scenarios import StressTestReport


REPORT_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>Risk Digest - {{ date }}</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }
        .container { max-width: 900px; margin: 0 auto; background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        h1 { color: #1a1a2e; border-bottom: 3px solid #4a69bd; padding-bottom: 10px; }
        h2 { color: #4a69bd; margin-top: 30px; }
        .metric-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 20px; margin: 20px 0; }
        .metric-box { background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; }
        .metric-value { font-size: 28px; font-weight: bold; color: #1a1a2e; }
        .metric-label { color: #666; font-size: 14px; margin-top: 5px; }
        .positive { color: #27ae60; }
        .negative { color: #e74c3c; }
        .warning { color: #f39c12; }
        table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        th, td { padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }
        th { background: #4a69bd; color: white; }
        tr:hover { background: #f5f5f5; }
        .action-plan { background: #e8f4f8; padding: 20px; border-radius: 8px; margin: 20px 0; }
        .action-plan li { margin: 10px 0; }
        .timestamp { color: #999; font-size: 12px; text-align: right; }
    </style>
</head>
<body>
    <div class="container">
        <h1>Daily Risk Digest</h1>
        <p class="timestamp">Generated: {{ timestamp }}</p>

        <h2>Portfolio Summary</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">${{ "{:,.0f}".format(portfolio_value) }}</div>
                <div class="metric-label">Portfolio Value</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {{ 'positive' if pnl >= 0 else 'negative' }}">${{ "{:+,.0f}".format(pnl) }}</div>
                <div class="metric-label">P&L Today</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ num_positions }}</div>
                <div class="metric-label">Positions</div>
            </div>
        </div>

        <h2>Greeks</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{{ "{:+.1f}".format(delta) }}</div>
                <div class="metric-label">Delta</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "{:+.4f}".format(gamma) }}</div>
                <div class="metric-label">Gamma</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${{ "{:+.0f}".format(theta) }}</div>
                <div class="metric-label">Theta / Day</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "{:+.1f}".format(vega) }}</div>
                <div class="metric-label">Vega</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${{ "{:,.0f}".format(dollar_delta) }}</div>
                <div class="metric-label">Dollar Delta</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${{ "{:,.0f}".format(dollar_gamma) }}</div>
                <div class="metric-label">Dollar Gamma</div>
            </div>
        </div>

        <h2>Volatility</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value">{{ "{:.1f}%".format(atm_iv * 100) }}</div>
                <div class="metric-label">ATM IV</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "{:.0f}".format(iv_percentile) }}th</div>
                <div class="metric-label">IV Percentile</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">{{ "{:.1f}%".format(realized_vol * 100) }}</div>
                <div class="metric-label">Realized Vol</div>
            </div>
        </div>

        <h2>Risk Metrics</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <div class="metric-value {{ 'warning' if var_percent > 1.5 else '' }}">${{ "{:,.0f}".format(var_95) }}</div>
                <div class="metric-label">VaR 95% ({{ "{:.1f}%".format(var_percent * 100) }})</div>
            </div>
            <div class="metric-box">
                <div class="metric-value">${{ "{:,.0f}".format(es_95) }}</div>
                <div class="metric-label">Expected Shortfall 95%</div>
            </div>
            <div class="metric-box">
                <div class="metric-value {{ 'negative' if drawdown > 2 else '' }}">{{ "{:.2f}%".format(drawdown * 100) }}</div>
                <div class="metric-label">Drawdown</div>
            </div>
        </div>

        <h2>Stress Test Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Scenario</th>
                    <th>Spot Change</th>
                    <th>IV Change</th>
                    <th>P&L Impact</th>
                    <th>New Delta</th>
                </tr>
            </thead>
            <tbody>
                {% for scenario in scenarios %}
                <tr>
                    <td>{{ scenario.name }}</td>
                    <td>{{ "{:+.1f}%".format(scenario.spot_change_pct) }}</td>
                    <td>{{ "{:+.1f}".format(scenario.iv_change_points) }} pts</td>
                    <td class="{{ 'positive' if scenario.pnl >= 0 else 'negative' }}">${{ "{:+,.0f}".format(scenario.pnl) }}</td>
                    <td>{{ "{:+.1f}".format(scenario.new_delta) }}</td>
                </tr>
                {% endfor %}
            </tbody>
        </table>

        <h2>Agent Action Plan</h2>
        <div class="action-plan">
            <ul>
                {% for action in action_plan %}
                <li>{{ action }}</li>
                {% endfor %}
            </ul>
        </div>

        <p class="timestamp">Delta-Hedging Agent v0.1.0</p>
    </div>
</body>
</html>
'''


class ReportGenerator:
    """Generate daily risk digest reports."""

    def __init__(self):
        self.reports_dir = settings.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    async def generate_daily_digest(
        self,
        snapshot: Optional[MarketSnapshot] = None,
        greeks: Optional[PortfolioGreeks] = None,
        risk_metrics: Optional[RiskMetrics] = None,
        stress_report: Optional[StressTestReport] = None,
        action_plan: Optional[list[str]] = None,
    ) -> Path:
        """
        Generate daily risk digest HTML report.

        Returns:
            Path to generated report
        """
        timestamp = datetime.now()
        report_date = date.today().isoformat()

        # Use defaults if not provided
        context = self._build_context(
            snapshot, greeks, risk_metrics, stress_report, action_plan, timestamp
        )

        # Render template
        from jinja2 import Template
        template = Template(REPORT_TEMPLATE)
        html_content = template.render(**context)

        # Save report
        report_path = self.reports_dir / f"risk_digest_{report_date}.html"
        report_path.write_text(html_content)

        logger.info(f"Generated report: {report_path}")
        return report_path

    def _build_context(
        self,
        snapshot: Optional[MarketSnapshot],
        greeks: Optional[PortfolioGreeks],
        risk_metrics: Optional[RiskMetrics],
        stress_report: Optional[StressTestReport],
        action_plan: Optional[list[str]],
        timestamp: datetime,
    ) -> dict:
        """Build template context from data."""
        # Default values
        context = {
            "date": date.today().isoformat(),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "portfolio_value": 100000,
            "pnl": 0,
            "num_positions": 0,
            "delta": 0,
            "gamma": 0,
            "theta": 0,
            "vega": 0,
            "dollar_delta": 0,
            "dollar_gamma": 0,
            "atm_iv": 0.25,
            "iv_percentile": 50,
            "realized_vol": 0.20,
            "var_95": 0,
            "es_95": 0,
            "var_percent": 0,
            "drawdown": 0,
            "scenarios": [],
            "action_plan": action_plan or ["No actions required"],
        }

        if snapshot:
            context["portfolio_value"] = snapshot.total_market_value
            context["num_positions"] = len(snapshot.positions)
            if snapshot.iv_surface:
                context["atm_iv"] = snapshot.iv_surface.atm_iv

        if greeks:
            context["delta"] = greeks.portfolio_delta
            context["gamma"] = greeks.portfolio_gamma
            context["theta"] = greeks.portfolio_theta
            context["vega"] = greeks.portfolio_vega
            context["dollar_delta"] = greeks.dollar_delta
            context["dollar_gamma"] = greeks.dollar_gamma

        if risk_metrics:
            context["var_95"] = risk_metrics.var_95
            context["es_95"] = risk_metrics.es_95
            context["var_percent"] = risk_metrics.var_percent
            context["drawdown"] = risk_metrics.drawdown_percent
            context["realized_vol"] = risk_metrics.realized_volatility
            context["iv_percentile"] = risk_metrics.iv_percentile
            context["pnl"] = risk_metrics.pnl_today

        if stress_report:
            context["scenarios"] = [
                {
                    "name": s.name,
                    "spot_change_pct": s.spot_change_pct,
                    "iv_change_points": s.iv_change_points,
                    "pnl": s.pnl,
                    "new_delta": s.new_delta,
                }
                for s in stress_report.scenarios
            ]

        return context


def generate_daily_report():
    """CLI entry point for report generation."""
    import asyncio
    generator = ReportGenerator()
    asyncio.run(generator.generate_daily_digest())


if __name__ == "__main__":
    generate_daily_report()
