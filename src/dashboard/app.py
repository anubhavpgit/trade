"""
Real-time risk monitoring dashboard using Dash.
Connected to local SQLite database for demo data.
"""

from datetime import datetime, timedelta
from typing import Optional

import dash
from dash import dcc, html, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import numpy as np
import pandas as pd

from src.config import settings
from src.data.demo_data_service import get_demo_service, DashboardData


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Delta-Hedging Agent Dashboard",
        suppress_callback_exceptions=True,
    )

    app.layout = create_layout()
    register_callbacks(app)

    return app


def create_layout() -> html.Div:
    """Create the dashboard layout."""
    return html.Div([
        # Header
        dbc.Navbar(
            dbc.Container([
                dbc.NavbarBrand([
                    html.I(className="fas fa-chart-line me-2"),
                    "Delta-Hedging Agent"
                ], className="ms-2 fw-bold"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/", active=True)),
                    dbc.NavItem(dbc.NavLink("Positions", href="/positions")),
                    dbc.NavItem(dbc.NavLink("Reports", href="/reports")),
                ], navbar=True),
                html.Div([
                    html.Span("SPY: ", className="text-muted me-1"),
                    html.Span(id="spot-price", className="text-success fw-bold me-2"),
                    html.Span(id="spot-change", className="me-3"),
                    html.Span("VIX: ", className="text-muted me-1"),
                    html.Span(id="vix-value", className="text-warning fw-bold me-3"),
                    html.Span(id="status-indicator", className="badge bg-success me-2"),
                    html.Span(id="last-update", className="text-muted small"),
                ]),
            ]),
            color="dark",
            dark=True,
            className="mb-4",
        ),

        # Main content
        dbc.Container([
            # Top row: Key metrics
            dbc.Row([
                dbc.Col(create_metric_card("portfolio-value", "Portfolio Value", "$0", "success"), md=3),
                dbc.Col(create_metric_card("pnl-today", "P&L Today", "$0", "info"), md=3),
                dbc.Col(create_metric_card("var-95", "VaR 95%", "$0", "warning"), md=3),
                dbc.Col(create_metric_card("drawdown", "Drawdown", "0%", "danger"), md=3),
            ], className="mb-4"),

            # Greeks row
            dbc.Row([
                dbc.Col(create_greek_card("delta", "Delta", "0.00", "Delta exposure in shares"), md=2),
                dbc.Col(create_greek_card("gamma", "Gamma", "0.0000", "Rate of delta change"), md=2),
                dbc.Col(create_greek_card("theta", "Theta/Day", "$0", "Daily time decay"), md=2),
                dbc.Col(create_greek_card("vega", "Vega", "0.00", "IV sensitivity"), md=2),
                dbc.Col(create_greek_card("iv-percentile", "IV %ile", "50th", "Current IV rank"), md=2),
                dbc.Col(create_greek_card("realized-vol", "Real Vol", "20%", "20-day realized"), md=2),
            ], className="mb-4"),

            # Charts row 1
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-chart-line me-2"),
                            "Portfolio Greeks Over Time"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="greeks-chart", style={"height": "320px"}),
                        ]),
                    ], className="h-100"),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-cube me-2"),
                            "P&L Stress Surface"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="pnl-surface", style={"height": "320px"}),
                        ]),
                    ], className="h-100"),
                ], md=6),
            ], className="mb-4"),

            # Charts row 2
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-wave-square me-2"),
                            "Implied Volatility Surface"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="iv-surface", style={"height": "320px"}),
                        ]),
                    ], className="h-100"),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-history me-2"),
                            "P&L History"
                        ]),
                        dbc.CardBody([
                            dcc.Graph(id="pnl-chart", style={"height": "320px"}),
                        ]),
                    ], className="h-100"),
                ], md=6),
            ], className="mb-4"),

            # Bottom row: Tables
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-folder-open me-2"),
                            "Active Positions"
                        ]),
                        dbc.CardBody([
                            html.Div(id="positions-table", style={"maxHeight": "250px", "overflowY": "auto"}),
                        ]),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader([
                            html.I(className="fas fa-bell me-2"),
                            "Recent Activity"
                        ]),
                        dbc.CardBody([
                            html.Div(id="alerts-table", style={"maxHeight": "250px", "overflowY": "auto"}),
                        ]),
                    ]),
                ], md=6),
            ], className="mb-4"),

            # Agent status bar
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.Div([
                                        html.Span("Agent Mode: ", className="text-muted"),
                                        html.Span(id="agent-mode", className="badge bg-primary ms-2"),
                                    ]),
                                ], md=2),
                                dbc.Col([
                                    html.Div([
                                        html.Span("Last Hedge: ", className="text-muted"),
                                        html.Span(id="last-hedge", className="text-info"),
                                    ]),
                                ], md=2),
                                dbc.Col([
                                    html.Div([
                                        html.Span("Trades Today: ", className="text-muted"),
                                        html.Span(id="trades-today", className="fw-bold"),
                                    ]),
                                ], md=2),
                                dbc.Col([
                                    html.Div([
                                        html.Span("Next Action: ", className="text-muted"),
                                        html.Span(id="next-action", className="text-warning fw-bold"),
                                    ]),
                                ], md=6),
                            ], align="center"),
                        ], className="py-2"),
                    ], color="dark"),
                ]),
            ]),

            # Auto-refresh interval
            dcc.Interval(
                id="interval-component",
                interval=settings.dashboard.refresh_interval_ms,
                n_intervals=0,
            ),

            # Data store
            dcc.Store(id="dashboard-data"),

        ], fluid=True),

        # Footer
        html.Footer([
            html.Hr(className="my-4"),
            html.P([
                "Delta-Hedging Agent v0.1.0 | ",
                html.Span("Demo Mode", className="badge bg-warning text-dark"),
                " | Data refreshes every ",
                f"{settings.dashboard.refresh_interval_ms // 1000}s"
            ], className="text-center text-muted small"),
        ]),
    ])


def create_metric_card(
    card_id: str,
    title: str,
    default_value: str,
    color: str = "dark",
) -> dbc.Card:
    """Create a metric display card."""
    color_class = {
        "success": "text-success",
        "info": "text-info",
        "warning": "text-warning",
        "danger": "text-danger",
    }.get(color, "")

    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted small"),
            html.H3(default_value, id=card_id, className=f"card-title mb-0 {color_class}"),
        ], className="text-center py-3"),
    ], className="h-100")


def create_greek_card(
    card_id: str,
    title: str,
    default_value: str,
    tooltip: str,
) -> dbc.Card:
    """Create a Greek metric card with tooltip."""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-1 text-muted small", title=tooltip),
            html.H4(default_value, id=card_id, className="card-title mb-0"),
        ], className="text-center py-2"),
    ], className="h-100")


def register_callbacks(app: dash.Dash):
    """Register all dashboard callbacks."""

    @app.callback(
        [
            Output("portfolio-value", "children"),
            Output("pnl-today", "children"),
            Output("pnl-today", "className"),
            Output("var-95", "children"),
            Output("drawdown", "children"),
            Output("delta", "children"),
            Output("delta", "className"),
            Output("gamma", "children"),
            Output("theta", "children"),
            Output("vega", "children"),
            Output("iv-percentile", "children"),
            Output("realized-vol", "children"),
            Output("spot-price", "children"),
            Output("spot-change", "children"),
            Output("spot-change", "className"),
            Output("vix-value", "children"),
            Output("status-indicator", "children"),
            Output("last-update", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def update_metrics(n):
        """Update all metric displays from database."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()

            # Format values
            portfolio_value = f"${data.portfolio_value:,.0f}"
            pnl_today = f"${data.pnl_today:+,.0f}"
            pnl_class = "card-title mb-0 text-success" if data.pnl_today >= 0 else "card-title mb-0 text-danger"
            var_95 = f"${data.var_95:,.0f}"
            drawdown = f"{data.drawdown * 100:.2f}%"

            # Greeks
            delta = f"{data.delta:+.1f}"
            delta_class = "card-title mb-0"
            if abs(data.delta) > 50:
                delta_class += " text-danger"
            elif abs(data.delta) > 20:
                delta_class += " text-warning"

            gamma = f"{data.gamma:.4f}"
            theta = f"${data.theta:+.0f}"
            vega = f"{data.vega:.1f}"
            iv_percentile = f"{data.iv_percentile:.0f}th"
            realized_vol = f"{data.realized_vol * 100:.1f}%"

            # Spot
            spot_price = f"${data.spot_price:.2f}"
            spot_change = f"{data.spot_change_pct:+.2f}%"
            spot_class = "text-success" if data.spot_change_pct >= 0 else "text-danger"

            vix_value = f"{data.vix_current:.1f}"

            return (
                portfolio_value,
                pnl_today,
                pnl_class,
                var_95,
                drawdown,
                delta,
                delta_class,
                gamma,
                theta,
                vega,
                iv_percentile,
                realized_vol,
                spot_price,
                spot_change,
                spot_class,
                vix_value,
                "LIVE",
                datetime.now().strftime("%H:%M:%S"),
            )
        except Exception as e:
            # Return defaults on error
            return (
                "$0", "$0", "card-title mb-0", "$0", "0%",
                "0.0", "card-title mb-0", "0.0000", "$0", "0.0",
                "50th", "20%", "$450.00", "0.00%", "text-muted",
                "18.0", "ERROR", str(e)[:20],
            )

    @app.callback(
        Output("greeks-chart", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_greeks_chart(n):
        """Update Greeks time series chart from database."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()
            df = data.greeks_history

            fig = go.Figure()

            if not df.empty:
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["delta"],
                    name="Delta",
                    line=dict(color="#00bc8c", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 188, 140, 0.1)",
                ))
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["gamma"] * 1000,
                    name="Gamma (×1000)",
                    line=dict(color="#3498db", width=2),
                    yaxis="y2",
                ))

                # Add delta bands
                fig.add_hline(y=10, line_dash="dash", line_color="yellow", opacity=0.5)
                fig.add_hline(y=-10, line_dash="dash", line_color="yellow", opacity=0.5)

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                xaxis_title="Time",
                yaxis=dict(title="Delta", side="left"),
                yaxis2=dict(title="Gamma (×1000)", side="right", overlaying="y"),
                hovermode="x unified",
            )
            return fig

        except Exception:
            return go.Figure().update_layout(template="plotly_dark")

    @app.callback(
        Output("pnl-surface", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_pnl_surface(n):
        """Update P&L stress surface from database."""
        try:
            service = get_demo_service()
            spot_changes, iv_changes, Z = service.get_pnl_surface_data()

            fig = go.Figure(data=[go.Surface(
                x=spot_changes,
                y=iv_changes,
                z=Z,
                colorscale="RdYlGn",
                showscale=True,
                colorbar=dict(title="P&L ($)", len=0.8),
            )])

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis_title="Spot Change (%)",
                    yaxis_title="IV Change (pts)",
                    zaxis_title="P&L ($)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
                ),
            )
            return fig

        except Exception:
            return go.Figure().update_layout(template="plotly_dark")

    @app.callback(
        Output("iv-surface", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_iv_surface(n):
        """Update IV surface from database."""
        try:
            service = get_demo_service()
            strikes, expiries, iv_matrix = service.get_iv_surface_data()

            fig = go.Figure(data=[go.Surface(
                x=strikes,
                y=expiries,
                z=iv_matrix * 100,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="IV (%)", len=0.8),
            )])

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=0, r=0, t=30, b=0),
                scene=dict(
                    xaxis_title="Strike ($)",
                    yaxis_title="Days to Exp",
                    zaxis_title="IV (%)",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1)),
                ),
            )
            return fig

        except Exception:
            return go.Figure().update_layout(template="plotly_dark")

    @app.callback(
        Output("pnl-chart", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_pnl_chart(n):
        """Update P&L history chart."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()
            df = data.pnl_history

            fig = go.Figure()

            if not df.empty:
                # Cumulative P&L
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["pnl_cumulative"],
                    name="Cumulative P&L",
                    line=dict(color="#00bc8c", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(0, 188, 140, 0.2)",
                ))

                # Portfolio value on secondary axis
                fig.add_trace(go.Scatter(
                    x=df["timestamp"],
                    y=df["portfolio_value"],
                    name="Portfolio Value",
                    line=dict(color="#f39c12", width=1, dash="dot"),
                    yaxis="y2",
                ))

            fig.update_layout(
                template="plotly_dark",
                margin=dict(l=50, r=50, t=30, b=50),
                legend=dict(orientation="h", y=1.15, x=0.5, xanchor="center"),
                xaxis_title="Time",
                yaxis=dict(title="P&L ($)", side="left"),
                yaxis2=dict(title="Portfolio ($)", side="right", overlaying="y"),
                hovermode="x unified",
            )
            return fig

        except Exception:
            return go.Figure().update_layout(template="plotly_dark")

    @app.callback(
        Output("positions-table", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update_positions(n):
        """Update positions table."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()
            positions = data.positions

            if not positions:
                return html.P("No active positions", className="text-muted text-center")

            return dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Symbol", style={"width": "25%"}),
                    html.Th("Type", style={"width": "15%"}),
                    html.Th("Qty", style={"width": "15%"}),
                    html.Th("Strike", style={"width": "15%"}),
                    html.Th("Exp", style={"width": "15%"}),
                    html.Th("Value", style={"width": "15%"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(p["symbol"][:15], title=p["symbol"]),
                        html.Td(dbc.Badge(
                            p["option_type"] or "STOCK",
                            color="primary" if p["option_type"] == "call" else "info" if p["option_type"] == "put" else "secondary"
                        )),
                        html.Td(
                            str(p["quantity"]),
                            className="text-success" if p["quantity"] > 0 else "text-danger"
                        ),
                        html.Td(f"${p['strike']:.0f}" if p["strike"] else "-"),
                        html.Td(p["expiration"][:10] if p["expiration"] else "-"),
                        html.Td(f"${p['market_value']:,.0f}"),
                    ])
                    for p in positions[:10]
                ]),
            ], striped=True, hover=True, size="sm", className="mb-0")

        except Exception:
            return html.P("Error loading positions", className="text-danger")

    @app.callback(
        Output("alerts-table", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update_alerts(n):
        """Update alerts/trades table."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()

            # Combine trades and alerts
            items = []

            for trade in data.recent_trades[:5]:
                items.append({
                    "time": trade["timestamp"],
                    "type": "TRADE",
                    "message": f"{trade['side'].upper()} {trade['quantity']} {trade['symbol']} @ ${trade['price']:.2f}",
                    "color": "success",
                })

            for alert in data.recent_alerts[:5]:
                items.append({
                    "time": alert["timestamp"],
                    "type": alert["type"],
                    "message": alert["message"],
                    "color": "warning" if alert["severity"] == "warning" else "info",
                })

            # Sort by time
            items.sort(key=lambda x: x["time"], reverse=True)

            if not items:
                return html.P("No recent activity", className="text-muted text-center")

            return dbc.Table([
                html.Thead(html.Tr([
                    html.Th("Time", style={"width": "25%"}),
                    html.Th("Type", style={"width": "20%"}),
                    html.Th("Details", style={"width": "55%"}),
                ])),
                html.Tbody([
                    html.Tr([
                        html.Td(item["time"].strftime("%H:%M:%S") if hasattr(item["time"], "strftime") else str(item["time"])[:8]),
                        html.Td(dbc.Badge(item["type"], color=item["color"])),
                        html.Td(item["message"][:50], title=item["message"]),
                    ])
                    for item in items[:10]
                ]),
            ], striped=True, hover=True, size="sm", className="mb-0")

        except Exception:
            return html.P("Error loading activity", className="text-danger")

    @app.callback(
        [
            Output("agent-mode", "children"),
            Output("last-hedge", "children"),
            Output("trades-today", "children"),
            Output("next-action", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def update_agent_status(n):
        """Update agent status display."""
        try:
            service = get_demo_service()
            data = service.get_dashboard_data()

            mode = data.agent_mode

            if data.last_hedge_time:
                delta = datetime.now() - data.last_hedge_time
                if delta.seconds < 60:
                    last_hedge = f"{delta.seconds}s ago"
                elif delta.seconds < 3600:
                    last_hedge = f"{delta.seconds // 60}m ago"
                else:
                    last_hedge = f"{delta.seconds // 3600}h ago"
            else:
                last_hedge = "Never"

            return (
                mode,
                last_hedge,
                str(data.trades_today),
                data.next_action,
            )

        except Exception:
            return ("ERROR", "-", "0", "Database not initialized")


def run_server():
    """Run the dashboard server."""
    app = create_app()
    print("\n" + "=" * 60)
    print("Delta-Hedging Agent Dashboard")
    print("=" * 60)
    print(f"Starting server at http://{settings.dashboard.host}:{settings.dashboard.port}")
    print("\nIf database is empty, run: python -m src.data.seed_demo_data")
    print("=" * 60 + "\n")

    app.run_server(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=settings.dashboard.debug,
    )


if __name__ == "__main__":
    run_server()
