"""
Real-time risk monitoring dashboard using Dash.
"""

from datetime import datetime
from typing import Optional

import dash
from dash import dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

from src.config import settings


def create_app() -> dash.Dash:
    """Create and configure the Dash application."""
    app = dash.Dash(
        __name__,
        external_stylesheets=[dbc.themes.DARKLY],
        title="Delta-Hedging Agent Dashboard",
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
                dbc.NavbarBrand("Delta-Hedging Agent", className="ms-2"),
                dbc.Nav([
                    dbc.NavItem(dbc.NavLink("Dashboard", href="/")),
                    dbc.NavItem(dbc.NavLink("Reports", href="/reports")),
                    dbc.NavItem(dbc.NavLink("Settings", href="/settings")),
                ], navbar=True),
                html.Div([
                    html.Span(id="status-indicator", className="badge bg-success me-2"),
                    html.Span(id="last-update", className="text-muted"),
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
                dbc.Col(create_metric_card("portfolio-value", "Portfolio Value", "$0"), md=3),
                dbc.Col(create_metric_card("pnl-today", "P&L Today", "$0"), md=3),
                dbc.Col(create_metric_card("var-95", "VaR 95%", "$0"), md=3),
                dbc.Col(create_metric_card("drawdown", "Drawdown", "0%"), md=3),
            ], className="mb-4"),

            # Greeks row
            dbc.Row([
                dbc.Col(create_metric_card("delta", "Delta", "0.00", color="primary"), md=2),
                dbc.Col(create_metric_card("gamma", "Gamma", "0.0000", color="info"), md=2),
                dbc.Col(create_metric_card("theta", "Theta/Day", "$0", color="warning"), md=2),
                dbc.Col(create_metric_card("vega", "Vega", "0.00", color="secondary"), md=2),
                dbc.Col(create_metric_card("iv-percentile", "IV %ile", "50th", color="light"), md=2),
                dbc.Col(create_metric_card("realized-vol", "Realized Vol", "20%", color="light"), md=2),
            ], className="mb-4"),

            # Charts row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Portfolio Greeks Over Time"),
                        dbc.CardBody([
                            dcc.Graph(id="greeks-chart", style={"height": "300px"}),
                        ]),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("P&L Surface (Stress Test)"),
                        dbc.CardBody([
                            dcc.Graph(id="pnl-surface", style={"height": "300px"}),
                        ]),
                    ]),
                ], md=6),
            ], className="mb-4"),

            # Bottom row
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Implied Volatility Surface"),
                        dbc.CardBody([
                            dcc.Graph(id="iv-surface", style={"height": "300px"}),
                        ]),
                    ]),
                ], md=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Recent Trades & Alerts"),
                        dbc.CardBody([
                            html.Div(id="alerts-table"),
                        ]),
                    ]),
                ], md=6),
            ], className="mb-4"),

            # Agent status
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Agent Status"),
                        dbc.CardBody([
                            dbc.Row([
                                dbc.Col([
                                    html.H6("Mode"),
                                    html.P(id="agent-mode", className="text-info"),
                                ], md=3),
                                dbc.Col([
                                    html.H6("Last Hedge"),
                                    html.P(id="last-hedge", className="text-muted"),
                                ], md=3),
                                dbc.Col([
                                    html.H6("Trades Today"),
                                    html.P(id="trades-today"),
                                ], md=3),
                                dbc.Col([
                                    html.H6("Action"),
                                    html.P(id="next-action", className="text-warning"),
                                ], md=3),
                            ]),
                        ]),
                    ]),
                ]),
            ]),

            # Auto-refresh
            dcc.Interval(
                id="interval-component",
                interval=settings.dashboard.refresh_interval_ms,
                n_intervals=0,
            ),

            # Store for data
            dcc.Store(id="dashboard-data"),
        ], fluid=True),
    ])


def create_metric_card(
    card_id: str,
    title: str,
    default_value: str,
    color: str = "dark",
) -> dbc.Card:
    """Create a metric display card."""
    return dbc.Card([
        dbc.CardBody([
            html.H6(title, className="card-subtitle mb-2 text-muted"),
            html.H3(default_value, id=card_id, className="card-title mb-0"),
        ], className="text-center"),
    ], color=color, outline=True)


def register_callbacks(app: dash.Dash):
    """Register all dashboard callbacks."""

    @app.callback(
        [
            Output("portfolio-value", "children"),
            Output("pnl-today", "children"),
            Output("var-95", "children"),
            Output("drawdown", "children"),
            Output("delta", "children"),
            Output("gamma", "children"),
            Output("theta", "children"),
            Output("vega", "children"),
            Output("iv-percentile", "children"),
            Output("realized-vol", "children"),
            Output("status-indicator", "children"),
            Output("last-update", "children"),
        ],
        Input("interval-component", "n_intervals"),
    )
    def update_metrics(n):
        """Update all metric displays."""
        # TODO: Get real data from trading system
        # For now, return mock data
        return (
            "$100,000",
            "+$500",
            "$2,000",
            "0.5%",
            "+12.5",
            "0.0234",
            "-$45",
            "125.3",
            "65th",
            "18.5%",
            "LIVE",
            datetime.now().strftime("%H:%M:%S"),
        )

    @app.callback(
        Output("greeks-chart", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_greeks_chart(n):
        """Update Greeks time series chart."""
        # Mock data
        times = list(range(100))
        delta = np.cumsum(np.random.randn(100)) + 10
        gamma = np.cumsum(np.random.randn(100) * 0.01) + 0.02

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=delta, name="Delta",
            line=dict(color="#00bc8c"),
        ))
        fig.add_trace(go.Scatter(
            x=times, y=gamma * 500, name="Gamma (scaled)",
            line=dict(color="#3498db"),
        ))

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=40, r=40, t=20, b=40),
            legend=dict(orientation="h", y=1.1),
            xaxis_title="Time",
            yaxis_title="Value",
        )
        return fig

    @app.callback(
        Output("pnl-surface", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_pnl_surface(n):
        """Update P&L surface chart."""
        spot_changes = np.linspace(-10, 10, 21)
        iv_changes = np.linspace(-5, 5, 11)
        X, Y = np.meshgrid(spot_changes, iv_changes)

        # Mock P&L surface
        delta = 10
        gamma = 0.05
        vega = 100
        Z = delta * X * 100 + 0.5 * gamma * (X * 100) ** 2 + vega * Y

        fig = go.Figure(data=[go.Surface(
            x=spot_changes,
            y=iv_changes,
            z=Z,
            colorscale="RdYlGn",
            showscale=True,
        )])

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis_title="Spot %",
                yaxis_title="IV pts",
                zaxis_title="P&L",
            ),
        )
        return fig

    @app.callback(
        Output("iv-surface", "figure"),
        Input("interval-component", "n_intervals"),
    )
    def update_iv_surface(n):
        """Update IV surface chart."""
        strikes = np.linspace(90, 110, 21)
        expiries = np.array([7, 14, 30, 60, 90]) / 365
        X, Y = np.meshgrid(strikes, expiries)

        # Mock IV surface with smile
        atm = 100
        base_iv = 0.20
        smile = 0.001 * (X - atm) ** 2
        term_structure = 0.02 * np.sqrt(Y * 365 / 30)
        Z = base_iv + smile + term_structure

        fig = go.Figure(data=[go.Surface(
            x=strikes,
            y=expiries * 365,
            z=Z * 100,
            colorscale="Viridis",
            showscale=True,
        )])

        fig.update_layout(
            template="plotly_dark",
            margin=dict(l=0, r=0, t=0, b=0),
            scene=dict(
                xaxis_title="Strike",
                yaxis_title="Days to Exp",
                zaxis_title="IV %",
            ),
        )
        return fig

    @app.callback(
        Output("alerts-table", "children"),
        Input("interval-component", "n_intervals"),
    )
    def update_alerts(n):
        """Update alerts table."""
        # Mock alerts
        alerts = [
            {"time": "14:32:15", "type": "HEDGE", "message": "Sold 25 shares SPY @ $450.23"},
            {"time": "14:30:00", "type": "INFO", "message": "Delta within band (+/- 10)"},
            {"time": "14:25:30", "type": "WARNING", "message": "VaR approaching limit (1.8%)"},
        ]

        return dbc.Table([
            html.Thead(html.Tr([
                html.Th("Time"),
                html.Th("Type"),
                html.Th("Message"),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(a["time"]),
                    html.Td(dbc.Badge(a["type"], color="success" if a["type"] == "HEDGE" else "warning" if a["type"] == "WARNING" else "info")),
                    html.Td(a["message"]),
                ])
                for a in alerts
            ]),
        ], striped=True, hover=True, size="sm")

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
        return (
            settings.agent.mode.upper(),
            "2 min ago",
            "5",
            "HOLD - Delta in band",
        )


def run_server():
    """Run the dashboard server."""
    app = create_app()
    app.run_server(
        host=settings.dashboard.host,
        port=settings.dashboard.port,
        debug=settings.dashboard.debug,
    )


if __name__ == "__main__":
    run_server()
