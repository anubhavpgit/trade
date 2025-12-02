# Delta-Hedging Agent

Real-Time Risk Management and Autonomous Delta-Hedging System

## Overview

A comprehensive system for:
- **Automated AI-driven trading bot** with delta-neutral hedging
- **Daily Greeks and Risk Digest** generator
- **Intraday stress testing** with VaR/ES monitoring and guardrail triggers

## Architecture

```
├── src/
│   ├── data/           # Market data ingestion, database, demo service
│   ├── analytics/      # Pricing, Greeks, VaR/ES, scenarios
│   ├── agent/          # Hedging decision engine
│   ├── execution/      # Paper/live trading execution
│   ├── reporting/      # Daily reports and alerts
│   ├── dashboard/      # Real-time monitoring UI
│   └── utils/          # Helpers and utilities
├── tests/              # Test suite
├── configs/            # Configuration files
├── data/               # SQLite database and data storage
├── reports/            # Generated reports
└── notebooks/          # Analysis notebooks
```

## Quick Start (Demo Mode)

```bash
# 1. Clone and enter the repository
cd trading_bot

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install the package
pip install -e .

# 4. Seed the database with demo data
python -m src.data.seed_demo_data
# OR use the CLI command:
seed-demo-data

# 5. Launch the dashboard
python -m src.dashboard.app
# OR use the CLI command:
run-dashboard

# 6. Open browser to http://127.0.0.1:8050
```

## Demo Data

The demo system uses:
- **SPY Historical Prices**: Downloaded from Yahoo Finance (June 2023 - present)
- **VIX Historical Data**: Downloaded from Yahoo Finance for IV context
- **Synthetic Option Chains**: Generated using Black-Scholes with realistic IV smile
- **Sample Portfolio**: Short straddle + protective put + stock hedge
- **Risk Snapshots**: 200 historical snapshots for time series charts

Data sources referenced:
- [Yahoo Finance](https://finance.yahoo.com/) - Free stock/ETF data
- [CBOE Historical Data](https://www.cboe.com/us/options/market_statistics/historical_data/) - VIX archives
- [FRED VIX Data](https://fred.stlouisfed.org/series/VIXCLS) - Alternative VIX source

## Features

### Data Layer
- SQLite database with SQLAlchemy ORM
- Historical price and volatility storage
- Option chain snapshots
- Position and trade tracking
- Risk metrics time series

### Analytics Layer
- Black-Scholes option pricing
- Greeks calculation (Delta, Gamma, Theta, Vega, Rho)
- IV surface fitting and interpolation
- Monte Carlo VaR and Expected Shortfall
- Cornish-Fisher VaR adjustment
- Stress scenario simulation

### Agent Layer
- Rule-based delta-neutral hedging
- Configurable delta bands and thresholds
- Risk guardrails (Delta, Gamma, Vega, VaR limits)
- Kill-switch for drawdown protection
- Optional RL overlay for hedge timing (coming soon)

### Execution Layer
- Paper trading with simulated fills
- Slippage and commission modeling
- Position and P&L tracking
- Order management

### Dashboard
- Real-time portfolio metrics
- Greeks time series with delta bands
- P&L stress surface (3D)
- Implied volatility surface (3D)
- Cumulative P&L chart
- Active positions table
- Recent trades and alerts feed
- Agent status and next action

### Reporting Layer
- Daily HTML risk digest
- Alert system with multiple channels

## Installation

```bash
# Clone the repository
cd trading_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"

# For RL features
pip install -e ".[rl]"
```

## Configuration

1. Copy the example environment file:
```bash
cp .env.example .env
```

2. Edit `.env` with your settings:
- Market data API keys (Polygon, IEX, or use Yahoo/mock)
- Risk parameters
- Agent configuration

## CLI Commands

| Command | Description |
|---------|-------------|
| `trading-bot` | Run the main trading system |
| `run-dashboard` | Start the Dash monitoring dashboard |
| `generate-report` | Generate daily risk digest report |
| `seed-demo-data` | Populate database with demo data |

## Risk Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_delta_exposure` | 100 | Maximum absolute delta |
| `max_gamma_exposure` | 50 | Maximum absolute gamma |
| `max_vega_exposure` | 1000 | Maximum absolute vega |
| `max_var_percent` | 2% | VaR limit as % of portfolio |
| `max_drawdown_percent` | 5% | Kill switch threshold |
| `delta_band_threshold` | 10 | Rehedge when delta exceeds |

## Project Status

- [x] Core infrastructure
- [x] Option pricing and Greeks
- [x] VaR/ES calculation
- [x] Stress testing engine
- [x] Rule-based hedging agent
- [x] Paper trading execution
- [x] Dashboard UI
- [x] Report generation
- [x] SQLite database integration
- [x] Demo data system
- [ ] RL overlay
- [ ] Backtesting framework
- [ ] Live trading support

## Screenshots

Dashboard displays:
- Portfolio value, P&L, VaR, drawdown
- Real-time Greeks (Delta, Gamma, Theta, Vega)
- IV percentile and realized volatility
- Interactive 3D stress test and IV surfaces
- Position details and activity feed

## License

MIT

## Author

Anubhab Patnaik (ap8909@nyu.edu)
