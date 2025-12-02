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
│   ├── data/           # Market data ingestion and storage
│   ├── analytics/      # Pricing, Greeks, VaR/ES, scenarios
│   ├── agent/          # Hedging decision engine
│   ├── execution/      # Paper/live trading execution
│   ├── reporting/      # Daily reports and alerts
│   ├── dashboard/      # Real-time monitoring UI
│   └── utils/          # Helpers and utilities
├── tests/              # Test suite
├── configs/            # Configuration files
├── data/               # Data storage
├── reports/            # Generated reports
└── notebooks/          # Analysis notebooks
```

## Features

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

### Reporting Layer
- Daily HTML risk digest
- Real-time Dash dashboard
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

## Usage

### Run the Trading System
```bash
trading-bot
```

### Run the Dashboard
```bash
run-dashboard
# Open http://127.0.0.1:8050 in your browser
```

### Generate Daily Report
```bash
generate-report
```

### Run Tests
```bash
pytest
```

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
- [ ] Database integration
- [ ] RL overlay
- [ ] Backtesting framework
- [ ] Live trading support

## License

MIT

## Author

Anubhab Patnaik (ap8909@nyu.edu)
