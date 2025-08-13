# Alpaquero - Simple Algorithmic Trader

A simple algorithmic trading bot built for Alpaca Markets using Python.

## Features

- **Real-time Market Data**: Stream live market data from Alpaca
- **Multiple Trading Strategies**: Implemented strategies include moving average crossover, RSI, and mean reversion
- **Risk Management**: Built-in position sizing and stop-loss mechanisms
- **Backtesting**: Test strategies on historical data before live trading
- **Configuration Management**: Easy-to-configure trading parameters
- **Logging**: Comprehensive logging for monitoring and debugging

## Project Structure

```
Alpaquero/
├── src/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py
│   ├── strategies/
│   │   ├── __init__.py
│   │   ├── base_strategy.py
│   │   ├── moving_average.py
│   │   ├── rsi_strategy.py
│   │   └── mean_reversion.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── market_data.py
│   ├── trading/
│   │   ├── __init__.py
│   │   ├── trader.py
│   │   └── risk_manager.py
│   └── utils/
│       ├── __init__.py
│       └── logger.py
├── tests/
│   ├── __init__.py
│   ├── test_strategies.py
│   └── test_trader.py
├── config/
│   ├── config.yaml
│   └── .env.template
├── main.py
├── backtest.py
├── requirements.txt
└── README.md
```

## Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/Alpaquero.git
   cd Alpaquero
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Alpaca API credentials**
   - Copy `.env.template` to `.env`
   - Add your Alpaca API key and secret
   ```
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_BASE_URL=https://paper-api.alpaca.markets  # For paper trading
   ```

4. **Configure trading parameters**
   - Edit `config/config.yaml` to set your trading preferences

## Usage

### Live Trading
```bash
python main.py
```

### Backtesting
```bash
python backtest.py --strategy moving_average --start 2024-01-01 --end 2024-12-31
```

## Strategies

### 1. Moving Average Crossover
- **Buy**: When short MA crosses above long MA
- **Sell**: When short MA crosses below long MA

### 2. RSI Strategy
- **Buy**: When RSI < 30 (oversold)
- **Sell**: When RSI > 70 (overbought)

### 3. Mean Reversion
- **Buy**: When price is below Bollinger Band lower bound
- **Sell**: When price is above Bollinger Band upper bound

## Risk Management

- **Position Sizing**: Configurable percentage of portfolio per trade
- **Stop Loss**: Automatic stop-loss orders to limit downside
- **Take Profit**: Automatic profit-taking at target levels
- **Maximum Positions**: Limit on concurrent open positions

## Configuration

All trading parameters can be configured in `config/config.yaml`:

```yaml
trading:
  symbols: ["AAPL", "MSFT", "GOOGL"]
  position_size_pct: 0.1  # 10% of portfolio per trade
  stop_loss_pct: 0.05     # 5% stop loss
  take_profit_pct: 0.10   # 10% take profit
  max_positions: 5

strategies:
  moving_average:
    short_window: 20
    long_window: 50
  rsi:
    period: 14
    oversold: 30
    overbought: 70
```

## Logging

All trading activities are logged with timestamps and details:
- Trade executions
- Strategy signals
- Errors and exceptions
- Market data updates

## Disclaimer

This software is for educational purposes only. Trading involves risk of loss. Always test strategies thoroughly in paper trading before using real money.

## License

Apache License 2.0 - see LICENSE file for details.
