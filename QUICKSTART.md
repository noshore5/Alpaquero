# Quick Start Guide for Alpaquero

## Option 1: Easy Setup (Recommended)

Run the setup script that handles dependency installation:

```bash
python setup_easy.py
```

## Option 2: Manual Installation

### Step 1: Install Core Dependencies
```bash
pip install alpaca-trade-api pyyaml python-dotenv requests schedule websockets pytest
```

### Step 2: Install Data Dependencies (Choose One)

**Option A: Using conda (Recommended)**
```bash
conda install pandas numpy matplotlib seaborn plotly -c conda-forge
```

**Option B: Using pip with pre-compiled wheels**
```bash
pip install --only-binary=all pandas numpy matplotlib seaborn plotly
```

**Option C: If above fails, try one by one**
```bash
pip install pandas
pip install numpy  
pip install matplotlib
pip install seaborn
pip install plotly
```

### Step 3: Configure Environment
1. Copy `.env.template` to `.env`
2. Add your Alpaca API credentials to `.env`

### Step 4: Test Installation
```bash
python -c "import alpaca_trade_api; print('✓ Ready to trade!')"
```

## Configuration

Edit `config/config.yaml` to:
- Choose symbols to trade
- Configure trading parameters
- Enable/disable strategies
- Set risk management rules

## Running

**Paper Trading (Safe):**
```bash
python main.py
```

**Backtesting:**
```bash
python backtest.py --strategy moving_average --start 2024-01-01 --end 2024-12-31
```

## Troubleshooting

1. **Pandas compilation errors on Windows**: Use conda or pre-compiled wheels
2. **TA-Lib installation issues**: Skip it for now, basic indicators are included
3. **API connection errors**: Check your .env file credentials

## Next Steps

1. Start with paper trading
2. Test strategies with backtesting
3. Monitor logs in `logs/trading.log`
4. Gradually adjust parameters based on performance
