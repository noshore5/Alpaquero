#!/usr/bin/env python3
"""
Alpaquero - Simple Algorithmic Trader
Main entry point for live trading.
"""

import os
import sys
import time
import signal
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import settings
from src.utils.logger import setup_logger
from src.data.market_data import MarketDataProvider
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.trading.trader import AlgoTrader
from src.trading.risk_manager import RiskManager

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    print("\nShutdown signal received. Stopping trading...")
    global trader
    if trader:
        trader.stop_trading()
    sys.exit(0)

def main():
    """Main function to start the algorithmic trader."""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Set up logging
    log_config = settings.logging_config
    logger = setup_logger(
        level=log_config.get('level', 'INFO'),
        log_file=log_config.get('file', 'logs/trading.log'),
        console=log_config.get('console', True)
    )
    
    logger.info("="*50)
    logger.info("Starting Alpaquero Algorithmic Trader")
    logger.info(f"Timestamp: {datetime.now()}")
    logger.info("="*50)
    
    try:
        # Initialize strategies
        strategies = []
        
        # Moving Average Strategy
        if settings.is_strategy_enabled('moving_average'):
            ma_config = settings.get_strategy_config('moving_average')
            ma_config.update(settings.risk_management)
            ma_config['position_size_pct'] = settings.position_size_pct
            strategies.append(MovingAverageStrategy(ma_config, logger))
            logger.info("Enabled Moving Average strategy")
        
        # RSI Strategy
        if settings.is_strategy_enabled('rsi'):
            rsi_config = settings.get_strategy_config('rsi')
            rsi_config.update(settings.risk_management)
            rsi_config['position_size_pct'] = settings.position_size_pct
            strategies.append(RSIStrategy(rsi_config, logger))
            logger.info("Enabled RSI strategy")
        
        # Mean Reversion Strategy
        if settings.is_strategy_enabled('mean_reversion'):
            mr_config = settings.get_strategy_config('mean_reversion')
            mr_config.update(settings.risk_management)
            mr_config['position_size_pct'] = settings.position_size_pct
            strategies.append(MeanReversionStrategy(mr_config, logger))
            logger.info("Enabled Mean Reversion strategy")
        
        if not strategies:
            logger.error("No strategies enabled. Please enable at least one strategy in config.yaml")
            return
        
        # Initialize risk manager
        risk_manager = RiskManager(settings.risk_management, logger)
        
        # Initialize trader
        global trader
        trader = AlgoTrader(
            api_key=settings.alpaca_api_key,
            secret_key=settings.alpaca_secret_key,
            base_url=settings.alpaca_base_url,
            strategies=strategies,
            risk_manager=risk_manager,
            logger=logger
        )
        
        # Get account summary
        account_summary = trader.get_account_summary()
        logger.info(f"Account Summary: {account_summary}")
        
        # Start trading
        symbols = settings.symbols
        logger.info(f"Trading symbols: {symbols}")
        
        trader.start_trading(symbols)
        
        # Keep running (in a real implementation, this would be a proper event loop)
        logger.info("Trading started. Press Ctrl+C to stop.")
        
        while trader.is_trading:
            time.sleep(60)  # Check every minute
            # In a real implementation, you'd have a proper scheduling system
            trader._trading_iteration(symbols)
        
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received. Shutting down...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
    finally:
        if 'trader' in locals():
            trader.stop_trading()
        logger.info("Alpaquero shutdown complete")

if __name__ == "__main__":
    main()
