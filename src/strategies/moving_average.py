import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy, Signal
import logging

class MovingAverageStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__("MovingAverage", config, logger)
        self.short_window = config.get('short_window', 20)
        self.long_window = config.get('long_window', 50)
        self.position_size_pct = config.get('position_size_pct', 0.1)
        
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        indicators: Dict[str, pd.Series]) -> Signal:
        """Generate signals based on moving average crossover."""
        
        if len(data) < self.long_window:
            return Signal.HOLD
            
        # Get the moving averages from indicators
        short_ma = indicators.get(f'sma_{self.short_window}')
        long_ma = indicators.get(f'sma_{self.long_window}')
        
        if short_ma is None or long_ma is None:
            # Calculate if not provided
            short_ma = data['close'].rolling(window=self.short_window).mean()
            long_ma = data['close'].rolling(window=self.long_window).mean()
        
        # Check for crossover
        current_short = short_ma.iloc[-1]
        current_long = long_ma.iloc[-1]
        prev_short = short_ma.iloc[-2]
        prev_long = long_ma.iloc[-2]
        
        # Golden cross (bullish signal)
        if (prev_short <= prev_long and current_short > current_long):
            self.logger.info(f"Golden cross detected for {symbol}")
            return Signal.BUY
            
        # Death cross (bearish signal)
        elif (prev_short >= prev_long and current_short < current_long):
            self.logger.info(f"Death cross detected for {symbol}")
            return Signal.SELL
            
        return Signal.HOLD
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              portfolio_value: float) -> int:
        """Calculate position size based on percentage of portfolio."""
        
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / current_price)
        
        self.logger.info(
            f"Calculated position size for {symbol}: {quantity} shares "
            f"(${position_value:.2f} at ${current_price:.2f}/share)"
        )
        
        return quantity
    
    def get_strategy_info(self) -> Dict:
        """Get strategy configuration and current state."""
        return {
            'name': self.name,
            'short_window': self.short_window,
            'long_window': self.long_window,
            'position_size_pct': self.position_size_pct,
            'current_positions': len(self.positions)
        }
