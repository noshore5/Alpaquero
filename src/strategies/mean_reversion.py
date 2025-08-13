import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy, Signal
import logging

class MeanReversionStrategy(BaseStrategy):
    """Bollinger Bands Mean Reversion Strategy."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__("MeanReversion", config, logger)
        self.period = config.get('period', 20)
        self.std_dev = config.get('std_dev', 2)
        self.position_size_pct = config.get('position_size_pct', 0.1)
        
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        indicators: Dict[str, pd.Series]) -> Signal:
        """Generate signals based on Bollinger Bands."""
        
        if len(data) < self.period:
            return Signal.HOLD
            
        # Get Bollinger Bands from indicators or calculate
        bb_upper = indicators.get('bb_upper')
        bb_lower = indicators.get('bb_lower')
        bb_middle = indicators.get('bb_middle')
        
        if bb_upper is None or bb_lower is None:
            bb_data = self._calculate_bollinger_bands(data['close'])
            bb_upper = bb_data['upper']
            bb_lower = bb_data['lower']
            bb_middle = bb_data['middle']
        
        current_price = data['close'].iloc[-1]
        prev_price = data['close'].iloc[-2]
        
        current_upper = bb_upper.iloc[-1]
        current_lower = bb_lower.iloc[-1]
        current_middle = bb_middle.iloc[-1]
        
        # Buy signal: Price touches or goes below lower band
        if current_price <= current_lower and prev_price > bb_lower.iloc[-2]:
            self.logger.info(
                f"Mean reversion buy signal for {symbol}: "
                f"Price {current_price:.2f} <= Lower Band {current_lower:.2f}"
            )
            return Signal.BUY
            
        # Sell signal: Price touches or goes above upper band
        elif current_price >= current_upper and prev_price < bb_upper.iloc[-2]:
            self.logger.info(
                f"Mean reversion sell signal for {symbol}: "
                f"Price {current_price:.2f} >= Upper Band {current_upper:.2f}"
            )
            return Signal.SELL
            
        return Signal.HOLD
    
    def should_exit_position(self, symbol: str, current_price: float, 
                           position: Dict) -> bool:
        """Override exit logic for mean reversion strategy."""
        
        # First check base class exit conditions (stop loss, take profit)
        if super().should_exit_position(symbol, current_price, position):
            return True
            
        # Mean reversion specific exit: close position when price returns to middle band
        # This would require current Bollinger Bands data which we don't have here
        # In a real implementation, you might store the bands in the position data
        
        return False
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              portfolio_value: float) -> int:
        """Calculate position size based on percentage of portfolio."""
        
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / current_price)
        
        return quantity
    
    def _calculate_bollinger_bands(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands."""
        middle = prices.rolling(window=self.period).mean()
        std = prices.rolling(window=self.period).std()
        
        return {
            'upper': middle + (std * self.std_dev),
            'middle': middle,
            'lower': middle - (std * self.std_dev)
        }
    
    def get_strategy_info(self) -> Dict:
        """Get strategy configuration and current state."""
        return {
            'name': self.name,
            'period': self.period,
            'std_dev': self.std_dev,
            'position_size_pct': self.position_size_pct,
            'current_positions': len(self.positions)
        }
