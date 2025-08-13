import pandas as pd
import numpy as np
from typing import Dict
from .base_strategy import BaseStrategy, Signal
import logging

class RSIStrategy(BaseStrategy):
    """RSI-based trading strategy."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        super().__init__("RSI", config, logger)
        self.period = config.get('period', 14)
        self.oversold = config.get('oversold', 30)
        self.overbought = config.get('overbought', 70)
        self.position_size_pct = config.get('position_size_pct', 0.1)
        
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        indicators: Dict[str, pd.Series]) -> Signal:
        """Generate signals based on RSI levels."""
        
        if len(data) < self.period + 1:
            return Signal.HOLD
            
        # Get RSI from indicators or calculate
        rsi = indicators.get('rsi')
        if rsi is None:
            rsi = self._calculate_rsi(data['close'])
        
        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        
        # Buy signal: RSI crosses above oversold level
        if prev_rsi <= self.oversold and current_rsi > self.oversold:
            self.logger.info(f"RSI buy signal for {symbol}: RSI = {current_rsi:.2f}")
            return Signal.BUY
            
        # Sell signal: RSI crosses below overbought level
        elif prev_rsi >= self.overbought and current_rsi < self.overbought:
            self.logger.info(f"RSI sell signal for {symbol}: RSI = {current_rsi:.2f}")
            return Signal.SELL
            
        return Signal.HOLD
    
    def calculate_position_size(self, symbol: str, current_price: float, 
                              portfolio_value: float) -> int:
        """Calculate position size based on percentage of portfolio."""
        
        position_value = portfolio_value * self.position_size_pct
        quantity = int(position_value / current_price)
        
        return quantity
    
    def _calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def get_strategy_info(self) -> Dict:
        """Get strategy configuration and current state."""
        return {
            'name': self.name,
            'period': self.period,
            'oversold': self.oversold,
            'overbought': self.overbought,
            'position_size_pct': self.position_size_pct,
            'current_positions': len(self.positions)
        }
