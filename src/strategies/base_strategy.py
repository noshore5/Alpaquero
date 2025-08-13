from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from enum import Enum
import logging

class Signal(Enum):
    """Trading signal types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class BaseStrategy(ABC):
    """Base class for all trading strategies."""
    
    def __init__(self, name: str, config: Dict, logger: logging.Logger):
        self.name = name
        self.config = config
        self.logger = logger
        self.positions: Dict[str, Dict] = {}  # Track current positions
        
    @abstractmethod
    def generate_signals(self, symbol: str, data: pd.DataFrame, 
                        indicators: Dict[str, pd.Series]) -> Signal:
        """Generate trading signals based on market data and indicators."""
        pass
    
    @abstractmethod
    def calculate_position_size(self, symbol: str, current_price: float, 
                              portfolio_value: float) -> int:
        """Calculate the position size for a trade."""
        pass
    
    def should_exit_position(self, symbol: str, current_price: float, 
                           position: Dict) -> bool:
        """Determine if we should exit an existing position."""
        if symbol not in self.positions:
            return False
            
        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        side = pos['side']
        
        # Calculate current P&L percentage
        if side == 'long':
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # short
            pnl_pct = (entry_price - current_price) / entry_price
            
        # Check stop loss
        stop_loss_pct = self.config.get('stop_loss_pct', 0.05)
        if pnl_pct <= -stop_loss_pct:
            self.logger.info(f"Stop loss triggered for {symbol}: {pnl_pct:.2%}")
            return True
            
        # Check take profit
        take_profit_pct = self.config.get('take_profit_pct', 0.10)
        if pnl_pct >= take_profit_pct:
            self.logger.info(f"Take profit triggered for {symbol}: {pnl_pct:.2%}")
            return True
            
        return False
    
    def update_position(self, symbol: str, side: str, quantity: int, 
                       entry_price: float, timestamp: str):
        """Update position tracking."""
        self.positions[symbol] = {
            'side': side,
            'quantity': quantity,
            'entry_price': entry_price,
            'timestamp': timestamp
        }
        
    def close_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.positions:
            del self.positions[symbol]
            
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position for a symbol."""
        return self.positions.get(symbol)
    
    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in the symbol."""
        return symbol in self.positions
    
    def validate_signal(self, symbol: str, signal: Signal, 
                       current_price: float) -> bool:
        """Validate if a signal should be executed."""
        # Don't open new position if we already have one
        if signal in [Signal.BUY, Signal.SELL] and self.has_position(symbol):
            return False
            
        # Add more validation logic here
        return signal != Signal.HOLD
    
    def get_risk_metrics(self, symbol: str, current_price: float) -> Dict:
        """Calculate risk metrics for current position."""
        if not self.has_position(symbol):
            return {}
            
        position = self.get_position(symbol)
        entry_price = position['entry_price']
        quantity = position['quantity']
        side = position['side']
        
        if side == 'long':
            unrealized_pnl = (current_price - entry_price) * quantity
            unrealized_pnl_pct = (current_price - entry_price) / entry_price
        else:
            unrealized_pnl = (entry_price - current_price) * quantity
            unrealized_pnl_pct = (entry_price - current_price) / entry_price
            
        return {
            'unrealized_pnl': unrealized_pnl,
            'unrealized_pnl_pct': unrealized_pnl_pct,
            'position_value': current_price * quantity,
            'entry_value': entry_price * quantity
        }
