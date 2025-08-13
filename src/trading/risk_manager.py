from typing import Dict, List, Optional
import logging

class RiskManager:
    """Manages risk for trading operations."""
    
    def __init__(self, config: Dict, logger: logging.Logger):
        self.config = config
        self.logger = logger
        self.max_portfolio_risk = config.get('max_portfolio_risk', 0.20)
        self.max_position_size = config.get('max_position_size', 0.15)
        self.min_cash_balance = config.get('min_cash_balance', 1000)
        
    def validate_trade(self, symbol: str, quantity: int, price: float, 
                      side: str, portfolio_value: float, 
                      current_positions: Dict, cash_balance: float) -> bool:
        """Validate if a trade meets risk management criteria."""
        
        trade_value = quantity * price
        
        # Check minimum cash balance
        if side == 'buy' and cash_balance - trade_value < self.min_cash_balance:
            self.logger.warning(
                f"Trade rejected: Would leave cash balance below minimum "
                f"(${cash_balance - trade_value:.2f} < ${self.min_cash_balance})"
            )
            return False
        
        # Check position size limit
        position_pct = trade_value / portfolio_value
        if position_pct > self.max_position_size:
            self.logger.warning(
                f"Trade rejected: Position size {position_pct:.1%} exceeds "
                f"maximum {self.max_position_size:.1%}"
            )
            return False
        
        # Check total portfolio risk
        total_risk = self._calculate_total_risk(
            current_positions, trade_value, portfolio_value
        )
        if total_risk > self.max_portfolio_risk:
            self.logger.warning(
                f"Trade rejected: Total portfolio risk {total_risk:.1%} exceeds "
                f"maximum {self.max_portfolio_risk:.1%}"
            )
            return False
        
        return True
    
    def calculate_position_size(self, symbol: str, price: float, 
                              portfolio_value: float, 
                              strategy_size_pct: float) -> int:
        """Calculate safe position size considering risk limits."""
        
        # Use smaller of strategy size or max position size
        size_pct = min(strategy_size_pct, self.max_position_size)
        
        max_position_value = portfolio_value * size_pct
        quantity = int(max_position_value / price)
        
        self.logger.info(
            f"Risk-adjusted position size for {symbol}: {quantity} shares "
            f"({size_pct:.1%} of portfolio)"
        )
        
        return quantity
    
    def calculate_stop_loss_price(self, entry_price: float, side: str, 
                                 stop_loss_pct: float) -> float:
        """Calculate stop loss price."""
        if side == 'buy':
            return entry_price * (1 - stop_loss_pct)
        else:  # sell/short
            return entry_price * (1 + stop_loss_pct)
    
    def calculate_take_profit_price(self, entry_price: float, side: str, 
                                   take_profit_pct: float) -> float:
        """Calculate take profit price."""
        if side == 'buy':
            return entry_price * (1 + take_profit_pct)
        else:  # sell/short
            return entry_price * (1 - take_profit_pct)
    
    def check_correlation_risk(self, new_symbol: str, current_positions: Dict, 
                              correlation_threshold: float = 0.7) -> bool:
        """Check if new position would create too much correlation risk."""
        # This is a simplified version - in practice, you'd want to 
        # calculate actual correlations between assets
        
        # For now, just check sector/industry concentration
        # You could expand this to use actual correlation data
        
        return True  # Placeholder
    
    def _calculate_total_risk(self, current_positions: Dict, 
                            new_trade_value: float, 
                            portfolio_value: float) -> float:
        """Calculate total portfolio risk exposure."""
        
        total_position_value = sum(
            pos.get('market_value', 0) for pos in current_positions.values()
        )
        
        total_exposure = (total_position_value + new_trade_value) / portfolio_value
        return total_exposure
    
    def get_risk_metrics(self, portfolio_value: float, 
                        current_positions: Dict, 
                        cash_balance: float) -> Dict:
        """Get current risk metrics."""
        
        total_position_value = sum(
            pos.get('market_value', 0) for pos in current_positions.values()
        )
        
        portfolio_risk = total_position_value / portfolio_value
        cash_pct = cash_balance / portfolio_value
        
        largest_position = 0
        if current_positions:
            largest_position = max(
                pos.get('market_value', 0) for pos in current_positions.values()
            ) / portfolio_value
        
        return {
            'portfolio_risk': portfolio_risk,
            'cash_percentage': cash_pct,
            'largest_position_pct': largest_position,
            'number_of_positions': len(current_positions),
            'total_exposure': portfolio_risk
        }
