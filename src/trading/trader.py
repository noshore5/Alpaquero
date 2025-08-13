import alpaca_trade_api as tradeapi
from typing import Dict, List, Optional, Any
from datetime import datetime, time
import pytz
import logging
from ..strategies.base_strategy import BaseStrategy, Signal
from ..data.market_data import MarketDataProvider
from .risk_manager import RiskManager

class AlgoTrader:
    """Main algorithmic trading engine."""
    
    def __init__(self, api_key: str, secret_key: str, base_url: str, 
                 strategies: List[BaseStrategy], risk_manager: RiskManager,
                 logger: logging.Logger):
        
        self.api = tradeapi.REST(api_key, secret_key, base_url, api_version='v2')
        self.strategies = strategies
        self.risk_manager = risk_manager
        self.logger = logger
        self.data_provider = MarketDataProvider(self.api, logger)
        
        # Trading state
        self.is_trading = False
        self.positions: Dict[str, Any] = {}
        self.orders: Dict[str, Any] = {}
        
        # Verify API connection
        try:
            account = self.api.get_account()
            self.logger.info(f"Connected to Alpaca API. Account: {account.account_number}")
        except Exception as e:
            self.logger.error(f"Failed to connect to Alpaca API: {e}")
            raise
    
    def start_trading(self, symbols: List[str]):
        """Start the trading loop."""
        self.logger.info("Starting algorithmic trading...")
        self.is_trading = True
        
        # Load initial data
        self.logger.info("Loading historical data...")
        self.data_provider.get_historical_data(symbols)
        
        # Get current account info
        account = self.api.get_account()
        self.logger.info(f"Account value: ${float(account.portfolio_value):,.2f}")
        
        # Update positions
        self._update_positions()
        
        # Main trading loop would go here
        # In a real implementation, this would run continuously
        # For now, we'll just run one iteration
        self._trading_iteration(symbols)
    
    def stop_trading(self):
        """Stop the trading loop."""
        self.logger.info("Stopping algorithmic trading...")
        self.is_trading = False
    
    def _trading_iteration(self, symbols: List[str]):
        """Single iteration of the trading loop."""
        
        if not self._is_market_open():
            self.logger.info("Market is closed, skipping trading iteration")
            return
        
        for symbol in symbols:
            try:
                self._process_symbol(symbol)
            except Exception as e:
                self.logger.error(f"Error processing {symbol}: {e}")
    
    def _process_symbol(self, symbol: str):
        """Process trading logic for a single symbol."""
        
        # Get current data and indicators
        data = self.data_provider.get_cached_data(symbol)
        if data is None or len(data) < 50:
            self.logger.warning(f"Insufficient data for {symbol}")
            return
        
        indicators = self.data_provider.calculate_technical_indicators(symbol)
        current_price = self.data_provider.get_latest_price(symbol)
        
        if current_price is None:
            self.logger.warning(f"Could not get current price for {symbol}")
            return
        
        # Check each strategy
        for strategy in self.strategies:
            if not strategy.config.get('enabled', False):
                continue
                
            # Generate signal
            signal = strategy.generate_signals(symbol, data, indicators)
            
            if signal == Signal.HOLD:
                continue
            
            # Check if we should exit existing positions first
            if strategy.has_position(symbol):
                if strategy.should_exit_position(symbol, current_price, 
                                               strategy.get_position(symbol)):
                    self._close_position(symbol, strategy)
                continue
            
            # Validate signal
            if not strategy.validate_signal(symbol, signal, current_price):
                continue
            
            # Calculate position size
            account = self.api.get_account()
            portfolio_value = float(account.portfolio_value)
            quantity = strategy.calculate_position_size(symbol, current_price, portfolio_value)
            
            if quantity <= 0:
                continue
            
            # Risk management check
            cash_balance = float(account.cash)
            side = 'buy' if signal == Signal.BUY else 'sell'
            
            if self.risk_manager.validate_trade(
                symbol, quantity, current_price, side, 
                portfolio_value, self.positions, cash_balance
            ):
                self._execute_trade(symbol, signal, quantity, strategy)
    
    def _execute_trade(self, symbol: str, signal: Signal, quantity: int, 
                      strategy: BaseStrategy):
        """Execute a trade."""
        
        try:
            side = 'buy' if signal == Signal.BUY else 'sell'
            
            # Submit market order
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=side,
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"Order submitted: {side} {quantity} shares of {symbol}")
            
            # Track the order
            self.orders[order.id] = {
                'symbol': symbol,
                'strategy': strategy.name,
                'signal': signal,
                'quantity': quantity,
                'order': order
            }
            
            # Update strategy position (assuming fill for now)
            # In reality, you'd wait for fill confirmation
            current_price = self.data_provider.get_latest_price(symbol)
            strategy.update_position(
                symbol, side, quantity, current_price, 
                datetime.now().isoformat()
            )
            
        except Exception as e:
            self.logger.error(f"Error executing trade for {symbol}: {e}")
    
    def _close_position(self, symbol: str, strategy: BaseStrategy):
        """Close an existing position."""
        
        position = strategy.get_position(symbol)
        if not position:
            return
        
        try:
            # Determine close side (opposite of original)
            close_side = 'sell' if position['side'] == 'long' else 'buy'
            quantity = position['quantity']
            
            order = self.api.submit_order(
                symbol=symbol,
                qty=quantity,
                side=close_side,
                type='market',
                time_in_force='day'
            )
            
            self.logger.info(f"Position closed: {close_side} {quantity} shares of {symbol}")
            
            # Remove from strategy tracking
            strategy.close_position(symbol)
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
    
    def _update_positions(self):
        """Update current positions from Alpaca."""
        try:
            positions = self.api.list_positions()
            self.positions = {}
            
            for pos in positions:
                self.positions[pos.symbol] = {
                    'symbol': pos.symbol,
                    'quantity': int(pos.qty),
                    'side': 'long' if int(pos.qty) > 0 else 'short',
                    'market_value': float(pos.market_value),
                    'unrealized_pl': float(pos.unrealized_pl),
                    'avg_entry_price': float(pos.avg_entry_price)
                }
                
            self.logger.info(f"Updated {len(self.positions)} positions")
            
        except Exception as e:
            self.logger.error(f"Error updating positions: {e}")
    
    def _is_market_open(self) -> bool:
        """Check if the market is currently open."""
        try:
            clock = self.api.get_clock()
            return clock.is_open
        except Exception as e:
            self.logger.error(f"Error checking market status: {e}")
            return False
    
    def get_account_summary(self) -> Dict:
        """Get account summary information."""
        try:
            account = self.api.get_account()
            
            return {
                'portfolio_value': float(account.portfolio_value),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'equity': float(account.equity),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'day_trade_count': int(account.daytrade_count),
                'pattern_day_trader': account.pattern_day_trader
            }
        except Exception as e:
            self.logger.error(f"Error getting account summary: {e}")
            return {}
    
    def get_performance_metrics(self) -> Dict:
        """Get performance metrics."""
        # This would calculate various performance metrics
        # For now, return basic info
        return {
            'active_strategies': len([s for s in self.strategies if s.config.get('enabled')]),
            'total_positions': len(self.positions),
            'pending_orders': len(self.orders)
        }
