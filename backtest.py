#!/usr/bin/env python3
"""
Alpaquero - Backtesting Engine
Test strategies on historical data.
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.config.settings import settings
from src.utils.logger import setup_logger
from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.base_strategy import Signal

class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        
    def run_backtest(self, strategy, symbol: str, data: pd.DataFrame, 
                    start_date: str, end_date: str) -> Dict:
        """Run backtest for a strategy on historical data."""
        
        # Filter data by date range
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        mask = (data['timestamp'] >= start_date) & (data['timestamp'] <= end_date)
        data = data[mask].reset_index(drop=True)
        
        if len(data) < 100:
            raise ValueError(f"Insufficient data for backtesting {symbol}")
        
        print(f"Backtesting {strategy.name} on {symbol} from {start_date} to {end_date}")
        print(f"Data points: {len(data)}")
        
        # Calculate indicators
        indicators = self._calculate_indicators(data)
        
        # Run backtest
        for i in range(50, len(data)):  # Start from index 50 to have enough history
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            current_time = data.iloc[i]['timestamp']
            
            # Generate signal
            signal = strategy.generate_signals(symbol, current_data, 
                                             {k: v.iloc[:i+1] for k, v in indicators.items()})
            
            # Process signal
            if signal == Signal.BUY and symbol not in self.positions:
                self._open_position(symbol, 'long', current_price, current_time, strategy)
            elif signal == Signal.SELL and symbol not in self.positions:
                self._open_position(symbol, 'short', current_price, current_time, strategy)
            elif symbol in self.positions:
                # Check exit conditions
                position = self.positions[symbol]
                if strategy.should_exit_position(symbol, current_price, position):
                    self._close_position(symbol, current_price, current_time)
            
            # Record equity
            equity = self._calculate_equity(current_price, {symbol: current_price})
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': equity,
                'price': current_price
            })
        
        # Close any remaining positions
        final_price = data.iloc[-1]['close']
        if symbol in self.positions:
            self._close_position(symbol, final_price, data.iloc[-1]['timestamp'])
        
        return self._calculate_performance_metrics()
    
    def _open_position(self, symbol: str, side: str, price: float, 
                      timestamp, strategy):
        """Open a new position."""
        
        # Calculate position size (simplified)
        position_value = self.capital * 0.1  # Use 10% of capital
        quantity = int(position_value / price)
        
        if quantity > 0:
            self.positions[symbol] = {
                'side': side,
                'quantity': quantity,
                'entry_price': price,
                'entry_time': timestamp
            }
            
            # Record trade
            self.trades.append({
                'symbol': symbol,
                'action': 'open',
                'side': side,
                'quantity': quantity,
                'price': price,
                'timestamp': timestamp,
                'value': quantity * price
            })
            
            print(f"Opened {side} position: {quantity} shares of {symbol} at ${price:.2f}")
    
    def _close_position(self, symbol: str, price: float, timestamp):
        """Close an existing position."""
        
        if symbol not in self.positions:
            return
        
        position = self.positions[symbol]
        quantity = position['quantity']
        entry_price = position['entry_price']
        side = position['side']
        
        # Calculate P&L
        if side == 'long':
            pnl = (price - entry_price) * quantity
        else:  # short
            pnl = (entry_price - price) * quantity
        
        self.capital += pnl
        
        # Record trade
        self.trades.append({
            'symbol': symbol,
            'action': 'close',
            'side': side,
            'quantity': quantity,
            'price': price,
            'timestamp': timestamp,
            'pnl': pnl,
            'return_pct': pnl / (entry_price * quantity)
        })
        
        print(f"Closed {side} position: {quantity} shares of {symbol} at ${price:.2f}, P&L: ${pnl:.2f}")
        
        del self.positions[symbol]
    
    def _calculate_equity(self, current_price: float, prices: Dict[str, float]) -> float:
        """Calculate current equity value."""
        
        equity = self.capital
        
        for symbol, position in self.positions.items():
            if symbol in prices:
                current_value = position['quantity'] * prices[symbol]
                if position['side'] == 'long':
                    unrealized_pnl = current_value - (position['quantity'] * position['entry_price'])
                else:
                    unrealized_pnl = (position['quantity'] * position['entry_price']) - current_value
                equity += unrealized_pnl
        
        return equity
    
    def _calculate_indicators(self, data: pd.DataFrame) -> Dict:
        """Calculate technical indicators."""
        
        indicators = {}
        
        # Moving averages
        indicators['sma_20'] = data['close'].rolling(window=20).mean()
        indicators['sma_50'] = data['close'].rolling(window=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        sma = data['close'].rolling(window=20).mean()
        std = data['close'].rolling(window=20).std()
        indicators['bb_upper'] = sma + (std * 2)
        indicators['bb_middle'] = sma
        indicators['bb_lower'] = sma - (std * 2)
        
        return indicators
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate performance metrics."""
        
        if not self.trades:
            return {'error': 'No trades executed'}
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        
        # Calculate returns
        equity_df['returns'] = equity_df['equity'].pct_change()
        
        # Basic metrics
        total_return = (self.capital - self.initial_capital) / self.initial_capital
        total_trades = len([t for t in self.trades if t['action'] == 'close'])
        
        if total_trades > 0:
            winning_trades = len([t for t in self.trades if t['action'] == 'close' and t['pnl'] > 0])
            win_rate = winning_trades / total_trades
            
            avg_win = np.mean([t['pnl'] for t in self.trades if t['action'] == 'close' and t['pnl'] > 0])
            avg_loss = np.mean([t['pnl'] for t in self.trades if t['action'] == 'close' and t['pnl'] < 0])
            
            if np.isnan(avg_win):
                avg_win = 0
            if np.isnan(avg_loss):
                avg_loss = 0
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
        
        # Risk metrics
        if len(equity_df) > 1:
            sharpe_ratio = equity_df['returns'].mean() / equity_df['returns'].std() * np.sqrt(252)
            max_drawdown = self._calculate_max_drawdown(equity_df['equity'])
        else:
            sharpe_ratio = 0
            max_drawdown = 0
        
        return {
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'final_capital': self.capital,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'equity_curve': equity_df
        }
    
    def _calculate_max_drawdown(self, equity_series: pd.Series) -> float:
        """Calculate maximum drawdown."""
        
        peak = equity_series.expanding().max()
        drawdown = (equity_series - peak) / peak
        return drawdown.min()

def load_sample_data(symbol: str, days: int = 365) -> pd.DataFrame:
    """Generate sample data for backtesting (since we can't access real historical data easily)."""
    
    print(f"Generating sample data for {symbol}...")
    
    # Generate synthetic price data
    np.random.seed(42)  # For reproducible results
    
    dates = pd.date_range(start=datetime.now() - timedelta(days=days), 
                         end=datetime.now(), freq='1H')
    
    # Generate price series with some trend and volatility
    returns = np.random.normal(0.0001, 0.02, len(dates))
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        high = price * (1 + abs(np.random.normal(0, 0.01)))
        low = price * (1 - abs(np.random.normal(0, 0.01)))
        volume = np.random.randint(1000, 10000)
        
        data.append({
            'timestamp': date,
            'open': price,
            'high': high,
            'low': low,
            'close': price,
            'volume': volume
        })
    
    return pd.DataFrame(data)

def main():
    """Main backtesting function."""
    
    parser = argparse.ArgumentParser(description='Backtest trading strategies')
    parser.add_argument('--strategy', required=True, 
                       choices=['moving_average', 'rsi', 'mean_reversion'],
                       help='Strategy to backtest')
    parser.add_argument('--symbol', default='AAPL', help='Symbol to test')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, 
                       help='Initial capital')
    
    args = parser.parse_args()
    
    # Set up logging
    logger = setup_logger(console=True)
    
    print("="*60)
    print("Alpaquero Backtesting Engine")
    print("="*60)
    
    try:
        # Initialize strategy
        if args.strategy == 'moving_average':
            config = settings.get_strategy_config('moving_average')
            strategy = MovingAverageStrategy(config, logger)
        elif args.strategy == 'rsi':
            config = settings.get_strategy_config('rsi')
            strategy = RSIStrategy(config, logger)
        elif args.strategy == 'mean_reversion':
            config = settings.get_strategy_config('mean_reversion')
            strategy = MeanReversionStrategy(config, logger)
        
        # Load data (using sample data for demo)
        data = load_sample_data(args.symbol)
        
        # Run backtest
        backtester = Backtester(args.capital)
        results = backtester.run_backtest(strategy, args.symbol, data, 
                                        args.start, args.end)
        
        # Print results
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        print(f"Strategy: {args.strategy}")
        print(f"Symbol: {args.symbol}")
        print(f"Period: {args.start} to {args.end}")
        print(f"Initial Capital: ${args.capital:,.2f}")
        print("-"*60)
        print(f"Final Capital: ${results['final_capital']:,.2f}")
        print(f"Total Return: {results['total_return_pct']:.2f}%")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1%}")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {results['max_drawdown']:.1%}")
        print("="*60)
        
    except Exception as e:
        print(f"Error during backtesting: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
