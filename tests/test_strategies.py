import unittest
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.strategies.moving_average import MovingAverageStrategy
from src.strategies.rsi_strategy import RSIStrategy
from src.strategies.mean_reversion import MeanReversionStrategy
from src.strategies.base_strategy import Signal
from src.utils.logger import setup_logger

class TestStrategies(unittest.TestCase):
    """Test cases for trading strategies."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = setup_logger(level='ERROR')  # Suppress logs during testing
        
        # Create sample data
        np.random.seed(42)
        dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
        prices = 100 + np.cumsum(np.random.randn(len(dates)) * 0.5)
        
        self.sample_data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': prices * 1.02,
            'low': prices * 0.98,
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
    def test_moving_average_strategy(self):
        """Test Moving Average Strategy."""
        
        config = {
            'short_window': 10,
            'long_window': 20,
            'position_size_pct': 0.1
        }
        
        strategy = MovingAverageStrategy(config, self.logger)
        
        # Test signal generation
        signal = strategy.generate_signals('TEST', self.sample_data, {})
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])
        
        # Test position size calculation
        position_size = strategy.calculate_position_size('TEST', 100.0, 10000.0)
        self.assertGreater(position_size, 0)
        self.assertEqual(position_size, 10)  # 10% of 10000 / 100
        
    def test_rsi_strategy(self):
        """Test RSI Strategy."""
        
        config = {
            'period': 14,
            'oversold': 30,
            'overbought': 70,
            'position_size_pct': 0.1
        }
        
        strategy = RSIStrategy(config, self.logger)
        
        # Test signal generation
        signal = strategy.generate_signals('TEST', self.sample_data, {})
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])
        
        # Test RSI calculation
        rsi = strategy._calculate_rsi(self.sample_data['close'])
        self.assertFalse(rsi.isna().all())
        self.assertTrue((rsi >= 0).all())
        self.assertTrue((rsi <= 100).all())
        
    def test_mean_reversion_strategy(self):
        """Test Mean Reversion Strategy."""
        
        config = {
            'period': 20,
            'std_dev': 2,
            'position_size_pct': 0.1
        }
        
        strategy = MeanReversionStrategy(config, self.logger)
        
        # Test signal generation
        signal = strategy.generate_signals('TEST', self.sample_data, {})
        self.assertIn(signal, [Signal.BUY, Signal.SELL, Signal.HOLD])
        
        # Test Bollinger Bands calculation
        bb_data = strategy._calculate_bollinger_bands(self.sample_data['close'])
        self.assertIn('upper', bb_data)
        self.assertIn('middle', bb_data)
        self.assertIn('lower', bb_data)
        
        # Upper band should be higher than lower band
        self.assertTrue((bb_data['upper'] > bb_data['lower']).all())
        
    def test_position_management(self):
        """Test position management functionality."""
        
        config = {'position_size_pct': 0.1}
        strategy = MovingAverageStrategy(config, self.logger)
        
        # Test position creation
        strategy.update_position('TEST', 'long', 100, 50.0, '2024-01-01')
        self.assertTrue(strategy.has_position('TEST'))
        
        position = strategy.get_position('TEST')
        self.assertEqual(position['quantity'], 100)
        self.assertEqual(position['entry_price'], 50.0)
        
        # Test position closing
        strategy.close_position('TEST')
        self.assertFalse(strategy.has_position('TEST'))
        
    def test_exit_conditions(self):
        """Test position exit conditions."""
        
        config = {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.10,
            'position_size_pct': 0.1
        }
        strategy = MovingAverageStrategy(config, self.logger)
        
        # Create a position
        strategy.update_position('TEST', 'long', 100, 100.0, '2024-01-01')
        
        # Test stop loss
        should_exit = strategy.should_exit_position('TEST', 94.0, strategy.get_position('TEST'))
        self.assertTrue(should_exit)  # 6% loss should trigger 5% stop loss
        
        # Test take profit
        should_exit = strategy.should_exit_position('TEST', 111.0, strategy.get_position('TEST'))
        self.assertTrue(should_exit)  # 11% gain should trigger 10% take profit
        
        # Test no exit
        should_exit = strategy.should_exit_position('TEST', 102.0, strategy.get_position('TEST'))
        self.assertFalse(should_exit)  # 2% gain should not trigger exit

if __name__ == '__main__':
    unittest.main()
