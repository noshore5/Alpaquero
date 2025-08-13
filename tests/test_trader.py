import unittest
from unittest.mock import Mock, MagicMock
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.trading.risk_manager import RiskManager
from src.utils.logger import setup_logger

class TestTrader(unittest.TestCase):
    """Test cases for trading functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.logger = setup_logger(level='ERROR')  # Suppress logs during testing
        
    def test_risk_manager_validation(self):
        """Test risk manager trade validation."""
        
        config = {
            'max_portfolio_risk': 0.20,
            'max_position_size': 0.15,
            'min_cash_balance': 1000
        }
        
        risk_manager = RiskManager(config, self.logger)
        
        # Test valid trade
        is_valid = risk_manager.validate_trade(
            symbol='TEST',
            quantity=10,
            price=100.0,
            side='buy',
            portfolio_value=10000.0,
            current_positions={},
            cash_balance=5000.0
        )
        self.assertTrue(is_valid)
        
        # Test trade that violates minimum cash balance
        is_valid = risk_manager.validate_trade(
            symbol='TEST',
            quantity=50,
            price=100.0,
            side='buy',
            portfolio_value=10000.0,
            current_positions={},
            cash_balance=2000.0
        )
        self.assertFalse(is_valid)  # Would leave only $2000 - $5000 = -$3000
        
        # Test trade that violates position size limit
        is_valid = risk_manager.validate_trade(
            symbol='TEST',
            quantity=20,
            price=100.0,
            side='buy',
            portfolio_value=10000.0,
            current_positions={},
            cash_balance=5000.0
        )
        self.assertFalse(is_valid)  # 20% position exceeds 15% limit
        
    def test_position_size_calculation(self):
        """Test position size calculation with risk limits."""
        
        config = {
            'max_position_size': 0.10  # 10% max position size
        }
        
        risk_manager = RiskManager(config, self.logger)
        
        # Test normal calculation
        position_size = risk_manager.calculate_position_size(
            symbol='TEST',
            price=100.0,
            portfolio_value=10000.0,
            strategy_size_pct=0.15  # Strategy wants 15%
        )
        
        # Should be limited to 10% (max_position_size)
        expected_quantity = int((10000.0 * 0.10) / 100.0)
        self.assertEqual(position_size, expected_quantity)
        
    def test_stop_loss_calculation(self):
        """Test stop loss price calculation."""
        
        config = {}
        risk_manager = RiskManager(config, self.logger)
        
        # Test long position stop loss
        stop_price = risk_manager.calculate_stop_loss_price(
            entry_price=100.0,
            side='buy',
            stop_loss_pct=0.05
        )
        self.assertEqual(stop_price, 95.0)  # 5% below entry
        
        # Test short position stop loss
        stop_price = risk_manager.calculate_stop_loss_price(
            entry_price=100.0,
            side='sell',
            stop_loss_pct=0.05
        )
        self.assertEqual(stop_price, 105.0)  # 5% above entry
        
    def test_take_profit_calculation(self):
        """Test take profit price calculation."""
        
        config = {}
        risk_manager = RiskManager(config, self.logger)
        
        # Test long position take profit
        take_profit_price = risk_manager.calculate_take_profit_price(
            entry_price=100.0,
            side='buy',
            take_profit_pct=0.10
        )
        self.assertEqual(take_profit_price, 110.0)  # 10% above entry
        
        # Test short position take profit
        take_profit_price = risk_manager.calculate_take_profit_price(
            entry_price=100.0,
            side='sell',
            take_profit_pct=0.10
        )
        self.assertEqual(take_profit_price, 90.0)  # 10% below entry
        
    def test_risk_metrics(self):
        """Test risk metrics calculation."""
        
        config = {}
        risk_manager = RiskManager(config, self.logger)
        
        current_positions = {
            'AAPL': {'market_value': 5000},
            'MSFT': {'market_value': 3000}
        }
        
        metrics = risk_manager.get_risk_metrics(
            portfolio_value=20000.0,
            current_positions=current_positions,
            cash_balance=12000.0
        )
        
        self.assertEqual(metrics['portfolio_risk'], 0.4)  # 8000/20000
        self.assertEqual(metrics['cash_percentage'], 0.6)  # 12000/20000
        self.assertEqual(metrics['largest_position_pct'], 0.25)  # 5000/20000
        self.assertEqual(metrics['number_of_positions'], 2)

if __name__ == '__main__':
    unittest.main()
