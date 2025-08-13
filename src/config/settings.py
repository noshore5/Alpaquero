import os
import yaml
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings:
    """Configuration settings for the trading bot."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        self.config_path = config_path
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r') as file:
                return yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML configuration: {e}")
    
    @property
    def alpaca_api_key(self) -> str:
        """Get Alpaca API key from environment."""
        key = os.getenv('ALPACA_API_KEY')
        if not key:
            raise ValueError("ALPACA_API_KEY not found in environment variables")
        return key
    
    @property
    def alpaca_secret_key(self) -> str:
        """Get Alpaca secret key from environment."""
        key = os.getenv('ALPACA_SECRET_KEY')
        if not key:
            raise ValueError("ALPACA_SECRET_KEY not found in environment variables")
        return key
    
    @property
    def alpaca_base_url(self) -> str:
        """Get Alpaca base URL from environment."""
        return os.getenv('ALPACA_BASE_URL', 'https://paper-api.alpaca.markets')
    
    @property
    def symbols(self) -> list:
        """Get list of symbols to trade."""
        return self._config.get('trading', {}).get('symbols', ['AAPL'])
    
    @property
    def position_size_pct(self) -> float:
        """Get position size percentage."""
        return self._config.get('trading', {}).get('position_size_pct', 0.1)
    
    @property
    def stop_loss_pct(self) -> float:
        """Get stop loss percentage."""
        return self._config.get('trading', {}).get('stop_loss_pct', 0.05)
    
    @property
    def take_profit_pct(self) -> float:
        """Get take profit percentage."""
        return self._config.get('trading', {}).get('take_profit_pct', 0.10)
    
    @property
    def max_positions(self) -> int:
        """Get maximum number of concurrent positions."""
        return self._config.get('trading', {}).get('max_positions', 3)
    
    @property
    def trading_hours(self) -> Dict[str, str]:
        """Get trading hours configuration."""
        return self._config.get('trading', {}).get('trading_hours', {
            'start': '09:30',
            'end': '16:00',
            'timezone': 'America/New_York'
        })
    
    @property
    def strategies(self) -> Dict[str, Any]:
        """Get strategies configuration."""
        return self._config.get('strategies', {})
    
    @property
    def risk_management(self) -> Dict[str, float]:
        """Get risk management configuration."""
        return self._config.get('risk_management', {})
    
    @property
    def data_config(self) -> Dict[str, Any]:
        """Get data configuration."""
        return self._config.get('data', {})
    
    @property
    def logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self._config.get('logging', {})
    
    def get_strategy_config(self, strategy_name: str) -> Dict[str, Any]:
        """Get configuration for a specific strategy."""
        return self.strategies.get(strategy_name, {})
    
    def is_strategy_enabled(self, strategy_name: str) -> bool:
        """Check if a strategy is enabled."""
        strategy_config = self.get_strategy_config(strategy_name)
        return strategy_config.get('enabled', False)

# Global settings instance
settings = Settings()
