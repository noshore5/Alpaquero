#!/usr/bin/env python3
"""
Setup script for Alpaquero
Helps users configure the trading bot for first use.
"""

import os
import shutil
from pathlib import Path

def setup_environment():
    """Set up the environment for first use."""
    
    print("=" * 50)
    print("Welcome to Alpaquero Setup")
    print("=" * 50)
    
    # Create .env file from template
    env_template = Path(".env.template")
    env_file = Path(".env")
    
    if not env_file.exists() and env_template.exists():
        print("\n1. Setting up environment variables...")
        shutil.copy(env_template, env_file)
        print(f"Created {env_file} from template")
        print("Please edit .env file with your Alpaca API credentials")
    else:
        print("\n1. Environment file already exists")
    
    # Check if logs directory exists
    logs_dir = Path("logs")
    if not logs_dir.exists():
        print("\n2. Creating logs directory...")
        logs_dir.mkdir()
        print("Created logs directory")
    else:
        print("\n2. Logs directory already exists")
    
    # Display next steps
    print("\n" + "=" * 50)
    print("SETUP COMPLETE")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit .env file with your Alpaca API credentials:")
    print("   - Get API keys from: https://alpaca.markets/")
    print("   - Use paper trading URL for testing")
    print("\n2. Review config/config.yaml and adjust settings:")
    print("   - Choose symbols to trade")
    print("   - Enable/disable strategies")
    print("   - Set risk parameters")
    print("\n3. Test the setup:")
    print("   python -m pytest tests/")
    print("\n4. Run backtesting:")
    print("   python backtest.py --strategy moving_average --start 2024-01-01 --end 2024-12-31")
    print("\n5. Start live trading (paper trading recommended):")
    print("   python main.py")
    print("\nImportant: Always test with paper trading first!")
    print("=" * 50)

if __name__ == "__main__":
    setup_environment()
