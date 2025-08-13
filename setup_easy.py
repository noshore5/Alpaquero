#!/usr/bin/env python3
"""
Setup script for Alpaquero trading bot.
This script helps with the initial setup and dependency installation.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors."""
    print(f"\n{description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✓ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"✗ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please install Python 3.8 or later")
        return False

def create_directories():
    """Create necessary directories."""
    directories = ['logs', 'data', 'backtest_results']
    
    for directory in directories:
        path = Path(directory)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created directory: {directory}")
        else:
            print(f"✓ Directory already exists: {directory}")

def install_basic_dependencies():
    """Install basic dependencies that usually work on all systems."""
    basic_deps = [
        "alpaca-trade-api>=3.0.0",
        "pyyaml>=6.0.0", 
        "python-dotenv>=1.0.0",
        "requests>=2.31.0",
        "schedule>=1.2.0",
        "websockets>=11.0.0",
        "pytest>=7.4.0"
    ]
    
    print("\nInstalling basic dependencies...")
    for dep in basic_deps:
        cmd = f"pip install \"{dep}\""
        if not run_command(cmd, f"Installing {dep.split('>=')[0]}"):
            return False
    return True

def install_data_dependencies():
    """Try to install data analysis dependencies."""
    print("\nAttempting to install data analysis dependencies...")
    print("Note: If this fails, you can install using conda instead:")
    print("conda install pandas numpy matplotlib seaborn plotly -c conda-forge")
    
    data_deps = ["pandas", "numpy", "matplotlib", "seaborn", "plotly"]
    
    failed_deps = []
    for dep in data_deps:
        cmd = f"pip install {dep}"
        if not run_command(cmd, f"Installing {dep}"):
            failed_deps.append(dep)
    
    if failed_deps:
        print(f"\n⚠️  Failed to install: {', '.join(failed_deps)}")
        print("You can install these manually using:")
        print("conda install pandas numpy matplotlib seaborn plotly -c conda-forge")
        print("or try installing pre-compiled wheels")
    
    return len(failed_deps) == 0

def setup_environment_file():
    """Setup the environment file."""
    env_file = Path(".env")
    template_file = Path(".env.template")
    
    if not env_file.exists() and template_file.exists():
        shutil.copy(template_file, env_file)
        print("✓ Created .env file from template")
        print("⚠️  Please edit .env file with your Alpaca API credentials")
    else:
        print("✓ .env file already exists")

def verify_installation():
    """Verify that key components can be imported."""
    print("\nVerifying installation...")
    
    try:
        import alpaca_trade_api
        print("✓ alpaca-trade-api imported successfully")
    except ImportError:
        print("✗ Failed to import alpaca-trade-api")
        return False
    
    try:
        import yaml
        print("✓ pyyaml imported successfully")
    except ImportError:
        print("✗ Failed to import pyyaml")
        return False
    
    try:
        import pandas
        print("✓ pandas imported successfully")
    except ImportError:
        print("⚠️  pandas not available - some features may not work")
    
    try:
        import numpy
        print("✓ numpy imported successfully")
    except ImportError:
        print("⚠️  numpy not available - some features may not work")
    
    return True

def main():
    """Main setup function."""
    print("=" * 50)
    print("Alpaquero Trading Bot Setup")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_basic_dependencies():
        print("\n✗ Failed to install basic dependencies")
        return False
    
    # Try to install data dependencies
    install_data_dependencies()
    
    # Setup environment file
    setup_environment_file()
    
    # Verify installation
    if not verify_installation():
        print("\n✗ Installation verification failed")
        return False
    
    print("\n" + "=" * 50)
    print("Setup completed!")
    print("=" * 50)
    print("\nNext steps:")
    print("1. Edit .env file with your Alpaca API credentials")
    print("2. Review config/config.yaml for trading parameters")
    print("3. Run: python main.py")
    print("\nFor backtesting:")
    print("python backtest.py --strategy moving_average --start 2024-01-01 --end 2024-12-31")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
