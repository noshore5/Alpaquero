import logging
import os
from datetime import datetime
from typing import Optional

def setup_logger(
    name: str = "alpaquero",
    level: str = "INFO",
    log_file: Optional[str] = None,
    console: bool = True
) -> logging.Logger:
    """Set up logger with file and console handlers."""
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
            
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def log_trade(logger: logging.Logger, action: str, symbol: str, quantity: int, 
              price: float, timestamp: Optional[datetime] = None):
    """Log trade execution details."""
    if timestamp is None:
        timestamp = datetime.now()
    
    logger.info(
        f"TRADE EXECUTED: {action} {quantity} shares of {symbol} "
        f"at ${price:.2f} on {timestamp.strftime('%Y-%m-%d %H:%M:%S')}"
    )

def log_signal(logger: logging.Logger, strategy: str, symbol: str, 
               signal: str, details: Optional[str] = None):
    """Log strategy signal generation."""
    message = f"SIGNAL: {strategy} generated {signal} signal for {symbol}"
    if details:
        message += f" - {details}"
    logger.info(message)

def log_error(logger: logging.Logger, operation: str, error: Exception):
    """Log error with context."""
    logger.error(f"ERROR in {operation}: {str(error)}", exc_info=True)
