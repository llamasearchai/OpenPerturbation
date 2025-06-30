"""
Logging configuration for OpenPerturbation.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import colorlog
import time

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    enable_colors: bool = True
) -> logging.Logger:
    """Setup logging configuration."""
    
    # Create logs directory if it doesn't exist
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert string level to logging level
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    
    # Create formatter
    if log_format is None:
        log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    formatter = logging.Formatter(log_format)
    
    # Create console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    
    # Add colors to console output
    if enable_colors and sys.stdout.isatty():
        colored_formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            log_colors={
                'DEBUG': 'cyan',
                'INFO': 'green',
                'WARNING': 'yellow',
                'ERROR': 'red',
                'CRITICAL': 'red,bg_white',
            },
            secondary_log_colors={},
            style='%'
        )
        console_handler.setFormatter(colored_formatter)
    else:
        console_handler.setFormatter(formatter)
    
    # Create file handler if log_file is specified
    handlers = [console_handler]
    
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        root_logger.addHandler(handler)
    
    # Set specific logger levels
    logging.getLogger("uvicorn").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("fastapi").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured with level: {level}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Return the configured logger
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger with the specified name."""
    return logging.getLogger(name)

class LoggerMixin:
    """Mixin class to add logging capabilities."""
    
    @property
    def logger(self) -> logging.Logger:
        """Get logger for this class."""
        return logging.getLogger(self.__class__.__name__)

class PerformanceLogger:
    """Performance logging utility."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.start_times: Dict[str, float] = {}
    
    def start_timer(self, operation: str) -> None:
        """Start timing an operation."""
        self.start_times[operation] = time.time()
        self.logger.debug(f"Started timing: {operation}")
    
    def end_timer(self, operation: str, log_level: str = "info") -> float:
        """End timing an operation and log the duration."""
        if operation not in self.start_times:
            self.logger.warning(f"No start time found for operation: {operation}")
            return 0.0
        
        duration = time.time() - self.start_times[operation]
        log_method = getattr(self.logger, log_level.lower(), self.logger.info)
        log_method(f"Operation '{operation}' completed in {duration:.4f}s")
        
        del self.start_times[operation]
        return duration
    
    def log_operation(self, operation: str, func, *args, **kwargs):
        """Decorator-style operation logging."""
        self.start_timer(operation)
        try:
            result = func(*args, **kwargs)
            self.end_timer(operation)
            return result
        except Exception as e:
            self.logger.error(f"Operation '{operation}' failed after {time.time() - self.start_times[operation]:.4f}s: {e}")
            raise

class StructuredLogger:
    """Structured logging utility."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
    
    def log_event(self, event_type: str, **kwargs) -> None:
        """Log a structured event."""
        event_data = {
            "event_type": event_type,
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.info(f"EVENT: {event_data}")
    
    def log_metric(self, metric_name: str, value: float, **kwargs) -> None:
        """Log a metric."""
        metric_data = {
            "metric": metric_name,
            "value": value,
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.info(f"METRIC: {metric_data}")
    
    def log_error(self, error_type: str, error_message: str, **kwargs) -> None:
        """Log an error."""
        error_data = {
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": time.time(),
            **kwargs
        }
        self.logger.error(f"ERROR: {error_data}")

def create_logging_config() -> Dict[str, Any]:
    """Create logging configuration dictionary."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "colored": {
                "()": "colorlog.ColoredFormatter",
                "format": "%(log_color)s%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "log_colors": {
                    "DEBUG": "cyan",
                    "INFO": "green",
                    "WARNING": "yellow",
                    "ERROR": "red",
                    "CRITICAL": "red,bg_white",
                }
            },
            "json": {
                "()": "logging_utilities.formatters.json_formatter.JsonFormatter",
                "format": {
                    "timestamp": "asctime",
                    "level": "levelname",
                    "name": "name",
                    "message": "message"
                }
            }
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "colored",
                "stream": "ext://sys.stdout"
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "formatter": "standard",
                "filename": "logs/openperturbation.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            },
            "error_file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "ERROR",
                "formatter": "standard",
                "filename": "logs/errors.log",
                "maxBytes": 10485760,  # 10MB
                "backupCount": 5
            }
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            },
            "openperturbation": {
                "handlers": ["console", "file"],
                "level": "DEBUG",
                "propagate": False
            },
            "uvicorn": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            },
            "fastapi": {
                "handlers": ["console"],
                "level": "WARNING",
                "propagate": False
            }
        },
    }

def setup_structured_logging(
    log_file: str = "logs/structured.log",
    enable_json: bool = True
) -> None:
    """Setup structured logging."""
    
    # Create logs directory
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Configure structured logging
    if enable_json:
        try:
            from logging_utilities.formatters.json_formatter import JsonFormatter
            formatter = JsonFormatter(
                {
                    "timestamp": "asctime",
                    "level": "levelname",
                    "name": "name",
                    "message": "message"
                }
            )
        except ImportError:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    
    # Get structured logger
    structured_logger = logging.getLogger("openperturbation.structured")
    structured_logger.addHandler(file_handler)
    structured_logger.setLevel(logging.INFO)

# Export logging components
__all__ = [
    'setup_logging',
    'get_logger',
    'LoggerMixin',
    'PerformanceLogger',
    'StructuredLogger',
    'create_logging_config',
    'setup_structured_logging'
] 