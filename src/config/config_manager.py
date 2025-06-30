"""
Configuration management for OpenPerturbation.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union, List, cast
from dataclasses import dataclass, field
import yaml
import json

try:
    from omegaconf import DictConfig, OmegaConf  # type: ignore
    OMEGACONF_AVAILABLE = True
except ImportError:
    OMEGACONF_AVAILABLE = False
    DictConfig = Dict[str, Any]

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Model configuration."""
    name: str
    type: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path: Optional[str] = None
    device: str = "auto"
    batch_size: int = 32
    learning_rate: float = 0.001
    num_epochs: int = 100

@dataclass
class DataConfig:
    """Data configuration."""
    data_type: str
    data_path: str
    batch_size: int = 32
    num_workers: int = 4
    shuffle: bool = True
    validation_split: float = 0.2
    test_split: float = 0.1
    max_file_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class TrainingConfig:
    """Training configuration."""
    num_epochs: int = 100
    learning_rate: float = 0.001
    weight_decay: float = 1e-5
    early_stopping_patience: int = 10
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"
    use_gpu: bool = True
    mixed_precision: bool = False

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    name: str
    description: str
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    random_seed: int = 42
    use_gpu: bool = True
    max_workers: int = 4

@dataclass
class APIConfig:
    """API configuration."""
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    rate_limit_per_minute: int = 60
    max_file_size: int = 100 * 1024 * 1024  # 100MB

@dataclass
class SystemConfig:
    """System configuration."""
    max_workers: int = 4
    memory_limit: str = "8GB"
    gpu_memory_fraction: float = 0.8
    temp_dir: str = "temp"
    cache_dir: str = "cache"

class ConfigManager:
    """Configuration manager for OpenPerturbation."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration manager."""
        self.config_path = config_path
        self.config: Dict[str, Any] = {}
        self.env_prefix = "OPENPERTURBATION"
        self.load_config()
    
    def load_config(self) -> None:
        """Load configuration from file or environment."""
        if self.config_path and Path(self.config_path).exists():
            self._load_from_file()
        else:
            self._load_default_config()
        
        # Override with environment variables
        self._load_from_environment()
        
        # Validate configuration
        validation_result = self.validate_config()
        if not validation_result["valid"]:
            logger.warning(f"Configuration validation failed: {validation_result['errors']}")
    
    def _load_from_file(self) -> None:
        """Load configuration from file."""
        if not self.config_path:
            self._load_default_config()
            return
            
        try:
            if OMEGACONF_AVAILABLE and self.config_path.endswith(('.yaml', '.yml')):
                loaded_config = OmegaConf.load(self.config_path)
                if isinstance(loaded_config, (dict, list)):
                    unboxed_config = OmegaConf.to_container(loaded_config, resolve=True)
                    if isinstance(unboxed_config, dict):
                        self.config = cast(Dict[str, Any], unboxed_config)
                    else:
                        logger.warning(f"Configuration from {self.config_path} is not a dictionary, using empty config.")
                        self.config = {}
                else:
                    logger.warning(f"YAML file at {self.config_path} did not load as a dictionary or list.")
                    self.config = {}

            else:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    if self.config_path.endswith('.json'):
                        self.config = json.load(f)
                    elif self.config_path.endswith(('.yaml', '.yml')):
                        self.config = yaml.safe_load(f)
                    else:
                        logger.warning(f"Unsupported config file format: {self.config_path}")
                        self._load_default_config()

            logger.info(f"Loaded configuration from {self.config_path}")
        except Exception as e:
            logger.warning(f"Could not load config from {self.config_path}: {e}")
            self._load_default_config()
    
    def _load_default_config(self) -> None:
        """Load default configuration."""
        self.config = {
            "api": {
                "host": "127.0.0.1",
                "port": 8000,
                "debug": False,
                "log_level": "INFO",
                "cors_origins": ["*"],
                "rate_limit_per_minute": 60,
                "max_file_size": 100 * 1024 * 1024
            },
            "models": {
                "default_model": "multimodal_fusion",
                "checkpoint_dir": "checkpoints",
                "device": "auto",
                "batch_size": 32,
                "learning_rate": 0.001,
                "num_epochs": 100
            },
            "data": {
                "upload_dir": "uploads",
                "output_dir": "outputs",
                "max_file_size": 100 * 1024 * 1024,
                "allowed_extensions": {
                    "genomics": [".csv", ".tsv", ".h5", ".xlsx", ".txt"],
                    "imaging": [".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp"],
                    "chemical": [".sdf", ".mol", ".csv", ".tsv", ".txt"]
                }
            },
            "experiments": {
                "default_batch_size": 32,
                "default_epochs": 100,
                "default_learning_rate": 0.001,
                "random_seed": 42
            },
            "system": {
                "max_workers": 4,
                "memory_limit": "8GB",
                "gpu_memory_fraction": 0.8,
                "temp_dir": "temp",
                "cache_dir": "cache"
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": None,
                "max_bytes": 10 * 1024 * 1024,
                "backup_count": 5
            },
            "security": {
                "allowed_hosts": ["*"],
                "trusted_proxies": [],
                "rate_limiting": True
            }
        }
        logger.info("Loaded default configuration")
    
    def _load_from_environment(self) -> None:
        """Override configuration with environment variables."""
        env_mappings = {
            "OPENPERTURBATION_API_HOST": ("api", "host"),
            "OPENPERTURBATION_API_PORT": ("api", "port"),
            "OPENPERTURBATION_LOG_LEVEL": ("api", "log_level"),
            "OPENPERTURBATION_DEBUG": ("api", "debug"),
            "OPENPERTURBATION_UPLOAD_DIR": ("data", "upload_dir"),
            "OPENPERTURBATION_OUTPUT_DIR": ("data", "output_dir"),
            "OPENPERTURBATION_MAX_FILE_SIZE": ("data", "max_file_size"),
            "OPENPERTURBATION_DEFAULT_BATCH_SIZE": ("experiments", "default_batch_size"),
            "OPENPERTURBATION_DEFAULT_EPOCHS": ("experiments", "default_epochs"),
            "OPENPERTURBATION_LEARNING_RATE": ("experiments", "default_learning_rate"),
            "OPENPERTURBATION_MAX_WORKERS": ("system", "max_workers"),
            "OPENPERTURBATION_MEMORY_LIMIT": ("system", "memory_limit"),
            "OPENPERTURBATION_GPU_MEMORY_FRACTION": ("system", "gpu_memory_fraction"),
            "OPENPERTURBATION_RATE_LIMIT": ("api", "rate_limit_per_minute"),
            "OPENPERTURBATION_CORS_ORIGINS": ("api", "cors_origins")
        }
        
        for env_var, config_path in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                self._set_nested_value(config_path, value)
    
    def _set_nested_value(self, path: tuple, value: Any) -> None:
        """Set a nested configuration value."""
        current = self.config
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        # Convert value type if needed
        if isinstance(value, str):
            if value.lower() in ('true', 'false'):
                value = value.lower() == 'true'
            elif value.isdigit():
                value = int(value)
            elif value.replace('.', '').isdigit():
                value = float(value)
            elif path[-1] == "cors_origins":
                value = [v.strip() for v in value.split(",")]
        
        current[path[-1]] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        current = self.config
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default
        
        return current
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        keys = key.split('.')
        self._set_nested_value(tuple(keys), value)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate configuration and return errors/warnings."""
        errors = []
        warnings = []
        
        # Check required fields
        required_fields = [
            ("api.host", str),
            ("api.port", int),
            ("data.upload_dir", str),
            ("data.output_dir", str)
        ]
        
        for field, expected_type in required_fields:
            value = self.get(field)
            if value is None:
                errors.append(f"Missing required field: {field}")
            elif not isinstance(value, expected_type):
                errors.append(f"Invalid type for {field}: expected {expected_type.__name__}")
        
        # Check port range
        port = self.get("api.port")
        if port is not None and (port < 1 or port > 65535):
            errors.append("Port must be between 1 and 65535")
        
        # Check file size limit
        max_file_size = self.get("data.max_file_size")
        if max_file_size is not None and max_file_size <= 0:
            errors.append("Max file size must be positive")
        
        # Check rate limit
        rate_limit = self.get("api.rate_limit_per_minute")
        if rate_limit is not None and rate_limit <= 0:
            errors.append("Rate limit must be positive")
        
        # Warnings
        batch_size = self.get("experiments.default_batch_size")
        if batch_size and batch_size > 128:
            warnings.append("Large batch size may cause memory issues")
        
        learning_rate = self.get("experiments.default_learning_rate")
        if learning_rate and learning_rate > 0.01:
            warnings.append("High learning rate may cause training instability")
        
        max_workers = self.get("system.max_workers")
        if max_workers and max_workers > 16:
            warnings.append("High number of workers may not improve performance")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }
    
    def save_config(self, path: Optional[str] = None) -> None:
        """Save configuration to file."""
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No path specified for saving configuration")
        
        try:
            # Create directory if it doesn't exist
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'w') as f:
                if save_path.endswith('.json'):
                    json.dump(self.config, f, indent=2, default=str)
                else:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Could not save configuration to {save_path}: {e}")
            raise
    
    def get_api_config(self) -> APIConfig:
        """Get API configuration as dataclass."""
        api_config = self.get("api", {})
        return APIConfig(
            host=api_config.get("host", "127.0.0.1"),
            port=api_config.get("port", 8000),
            debug=api_config.get("debug", False),
            log_level=api_config.get("log_level", "INFO"),
            cors_origins=api_config.get("cors_origins", ["*"]),
            rate_limit_per_minute=api_config.get("rate_limit_per_minute", 60),
            max_file_size=api_config.get("max_file_size", 100 * 1024 * 1024)
        )
    
    def get_system_config(self) -> SystemConfig:
        """Get system configuration as dataclass."""
        system_config = self.get("system", {})
        return SystemConfig(
            max_workers=system_config.get("max_workers", 4),
            memory_limit=system_config.get("memory_limit", "8GB"),
            gpu_memory_fraction=system_config.get("gpu_memory_fraction", 0.8),
            temp_dir=system_config.get("temp_dir", "temp"),
            cache_dir=system_config.get("cache_dir", "cache")
        )
    
    def create_experiment_config(self, name: str, description: str) -> ExperimentConfig:
        """Create experiment configuration."""
        model_config = ModelConfig(
            name=self.get("models.default_model", "multimodal_fusion"),
            type="multimodal_fusion",
            batch_size=self.get("experiments.default_batch_size", 32),
            learning_rate=self.get("experiments.default_learning_rate", 0.001),
            num_epochs=self.get("experiments.default_epochs", 100)
        )
        
        data_config = DataConfig(
            data_type="multimodal",
            data_path="",
            batch_size=self.get("experiments.default_batch_size", 32),
            max_file_size=self.get("data.max_file_size", 100 * 1024 * 1024)
        )
        
        training_config = TrainingConfig(
            num_epochs=self.get("experiments.default_epochs", 100),
            learning_rate=self.get("experiments.default_learning_rate", 0.001),
            use_gpu=self.get("system.gpu_memory_fraction", 0.8) > 0
        )
        
        return ExperimentConfig(
            name=name,
            description=description,
            model=model_config,
            data=data_config,
            training=training_config,
            random_seed=self.get("experiments.random_seed", 42),
            max_workers=self.get("system.max_workers", 4)
        )

# Global configuration instance
config_manager = ConfigManager()

def get_config() -> ConfigManager:
    """Get global configuration manager."""
    return config_manager

def get_api_config() -> APIConfig:
    """Get API configuration."""
    return config_manager.get_api_config()

def get_system_config() -> SystemConfig:
    """Get system configuration."""
    return config_manager.get_system_config()

# Export configuration components
__all__ = [
    'ConfigManager',
    'APIConfig',
    'SystemConfig',
    'ModelConfig',
    'DataConfig',
    'TrainingConfig',
    'ExperimentConfig',
    'get_config',
    'get_api_config',
    'get_system_config'
] 