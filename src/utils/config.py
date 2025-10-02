import yaml
import os
from pathlib import Path
from typing import Dict, Any

class Config:
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def load(self, config_path: str = "config/config.yaml"):
        """Load configuration from YAML file."""
        config_file = Path(config_path)
        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_file, 'r') as file:
            self._config = yaml.safe_load(file)
        
        # Load model parameters
        model_params_path = config_file.parent / "model_params.yaml"
        if model_params_path.exists():
            with open(model_params_path, 'r') as file:
                model_params = yaml.safe_load(file)
                self._config['model_params'] = model_params
    
    def get(self, key: str, default: Any = None):
        """Get configuration value by key (supports dot notation)."""
        if self._config is None:
            self.load()
        
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

def load_config(config_path: str = "config/config.yaml"):
    """Load configuration from file."""
    config = Config()
    config.load(config_path)
    return config

def get_config():
    """Get the global configuration instance."""
    return Config()