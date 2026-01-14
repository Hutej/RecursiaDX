"""
Configuration loading and validation utilities.
"""
import yaml
from pathlib import Path
from typing import Dict, Any
from .logger import get_logger

logger = get_logger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If config file is invalid
    """
    config_file = Path(config_path)
    
    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        logger.info(f"✓ Configuration loaded from {config_path}")
        return config
    
    except yaml.YAMLError as e:
        logger.error(f"Error parsing config file: {e}")
        raise


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        True if valid, raises ValueError otherwise
    """
    required_sections = ['experiment', 'preprocessing', 'feature_extraction', 
                        'sampling', 'mil', 'hardware', 'paths']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required config section: {section}")
    
    # Validate experiment config
    if config['experiment']['seed'] < 0:
        raise ValueError("Seed must be non-negative")
    
    # Validate preprocessing config
    if config['preprocessing']['tile_size'] <= 0:
        raise ValueError("Tile size must be positive")
    
    # Validate sampling config
    if config['sampling']['k'] <= 0:
        raise ValueError("K must be positive")
    
    # Validate MIL config
    if config['mil']['num_classes'] < 2:
        raise ValueError("Number of classes must be at least 2")
    
    logger.info("✓ Configuration validated successfully")
    return True


def get_config_value(config: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """
    Get nested configuration value using dot notation.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., 'mil.training.batch_size')
        default: Default value if key not found
    
    Returns:
        Configuration value or default
    
    Example:
        >>> config = {'mil': {'training': {'batch_size': 32}}}
        >>> get_config_value(config, 'mil.training.batch_size')
        32
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
    
    return value
