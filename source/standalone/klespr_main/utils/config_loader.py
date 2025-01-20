# utils/config_loader.py
import os
from typing import Dict, Any, Optional
from pathlib import Path
import yaml
from skrl.resources.preprocessors.torch.running_standard_scaler import RunningStandardScaler

# Dictionary mapping string names to actual class references
CLASS_MAPPINGS = {"RunningStandardScaler": RunningStandardScaler}


class ConfigLoadError(Exception):
    """Exception raised for errors in loading configuration."""

    pass


def get_config_dir() -> Path:
    """
    Get the configuration directory path.

    Returns:
        Path: Path to the configs directory
    """
    project_root = Path(__file__).parent.parent
    return project_root / "configs"


def process_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process configuration values, resolving any class references.

    Args:
        config: Raw configuration dictionary

    Returns:
        Processed configuration dictionary
    """
    processed_config = {}

    for key, value in config.items():
        if isinstance(value, dict):
            processed_config[key] = process_config_values(value)
        elif isinstance(value, str) and value in CLASS_MAPPINGS:
            processed_config[key] = CLASS_MAPPINGS[value]
        else:
            processed_config[key] = value

    return processed_config


def load_config(task_name: str, config_type: str = "es") -> Dict[str, Any]:
    """
    Load configuration for a specific task and type from YAML files.

    Args:
        task_name: Name of the task (e.g., 'cartpole', 'ant')
        config_type: Type of configuration ('es' or 'hybrid')

    Returns:
        Dict[str, Any]: Configuration dictionary for the specified task and type

    Raises:
        ConfigLoadError: If there's an error loading or parsing the configuration
    """
    task_name = task_name.lower()
    config_type = config_type.lower()

    config_dir = get_config_dir()
    config_path = config_dir / f"{task_name}.yaml"

    try:
        if not config_path.exists():
            raise ConfigLoadError(f"Configuration file not found: {config_path}")

        # Load and parse YAML file
        with open(config_path, "r") as f:
            try:
                config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ConfigLoadError(f"Error parsing YAML file {config_path}: {str(e)}")

        # Validate config structure
        if not isinstance(config, dict):
            raise ConfigLoadError(f"Invalid config format in {config_path}")

        # Get task-specific config
        if task_name not in config:
            raise ConfigLoadError(f"Task '{task_name}' not found in config file")

        task_config = config[task_name]

        # Get config type
        if config_type not in task_config:
            raise ConfigLoadError(f"Config type '{config_type}' not found for task '{task_name}'")

        # Process the configuration values
        raw_config = task_config[config_type]
        processed_config = process_config_values(raw_config)

        return processed_config

    except Exception as e:
        if not isinstance(e, ConfigLoadError):
            raise ConfigLoadError(f"Unexpected error loading config: {str(e)}")
        raise


def validate_config(config: Dict[str, Any], required_fields: Optional[list] = None) -> None:
    """
    Validate that a configuration has all required fields.

    Args:
        config: Configuration dictionary to validate
        required_fields: List of required field names

    Raises:
        ConfigLoadError: If validation fails
    """
    if required_fields is None:
        return

    missing_fields = [field for field in required_fields if field not in config]
    if missing_fields:
        raise ConfigLoadError(f"Missing required fields in config: {', '.join(missing_fields)}")
