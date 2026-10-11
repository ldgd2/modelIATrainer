import os
import yaml
from typing import Any, Dict

def load_yaml_config(config_path: str) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return config

def merge_cli_args(config: Dict[str, Any], cli_args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Overwrites YAML configuration with values explicitly provided via CLI.
    Ignores None values from CLI.
    """
    updated = config.copy()
    for key, value in cli_args.items():
        if value is not None:
            # Handle nested keys if needed, or flat top-level parameters
            updated[key] = value
    return updated
