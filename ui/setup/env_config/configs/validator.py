"""
File: smartcash/ui/setup/env_config/configs/validator.py

Configuration validator untuk environment setup.
"""

from typing import Dict, Any, List, Optional
import os
from pathlib import Path

def validate_config(config: Dict[str, Any]) -> bool:
    """Validate environment configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        True jika valid, False jika tidak
    """
    try:
        # Required top-level keys
        required_keys = ['version', 'environment', 'setup', 'paths']
        for key in required_keys:
            if key not in config:
                return False
        
        # Validate environment config
        if not _validate_environment_config(config.get('environment', {})):
            return False
        
        # Validate setup config
        if not _validate_setup_config(config.get('setup', {})):
            return False
        
        # Validate paths config
        if not _validate_paths_config(config.get('paths', {})):
            return False
        
        return True
        
    except Exception:
        return False

def _validate_environment_config(env_config: Dict[str, Any]) -> bool:
    """Validate environment configuration."""
    required_keys = ['type', 'auto_mount_drive', 'mount_path', 'base_path']
    
    for key in required_keys:
        if key not in env_config:
            return False
    
    # Validate environment type
    if env_config['type'] not in ['colab', 'local', 'jupyter']:
        return False
    
    # Validate boolean values
    if not isinstance(env_config['auto_mount_drive'], bool):
        return False
    
    # Validate paths exist or can be created
    paths_to_check = ['mount_path', 'base_path']
    for path_key in paths_to_check:
        path_value = env_config.get(path_key)
        if not path_value or not isinstance(path_value, str):
            return False
    
    return True

def _validate_setup_config(setup_config: Dict[str, Any]) -> bool:
    """Validate setup configuration."""
    required_keys = ['auto_start', 'stop_on_error', 'verify_setup', 'max_retries', 'stages']
    
    for key in required_keys:
        if key not in setup_config:
            return False
    
    # Validate boolean values
    bool_keys = ['auto_start', 'stop_on_error', 'verify_setup']
    for key in bool_keys:
        if not isinstance(setup_config[key], bool):
            return False
    
    # Validate integer values
    if not isinstance(setup_config['max_retries'], int) or setup_config['max_retries'] < 0:
        return False
    
    # Validate stages
    stages = setup_config.get('stages', [])
    if not isinstance(stages, list) or len(stages) == 0:
        return False
    
    valid_stages = ['drive_mount', 'config_sync', 'folder_setup', 'verify']
    for stage in stages:
        if stage not in valid_stages:
            return False
    
    return True

def _validate_paths_config(paths_config: Dict[str, Any]) -> bool:
    """Validate paths configuration."""
    required_keys = ['drive_base', 'colab_base', 'config_dir', 'data_dir']
    
    for key in required_keys:
        if key not in paths_config:
            return False
        
        path_value = paths_config[key]
        if not isinstance(path_value, str) or not path_value.strip():
            return False
    
    return True

def get_validation_errors(config: Dict[str, Any]) -> List[str]:
    """Get detailed validation errors.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of error messages
    """
    errors = []
    
    try:
        # Check required top-level keys
        required_keys = ['version', 'environment', 'setup', 'paths']
        for key in required_keys:
            if key not in config:
                errors.append(f"Missing required key: {key}")
        
        # Validate environment config
        env_config = config.get('environment', {})
        env_errors = _get_environment_errors(env_config)
        errors.extend(env_errors)
        
        # Validate setup config
        setup_config = config.get('setup', {})
        setup_errors = _get_setup_errors(setup_config)
        errors.extend(setup_errors)
        
        # Validate paths config
        paths_config = config.get('paths', {})
        paths_errors = _get_paths_errors(paths_config)
        errors.extend(paths_errors)
        
    except Exception as e:
        errors.append(f"Configuration validation error: {str(e)}")
    
    return errors

def _get_environment_errors(env_config: Dict[str, Any]) -> List[str]:
    """Get environment validation errors."""
    errors = []
    
    required_keys = ['type', 'auto_mount_drive', 'mount_path', 'base_path']
    for key in required_keys:
        if key not in env_config:
            errors.append(f"Missing environment key: {key}")
    
    # Validate environment type
    env_type = env_config.get('type')
    if env_type and env_type not in ['colab', 'local', 'jupyter']:
        errors.append(f"Invalid environment type: {env_type}")
    
    # Validate boolean values
    auto_mount = env_config.get('auto_mount_drive')
    if auto_mount is not None and not isinstance(auto_mount, bool):
        errors.append("auto_mount_drive must be boolean")
    
    return errors

def _get_setup_errors(setup_config: Dict[str, Any]) -> List[str]:
    """Get setup validation errors."""
    errors = []
    
    required_keys = ['auto_start', 'stop_on_error', 'verify_setup', 'max_retries', 'stages']
    for key in required_keys:
        if key not in setup_config:
            errors.append(f"Missing setup key: {key}")
    
    # Validate max_retries
    max_retries = setup_config.get('max_retries')
    if max_retries is not None and (not isinstance(max_retries, int) or max_retries < 0):
        errors.append("max_retries must be non-negative integer")
    
    # Validate stages
    stages = setup_config.get('stages', [])
    if not isinstance(stages, list):
        errors.append("stages must be a list")
    elif len(stages) == 0:
        errors.append("stages list cannot be empty")
    else:
        valid_stages = ['drive_mount', 'config_sync', 'folder_setup', 'verify']
        for stage in stages:
            if stage not in valid_stages:
                errors.append(f"Invalid stage: {stage}")
    
    return errors

def _get_paths_errors(paths_config: Dict[str, Any]) -> List[str]:
    """Get paths validation errors."""
    errors = []
    
    required_keys = ['drive_base', 'colab_base', 'config_dir', 'data_dir']
    for key in required_keys:
        if key not in paths_config:
            errors.append(f"Missing path key: {key}")
        else:
            path_value = paths_config[key]
            if not isinstance(path_value, str) or not path_value.strip():
                errors.append(f"Path {key} must be non-empty string")
    
    return errors