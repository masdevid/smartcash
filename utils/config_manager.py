"""
File: smartcash/utils/config_manager.py
Refactored configuration manager with improved environment integration
"""

import os
import yaml
import pickle
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from smartcash.utils.environment_manager import EnvironmentManager

class ConfigManager:
    """
    Centralized configuration manager for SmartCash with improved environment integration.
    
    Provides consistent interface for:
    - Loading configurations from various sources
    - Saving configurations with backup
    - Accessing and updating configuration values
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, *args, **kwargs):
        """Singleton pattern implementation."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ConfigManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, base_dir: Optional[str] = None, logger = None):
        """Initialize ConfigManager with environment management."""
        if self._initialized:
            return
        
        self.env_manager = EnvironmentManager(base_dir, logger)
        self.logger = logger
        self.config_dir = self.env_manager.base_dir / 'configs'
        self.config = {}
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Default file paths
        self.default_config_path = self.config_dir / 'base_config.yaml'
        self.experiment_config_path = self.config_dir / 'experiment_config.yaml'
        self.pickle_path = self.env_manager.base_dir / 'config.pkl'
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls, base_dir: Optional[str] = None, logger = None) -> 'ConfigManager':
        """Get singleton instance."""
        return cls(base_dir, logger)
    
    @classmethod
    def load_config(cls, 
                  filename: Optional[str] = None,
                  fallback_to_pickle: bool = True,
                  default_config: Optional[Dict[str, Any]] = None,
                  logger: Optional[Any] = None,
                  use_singleton: bool = True) -> Dict[str, Any]:
        """
        Load configuration from YAML or pickle with clear priority.
        
        Args:
            filename: Configuration filename (optional)
            fallback_to_pickle: Fallback to pickle if YAML not found
            default_config: Default configuration if no file found
            logger: Optional logger for messages
            use_singleton: Use singleton instance for storing config
            
        Returns:
            Configuration dictionary
        """
        # Get or create singleton instance
        cm = cls.get_instance(logger=logger) if use_singleton else cls(logger=logger)
        
        # Return existing config if not empty and no filename specified
        if cm.config and not filename:
            return cm.config
        
        # Define files to try loading
        files_to_try = []
        if filename:
            # Full path or filename in configs directory
            files_to_try.append(filename if os.path.isabs(filename) or '/' in filename 
                                else os.path.join('configs', filename))
        
        # Add default configuration files
        files_to_try.extend([
            'configs/experiment_config.yaml',
            'configs/training_config.yaml',
            'configs/base_config.yaml'
        ])
        
        # Check Drive path if available
        if cm.env_manager.is_colab and cm.env_manager.is_drive_mounted:
            drive_files = [os.path.join('/content/drive/MyDrive/SmartCash', f) for f in files_to_try]
            files_to_try = drive_files + files_to_try
        
        # Try loading from YAML
        for file_path in files_to_try:
            try:
                if os.path.exists(file_path):
                    with open(file_path, 'r') as f:
                        config = yaml.safe_load(f)
                    if logger:
                        logger.info(f"📝 Configuration loaded from {file_path}")
                    
                    # Save to singleton if using
                    if use_singleton:
                        cm.config = config
                    
                    return config
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Failed to load configuration from {file_path}: {str(e)}")
        
        # Fallback to pickle if enabled
        if fallback_to_pickle:
            pickle_files = ['config.pkl']
            if cm.env_manager.is_colab and cm.env_manager.is_drive_mounted:
                pickle_files.insert(0, str(cm.env_manager.drive_path / 'config.pkl'))
            
            for pickle_path in pickle_files:
                if os.path.exists(pickle_path):
                    try:
                        with open(pickle_path, 'rb') as f:
                            config = pickle.load(f)
                        if logger:
                            logger.info(f"📝 Configuration loaded from {pickle_path}")
                        
                        # Save to singleton if using
                        if use_singleton:
                            cm.config = config
                        
                        return config
                    except Exception as e:
                        if logger:
                            logger.warning(f"⚠️ Failed to load configuration from {pickle_path}: {str(e)}")
        
        # Use default configuration if all else fails
        if default_config:
            if logger:
                logger.warning("⚠️ Using default configuration")
            
            # Save to singleton if using
            if use_singleton:
                cm.config = default_config
            
            return default_config
        
        # Return empty dict if no configuration found
        if logger:
            logger.warning("⚠️ No configuration loaded, returning empty dictionary")
        
        # Save empty config to singleton if using
        if use_singleton:
            cm.config = {}
        
        return {}
    
    def save_config(self, 
                  config: Optional[Dict[str, Any]] = None,
                  filename: Optional[str] = None,
                  backup: bool = True,
                  sync_to_drive: bool = True) -> bool:
        """
        Save configuration to file with optional backup and Drive sync.
        
        Args:
            config: Configuration to save (uses self.config if None)
            filename: Filename for saving
            backup: Create backup before saving
            sync_to_drive: Sync to Google Drive if available
            
        Returns:
            Success status
        """
        config = config or self.config
        
        # Determine filename
        filename = filename or self.default_config_path
        if not os.path.isabs(filename):
            filename = self.config_dir / filename
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Create backup if needed
        if backup and os.path.exists(filename):
            backup_path = f"{filename}.bak"
            try:
                Path(backup_path).write_bytes(Path(filename).read_bytes())
                if self.logger:
                    self.logger.info(f"📝 Configuration backup created: {backup_path}")
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"⚠️ Failed to create backup: {str(e)}")
        
        try:
            # Save to YAML
            with open(filename, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            
            # Save to pickle
            pickle_path = str(Path(filename).with_suffix('.pkl'))
            with open(pickle_path, 'wb') as f:
                pickle.dump(config, f)
            
            if self.logger:
                self.logger.info(f"💾 Configuration saved to {filename}")
            
            # Sync to Drive if needed
            if sync_to_drive and self.env_manager.is_colab and self.env_manager.is_drive_mounted:
                drive_config_dir = self.env_manager.drive_path / 'configs'
                drive_config_dir.mkdir(parents=True, exist_ok=True)
                
                # Save to Drive
                drive_yaml_path = drive_config_dir / Path(filename).name
                drive_pickle_path = drive_config_dir / Path(filename).with_suffix('.pkl').name
                
                with open(drive_yaml_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)
                
                with open(drive_pickle_path, 'wb') as f:
                    pickle.dump(config, f)
                
                if self.logger:
                    self.logger.info(f"☁️ Configuration synced to Drive: {drive_yaml_path}")
            
            return True
        
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error saving configuration: {str(e)}")
            return False
    
    def update_config(self, 
                     updates: Dict[str, Any], 
                     save: bool = True, 
                     filename: Optional[str] = None,
                     sync_to_drive: bool = True) -> Dict[str, Any]:
        """
        Update configuration recursively.
        
        Args:
            updates: Dict with new values to update
            save: Flag to save changes to file
            filename: Filename for saving
            sync_to_drive: Sync to Google Drive if available
            
        Returns:
            Updated configuration dictionary
        """
        # Deep update config
        self._deep_update(self.config, updates)
        
        # Save if needed
        if save:
            self.save_config(self.config, filename, sync_to_drive=sync_to_drive)
        
        return self.config
    
    def _deep_update(self, target: Dict, source: Dict) -> None:
        """
        Recursively update nested dictionary.
        
        Args:
            target: Target dictionary to update
            source: Source dictionary with new values
        """
        for key, value in source.items():
            if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                self._deep_update(target[key], value)
            else:
                target[key] = value

def get_config_manager(logger = None) -> ConfigManager:
    """Get singleton ConfigManager instance."""
    return ConfigManager.get_instance(logger=logger)