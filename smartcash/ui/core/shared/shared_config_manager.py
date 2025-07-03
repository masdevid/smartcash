# smartcash/ui/core/shared/shared_config_manager.py
"""
Shared configuration manager for managing persistent configuration storage.
"""
from typing import Dict, Any, Optional, Union
import logging
import os
import json
from pathlib import Path

from smartcash.ui.core.shared.logger import get_ui_logger


class SharedConfigManager:
    """
    Shared configuration manager for persistent configuration storage.
    
    This class provides functionality for loading, saving, and managing
    configuration files for SmartCash UI modules.
    """
    
    def __init__(
        self,
        module_name: str,
        parent_module: str,
        config_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the shared configuration manager.
        
        Args:
            module_name: Name of the module
            parent_module: Parent module name
            config_dir: Optional directory for configuration files
            logger: Optional logger instance
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.logger = logger or get_ui_logger(
            module_name="config_manager",
            parent_module="ui.core.shared"
        )
        
        # Set up config directory
        if config_dir:
            self.config_dir = Path(config_dir)
        else:
            # Default to ~/.smartcash/config
            self.config_dir = Path.home() / ".smartcash" / "config"
        
        # Ensure config directory exists
        os.makedirs(self.config_dir, exist_ok=True)
        
        # Config file path
        self.config_file = self.config_dir / f"{parent_module}_{module_name}.json"
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file.
        
        Returns:
            Loaded configuration dictionary
        """
        if not self.config_file.exists():
            self.logger.debug(f"Config file {self.config_file} does not exist")
            return {}
        
        try:
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            self.logger.debug(f"Loaded config from {self.config_file}")
            return config
        except Exception as e:
            self.logger.error(f"Error loading config from {self.config_file}: {str(e)}")
            return {}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary to save
            
        Returns:
            True if successful, False otherwise
        """
        try:
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.logger.debug(f"Saved config to {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error saving config to {self.config_file}: {str(e)}")
            return False
    
    def delete_config(self) -> bool:
        """
        Delete configuration file.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.config_file.exists():
            self.logger.debug(f"Config file {self.config_file} does not exist")
            return True
        
        try:
            os.remove(self.config_file)
            self.logger.debug(f"Deleted config file {self.config_file}")
            return True
        except Exception as e:
            self.logger.error(f"Error deleting config file {self.config_file}: {str(e)}")
            return False
    
    def get_config_path(self) -> str:
        """
        Get the path to the configuration file.
        
        Returns:
            Path to the configuration file
        """
        return str(self.config_file)
