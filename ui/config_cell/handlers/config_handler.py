"""
File: smartcash/ui/config_cell/handlers/config_handler.py
Deskripsi: Config cell handler for managing configuration state and persistence
"""

import os
import yaml
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

class ConfigCellHandler:
    """Handler for configuration management with YAML persistence"""
    
    def __init__(self, module_name: str, parent_module: str = None):
        """Initialize config handler
        
        Args:
            module_name: Name of the module
            parent_module: Optional parent module name
        """
        self.module_name = module_name
        self.parent_module = parent_module
        self.config = {}
        self.listeners: List[Callable] = []
        self._setup_config_file()
    
    def _setup_config_file(self) -> None:
        """Setup config file path and load if exists"""
        config_dir = os.path.join(os.getcwd(), 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        filename = f"{self.parent_module}_{self.module_name}.yaml" if self.parent_module else f"{self.module_name}.yaml"
        self.file_path = os.path.join(config_dir, filename)
        
        if os.path.exists(self.file_path):
            self.load()
    
    def load(self) -> Dict[str, Any]:
        """Load config from file"""
        try:
            with open(self.file_path, 'r') as f:
                self.config = yaml.safe_load(f) or {}
                self._notify_listeners('load', self.config)
                return self.config
        except Exception as e:
            print(f"Error loading config: {str(e)}")
            return {}
    
    def save(self) -> bool:
        """Save config to file"""
        try:
            with open(self.file_path, 'w') as f:
                yaml.dump(self.config, f)
                self._notify_listeners('save', self.config)
                return True
        except Exception as e:
            print(f"Error saving config: {str(e)}")
            return False
    
    def update(self, new_config: Dict[str, Any]) -> None:
        """Update config and notify listeners"""
        self.config.update(new_config)
        self._notify_listeners('update', self.config)
    
    def add_listener(self, callback: Callable) -> None:
        """Add a listener for config changes"""
        if callback not in self.listeners:
            self.listeners.append(callback)
    
    def remove_listener(self, callback: Callable) -> None:
        """Remove a listener"""
        if callback in self.listeners:
            self.listeners.remove(callback)
    
    def _notify_listeners(self, event: str, data: Any) -> None:
        """Notify all listeners about config changes"""
        for listener in self.listeners:
            try:
                if hasattr(listener, f'on_{event}'):
                    getattr(listener, f'on_{event}')(data)
            except Exception as e:
                print(f"Error notifying listener: {str(e)}")
