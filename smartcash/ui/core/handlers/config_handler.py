"""
File: smartcash/ui/core/handlers/config_handler.py
Deskripsi: Config handler hierarchy dengan in-memory, persistent, dan shared configuration support.
Menggunakan composition over inheritance untuk mengurangi kompleksitas.
"""

import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable, Type, TypeVar

from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.common.config.manager import get_config_manager, SimpleConfigManager as ConfigManager
from smartcash.ui.core.shared.shared_config_manager import SharedConfigManager
from smartcash.ui.core.errors import ErrorLevel, handle_errors

class ConfigHandler(BaseHandler):
    """Base config handler dengan in-memory configuration.
    
    Features:
    - 📝 In-memory config storage
    - 🔄 Config change callbacks
    - ✅ Config validation hooks
    - 🎯 Default config support
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 default_config: Optional[Dict[str, Any]] = None):
        """Initialize config handler."""
        super().__init__(module_name, parent_module)
        
        # Config state
        self._config: Dict[str, Any] = default_config or {}
        self._config_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self._default_config = default_config or {}
        
        self.logger.debug(f"📋 ConfigHandler initialized with {len(self._config)} default configs")
    
    # === Config Property ===
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration (copy untuk safety)."""
        return self._config.copy()
    
    @config.setter
    def config(self, new_config: Dict[str, Any]) -> None:
        """Set configuration dengan validation dan callbacks."""
        with self.error_context("Setting config", fail_fast=False):
            # Validate
            if not self.validate_config(new_config):
                raise ValueError("Invalid configuration")
            
            # Update
            old_config = self._config.copy()
            self._config = new_config.copy()
            
            # Notify callbacks
            self._notify_config_change(old_config, self._config)
            
            self.logger.info(f"📋 Config updated: {len(self._config)} items")
    
    # === Config Operations ===
    
    def update_config(self, updates: Dict[str, Any], merge: bool = True) -> None:
        """Update configuration dengan merge option."""
        if merge:
            new_config = self._config.copy()
            new_config.update(updates)
        else:
            new_config = updates
        
        self.config = new_config  # Trigger setter validation
    
    def reset_config(self) -> None:
        """Reset ke default configuration."""
        self.config = self._default_config.copy()
        self.logger.info("🔄 Config reset to defaults")
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get single config value dengan dot notation support."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set single config value dengan dot notation support."""
        keys = key.split('.')
        config = self._config.copy()
        
        # Navigate to parent
        parent = config
        for k in keys[:-1]:
            if k not in parent:
                parent[k] = {}
            parent = parent[k]
        
        # Set value
        parent[keys[-1]] = value
        
        # Update config (trigger validation)
        self.config = config
    
    # === Validation ===
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration. Override untuk custom validation."""
        if not isinstance(config, dict):
            self.logger.error(f"❌ Config must be dict, got: {type(config)}")
            return False
        
        # Subclass bisa override untuk validation spesifik
        return True
    
    # === Callbacks ===
    
    def add_config_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Add callback untuk config changes."""
        if callback not in self._config_callbacks:
            self._config_callbacks.append(callback)
            self.logger.debug(f"📌 Added config callback: {callback.__name__}")
    
    def remove_config_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """Remove config callback."""
        if callback in self._config_callbacks:
            self._config_callbacks.remove(callback)
            self.logger.debug(f"📌 Removed config callback: {callback.__name__}")
    
    def _notify_config_change(self, old_config: Dict[str, Any], new_config: Dict[str, Any]) -> None:
        """Notify semua callbacks tentang config change."""
        for callback in self._config_callbacks:
            try:
                callback(new_config)
            except Exception as e:
                self.logger.error(f"❌ Config callback error: {e}")


class ConfigurableHandler(ConfigHandler):
    """Handler dengan UI config extraction dan update support.
    
    Menambahkan:
    - 🎨 UI component config extraction
    - 🔄 UI update dari config
    - 📦 Config bundling untuk save/load
    """
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration dari UI components.
        
        Override method ini untuk extract config dari UI components.
        Default implementation return empty dict.
        """
        return {}
    
    def update_ui_from_config(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Update UI components dari configuration.
        
        Override method ini untuk update UI dari config.
        Default implementation tidak melakukan apa-apa.
        """
        pass
    
    def sync_config_with_ui(self) -> None:
        """Sync config dengan UI (extract kemudian update internal config)."""
        with self.error_context("Syncing config with UI"):
            ui_config = self.extract_config_from_ui()
            if ui_config:
                self.update_config(ui_config)
                self.logger.info(f"🔄 Synced {len(ui_config)} config items from UI")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI dengan config (update UI dari internal config)."""
        with self.error_context("Syncing UI with config"):
            self.update_ui_from_config(self.config)
            self.logger.info("🔄 UI synced with config")


class PersistentConfigHandler(ConfigurableHandler):
    """Handler dengan persistent storage support.
    
    Menambahkan:
    - 💾 Save/load config ke/dari file
    - 🔐 Config versioning
    - 📝 Config metadata
    """
    
    CONFIG_VERSION = "1.0.0"
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 default_config: Optional[Dict[str, Any]] = None,
                 config_manager: Optional[ConfigManager] = None):
        """Initialize dengan persistent storage support."""
        super().__init__(module_name, parent_module, default_config)
        
        # Config manager untuk file operations
        self._config_manager = config_manager or get_config_manager()
        self._config_dir = config_dir
        
        # Auto-load config saat init
        self.load_config()
    
    def load_config(self, name: Optional[str] = None) -> bool:
        """Load configuration dari file."""
        @handle_errors(error_msg="Failed to load configuration", level=ErrorLevel.ERROR, reraise=True)
        def _load_config():
            config_name = name or f"{self.module_name}_config"
            
            # Load via config manager
            loaded_config = self._config_manager.get_config(config_name)
            
            if loaded_config:
                # Validate version
                if 'metadata' in loaded_config:
                    version = loaded_config['metadata'].get('version', '0.0.0')
                    if version != self.CONFIG_VERSION:
                        self.logger.warning(f"⚠️ Config version mismatch: {version} != {self.CONFIG_VERSION}")
                
                # Extract actual config
                config_data = loaded_config.get('config', loaded_config)
                self.config = config_data
                
                self.logger.info(f"📂 Loaded config: {config_name}")
                return True
            else:
                self.logger.info(f"📂 No existing config found: {config_name}")
                return False
        
        try:
            return _load_config()
        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            return False
    
    @handle_errors(error_msg="Failed to save configuration", level=ErrorLevel.ERROR, reraise=True)
    def save_config(self, name: Optional[str] = None) -> bool:
        """Save configuration ke file."""
        config_name = name or f"{self.module_name}_config"
        
        # Bundle config dengan metadata
        config_bundle = {
            'metadata': {
                'version': self.CONFIG_VERSION,
                'module': self.full_module_name,
                'saved_at': datetime.now().isoformat(),
            },
            'config': self.config
        }
        
        # Save via config manager
        self._config_manager.save_config(config_bundle, config_name)
        
        self.logger.info(f"💾 Saved config: {config_name}")
        self.update_status("Configuration saved successfully", 'success')
        return True
    
    @handle_errors(error_msg="Failed to delete configuration", level=ErrorLevel.ERROR, reraise=True)
    def delete_config(self, name: Optional[str] = None) -> bool:
        """Delete configuration file."""
        config_path = self.get_config_path(name)
        
        if config_path.exists():
            config_path.unlink()
            self.logger.info(f"🗑️ Deleted config: {config_path.name}")
            return True
        else:
            self.logger.warning(f"⚠️ Config not found: {config_path.name}")
            return False
    
    def get_config_path(self, name: Optional[str] = None) -> Path:
        """Get path untuk config file."""
        config_name = name or f"{self.module_name}_config"
        
        if self._config_dir:
            return self._config_dir / f"{config_name}.json"
        
        # Use config manager default
        return Path(self._config_manager.get_config_path(config_name))


class SharedConfigHandler(PersistentConfigHandler):
    """Handler dengan shared configuration support antar modules.
    
    Menambahkan:
    - 🔗 Config sharing antar modules
    - 📡 Auto-sync dengan shared config manager
    - 🔄 Config change broadcasting
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 default_config: Optional[Dict[str, Any]] = None,
                 config_dir: Optional[Path] = None,
                 enable_sharing: bool = True):
        """Initialize dengan shared config support."""
        # Initialize parent first
        super().__init__(module_name, parent_module, default_config, config_dir)
        
        # Shared config setup
        self._shared_manager: Optional[SharedConfigManager] = None
        self._unsubscribe_fn: Optional[Callable] = None
        self._enable_sharing = enable_sharing
        
        # Setup sharing jika enabled dan ada parent module
        if enable_sharing and parent_module:
            self._setup_shared_config()
    
    def _setup_shared_config(self) -> None:
        """Setup shared configuration manager."""
        try:
            # Get shared manager instance
            self._shared_manager = get_shared_config_manager(self.parent_module)
            
            # Subscribe untuk updates
            self._unsubscribe_fn = self._shared_manager.subscribe(
                self.module_name, 
                self._on_shared_config_update
            )
            
            # Load initial shared config jika ada
            if shared_config := self._shared_manager.get_config(self.module_name):
                self._config.update(shared_config)
                self.logger.info(f"🔗 Loaded shared config: {len(shared_config)} items")
            
            self.logger.info(f"🔗 Shared config enabled for: {self.parent_module}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to setup shared config: {e}")
            self._shared_manager = None
    
    def _on_shared_config_update(self, config: Dict[str, Any]) -> None:
        """Handle shared config updates dari modules lain."""
        self.logger.debug(f"📡 Received shared config update: {len(config)} items")
        
        # Update internal config tanpa trigger callbacks (untuk avoid loop)
        self._config.update(config)
        
        # Update UI jika ada
        self.sync_ui_with_config()
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get config dengan merge dari shared config."""
        base_config = super().config
        
        # Merge dengan shared config jika ada
        if self._shared_manager and self._enable_sharing:
            if shared := self._shared_manager.get_config(self.module_name):
                base_config.update(shared)
        
        return base_config
    
    @config.setter
    def config(self, new_config: Dict[str, Any]) -> None:
        """Set config dan broadcast ke shared manager."""
        # Set via parent
        super(SharedConfigHandler, self.__class__).config.fset(self, new_config)
        
        # Broadcast jika sharing enabled
        if self._shared_manager and self._enable_sharing:
            try:
                self._shared_manager.update_config(self.module_name, new_config)
                self.logger.debug(f"📡 Broadcasted config update: {len(new_config)} items")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to broadcast config: {e}")
    
    def enable_sharing(self, enabled: bool = True) -> None:
        """Enable/disable config sharing."""
        self._enable_sharing = enabled
        
        if enabled and not self._shared_manager and self.parent_module:
            self._setup_shared_config()
        
        self.logger.info(f"🔗 Config sharing: {'enabled' if enabled else 'disabled'}")
    
    def get_shared_modules(self) -> List[str]:
        """Get list modules yang sharing config."""
        if self._shared_manager:
            return list(self._shared_manager.get_all_configs().keys())
        return []
    
    def initialize(self, **kwargs) -> Dict[str, Any]:
        """Initialize the shared config handler.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            Dictionary containing initialization result with 'status' key
        """
        try:
            # If sharing is enabled and not already set up, set it up
            if self._enable_sharing and self.parent_module and not self._shared_manager:
                self._setup_shared_config()
                
                # If we have a shared manager, load the initial config
                if self._shared_manager:
                    if shared_config := self._shared_manager.get_config(self.module_name):
                        self._config.update(shared_config)
                        self.logger.info(f"🔗 Loaded shared config: {len(shared_config)} items")
            
            return {
                'status': True,
                'initialized': True,
                'sharing_enabled': self._enable_sharing and bool(self._shared_manager),
                'message': 'Shared config handler initialized successfully'
            }
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize shared config handler: {e}", exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': f'Failed to initialize shared config: {e}'
            }
    
    def __del__(self):
        """Cleanup shared config subscription."""
        if self._unsubscribe_fn:
            try:
                self._unsubscribe_fn()
            except:
                pass  # Ignore errors during cleanup