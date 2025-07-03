"""
File: smartcash/ui/core/initializers/config_initializer.py
Deskripsi: Initializer dengan configuration management terintegrasi.
Extends BaseInitializer dengan config loading dan validation.
"""

from typing import Dict, Any, Optional, Type
from pathlib import Path

from smartcash.ui.core.initializers.base_initializer import BaseInitializer
from smartcash.ui.core.handlers.config_handler import SharedConfigHandler
from smartcash.ui.handlers.config_handlers import ConfigHandler as LegacyConfigHandler


class ConfigurableInitializer(BaseInitializer):
    """Initializer dengan configuration management.
    
    Features:
    - 📋 Auto config loading saat init
    - ✅ Config validation
    - 🔄 Config-UI synchronization
    - 💾 Config persistence support
    """
    
    def __init__(self, 
                 module_name: str, 
                 parent_module: Optional[str] = None,
                 config_handler_class: Optional[Type] = None,
                 enable_shared_config: bool = True):
        """Initialize dengan config support.
        
        Args:
            module_name: Nama module
            parent_module: Parent module untuk shared config
            config_handler_class: Custom config handler class
            enable_shared_config: Enable shared config antar modules
        """
        super().__init__(module_name, parent_module)
        
        # Setup config handler
        if config_handler_class:
            # Use custom handler (untuk backward compatibility)
            self.config_handler = config_handler_class(module_name)
        else:
            # Use new SharedConfigHandler
            self.config_handler = SharedConfigHandler(
                module_name, 
                parent_module,
                enable_sharing=enable_shared_config
            )
        
        self.logger.debug(f"📋 ConfigurableInitializer setup with {type(self.config_handler).__name__}")
    
    # === Config Properties ===
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get current configuration."""
        return self.config_handler.config
    
    @config.setter
    def config(self, new_config: Dict[str, Any]) -> None:
        """Set configuration."""
        self.config_handler.config = new_config
    
    # === Extended Initialization ===
    
    def pre_initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """Load configuration before initialization."""
        super().pre_initialize(config, **kwargs)
        
        # Load config
        if config:
            # Use provided config
            self.logger.info(f"📋 Using provided config: {len(config)} items")
            self.config = config
        else:
            # Load from file
            self.logger.info("📋 Loading configuration from file...")
            self.load_config()
    
    def post_initialize(self, **kwargs) -> None:
        """Sync UI dengan config after initialization."""
        super().post_initialize(**kwargs)
        
        # Sync UI jika handler support
        if hasattr(self.config_handler, 'update_ui_from_config'):
            try:
                self.config_handler.set_ui_components(self._ui_components)
                self.config_handler.update_ui_from_config(self.config)
                self.logger.info("🔄 UI synced with configuration")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to sync UI with config: {e}")
    
    # === Config Operations ===
    
    def load_config(self, name: Optional[str] = None) -> bool:
        """Load configuration dari file.
        
        Args:
            name: Optional config name
            
        Returns:
            True jika berhasil load
        """
        try:
            # Check jika handler support load
            if hasattr(self.config_handler, 'load_config'):
                return self.config_handler.load_config(name)
            else:
                self.logger.warning("⚠️ Config handler doesn't support loading")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load config: {e}")
            return False
    
    def save_config(self, name: Optional[str] = None) -> bool:
        """Save configuration ke file.
        
        Args:
            name: Optional config name
            
        Returns:
            True jika berhasil save
        """
        try:
            # Extract dari UI dulu jika ada
            if hasattr(self.config_handler, 'extract_config_from_ui'):
                self.config_handler.sync_config_with_ui()
            
            # Save
            if hasattr(self.config_handler, 'save_config'):
                return self.config_handler.save_config(name)
            else:
                self.logger.warning("⚠️ Config handler doesn't support saving")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save config: {e}")
            return False
    
    def reset_config(self) -> None:
        """Reset configuration ke defaults."""
        try:
            self.config_handler.reset_config()
            
            # Update UI jika ada
            if hasattr(self.config_handler, 'update_ui_from_config'):
                self.config_handler.update_ui_from_config(self.config)
                
            self.logger.info("🔄 Configuration reset to defaults")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to reset config: {e}")
    
    # === Config Validation ===
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration.
        
        Args:
            config: Config untuk validate (default: current config)
            
        Returns:
            True jika valid
        """
        config = config or self.config
        
        try:
            return self.config_handler.validate_config(config)
        except Exception as e:
            self.logger.error(f"❌ Config validation error: {e}")
            return False
    
    def get_config_errors(self, config: Optional[Dict[str, Any]] = None) -> List[str]:
        """Get list of config validation errors.
        
        Override di subclass untuk custom validation.
        """
        return []
    
    # === Utility Methods ===
    
    def get_config_value(self, key: str, default: Any = None) -> Any:
        """Get config value dengan dot notation."""
        return self.config_handler.get_config_value(key, default)
    
    def set_config_value(self, key: str, value: Any) -> None:
        """Set config value dengan dot notation."""
        self.config_handler.set_config_value(key, value)
    
    def export_config(self) -> Dict[str, Any]:
        """Export current configuration untuk sharing."""
        return {
            'module': self.full_module_name,
            'version': getattr(self.config_handler, 'CONFIG_VERSION', '1.0.0'),
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }
    
    def import_config(self, config_data: Dict[str, Any]) -> bool:
        """Import configuration dari export."""
        try:
            if 'config' not in config_data:
                raise ValueError("Invalid config data: missing 'config' key")
            
            # Validate version jika ada
            if 'version' in config_data:
                current_version = getattr(self.config_handler, 'CONFIG_VERSION', '1.0.0')
                if config_data['version'] != current_version:
                    self.logger.warning(
                        f"⚠️ Version mismatch: {config_data['version']} != {current_version}"
                    )
            
            # Import config
            self.config = config_data['config']
            
            # Update UI
            if hasattr(self.config_handler, 'update_ui_from_config'):
                self.config_handler.update_ui_from_config(self.config)
            
            self.logger.info("📥 Configuration imported successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to import config: {e}")
            return False