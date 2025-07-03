"""
File: smartcash/ui/setup/env_config/handlers/config_handler.py

Configuration Handler - Refactored dengan arsitektur baru.

Handler khusus untuk mengelola konfigurasi environment dengan
in-memory storage saja (tidak ada persistence ke drive).
"""

from typing import Dict, Any, Optional, List
from smartcash.ui.core.shared.logger import get_enhanced_logger
from pathlib import Path

# Import core handlers dari arsitektur baru
from smartcash.ui.core.handlers.config_handler import ConfigurableHandler
from smartcash.ui.core.shared.logger import get_module_logger

# Import configs
from smartcash.ui.setup.env_config.configs.defaults import DEFAULT_CONFIG
from smartcash.ui.setup.env_config.configs.validator import validate_config


class ConfigHandler(ConfigurableHandler):
    """Configuration handler untuk environment setup.
    
    Handler ini mengelola konfigurasi environment dengan fitur:
    - Persistent storage menggunakan file system
    - Config validation
    - UI synchronization
    - Template management
    """
    
    def __init__(self):
        """Initialize configuration handler."""
        super().__init__(
            module_name='env_config',
            parent_module='setup'
        )
        
        # Initialize with default config
        self.config = DEFAULT_CONFIG.copy()
        
        self.logger = get_enhanced_logger(__name__)
        self.logger.info("🔧 Environment config handler initialized")
    
    def initialize(self) -> Dict[str, Any]:
        """Initialize the configuration handler.
        
        Returns:
            Dictionary containing initialization status and any relevant data
        """
        try:
            # Mark as initialized
            self._is_initialized = True
            
            # Load any existing configuration
            if hasattr(self, 'load_config'):
                loaded = self.load_config()
                if loaded:
                    self.logger.info("✅ Successfully loaded existing configuration")
                else:
                    self.logger.info("ℹ️ No existing configuration found, using defaults")
            
            return {
                'status': True,
                'initialized': True,
                'config': self.config,
                'message': 'Configuration handler initialized successfully'
            }
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize config handler: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': error_msg,
                'message': 'Failed to initialize configuration handler'
            }
    
    def validate_config(self, config: Optional[Dict[str, Any]] = None) -> bool:
        """Validate configuration.
        
        Args:
            config: Optional config to validate (uses self.config if None)
            
        Returns:
            True jika valid, False jika tidak
        """
        try:
            config = config or self.config
            return validate_config(config)
            
        except Exception as e:
            self.logger.error(f"❌ Config validation error: {str(e)}")
            return False
    
    def extract_config_from_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration dari UI components.
        
        Args:
            ui_components: Dictionary berisi UI components
            
        Returns:
            Dictionary berisi extracted config
        """
        try:
            extracted_config = {}
            
            # Extract dari berbagai UI components
            if 'config_form' in ui_components:
                form_data = ui_components['config_form'].get_values()
                extracted_config.update(form_data)
            
            if 'path_selector' in ui_components:
                selected_paths = ui_components['path_selector'].get_selected_paths()
                extracted_config['paths'] = selected_paths
            
            if 'options_panel' in ui_components:
                options = ui_components['options_panel'].get_options()
                extracted_config['options'] = options
            
            self.logger.debug(f"📤 Extracted config: {extracted_config}")
            return extracted_config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {str(e)}")
            return {}
    
    def update_ui_from_config(self, ui_components: Dict[str, Any]):
        """Update UI components dari current config.
        
        Args:
            ui_components: Dictionary berisi UI components
        """
        try:
            # Update config form
            if 'config_form' in ui_components:
                ui_components['config_form'].set_values(self.config)
            
            # Update path selector
            if 'path_selector' in ui_components and 'paths' in self.config:
                ui_components['path_selector'].set_selected_paths(self.config['paths'])
            
            # Update options panel
            if 'options_panel' in ui_components and 'options' in self.config:
                ui_components['options_panel'].set_options(self.config['options'])
            
            self.logger.debug("📥 UI updated from config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI from config: {str(e)}")
    
    def sync_config_with_ui(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Sync config dengan UI dalam kedua arah.
        
        Args:
            ui_components: Dictionary berisi UI components
            
        Returns:
            Dictionary berisi hasil sync
        """
        try:
            # Extract config dari UI
            ui_config = self.extract_config_from_ui(ui_components)
            
            # Merge dengan current config
            merged_config = self.config.copy()
            merged_config.update(ui_config)
            
            # Validate merged config
            if self.validate_config(merged_config):
                # Update config
                self.config = merged_config
                
                # Update UI untuk reflect any changes
                self.update_ui_from_config(ui_components)
                
                return {
                    'status': True,
                    'message': '✅ Config berhasil di-sync dengan UI'
                }
            else:
                return {
                    'status': False,
                    'message': '❌ Config tidak valid setelah sync'
                }
                
        except Exception as e:
            error_msg = f"❌ Error syncing config with UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def get_config_template(self, template_name: str) -> Dict[str, Any]:
        """Get config template by name.
        
        Args:
            template_name: Nama template
            
        Returns:
            Dictionary berisi template config
        """
        try:
            templates = {
                'default': DEFAULT_CONFIG.copy(),
                'development': {
                    **DEFAULT_CONFIG,
                    'debug': True,
                    'verbose': True
                },
                'production': {
                    **DEFAULT_CONFIG,
                    'debug': False,
                    'verbose': False
                }
            }
            
            return templates.get(template_name, DEFAULT_CONFIG.copy())
            
        except Exception as e:
            self.logger.error(f"❌ Error getting config template: {str(e)}")
            return DEFAULT_CONFIG.copy()
    
    def apply_config_template(self, template_name: str) -> Dict[str, Any]:
        """Apply config template.
        
        Args:
            template_name: Nama template
            
        Returns:
            Dictionary berisi hasil apply
        """
        try:
            template = self.get_config_template(template_name)
            
            if self.validate_config(template):
                self.config = template
                return {
                    'status': True,
                    'message': f'✅ Template {template_name} berhasil diterapkan'
                }
            else:
                return {
                    'status': False,
                    'message': f'❌ Template {template_name} tidak valid'
                }
                
        except Exception as e:
            error_msg = f"❌ Error applying config template: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def backup_config(self, backup_name: Optional[str] = None) -> Dict[str, Any]:
        """Backup current config.
        
        Args:
            backup_name: Optional nama backup
            
        Returns:
            Dictionary berisi hasil backup
        """
        try:
            from datetime import datetime
            
            if not backup_name:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"env_config_backup_{timestamp}"
            
            success = self.save_config(backup_name)
            
            if success:
                return {
                    'status': True,
                    'message': f'✅ Config berhasil di-backup ke {backup_name}'
                }
            else:
                return {
                    'status': False,
                    'message': f'❌ Gagal backup config ke {backup_name}'
                }
                
        except Exception as e:
            error_msg = f"❌ Error backing up config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def restore_config(self, backup_name: str) -> Dict[str, Any]:
        """Restore config dari backup.
        
        Args:
            backup_name: Nama backup untuk restore
            
        Returns:
            Dictionary berisi hasil restore
        """
        try:
            # Load config dari backup name
            backup_config = self.get_config_template(backup_name)
            
            if self.validate_config(backup_config):
                self.config = backup_config
                return {
                    'status': True,
                    'message': f'✅ Config berhasil di-restore dari {backup_name}'
                }
            else:
                return {
                    'status': False,
                    'message': f'❌ Backup config {backup_name} tidak valid'
                }
                
        except Exception as e:
            error_msg = f"❌ Error restoring config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }
    
    def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset config ke default values.
        
        Returns:
            Dictionary berisi hasil reset
        """
        try:
            self.config = DEFAULT_CONFIG.copy()
            return {
                'status': True,
                'message': '✅ Config berhasil di-reset ke defaults'
            }
            
        except Exception as e:
            error_msg = f"❌ Error resetting config: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {
                'status': False,
                'error': str(e),
                'message': error_msg
            }