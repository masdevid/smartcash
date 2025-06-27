# =============================================================================
# File: smartcash/ui/setup/dependency/handlers/config_event_handlers.py - MISSING FILE
# Deskripsi: Config event handlers yang dipanggil di event_handlers.py
# =============================================================================

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from .base_handler import BaseDependencyHandler

class ConfigEventHandler(BaseDependencyHandler):
    """Handler untuk config operations (save/reset/load)"""
    
    def __init__(self, ui_components: Dict[str, Any], config_handler: ConfigHandler):
        super().__init__(ui_components)
        self.config_handler = config_handler
    
    def save_config(self, *args):
        """Save current configuration"""
        try:
            self.log_info("💾 Menyimpan konfigurasi...")
            
            # Extract config dari UI
            config = self.config_handler.extract_config(self.ui_components)
            
            # Validate config
            if not self._validate_config(config):
                self.log_error("❌ Konfigurasi tidak valid")
                return
            
            # Save config
            self.config_handler.save_config(config)
            
            self.log_success("💾 Konfigurasi berhasil disimpan")
            
            # Update status panel
            from smartcash.ui.components.status_panel import update_status_panel
            status_panel = self.ui_components.get('status_panel')
            if status_panel:
                update_status_panel(status_panel, "💾 Config saved", 'success')
                
        except Exception as e:
            self.log_error(f"❌ Error save config: {str(e)}")
    
    def reset_config(self, *args):
        """Reset configuration to default"""
        try:
            self.log_info("🔄 Reset konfigurasi ke default...")
            
            # Get default config
            default_config = self.config_handler.get_default_config()
            
            # Update UI dengan default config
            self.config_handler.update_ui(self.ui_components, default_config)
            
            self.log_success("🔄 Konfigurasi direset ke default")
            
            # Update status panel
            from smartcash.ui.components.status_panel import update_status_panel
            status_panel = self.ui_components.get('status_panel')
            if status_panel:
                update_status_panel(status_panel, "🔄 Config reset", 'info')
                
        except Exception as e:
            self.log_error(f"❌ Error reset config: {str(e)}")
    
    def load_config(self, config_path: str = None):
        """Load configuration dari file"""
        try:
            self.log_info(f"📂 Loading konfigurasi{' dari ' + config_path if config_path else ''}...")
            
            # Load config
            if config_path:
                config = self.config_handler.load_config(config_path)
            else:
                config = self.config_handler.load_config()
            
            if not config:
                self.log_warning("⚠️ Tidak ada konfigurasi yang ditemukan, menggunakan default")
                config = self.config_handler.get_default_config()
            
            # Validate loaded config
            if not self._validate_config(config):
                self.log_warning("⚠️ Konfigurasi tidak valid, menggunakan default")
                config = self.config_handler.get_default_config()
            
            # Update UI dengan loaded config
            self.config_handler.update_ui(self.ui_components, config)
            
            self.log_success("📂 Konfigurasi berhasil dimuat")
            
            # Update status panel
            from smartcash.ui.components.status_panel import update_status_panel
            status_panel = self.ui_components.get('status_panel')
            if status_panel:
                update_status_panel(status_panel, "📂 Config loaded", 'success')
                
        except Exception as e:
            self.log_error(f"❌ Error load config: {str(e)}")
    
    def _validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        if not config or not isinstance(config, dict):
            return False
        
        required_keys = ['module_name', 'selected_packages']
        for key in required_keys:
            if key not in config:
                self.log_warning(f"⚠️ Missing required config key: {key}")
                return False
        
        # Check if selected_packages is list
        if not isinstance(config.get('selected_packages'), list):
            self.log_warning("⚠️ selected_packages harus berupa list")
            return False
        
        # Check if custom_packages is string
        custom_packages = config.get('custom_packages', '')
        if not isinstance(custom_packages, str):
            self.log_warning("⚠️ custom_packages harus berupa string")
            return False
        
        return True

# Factory function
def setup_config_handlers(ui_components: Dict[str, Any], config_handler: ConfigHandler) -> Dict[str, Any]:
    """Setup config event handlers"""
    if not config_handler:
        raise ValueError("ConfigHandler tidak boleh None")
    
    handler = ConfigEventHandler(ui_components, config_handler)
    
    return {
        'save': handler.save_config,
        'reset': handler.reset_config,
        'load': handler.load_config
    }