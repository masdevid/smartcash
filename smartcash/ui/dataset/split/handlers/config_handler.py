"""
File: smartcash/ui/dataset/split/handlers/config_handler.py
Deskripsi: Configuration handler untuk dataset split dengan logger bridge integration
"""

from typing import Dict, Any, Optional, Tuple
from smartcash.ui.config_cell.handlers.config_handler import ConfigCellHandler
from smartcash.ui.utils.logger_bridge import UILoggerBridge
from .config_extractor import extract_split_config
from .config_updater import update_split_ui, reset_ui_to_defaults
from .defaults import get_default_split_config

class SplitConfigHandler(ConfigCellHandler):
    """Handler untuk dataset split configuration dengan logger bridge support.
    
    Handler ini mengintegrasikan dengan parent component untuk:
    - Logging terpusat melalui logger bridge
    - Update status panel otomatis
    - Error handling konsisten
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize split config handler.
        
        Args:
            config: Optional initial configuration
        """
        super().__init__(config or {})
        self._logger_bridge: Optional[UILoggerBridge] = None
        
    def set_logger_bridge(self, logger_bridge: UILoggerBridge) -> None:
        """Set logger bridge untuk integrasi dengan parent component.
        
        Args:
            logger_bridge: Instance UILoggerBridge dari parent
        """
        self._logger_bridge = logger_bridge
        
    @property
    def logger(self):
        """Get logger instance, prefer logger bridge jika tersedia."""
        if self._logger_bridge:
            return self._logger_bridge
        return super().logger
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract configuration dari UI components.
        
        Args:
            ui_components: Dictionary of UI components
            
        Returns:
            Dictionary containing extracted configuration
        """
        try:
            config = extract_split_config(ui_components)
            
            # Log sukses via logger bridge
            if self._logger_bridge:
                self._logger_bridge.info("✅ Berhasil extract konfigurasi split")
            
            # Update status panel jika ada
            self._update_status_panel(ui_components, "✅ Konfigurasi berhasil di-extract", "success")
            
            return config
            
        except Exception as e:
            error_msg = f"❌ Gagal extract config: {str(e)}"
            
            # Log error via logger bridge
            if self._logger_bridge:
                self._logger_bridge.error(error_msg)
            
            # Update status panel dengan error
            self._update_status_panel(ui_components, error_msg, "error")
            
            raise
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI components dengan konfigurasi.
        
        Args:
            ui_components: Dictionary of UI components
            config: Configuration to apply
        """
        try:
            update_split_ui(ui_components, config)
            
            # Log sukses
            if self._logger_bridge:
                self._logger_bridge.info("✅ UI berhasil di-update dengan konfigurasi")
                
            # Update status panel
            self._update_status_panel(ui_components, "✅ UI berhasil di-update", "success")
            
        except Exception as e:
            error_msg = f"❌ Gagal update UI: {str(e)}"
            
            if self._logger_bridge:
                self._logger_bridge.error(error_msg)
                
            self._update_status_panel(ui_components, error_msg, "error")
            raise
    
    def reset_ui(self, ui_components: Dict[str, Any]) -> None:
        """Reset UI components ke nilai default.
        
        Args:
            ui_components: Dictionary of UI components to reset
        """
        try:
            reset_ui_to_defaults(ui_components)
            
            if self._logger_bridge:
                self._logger_bridge.info("🔄 UI berhasil di-reset ke default")
                
            self._update_status_panel(ui_components, "🔄 UI di-reset ke default", "info")
            
        except Exception as e:
            error_msg = f"❌ Gagal reset UI: {str(e)}"
            
            if self._logger_bridge:
                self._logger_bridge.error(error_msg)
                
            self._update_status_panel(ui_components, error_msg, "error")
            raise
        
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, str]:
        """Validate konfigurasi split.
        
        Args:
            config: Configuration to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            ratios = config.get('data', {}).get('split_ratios', {})
            train = ratios.get('train', 0)
            valid = ratios.get('valid', 0)
            test = ratios.get('test', 0)
            
            # Validasi ratio sum = 1.0
            if not self.validate_split_ratios(train, valid, test):
                return False, "Split ratios harus berjumlah 1.0"
            
            # Validasi path configuration
            paths = config.get('dataset', {}).get('paths', {})
            if not paths.get('base_dir'):
                return False, "Base directory tidak boleh kosong"
                
            # Log validasi sukses
            if self._logger_bridge:
                self._logger_bridge.debug(f"✅ Validasi sukses - Train: {train}, Valid: {valid}, Test: {test}")
                
            return True, ""
            
        except Exception as e:
            error_msg = f"Validation error: {str(e)}"
            
            if self._logger_bridge:
                self._logger_bridge.error(f"❌ {error_msg}")
                
            return False, error_msg
    
    def save_config(self, ui_components: Dict[str, Any]) -> bool:
        """Save konfigurasi dengan status update.
        
        Args:
            ui_components: UI components untuk extract config
            
        Returns:
            True jika berhasil save
        """
        try:
            # Extract current config
            config = self.extract_config(ui_components)
            
            # Validate sebelum save
            is_valid, error_msg = self.validate_config(config)
            if not is_valid:
                raise ValueError(error_msg)
            
            # Save menggunakan parent method
            result = super().save_config(ui_components)
            
            if result:
                if self._logger_bridge:
                    self._logger_bridge.info("💾 Konfigurasi berhasil disimpan")
                self._update_status_panel(ui_components, "💾 Konfigurasi berhasil disimpan", "success")
            
            return result
            
        except Exception as e:
            error_msg = f"❌ Gagal save config: {str(e)}"
            
            if self._logger_bridge:
                self._logger_bridge.error(error_msg)
                
            self._update_status_panel(ui_components, error_msg, "error")
            return False
    
    def _update_status_panel(self, ui_components: Dict[str, Any], message: str, status_type: str) -> None:
        """Update status panel jika tersedia.
        
        Args:
            ui_components: Dictionary containing UI components
            message: Status message
            status_type: Status type (success, error, info, warning)
        """
        # Cari status panel dari parent atau ui_components
        status_panel = ui_components.get('status_panel')
        
        # Coba dari parent jika ada
        if not status_panel and 'parent' in ui_components:
            parent = ui_components['parent']
            if hasattr(parent, 'status_panel'):
                status_panel = parent.status_panel
        
        if status_panel:
            try:
                from smartcash.ui.components import update_status_panel
                update_status_panel(status_panel, message, status_type)
            except Exception as e:
                # Log tapi jangan fail operasi utama
                if self._logger_bridge:
                    self._logger_bridge.debug(f"⚠️ Gagal update status panel: {str(e)}")
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """Get default configuration untuk handler ini.
        
        Returns:
            Default configuration dictionary
        """
        return get_default_split_config()