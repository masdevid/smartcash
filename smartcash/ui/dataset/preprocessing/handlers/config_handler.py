"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: DRY config handler dengan defaults sebagai base structure
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """DRY config handler dengan defaults sebagai base"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'preprocessing_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract dengan DRY approach"""
        return extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_preprocessing_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default dari defaults.py"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def load_config(self, config_filename: str = None) -> Dict[str, Any]:
        """Load dengan fallback ke defaults"""
        try:
            filename = config_filename or self.config_filename
            config = self.config_manager.load_config(filename)
            
            if not config:
                self.logger.warning("⚠️ Config kosong, menggunakan default")
                return self.get_default_config()
            
            # Merge dengan defaults untuk missing keys
            return self._merge_with_defaults(config)
            
        except Exception as e:
            self.logger.error(f"❌ Error loading config: {str(e)}")
            return self.get_default_config()
    
    def _merge_with_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Merge config dengan defaults untuk missing keys"""
        import copy
        
        defaults = self.get_default_config()
        merged = copy.deepcopy(defaults)
        
        # Deep merge untuk preserve structure
        for section in ['preprocessing', 'performance', 'cleanup']:
            if section in config:
                merged[section].update(config[section])
                
                # Deep merge untuk nested dicts
                if section == 'preprocessing':
                    for nested in ['validate', 'normalization', 'analysis', 'balance']:
                        if nested in config[section] and nested in merged[section]:
                            if isinstance(merged[section][nested], dict):
                                merged[section][nested].update(config[section][nested])
        
        # Preserve inheritance marker
        if '_base_' in config:
            merged['_base_'] = config['_base_']
        
        return merged
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save dengan auto refresh"""
        try:
            filename = config_filename or self.config_filename
            ui_config = self.extract_config(ui_components)
            
            success = self.config_manager.save_config(ui_config, filename)
            
            if success:
                self.logger.success(f"✅ Config tersimpan ke {filename}")
                self._refresh_ui(ui_components, filename)
                return True
            else:
                self.logger.error(f"❌ Gagal simpan config")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error save: {str(e)}")
            return False
    
    def reset_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Reset dengan auto refresh"""
        try:
            filename = config_filename or self.config_filename
            default_config = self.get_default_config()
            
            success = self.config_manager.save_config(default_config, filename)
            
            if success:
                self.logger.success(f"🔄 Config direset ke default")
                self.update_ui(ui_components, default_config)
                return True
            else:
                self.logger.error(f"❌ Gagal reset config")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error reset: {str(e)}")
            return False
    
    def _refresh_ui(self, ui_components: Dict[str, Any], filename: str):
        """Auto refresh UI setelah save"""
        try:
            saved_config = self.config_manager.load_config(filename)
            if saved_config:
                self.update_ui(ui_components, saved_config)
                self.logger.info("🔄 UI direfresh")
        except Exception as e:
            self.logger.warning(f"⚠️ Error refresh: {str(e)}")