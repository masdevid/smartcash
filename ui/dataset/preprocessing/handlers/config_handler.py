"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Config handler untuk preprocessing dengan dataset integration
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Config handler untuk preprocessing dengan dataset integration"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'dataset_config.yaml'  # Use dataset_config.yaml untuk consistency
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari preprocessing UI components"""
        return extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_preprocessing_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan optimal workers"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan merge strategy untuk dataset_config.yaml"""
        try:
            filename = config_filename or self.config_filename
            
            # Extract current config dari UI
            current_config = self.extract_config(ui_components)
            
            # Validate sebelum save
            validation = self.validate_config(current_config)
            if not validation['valid']:
                self.logger.error(f"❌ Config tidak valid: {'; '.join(validation['errors'])}")
                return False
            
            # Load existing config untuk merge
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Merge dengan strategy yang aman
            merged_config = self._merge_preprocessing_config(existing_config, current_config)
            
            # Save merged config
            success = self.config_manager.save_config(merged_config, filename)
            
            if success:
                self.logger.success(f"✅ Config tersimpan ke {filename}")
                return True
            else:
                self.logger.error(f"❌ Gagal menyimpan config ke {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving config: {str(e)}")
            return False
    
    def _merge_preprocessing_config(self, existing: Dict[str, Any], new_preprocessing: Dict[str, Any]) -> Dict[str, Any]:
        """Merge preprocessing config dengan existing dataset_config.yaml"""
        merged = dict(existing) if existing else {}
        
        # Preprocessing-specific sections yang akan di-merge
        preprocessing_sections = ['preprocessing', 'performance']
        
        for section in preprocessing_sections:
            if section in new_preprocessing:
                merged[section] = new_preprocessing[section]
        
        # Preserve config metadata
        merged['config_version'] = new_preprocessing.get('config_version', '1.0')
        merged['updated_at'] = new_preprocessing.get('updated_at')
        merged['_base_'] = new_preprocessing.get('_base_', 'base_config.yaml')
        
        return merged