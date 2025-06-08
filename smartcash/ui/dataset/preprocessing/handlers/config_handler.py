"""
File: smartcash/ui/dataset/preprocessing/handlers/config_handler.py
Deskripsi: Fixed config handler yang mempertahankan struktur defaults dan normalisasi
"""

from typing import Dict, Any
from smartcash.ui.handlers.config_handlers import ConfigHandler
from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
from smartcash.ui.dataset.preprocessing.handlers.config_updater import update_preprocessing_ui
from smartcash.common.config.manager import get_config_manager

class PreprocessingConfigHandler(ConfigHandler):
    """Config handler untuk preprocessing dengan struktur yang konsisten"""
    
    def __init__(self, module_name: str = 'preprocessing', parent_module: str = 'dataset'):
        super().__init__(module_name, parent_module)
        self.config_manager = get_config_manager()
        self.config_filename = 'dataset_config.yaml'
    
    def extract_config(self, ui_components: Dict[str, Any]) -> Dict[str, Any]:
        """Extract config dari preprocessing UI components"""
        return extract_preprocessing_config(ui_components)
    
    def update_ui(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Update UI dari config"""
        update_preprocessing_ui(ui_components, config)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default config dengan struktur lengkap"""
        from smartcash.ui.dataset.preprocessing.handlers.defaults import get_default_preprocessing_config
        return get_default_preprocessing_config()
    
    def save_config(self, ui_components: Dict[str, Any], config_filename: str = None) -> bool:
        """Save config dengan mempertahankan struktur defaults"""
        try:
            filename = config_filename or self.config_filename
            
            # Extract UI values dengan struktur lengkap
            ui_config = self.extract_config(ui_components)
            
            # Load existing config
            existing_config = self.config_manager.load_config(filename) or {}
            
            # Load default structure sebagai base
            default_config = self.get_default_config()
            
            # Merge dengan prioritas: defaults -> existing -> UI values
            merged_config = self._deep_merge_configs(default_config, existing_config, ui_config)
            
            # Validate sebelum save
            validation = self.validate_config(merged_config)
            if not validation['valid']:
                self.logger.error(f"❌ Config tidak valid: {'; '.join(validation['errors'])}")
                return False
            
            # Save merged config
            success = self.config_manager.save_config(merged_config, filename)
            
            if success:
                self.logger.success(f"✅ Config preprocessing tersimpan ke {filename}")
                return True
            else:
                self.logger.error(f"❌ Gagal menyimpan config ke {filename}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error saving preprocessing config: {str(e)}")
            return False
    
    def _deep_merge_configs(self, default: Dict[str, Any], existing: Dict[str, Any], ui_values: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configs dengan mempertahankan struktur defaults"""
        import copy
        
        # Start dengan default structure
        merged = copy.deepcopy(default)
        
        # Apply existing values (non-UI fields)
        self._deep_update(merged, existing, preserve_structure=True)
        
        # Apply UI values dengan mapping yang tepat
        self._apply_ui_values(merged, ui_values)
        
        return merged
    
    def _deep_update(self, base: Dict[str, Any], update: Dict[str, Any], preserve_structure: bool = False):
        """Deep update dictionary dengan option preserve structure"""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value, preserve_structure)
            elif not preserve_structure or key in base:
                base[key] = value
    
    def _apply_ui_values(self, merged: Dict[str, Any], ui_values: Dict[str, Any]):
        """Apply UI values ke config structure yang tepat"""
        # Extract UI values
        preprocessing_ui = ui_values.get('preprocessing', {})
        performance_ui = ui_values.get('performance', {})
        cleanup_ui = ui_values.get('cleanup', {})
        
        # Apply preprocessing settings dengan struktur lengkap
        if 'preprocessing' in merged and preprocessing_ui:
            # Complete normalization structure
            if 'normalization' in preprocessing_ui:
                norm_ui = preprocessing_ui['normalization']
                merged['preprocessing']['normalization'] = {
                    'enabled': norm_ui.get('enabled', True),
                    'method': norm_ui.get('method', 'minmax'),
                    'target_size': norm_ui.get('target_size', [640, 640]),
                    'preserve_aspect_ratio': norm_ui.get('preserve_aspect_ratio', True),
                    'normalize_pixel_values': norm_ui.get('normalize_pixel_values', True),
                    'pixel_range': norm_ui.get('pixel_range', [0, 1])
                }
            
            # Target split
            if 'target_split' in preprocessing_ui:
                merged['preprocessing']['target_split'] = preprocessing_ui['target_split']
            
            # Force reprocess
            if 'force_reprocess' in preprocessing_ui:
                merged['preprocessing']['force_reprocess'] = preprocessing_ui['force_reprocess']
            
            # Validate settings
            if 'validate' in preprocessing_ui:
                merged['preprocessing']['validate'].update(preprocessing_ui['validate'])
            
            # Analysis settings  
            if 'analysis' in preprocessing_ui:
                merged['preprocessing']['analysis'].update(preprocessing_ui['analysis'])
            
            # Balance settings
            if 'balance' in preprocessing_ui:
                merged['preprocessing']['balance'].update(preprocessing_ui['balance'])
        
        # Apply performance settings
        if 'performance' in merged and performance_ui:
            merged['performance'].update(performance_ui)
        
        # Apply cleanup settings
        if 'cleanup' in merged and cleanup_ui:
            merged['cleanup'].update(cleanup_ui)
        
        # Update metadata
        merged['updated_at'] = ui_values.get('updated_at')
        merged['config_version'] = ui_values.get('config_version', '1.0')
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate preprocessing config"""
        errors = []
        warnings = []
        
        preprocessing = config.get('preprocessing', {})
        performance = config.get('performance', {})
        
        # Validate normalization
        normalization = preprocessing.get('normalization', {})
        method = normalization.get('method', 'minmax')
        if method not in ['minmax', 'standard', 'none']:
            errors.append("Metode normalisasi tidak valid")
        
        target_size = normalization.get('target_size', [640, 640])
        if not isinstance(target_size, list) or len(target_size) != 2:
            errors.append("Target size harus berupa list [width, height]")
        
        # Validate workers
        num_workers = performance.get('num_workers', 8)
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 16:
            warnings.append("Number of workers sebaiknya antara 1-16")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }