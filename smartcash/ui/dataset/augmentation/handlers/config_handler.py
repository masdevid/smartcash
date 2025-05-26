"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Fixed SRP handler untuk konfigurasi augmentasi dengan SimpleConfigManager yang benar
"""

from typing import Dict, Any, Optional, List
from smartcash.common.config import get_config_manager
from smartcash.dataset.augmentor.config import extract_ui_config, create_aug_config

class ConfigHandler:
    """SRP handler untuk mengelola konfigurasi augmentasi dengan SimpleConfigManager yang benar."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.config_manager = get_config_manager()
        self.config_file = 'augmentation_config.yaml'
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract konfigurasi dari UI components."""
        try:
            # Extract basic parameters dengan safe attribute access
            config = {
                'augmentation': {
                    'num_variations': self._get_widget_value('num_variations', 2),
                    'target_count': self._get_widget_value('target_count', 500),
                    'output_prefix': self._get_widget_value('output_prefix', 'aug'),
                    'balance_classes': self._get_widget_value('balance_classes', False),
                    
                    # Advanced parameters dengan nilai moderat
                    'fliplr': self._get_widget_value('fliplr', 0.5),
                    'degrees': self._get_widget_value('degrees', 10),
                    'translate': self._get_widget_value('translate', 0.1),
                    'scale': self._get_widget_value('scale', 0.1),
                    'hsv_h': self._get_widget_value('hsv_h', 0.015),
                    'hsv_s': self._get_widget_value('hsv_s', 0.7),
                    'brightness': self._get_widget_value('brightness', 0.2),
                    'contrast': self._get_widget_value('contrast', 0.2),
                    
                    # Types dan split
                    'types': list(self._get_widget_value('augmentation_types', ['combined'])),
                    'target_split': self._get_widget_value('target_split', 'train'),
                    'output_dir': 'data/augmented'
                },
                'data': {'dir': self.ui_components.get('data_dir', 'data')},
                'preprocessing': {'output_dir': 'data/preprocessed'}
            }
            
            return config
            
        except Exception as e:
            # Fallback ke extract_ui_config
            return extract_ui_config(self.ui_components)
    
    # One-liner widget value extractor
    _get_widget_value = lambda self, key, default: getattr(self.ui_components.get(key), 'value', default)
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Simpan konfigurasi menggunakan SimpleConfigManager dengan one-liner style."""
        config = config or self.extract_config_from_ui()
        
        try:
            success = self.config_manager.save_config(config, self.config_file)
            status_msg = '✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive' if success else '❌ Gagal menyimpan konfigurasi'
            self._log_message(status_msg, 'success' if success else 'error')
            
            return {'status': 'success' if success else 'error', 'message': status_msg, 'config': config}
        except Exception as e:
            error_msg = f'❌ Error save config: {str(e)}'
            self._log_message(error_msg, 'error')
            return {'status': 'error', 'message': error_msg, 'config': config or {}}
    
    def load_config(self) -> Dict[str, Any]:
        """Load konfigurasi dengan one-liner fallback."""
        try:
            saved_config = self.config_manager.load_config(self.config_file)
            is_valid = saved_config and 'augmentation' in saved_config
            result_config = saved_config if is_valid else self.get_default_config()
            
            self._log_message('📂 Konfigurasi berhasil dimuat dari storage' if is_valid else '🔧 Menggunakan konfigurasi default', 'info')
            return result_config
        except Exception as e:
            self._log_message(f'⚠️ Error load config: {str(e)}, menggunakan default', 'warning')
            return self.get_default_config()
    
    def reset_to_default(self) -> Dict[str, Any]:
        """Reset konfigurasi ke default dengan one-liner flow."""
        try:
            default_config = self.get_default_config()
            self.apply_config_to_ui(default_config)
            save_result = self.save_config(default_config)
            
            self._log_message('🔄 Konfigurasi berhasil direset ke default', 'success')
            return {'status': 'success', 'message': '🔄 Konfigurasi berhasil direset ke default', 'config': default_config, 'save_result': save_result}
        except Exception as e:
            error_msg = f'❌ Error reset config: {str(e)}'
            self._log_message(error_msg, 'error')
            return {'status': 'error', 'message': error_msg}
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> bool:
        """Apply konfigurasi ke UI dengan one-liner mappings."""
        try:
            aug_config = config.get('augmentation', {})
            
            # One-liner UI mappings
            ui_mappings = {k: aug_config.get(k, v) for k, v in {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug', 'balance_classes': False,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2, 'target_split': 'train'
            }.items()}
            
            # Apply values dengan one-liner
            [setattr(widget, 'value', value) for ui_key, value in ui_mappings.items() 
             if (widget := self.ui_components.get(ui_key)) and hasattr(widget, 'value')]
            
            # Special handling untuk augmentation types
            aug_types_widget = self.ui_components.get('augmentation_types')
            if aug_types_widget and hasattr(aug_types_widget, 'value'):
                aug_types_widget.value = list(aug_config.get('types', ['combined']))
            
            return True
        except Exception as e:
            self._log_message(f'⚠️ Error apply config to UI: {str(e)}', 'warning')
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """Dapatkan konfigurasi default dengan nilai moderat untuk penelitian."""
        return {
            'augmentation': {
                # Basic parameters
                'num_variations': 2,
                'target_count': 500,
                'output_prefix': 'aug',
                'balance_classes': False,
                
                # Advanced parameters - nilai moderat untuk penelitian
                'fliplr': 0.5,
                'degrees': 10,
                'translate': 0.1,
                'scale': 0.1,
                'hsv_h': 0.015,
                'hsv_s': 0.7,
                'brightness': 0.2,
                'contrast': 0.2,
                
                # Pipeline research types
                'types': ['combined'],
                'target_split': 'train',
                'intensity': 0.7,
                'output_dir': 'data/augmented'
            },
            'data': {'dir': 'data'},
            'preprocessing': {'output_dir': 'data/preprocessed'}
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi konfigurasi dengan one-liner checks."""
        aug_config = config.get('augmentation', {})
        
        # One-liner validation checks
        errors = [msg for check, msg in [
            (aug_config.get('num_variations', 0) <= 0, 'Jumlah variasi harus > 0'),
            (aug_config.get('target_count', 0) <= 0, 'Target count harus > 0'),
            (not aug_config.get('types', []), 'Jenis augmentasi harus dipilih minimal 1')
        ] if check]
        
        return {'valid': len(errors) == 0, 'warnings': [], 'errors': errors}
    
    # One-liner log helper
    _log_message = lambda self, msg, level='info': getattr(self.ui_components.get('logger'), level, lambda x: print(f"[{level.upper()}] {x}"))(msg) if 'logger' in self.ui_components else None

# Factory function
def create_config_handler(ui_components: Dict[str, Any]) -> ConfigHandler:
    """Factory function untuk create config handler."""
    return ConfigHandler(ui_components)

# One-liner utilities  
extract_config = lambda ui_components: ConfigHandler(ui_components).extract_config_from_ui()
save_augmentation_config = lambda ui_components, config=None: ConfigHandler(ui_components).save_config(config)
load_augmentation_config = lambda ui_components: ConfigHandler(ui_components).load_config()
reset_augmentation_config = lambda ui_components: ConfigHandler(ui_components).reset_to_default()
validate_augmentation_config = lambda ui_components, config: ConfigHandler(ui_components).validate_config(config)