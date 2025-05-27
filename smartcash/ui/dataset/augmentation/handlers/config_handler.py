"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: Fixed config handler dengan status panel updates dan train split enforcement
"""

from typing import Dict, Any, Optional
from smartcash.common.config import get_config_manager
from smartcash.dataset.augmentor.config import extract_ui_config, create_aug_config

class ConfigHandler:
    """Fixed config handler dengan train split enforcement dan status updates"""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.config_manager = get_config_manager()
        self.config_file = 'augmentation_config.yaml'
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Fixed config extraction dengan train split only"""
        try:
            # Extract augmentation types - handle all possible widget structures
            aug_types = self._get_augmentation_types()
            
            # Fixed config dengan train split enforcement
            config = {
                'augmentation': {
                    'num_variations': self._get_widget_value('num_variations', 2),
                    'target_count': self._get_widget_value('target_count', 500),
                    'output_prefix': self._get_widget_value('output_prefix', 'aug'),
                    'balance_classes': self._get_widget_value('balance_classes', True),
                    
                    # Advanced parameters
                    'fliplr': self._get_widget_value('fliplr', 0.5),
                    'degrees': self._get_widget_value('degrees', 10),
                    'translate': self._get_widget_value('translate', 0.1),
                    'scale': self._get_widget_value('scale', 0.1),
                    'hsv_h': self._get_widget_value('hsv_h', 0.015),
                    'hsv_s': self._get_widget_value('hsv_s', 0.7),
                    'brightness': self._get_widget_value('brightness', 0.2),
                    'contrast': self._get_widget_value('contrast', 0.2),
                    
                    # Fixed: enforce train split only
                    'types': aug_types,
                    'target_split': 'train',  # Always train only
                    'output_dir': 'data/augmented'
                },
                'data': {'dir': self.ui_components.get('data_dir', 'data')},
                'preprocessing': {'output_dir': 'data/preprocessed'}
            }
            
            return config
            
        except Exception as e:
            # Fallback ke extract_ui_config dengan train override
            config = extract_ui_config(self.ui_components)
            config['augmentation']['target_split'] = 'train'  # Force train split
            return config
    
    def _get_augmentation_types(self) -> list:
        """Fixed augmentation types extraction dari berbagai widget structures"""
        # Try multiple widget patterns
        for widget_key in ['augmentation_types', 'aug_options', 'types_widget']:
            widget = self.ui_components.get(widget_key)
            if widget:
                # Direct value access
                if hasattr(widget, 'value') and widget.value:
                    return list(widget.value)
                
                # Container with children
                if hasattr(widget, 'children') and widget.children:
                    for child in widget.children:
                        if hasattr(child, 'value') and child.value:
                            return list(child.value)
        
        # Fallback default
        return ['combined']
    
    # One-liner widget value extractor dengan fallback
    _get_widget_value = lambda self, key, default: getattr(self.ui_components.get(key), 'value', default)
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Save config dengan train split enforcement dan status update"""
        config = config or self.extract_config_from_ui()
        
        # Enforce train split
        config['augmentation']['target_split'] = 'train'
        
        try:
            success = self.config_manager.save_config(config, self.config_file)
            status_msg = '✅ Konfigurasi berhasil disimpan (Train split only)' if success else '❌ Gagal menyimpan konfigurasi'
            
            # Update status panel jika ada
            self._update_status_panel(status_msg, 'success' if success else 'error')
            
            return {'status': 'success' if success else 'error', 'message': status_msg, 'config': config}
        except Exception as e:
            error_msg = f'❌ Error save config: {str(e)}'
            self._update_status_panel(error_msg, 'error')
            return {'status': 'error', 'message': error_msg, 'config': config or {}}
    
    def load_config(self) -> Dict[str, Any]:
        """Load config dengan train split enforcement"""
        try:
            saved_config = self.config_manager.load_config(self.config_file)
            is_valid = saved_config and 'augmentation' in saved_config
            result_config = saved_config if is_valid else self.get_default_config()
            
            # Force train split
            result_config['augmentation']['target_split'] = 'train'
            
            return result_config
        except Exception as e:
            return self.get_default_config()
    
    def reset_to_default(self) -> Dict[str, Any]:
        """Reset dengan train split enforcement dan status update"""
        try:
            default_config = self.get_default_config()
            self.apply_config_to_ui(default_config)
            save_result = self.save_config(default_config)
            
            status_msg = '🔄 Konfigurasi direset ke default (Train split only)'
            self._update_status_panel(status_msg, 'success')
            
            return {'status': 'success', 'message': status_msg, 'config': default_config}
        except Exception as e:
            error_msg = f'❌ Error reset config: {str(e)}'
            self._update_status_panel(error_msg, 'error')
            return {'status': 'error', 'message': error_msg}
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> bool:
        """Apply config ke UI dengan train split enforcement"""
        try:
            aug_config = config.get('augmentation', {})
            
            # Apply basic config
            ui_mappings = {k: aug_config.get(k, v) for k, v in {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug', 'balance_classes': False,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2
            }.items()}
            
            # Apply values
            [setattr(widget, 'value', value) for ui_key, value in ui_mappings.items() 
             if (widget := self.ui_components.get(ui_key)) and hasattr(widget, 'value')]
            
            # Force train split in target_split widget
            target_split_widget = self.ui_components.get('target_split')
            if target_split_widget and hasattr(target_split_widget, 'value'):
                target_split_widget.value = 'train'
            
            # Apply augmentation types
            aug_types_widget = self.ui_components.get('augmentation_types')
            if aug_types_widget and hasattr(aug_types_widget, 'value'):
                aug_types_widget.value = list(aug_config.get('types', ['combined']))
            
            return True
        except Exception:
            return False
    
    def _update_status_panel(self, message: str, status_type: str):
        """One-liner update status panel dengan new message"""
        status_panel = self.ui_components.get('status_panel')
        if status_panel and hasattr(status_panel, 'value'):
            from smartcash.ui.utils.alert_utils import create_alert_html
            status_panel.value = create_alert_html(message, status_type)
    
    def get_default_config(self) -> Dict[str, Any]:
        """Default config dengan train split only"""
        return {
            'augmentation': {
                'num_variations': 2, 'target_count': 500, 'output_prefix': 'aug', 'balance_classes': False,
                'fliplr': 0.5, 'degrees': 10, 'translate': 0.1, 'scale': 0.1,
                'hsv_h': 0.015, 'hsv_s': 0.7, 'brightness': 0.2, 'contrast': 0.2,
                'types': ['combined'], 'target_split': 'train', 'intensity': 0.7, 'output_dir': 'data/augmented'
            },
            'data': {'dir': 'data'},
            'preprocessing': {'output_dir': 'data/preprocessed'}
        }

# Factory function
def create_config_handler(ui_components: Dict[str, Any]) -> ConfigHandler:
    return ConfigHandler(ui_components)

# One-liner utilities dengan train enforcement
extract_config = lambda ui_components: ConfigHandler(ui_components).extract_config_from_ui()
save_config = lambda ui_components, config=None: ConfigHandler(ui_components).save_config(config)
reset_config = lambda ui_components: ConfigHandler(ui_components).reset_to_default()