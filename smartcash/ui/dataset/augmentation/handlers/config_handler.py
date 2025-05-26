"""
File: smartcash/ui/dataset/augmentation/handlers/config_handler.py
Deskripsi: SRP handler untuk konfigurasi augmentasi - save/load/extract/validate
"""

from typing import Dict, Any, Optional, List
from smartcash.common.config import get_config_manager
from smartcash.dataset.augmentor.config import extract_ui_config, create_aug_config

class ConfigHandler:
    """SRP handler untuk mengelola konfigurasi augmentasi."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.config_manager = get_config_manager()
        self.module_name = 'augmentation'
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """
        Extract konfigurasi dari UI components.
        
        Returns:
            Dictionary konfigurasi yang diekstrak
        """
        try:
            # Extract basic parameters
            config = {
                'augmentation': {
                    'num_variations': getattr(self.ui_components.get('num_variations'), 'value', 2),
                    'target_count': getattr(self.ui_components.get('target_count'), 'value', 500),
                    'output_prefix': getattr(self.ui_components.get('output_prefix'), 'value', 'aug'),
                    'balance_classes': getattr(self.ui_components.get('balance_classes'), 'value', False),
                    
                    # Advanced parameters dengan nilai yang lebih moderat
                    'fliplr': getattr(self.ui_components.get('fliplr'), 'value', 0.5),
                    'degrees': getattr(self.ui_components.get('degrees'), 'value', 10),  # Reduced from 15
                    'translate': getattr(self.ui_components.get('translate'), 'value', 0.1),  # Reduced from 0.15
                    'scale': getattr(self.ui_components.get('scale'), 'value', 0.1),  # Reduced from 0.15
                    'hsv_h': getattr(self.ui_components.get('hsv_h'), 'value', 0.015),  # Reduced from 0.025
                    'hsv_s': getattr(self.ui_components.get('hsv_s'), 'value', 0.7),
                    'brightness': getattr(self.ui_components.get('brightness'), 'value', 0.2),  # Reduced from 0.3
                    'contrast': getattr(self.ui_components.get('contrast'), 'value', 0.2),  # Reduced from 0.3
                    
                    # Types dan split
                    'types': list(getattr(self.ui_components.get('augmentation_types'), 'value', ['combined'])),
                    'target_split': getattr(self.ui_components.get('target_split'), 'value', 'train'),
                    
                    # Output directories
                    'output_dir': 'data/augmented'
                },
                'data': {
                    'dir': getattr(self.ui_components, 'get', lambda x, default: default)('data_dir', 'data')
                },
                'preprocessing': {
                    'output_dir': 'data/preprocessed'
                }
            }
            
            return config
            
        except Exception as e:
            # Fallback ke extract_ui_config
            return extract_ui_config(self.ui_components)
    
    def save_config(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Simpan konfigurasi ke storage.
        
        Args:
            config: Konfigurasi untuk disimpan (None = extract dari UI)
            
        Returns:
            Result dictionary dengan status
        """
        try:
            # Extract config jika tidak disediakan
            if config is None:
                config = self.extract_config_from_ui()
            
            # Save dengan config manager
            aug_config = config.get('augmentation', {})
            success = self.config_manager.save_module_config(self.module_name, aug_config)
            
            if success:
                return {
                    'status': 'success',
                    'message': '✅ Konfigurasi berhasil disimpan dan disinkronkan ke Google Drive',
                    'config': config
                }
            else:
                return {
                    'status': 'error',
                    'message': '❌ Gagal menyimpan konfigurasi',
                    'config': config
                }
                
        except Exception as e:
            return {
                'status': 'error',
                'message': f'❌ Error save config: {str(e)}',
                'config': config or {}
            }
    
    def load_config(self) -> Dict[str, Any]:
        """
        Load konfigurasi dari storage.
        
        Returns:
            Dictionary konfigurasi yang dimuat
        """
        try:
            # Load dari config manager
            saved_config = self.config_manager.get_module_config(self.module_name)
            
            if saved_config:
                # Wrap dalam struktur yang sesuai
                return {
                    'augmentation': saved_config,
                    'data': {'dir': 'data'},
                    'preprocessing': {'output_dir': 'data/preprocessed'}
                }
            else:
                # Return default config
                return self.get_default_config()
                
        except Exception as e:
            return self.get_default_config()
    
    def reset_to_default(self) -> Dict[str, Any]:
        """
        Reset konfigurasi ke default dan apply ke UI.
        
        Returns:
            Result dictionary dengan status
        """
        try:
            default_config = self.get_default_config()
            
            # Apply ke UI components
            self.apply_config_to_ui(default_config)
            
            # Save default config
            save_result = self.save_config(default_config)
            
            return {
                'status': 'success',
                'message': '🔄 Konfigurasi berhasil direset ke default',
                'config': default_config,
                'save_result': save_result
            }
            
        except Exception as e:
            return {
                'status': 'error',
                'message': f'❌ Error reset config: {str(e)}'
            }
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> bool:
        """
        Apply konfigurasi ke UI components.
        
        Args:
            config: Konfigurasi untuk diapply
            
        Returns:
            True jika berhasil diapply
        """
        try:
            aug_config = config.get('augmentation', {})
            
            # Apply basic parameters
            ui_mappings = {
                'num_variations': aug_config.get('num_variations', 2),
                'target_count': aug_config.get('target_count', 500),
                'output_prefix': aug_config.get('output_prefix', 'aug'),
                'balance_classes': aug_config.get('balance_classes', False),
                
                # Advanced parameters dengan nilai moderat
                'fliplr': aug_config.get('fliplr', 0.5),
                'degrees': aug_config.get('degrees', 10),
                'translate': aug_config.get('translate', 0.1),
                'scale': aug_config.get('scale', 0.1),
                'hsv_h': aug_config.get('hsv_h', 0.015),
                'hsv_s': aug_config.get('hsv_s', 0.7),
                'brightness': aug_config.get('brightness', 0.2),
                'contrast': aug_config.get('contrast', 0.2),
                
                # Types dan split
                'target_split': aug_config.get('target_split', 'train')
            }
            
            # Apply values ke UI widgets
            for ui_key, value in ui_mappings.items():
                widget = self.ui_components.get(ui_key)
                if widget and hasattr(widget, 'value'):
                    widget.value = value
            
            # Special handling untuk augmentation types
            aug_types_widget = self.ui_components.get('augmentation_types')
            if aug_types_widget and hasattr(aug_types_widget, 'value'):
                types = aug_config.get('types', ['combined'])
                aug_types_widget.value = list(types)  # Ensure it's a list
            
            return True
            
        except Exception as e:
            return False
    
    def get_default_config(self) -> Dict[str, Any]:
        """
        Dapatkan konfigurasi default dengan nilai yang moderat untuk penelitian.
        
        Returns:
            Dictionary konfigurasi default
        """
        return {
            'augmentation': {
                # Basic parameters
                'num_variations': 2,
                'target_count': 500,
                'output_prefix': 'aug',
                'balance_classes': False,
                
                # Advanced parameters - nilai moderat untuk penelitian
                'fliplr': 0.5,
                'degrees': 10,          # Reduced from 15 - lebih konservatif
                'translate': 0.1,       # Reduced from 0.15 - tidak terlalu ekstrim
                'scale': 0.1,           # Reduced from 0.15 - scaling minimal
                'hsv_h': 0.015,         # Reduced from 0.025 - hue shift minimal
                'hsv_s': 0.7,           # Saturation adjustment moderat
                'brightness': 0.2,      # Reduced from 0.3 - brightness moderat
                'contrast': 0.2,        # Reduced from 0.3 - contrast moderat
                
                # Pipeline research types
                'types': ['combined'],  # Default ke combined untuk penelitian
                'target_split': 'train',
                'intensity': 0.7,       # Moderate intensity
                'output_dir': 'data/augmented'
            },
            'data': {
                'dir': 'data'
            },
            'preprocessing': {
                'output_dir': 'data/preprocessed'
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validasi konfigurasi augmentasi.
        
        Args:
            config: Konfigurasi untuk divalidasi
            
        Returns:
            Result validation dengan status dan messages
        """
        validation_result = {
            'valid': True,
            'warnings': [],
            'errors': []
        }
        
        try:
            aug_config = config.get('augmentation', {})
            
            # Validate basic parameters
            if aug_config.get('num_variations', 0) <= 0:
                validation_result['errors'].append('Jumlah variasi harus > 0')
            
            if aug_config.get('target_count', 0) <= 0:
                validation_result['errors'].append('Target count harus > 0')
            
            # Validate advanced parameters untuk menghindari nilai ekstrim
            advanced_limits = {
                'degrees': (0, 30, 'Rotasi'),
                'translate': (0, 0.3, 'Translasi'),
                'scale': (0, 0.3, 'Skala'),
                'hsv_h': (0, 0.1, 'HSV Hue'),
                'brightness': (0, 0.5, 'Brightness'),
                'contrast': (0, 0.5, 'Contrast')
            }
            
            for param, (min_val, max_val, name) in advanced_limits.items():
                value = aug_config.get(param, 0)
                if value < min_val or value > max_val:
                    validation_result['warnings'].append(f'{name} nilai {value} di luar range optimal {min_val}-{max_val}')
            
            # Validate types
            types = aug_config.get('types', [])
            if not types or not isinstance(types, list):
                validation_result['errors'].append('Jenis augmentasi harus dipilih minimal 1')
            
            # Set overall validity
            validation_result['valid'] = len(validation_result['errors']) == 0
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['errors'].append(f'Error validasi: {str(e)}')
        
        return validation_result

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