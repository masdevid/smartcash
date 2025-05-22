"""
File: smartcash/ui/dataset/augmentation/utils/parameter_extractor.py
Deskripsi: Konsolidasi ekstraksi parameter dari UI components untuk menghindari duplikasi
"""

from typing import Dict, Any, List, Optional, Tuple
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class AugmentationParameterExtractor:
    """Extractor tunggal untuk parameter augmentasi dari UI components."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi parameter extractor.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
    
    def extract_all_parameters(self) -> Dict[str, Any]:
        """
        Ekstrak semua parameter augmentasi dari UI components.
        
        Returns:
            Dictionary parameter lengkap untuk augmentasi
        """
        return {
            'basic_params': self._extract_basic_parameters(),
            'advanced_params': self._extract_advanced_parameters(),
            'types_params': self._extract_types_parameters(),
            'validation_params': self._extract_validation_parameters(),
            'path_params': self._extract_path_parameters()
        }
    
    def extract_service_parameters(self) -> Dict[str, Any]:
        """
        Ekstrak parameter dalam format yang dibutuhkan service.
        
        Returns:
            Dictionary parameter untuk AugmentationService
        """
        all_params = self.extract_all_parameters()
        
        return {
            'data_dir': all_params['path_params']['data_dir'],
            'split': all_params['types_params']['target_split'],
            'types': all_params['types_params']['augmentation_types'],
            'num_variations': all_params['basic_params']['num_variations'],
            'target_count': all_params['basic_params']['target_count'],
            'output_prefix': all_params['basic_params']['output_prefix'],
            'balance_classes': all_params['basic_params']['balance_classes'],
            'validate_results': all_params['validation_params']['validate_results'],
            'create_symlinks': True,
            'num_workers': 1  # Synchronous processing
        }
    
    def _extract_basic_parameters(self) -> Dict[str, Any]:
        """Ekstrak parameter dasar dari UI."""
        return {
            'num_variations': self._get_widget_value(['num_variations', 'variations'], 2),
            'target_count': self._get_widget_value(['target_count', 'count'], 500),
            'output_prefix': self._get_widget_value(['output_prefix', 'prefix'], 'aug_'),
            'balance_classes': self._get_widget_value(['balance_classes'], False)
        }
    
    def _extract_advanced_parameters(self) -> Dict[str, Any]:
        """Ekstrak parameter lanjutan dari UI."""
        return {
            'position': {
                'fliplr': self._get_widget_value(['fliplr'], 0.5),
                'degrees': self._get_widget_value(['degrees'], 15),
                'translate': self._get_widget_value(['translate'], 0.15),
                'scale': self._get_widget_value(['scale'], 0.15),
                'shear_max': self._get_widget_value(['shear_max'], 10)
            },
            'lighting': {
                'hsv_h': self._get_widget_value(['hsv_h'], 0.025),
                'hsv_s': self._get_widget_value(['hsv_s'], 0.7),
                'hsv_v': self._get_widget_value(['hsv_v'], 0.4),
                'contrast': [
                    self._get_widget_value(['contrast_min'], 0.7),
                    self._get_widget_value(['contrast_max'], 1.3)
                ],
                'brightness': [
                    self._get_widget_value(['brightness_min'], 0.7),
                    self._get_widget_value(['brightness_max'], 1.3)
                ],
                'blur': self._get_widget_value(['blur'], 0.2),
                'noise': self._get_widget_value(['noise'], 0.1)
            }
        }
    
    def _extract_types_parameters(self) -> Dict[str, Any]:
        """Ekstrak parameter jenis augmentasi dan target split."""
        aug_types = self._get_widget_value(['augmentation_types', 'types'], ['combined'])
        if isinstance(aug_types, (tuple, list)):
            aug_types = list(aug_types)
        else:
            aug_types = [aug_types] if aug_types else ['combined']
            
        return {
            'augmentation_types': aug_types,
            'target_split': self._get_widget_value(['target_split', 'split'], 'train')
        }
    
    def _extract_validation_parameters(self) -> Dict[str, Any]:
        """Ekstrak parameter validasi."""
        return {
            'validate_results': self._get_widget_value(['validate_results'], True),
            'process_bboxes': True  # Always true untuk augmentasi
        }
    
    def _extract_path_parameters(self) -> Dict[str, Any]:
        """Ekstrak parameter path."""
        return {
            'data_dir': self.ui_components.get('data_dir', 'data'),
            'augmented_dir': self.ui_components.get('augmented_dir', 'data/augmented'),
            'output_dir': self.ui_components.get('output_dir', 'data/augmented')
        }
    
    def _get_widget_value(self, possible_keys: List[str], default_value: Any) -> Any:
        """
        Dapatkan nilai widget dengan multiple fallback keys.
        
        Args:
            possible_keys: List kemungkinan key
            default_value: Nilai default
            
        Returns:
            Nilai widget atau default
        """
        for key in possible_keys:
            if key in self.ui_components and hasattr(self.ui_components[key], 'value'):
                return self.ui_components[key].value
        return default_value
    
    def validate_parameters(self) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validasi parameter yang diekstrak.
        
        Returns:
            Tuple (is_valid, error_message, validated_params)
        """
        try:
            params = self.extract_service_parameters()
            
            # Validasi augmentation types
            if not params['types'] or not isinstance(params['types'], list):
                return False, "Jenis augmentasi tidak valid", {}
            
            # Validasi num_variations
            if params['num_variations'] <= 0 or params['num_variations'] > 10:
                return False, "Jumlah variasi harus antara 1-10", {}
            
            # Validasi target_count
            if params['target_count'] <= 0 or params['target_count'] > 5000:
                return False, "Target count harus antara 1-5000", {}
            
            # Validasi split
            if params['split'] not in ['train', 'valid', 'test']:
                return False, f"Split tidak valid: {params['split']}", {}
            
            # Validasi output_prefix
            if not params['output_prefix'] or not str(params['output_prefix']).strip():
                return False, "Output prefix tidak boleh kosong", {}
            
            log_message(self.ui_components, "✅ Parameter validation berhasil", "success")
            return True, "Parameter valid", params
            
        except Exception as e:
            return False, f"Error validasi parameter: {str(e)}", {}

def get_parameter_extractor(ui_components: Dict[str, Any]) -> AugmentationParameterExtractor:
    """
    Factory function untuk mendapatkan parameter extractor.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance AugmentationParameterExtractor
    """
    return AugmentationParameterExtractor(ui_components)