"""
File: smartcash/ui/dataset/preprocessing/utils/config_extractor.py
Deskripsi: Ekstraksi config dari UI components untuk service layer integration
"""

from typing import Dict, Any
from smartcash.dataset.preprocessor.utils.preprocessing_config import PreprocessingConfig

class UIConfigExtractor:
    """Extractor untuk mengambil dan normalize config dari UI components."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
    
    def extract_processing_parameters(self) -> Dict[str, Any]:
        """Extract parameters dari UI untuk processing."""
        try:
            # Extract dari UI widgets
            ui_values = self._extract_ui_values()
            
            # Create config manager untuk validation
            base_config = self._get_base_config()
            config_manager = PreprocessingConfig(base_config, self.logger)
            
            # Extract dan validate
            extracted_config = config_manager.extract_ui_config(ui_values)
            validation_result = config_manager.validate_config_parameters(extracted_config)
            
            if not validation_result['valid']:
                self.logger and self.logger.warning(f"⚠️ Config issues: {validation_result['errors']}")
            
            return {
                'config': validation_result['normalized_config'],
                'split': extracted_config.get('split', 'all'),
                'force_reprocess': extracted_config.get('force_reprocess', False),
                'summary': config_manager.get_config_summary(validation_result['normalized_config']),
                'valid': validation_result['valid']
            }
            
        except Exception as e:
            self.logger and self.logger.error(f"❌ Config extraction error: {str(e)}")
            return self._get_safe_defaults()
    
    def get_full_config(self) -> Dict[str, Any]:
        """Get full config untuk service instantiation."""
        params = self.extract_processing_parameters()
        
        return {
            'data': {'dir': 'data'},
            'preprocessing': {
                **params['config'],
                'output_dir': 'data/preprocessed'
            }
        }
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> None:
        """Apply config values ke UI components."""
        try:
            preprocessing_config = config.get('preprocessing', {})
            
            # Update resolution dropdown
            if 'resolution_dropdown' in self.ui_components:
                img_size = preprocessing_config.get('img_size', [640, 640])
                resolution_str = f"{img_size[0]}x{img_size[1]}"
                if resolution_str in ['320x320', '416x416', '512x512', '640x640']:
                    self.ui_components['resolution_dropdown'].value = resolution_str
            
            # Update normalization dropdown
            if 'normalization_dropdown' in self.ui_components:
                normalization = preprocessing_config.get('normalization_method', 'minmax')
                if not preprocessing_config.get('normalize', True):
                    normalization = 'none'
                self.ui_components['normalization_dropdown'].value = normalization
            
            # Update worker slider
            if 'worker_slider' in self.ui_components:
                workers = preprocessing_config.get('num_workers', 4)
                self.ui_components['worker_slider'].value = max(1, min(10, workers))
            
            # Update split dropdown
            if 'split_dropdown' in self.ui_components:
                split = preprocessing_config.get('split', 'all')
                self.ui_components['split_dropdown'].value = split
                
        except Exception as e:
            self.logger and self.logger.error(f"❌ UI config application error: {str(e)}")
    
    def _extract_ui_values(self) -> Dict[str, Any]:
        """Extract values dari UI widgets dengan fallbacks."""
        return {
            'resolution': getattr(self.ui_components.get('resolution_dropdown'), 'value', '640x640'),
            'normalization': getattr(self.ui_components.get('normalization_dropdown'), 'value', 'minmax'),
            'num_workers': getattr(self.ui_components.get('worker_slider'), 'value', 4),
            'split': getattr(self.ui_components.get('split_dropdown'), 'value', 'all'),
            'preserve_aspect_ratio': True,
            'force_reprocess': False
        }
    
    def _get_base_config(self) -> Dict[str, Any]:
        """Get base config untuk preprocessing."""
        return {
            'data': {'dir': 'data'},
            'preprocessing': {
                'img_size': [640, 640],
                'normalize': True,
                'num_workers': 4,
                'output_dir': 'data/preprocessed'
            }
        }
    
    def _get_safe_defaults(self) -> Dict[str, Any]:
        """Get safe default parameters."""
        return {
            'config': {'img_size': [640, 640], 'normalize': True, 'num_workers': 4},
            'split': 'all',
            'force_reprocess': False,
            'summary': '🖼️ Size: 640x640 | 🔧 Normalize: Yes | ⚙️ Workers: 4 | 📂 Split: all',
            'valid': True
        }

def get_config_extractor(ui_components: Dict[str, Any]) -> UIConfigExtractor:
    """Factory untuk membuat config extractor instance."""
    return UIConfigExtractor(ui_components)