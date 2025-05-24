"""
File: smartcash/ui/dataset/preprocessing/utils/config_extractor.py
Deskripsi: Utility untuk extract dan manage konfigurasi preprocessing dari UI components
"""

from typing import Dict, Any

class ConfigExtractor:
    """Utility untuk extract dan manage konfigurasi preprocessing."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Extract konfigurasi preprocessing dari UI components."""
        resolution = self.ui_components['resolution_dropdown'].value.split('x')
        
        return {
            'img_size': [int(resolution[0]), int(resolution[1])],
            'normalization': self.ui_components['normalization_dropdown'].value,
            'num_workers': self.ui_components['worker_slider'].value,
            'split': self.ui_components['split_dropdown'].value,
            'raw_dataset_dir': self.ui_components.get('data_dir', 'data'),
            'preprocessed_dir': self.ui_components.get('preprocessed_dir', 'data/preprocessed'),
            'normalize': self.ui_components['normalization_dropdown'].value != 'none',
            'preserve_aspect_ratio': True
        }
    
    def get_current_ui_config(self) -> Dict[str, Any]:
        """Extract konfigurasi saat ini untuk save operations."""
        resolution = self.ui_components['resolution_dropdown'].value.split('x')
        
        return {
            'preprocessing': {
                'img_size': [int(resolution[0]), int(resolution[1])],
                'normalization': self.ui_components['normalization_dropdown'].value,
                'num_workers': self.ui_components['worker_slider'].value,
                'split': self.ui_components['split_dropdown'].value,
                'normalize': self.ui_components['normalization_dropdown'].value != 'none',
                'preserve_aspect_ratio': True
            },
            'paths': {
                'data_dir': self.ui_components.get('data_dir'),
                'preprocessed_dir': self.ui_components.get('preprocessed_dir')
            }
        }
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> list:
        """Apply konfigurasi ke UI components dengan validation."""
        errors = []
        
        try:
            preprocessing_config = config.get('preprocessing', {})
            
            # Resolution dengan validation
            img_size = preprocessing_config.get('img_size', [640, 640])
            if isinstance(img_size, (list, tuple)) and len(img_size) == 2:
                resolution_str = f"{img_size[0]}x{img_size[1]}"
                if resolution_str in self.ui_components['resolution_dropdown'].options:
                    self.ui_components['resolution_dropdown'].value = resolution_str
                else:
                    errors.append(f"Resolution {resolution_str} tidak tersedia")
            else:
                errors.append("img_size format tidak valid")
            
            # Normalization dengan validation
            normalization = preprocessing_config.get('normalization', 'minmax')
            if normalization in self.ui_components['normalization_dropdown'].options:
                self.ui_components['normalization_dropdown'].value = normalization
            else:
                errors.append(f"Normalization {normalization} tidak tersedia")
            
            # Workers dengan range validation
            num_workers = preprocessing_config.get('num_workers', 4)
            if isinstance(num_workers, int) and 1 <= num_workers <= 10:
                self.ui_components['worker_slider'].value = num_workers
            else:
                errors.append("num_workers harus antara 1-10")
            
            # Split dengan validation dan val->valid mapping
            split = preprocessing_config.get('split', 'all')
            if split == 'val':  # Map val to valid
                split = 'valid'
            if split in self.ui_components['split_dropdown'].options:
                self.ui_components['split_dropdown'].value = split
            else:
                errors.append(f"Split {split} tidak tersedia")
                
        except Exception as e:
            errors.append(f"Error applying config: {str(e)}")
        
        return errors
    
    def validate_config(self, config: Dict[str, Any]) -> list:
        """Validate konfigurasi dan return list errors."""
        errors = []
        preprocessing_config = config.get('preprocessing', {})
        
        # Validate img_size
        img_size = preprocessing_config.get('img_size', [])
        if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
            errors.append("img_size harus berupa list/tuple dengan 2 elemen")
        elif not all(isinstance(x, int) and x > 0 for x in img_size):
            errors.append("img_size harus berupa integer positif")
        
        # Validate num_workers
        num_workers = preprocessing_config.get('num_workers', 0)
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 10:
            errors.append("num_workers harus antara 1-10")
        
        # Validate normalization
        valid_normalizations = ['none', 'minmax', 'standard', 'robust']
        normalization = preprocessing_config.get('normalization', '')
        if normalization not in valid_normalizations:
            errors.append(f"normalization harus salah satu dari: {', '.join(valid_normalizations)}")
        
        return errors
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get summary konfigurasi untuk logging."""
        preprocessing_config = config.get('preprocessing', {})
        img_size = preprocessing_config.get('img_size', [])
        
        parts = [
            f"Resolusi: {img_size[0]}x{img_size[1]}" if len(img_size) == 2 else "Resolusi: invalid",
            f"Normalisasi: {preprocessing_config.get('normalization', 'unknown')}",
            f"Workers: {preprocessing_config.get('num_workers', 0)}",
            f"Split: {preprocessing_config.get('split', 'unknown')}"
        ]
        
        return " | ".join(parts)

def get_config_extractor(ui_components: Dict[str, Any]) -> ConfigExtractor:
    """Factory function untuk mendapatkan config extractor."""
    if 'config_extractor' not in ui_components:
        ui_components['config_extractor'] = ConfigExtractor(ui_components)
    return ui_components['config_extractor']