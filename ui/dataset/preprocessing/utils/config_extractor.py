"""
File: smartcash/ui/dataset/preprocessing/utils/config_extractor.py
Deskripsi: Fixed config extractor dengan normalization mapping dan validation yang tepat
"""

from typing import Dict, Any

class ConfigExtractor:
    """Fixed utility untuk extract dan manage konfigurasi preprocessing dengan normalization mapping."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        self.ui_components = ui_components
    
    def _extract_normalization_method(self, config_value: Any) -> str:
        """
        Extract normalization method dari berbagai format config.
        
        Args:
            config_value: Nilai normalization dari config (bisa string atau dict)
            
        Returns:
            String method yang valid untuk dropdown
        """
        # Jika string sederhana
        if isinstance(config_value, str):
            return config_value if config_value in ['minmax', 'standard', 'none'] else 'minmax'
        
        # Jika dict dengan method
        if isinstance(config_value, dict):
            method = config_value.get('method', 'minmax')
            enabled = config_value.get('enabled', True)
            
            # Map method names
            method_mapping = {
                'minmax': 'minmax',
                'min-max': 'minmax', 
                'zscore': 'standard',
                'z-score': 'standard',
                'standardization': 'standard',
                'normalize': 'minmax',
                'whitening': 'standard'
            }
            
            if not enabled:
                return 'none'
            
            return method_mapping.get(method.lower(), 'minmax')
        
        # Default fallback
        return 'minmax'
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Extract konfigurasi preprocessing dari UI components dengan error handling."""
        try:
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
        except Exception as e:
            # Fallback config jika UI components bermasalah
            return {
                'img_size': [640, 640],
                'normalization': 'minmax',
                'num_workers': 4,
                'split': 'all',
                'normalize': True,
                'preserve_aspect_ratio': True
            }
    
    def get_current_ui_config(self) -> Dict[str, Any]:
        """Extract konfigurasi saat ini untuk save operations dengan error handling."""
        try:
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
        except Exception:
            # Fallback untuk save operations
            return {
                'preprocessing': {
                    'img_size': [640, 640],
                    'normalization': 'minmax',
                    'num_workers': 4,
                    'split': 'all'
                }
            }
    
    def apply_config_to_ui(self, config: Dict[str, Any]) -> list:
        """Apply konfigurasi ke UI components dengan improved normalization handling."""
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
                    errors.append(f"⚠️ Resolution {resolution_str} tidak tersedia, menggunakan default")
                    self.ui_components['resolution_dropdown'].value = '640x640'
            else:
                errors.append("⚠️ Format img_size tidak valid, menggunakan default")
                self.ui_components['resolution_dropdown'].value = '640x640'
            
            # Improved normalization handling
            normalization_raw = preprocessing_config.get('normalization', 'minmax')
            normalization_method = self._extract_normalization_method(normalization_raw)
            
            if normalization_method in self.ui_components['normalization_dropdown'].options:
                self.ui_components['normalization_dropdown'].value = normalization_method
            else:
                errors.append(f"⚠️ Normalization method '{normalization_method}' tidak tersedia, menggunakan 'minmax'")
                self.ui_components['normalization_dropdown'].value = 'minmax'
            
            # Workers dengan range validation
            num_workers = preprocessing_config.get('num_workers', 4)
            if isinstance(num_workers, int) and 1 <= num_workers <= 10:
                self.ui_components['worker_slider'].value = num_workers
            else:
                errors.append(f"⚠️ num_workers {num_workers} diluar range 1-10, menggunakan 4")
                self.ui_components['worker_slider'].value = 4
            
            # Split dengan validation dan val->valid mapping
            split = preprocessing_config.get('split', 'all')
            if split == 'val':  # Map val to valid
                split = 'valid'
            if split in self.ui_components['split_dropdown'].options:
                self.ui_components['split_dropdown'].value = split
            else:
                available_splits = ', '.join(self.ui_components['split_dropdown'].options)
                errors.append(f"⚠️ Split '{split}' tidak tersedia (tersedia: {available_splits}), menggunakan 'all'")
                self.ui_components['split_dropdown'].value = 'all'
                
        except Exception as e:
            errors.append(f"❌ Error applying config: {str(e)}")
        
        return errors
    
    def validate_config(self, config: Dict[str, Any]) -> list:
        """Validate konfigurasi dan return list errors dengan improved validation."""
        errors = []
        preprocessing_config = config.get('preprocessing', {})
        
        # Validate img_size
        img_size = preprocessing_config.get('img_size', [])
        if not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
            errors.append("❌ img_size harus berupa list/tuple dengan 2 elemen")
        elif not all(isinstance(x, int) and x > 0 for x in img_size):
            errors.append("❌ img_size harus berupa integer positif")
        
        # Validate num_workers
        num_workers = preprocessing_config.get('num_workers', 0)
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 10:
            errors.append("❌ num_workers harus antara 1-10")
        
        # Improved normalization validation
        normalization = preprocessing_config.get('normalization', '')
        if isinstance(normalization, dict):
            method = normalization.get('method', 'minmax')
            valid_methods = ['minmax', 'min-max', 'standard', 'zscore', 'z-score', 'standardization', 'whitening', 'none']
            if method.lower() not in valid_methods:
                errors.append(f"❌ normalization method '{method}' tidak valid")
        elif isinstance(normalization, str):
            valid_normalizations = ['none', 'minmax', 'standard']
            if normalization not in valid_normalizations:
                errors.append(f"❌ normalization '{normalization}' harus salah satu dari: {', '.join(valid_normalizations)}")
        else:
            errors.append("❌ normalization harus berupa string atau dict")
        
        return errors
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Get summary konfigurasi untuk logging dengan improved formatting."""
        preprocessing_config = config.get('preprocessing', {})
        img_size = preprocessing_config.get('img_size', [])
        
        # Format normalization summary
        normalization_raw = preprocessing_config.get('normalization', 'unknown')
        if isinstance(normalization_raw, dict):
            method = normalization_raw.get('method', 'unknown')
            enabled = normalization_raw.get('enabled', True)
            norm_summary = f"{method} ({'enabled' if enabled else 'disabled'})"
        else:
            norm_summary = str(normalization_raw)
        
        parts = [
            f"🖼️ Resolusi: {img_size[0]}x{img_size[1]}" if len(img_size) == 2 else "🖼️ Resolusi: invalid",
            f"🔧 Normalisasi: {norm_summary}",
            f"⚙️ Workers: {preprocessing_config.get('num_workers', 0)}",
            f"📂 Split: {preprocessing_config.get('split', 'unknown')}"
        ]
        
        return " | ".join(parts)

def get_config_extractor(ui_components: Dict[str, Any]) -> ConfigExtractor:
    """Factory function untuk mendapatkan fixed config extractor."""
    if 'config_extractor' not in ui_components:
        ui_components['config_extractor'] = ConfigExtractor(ui_components)
    return ui_components['config_extractor']