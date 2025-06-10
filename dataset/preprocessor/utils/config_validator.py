"""
File: smartcash/dataset/preprocessor/utils/config_validator.py
Deskripsi: Validator konfigurasi untuk modul preprocessor
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

class PreprocessingConfigValidator:
    """✅ Validator untuk konfigurasi preprocessing"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._default_config = None
        self._config_paths = [
            Path(__file__).parent.parent.parent.parent / 'configs' / 'preprocessing_config.yaml',
            Path('configs/preprocessing_config.yaml'),
            Path('preprocessing_config.yaml')
        ]
    
    def get_default_config(self) -> Dict[str, Any]:
        """Muat konfigurasi default dari file YAML dengan fallback"""
        if self._default_config is not None:
            return self._default_config
        
        # Coba muat dari file YAML
        for config_path in self._config_paths:
            try:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._default_config = yaml.safe_load(f)
                        self.logger.info(f"📋 Konfigurasi default dimuat dari: {config_path}")
                        return self._default_config
            except Exception as e:
                self.logger.debug(f"Gagal memuat {config_path}: {str(e)}")
                continue
        
        # Fallback ke konfigurasi default hardcoded
        self.logger.warning("⚠️ Menggunakan konfigurasi fallback - File YAML tidak ditemukan")
        self._default_config = self._get_fallback_config()
        return self._default_config
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Konfigurasi fallback jika file tidak ditemukan"""
        return {
            'preprocessing': {
                'enabled': True,
                'validation': {
                    'enabled': True,
                    'move_invalid': True,
                    'fix_issues': False
                },
                'normalization': {
                    'method': 'minmax',  # minmax|standard|imagenet|none
                    'target_size': [640, 640],
                    'preserve_aspect_ratio': False,
                    'denormalize': False
                },
                'output': {
                    'create_npy': True,
                    'organize_by_split': True
                }
            },
            'file_naming': {
                'preprocessed_pattern': 'pre_{nominal}_{uuid}_{increment}',
                'preserve_uuid': True
            }
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validasi dan normalisasi konfigurasi
        
        Args:
            config: Konfigurasi yang akan divalidasi
            
        Returns:
            Dict: Konfigurasi yang sudah divalidasi atau dict dengan key 'error' jika validasi gagal
        """
        if not config:
            return {'error': 'Konfigurasi tidak boleh kosong'}
            
        # Lakukan deep copy untuk menghindari modifikasi input
        validated = self.get_default_config().copy()
        
        try:
            # Update dengan konfigurasi yang diberikan
            self._update_nested_dict(validated, config)
            
            # Validasi nilai-nilai penting
            if 'preprocessing' in config:
                prep = config['preprocessing']
                if 'normalization' in prep:
                    norm = prep['normalization']
                    if 'method' in norm and norm['method'] not in ['minmax', 'standard', 'imagenet', 'none']:
                        error_msg = f"Metode normalisasi tidak valid: {norm['method']}"
                        self.logger.error(error_msg)
                        return {'error': error_msg}
            
            return validated
            
        except Exception as e:
            error_msg = f"Gagal memvalidasi konfigurasi: {str(e)}"
            self.logger.error(error_msg)
            return {'error': error_msg}
    
    def _update_nested_dict(self, original: Dict[str, Any], updates: Dict[str, Any]) -> None:
        """Update nested dictionary secara rekursif"""
        for key, value in updates.items():
            if key in original and isinstance(original[key], dict) and isinstance(value, dict):
                self._update_nested_dict(original[key], value)
            else:
                original[key] = value

# Global validator instance
_validator = PreprocessingConfigValidator()

def validate_preprocessing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validasi konfigurasi preprocessing
    
    Args:
        config: Konfigurasi yang akan divalidasi
        
    Returns:
        Dict: Konfigurasi yang sudah divalidasi
        
    Raises:
        ValueError: Jika konfigurasi tidak valid
    """
    if config is None:
        raise ValueError("Konfigurasi tidak boleh None")
        
    validated = _validator.validate_config(config)
    
    # Periksa apakah ada error dalam validasi
    if isinstance(validated, dict) and 'error' in validated:
        raise ValueError(validated['error'])
        
    return validated

def get_default_preprocessing_config() -> Dict[str, Any]:
    """Dapatkan konfigurasi default"""
    return _validator.get_default_config()

def load_config_from_file(config_path: str) -> Dict[str, Any]:
    """Muat konfigurasi dari file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return validate_preprocessing_config(config)
    except Exception as e:
        get_logger(__name__).error(f"Gagal memuat konfigurasi dari {config_path}: {str(e)}")
        return get_default_preprocessing_config()

def reload_default_config() -> Dict[str, Any]:
    """Muat ulang konfigurasi default"""
    _validator._default_config = None
    return _validator.get_default_config()
