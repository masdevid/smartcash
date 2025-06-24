"""
File: smartcash/dataset/preprocessor/utils/config_validator.py
Deskripsi: Enhanced config validator dengan fallback dan validation yang lebih robust
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List

from smartcash.common.logger import get_logger

class PreprocessingConfigValidator:
    """✅ Enhanced validator dengan robust fallback dan validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._default_config = None
        self._config_paths = [
            Path(__file__).parent.parent.parent.parent / 'configs' / 'preprocessing_config.yaml',
            Path('configs/preprocessing_config.yaml'),
            Path('preprocessing_config.yaml')
        ]
    
    def get_default_config(self) -> Dict[str, Any]:
        """🔧 Load default config dengan enhanced fallback"""
        if self._default_config is not None:
            return self._default_config
        
        # Try load dari YAML files
        for config_path in self._config_paths:
            try:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        config = yaml.safe_load(f)
                        if config and self._validate_config_structure(config):
                            self._default_config = config
                            self.logger.debug(f"📋 Config loaded from: {config_path}")
                            return self._default_config
            except Exception as e:
                self.logger.debug(f"❌ Failed to load {config_path}: {str(e)}")
        
        # Enhanced fallback config
        self.logger.info("📋 Using enhanced fallback config")
        self._default_config = self._get_enhanced_fallback_config()
        return self._default_config
    
    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """✅ Enhanced validation dengan detailed checks"""
        if not config:
            return self._create_validation_error('Konfigurasi tidak boleh kosong')
        
        try:
            validated = self._deep_merge(self.get_default_config(), config)
            
            # Enhanced validation checks
            validation_errors = []
            validation_errors.extend(self._validate_preprocessing_section(validated.get('preprocessing', {})))
            validation_errors.extend(self._validate_performance_section(validated.get('performance', {})))
            validation_errors.extend(self._validate_data_section(validated.get('data', {})))
            
            if validation_errors:
                error_msg = '; '.join(validation_errors)
                return self._create_validation_error(error_msg)
            
            return validated
            
        except Exception as e:
            return self._create_validation_error(f"Validation error: {str(e)}")
    
    def _validate_preprocessing_section(self, preprocessing: Dict[str, Any]) -> List[str]:
        """✅ Validate preprocessing section"""
        errors = []
        
        # Validation settings
        if 'validation' in preprocessing:
            validation = preprocessing['validation']
            if not isinstance(validation.get('enabled', True), bool):
                errors.append("validation.enabled harus boolean")
        
        # Normalization settings
        if 'normalization' in preprocessing:
            norm = preprocessing['normalization']
            valid_methods = ['minmax', 'standard', 'imagenet', 'none']
            method = norm.get('method', 'minmax')
            
            if method not in valid_methods:
                errors.append(f"normalization.method '{method}' tidak valid. Valid: {valid_methods}")
            
            target_size = norm.get('target_size', [640, 640])
            if not isinstance(target_size, list) or len(target_size) != 2:
                errors.append("normalization.target_size harus list [width, height]")
            elif not all(isinstance(x, int) and x > 0 for x in target_size):
                errors.append("normalization.target_size harus positive integers")
        
        # Target splits validation
        target_splits = preprocessing.get('target_splits', ['train', 'valid'])
        valid_splits = ['train', 'valid', 'test', 'all']
        
        if isinstance(target_splits, str):
            if target_splits not in valid_splits:
                errors.append(f"target_splits '{target_splits}' tidak valid")
        elif isinstance(target_splits, list):
            invalid_splits = [s for s in target_splits if s not in valid_splits[:-1]]  # Exclude 'all'
            if invalid_splits:
                errors.append(f"Invalid splits: {invalid_splits}")
        
        return errors
    
    def _validate_performance_section(self, performance: Dict[str, Any]) -> List[str]:
        """✅ Validate performance section"""
        errors = []
        
        batch_size = performance.get('batch_size', 32)
        if not isinstance(batch_size, int) or batch_size <= 0:
            errors.append("performance.batch_size harus positive integer")
        
        return errors
    
    def _validate_data_section(self, data: Dict[str, Any]) -> List[str]:
        """✅ Validate data section"""
        errors = []
        
        # Check local paths jika ada
        if 'local' in data:
            local_paths = data['local']
            for split, path in local_paths.items():
                if not isinstance(path, (str, Path)):
                    errors.append(f"data.local.{split} harus string atau Path")
        
        return errors
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> bool:
        """✅ Basic structure validation"""
        required_sections = ['preprocessing']
        return all(section in config for section in required_sections)
    
    def _get_enhanced_fallback_config(self) -> Dict[str, Any]:
        """🔧 Enhanced fallback config dengan comprehensive settings"""
        return {
            'preprocessing': {
                'enabled': True,
                'target_splits': ['train', 'valid'],
                'output_dir': 'data/preprocessed',
                'validation': {
                    'enabled': True,
                    'move_invalid': True,
                    'fix_issues': False,
                    'check_image_quality': True,
                    'check_labels': True,
                    'check_coordinates': True
                },
                'normalization': {
                    'enabled': True,
                    'method': 'minmax',
                    'target_size': [640, 640],
                    'preserve_aspect_ratio': True,
                    'normalize_pixel_values': True
                },
                'output': {
                    'create_npy': True,
                    'organize_by_split': True
                }
            },
            'performance': {
                'batch_size': 32,
                'use_gpu': True,
                'threading': {
                    'io_workers': 8,
                    'cpu_workers': None
                }
            },
            'data': {
                'dir': 'data',
                'local': {}
            },
            'file_naming': {
                'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{increment}_{variance}',
                'preserve_uuid': True
            }
        }
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Deep merge configs"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _create_validation_error(self, message: str) -> Dict[str, str]:
        """❌ Create validation error response"""
        self.logger.error(f"❌ Config validation: {message}")
        return {'error': message}

# Global validator instance
_validator = PreprocessingConfigValidator()

def validate_preprocessing_config(config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """✅ Main validation function"""
    if config is None:
        raise ValueError("Konfigurasi tidak boleh None")
    
    validated = _validator.validate_config(config)
    
    if 'error' in validated:
        raise ValueError(validated['error'])
    
    return validated

def get_default_preprocessing_config() -> Dict[str, Any]:
    """🔧 Get default config"""
    return _validator.get_default_config()

def reload_default_config() -> Dict[str, Any]:
    """🔄 Reload default config"""
    _validator._default_config = None
    return _validator.get_default_config()