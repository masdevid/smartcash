"""
File: smartcash/dataset/augmentor/utils/config_validator.py
Deskripsi: Enhanced config validator dengan YAML loading dan validation
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

class ConfigValidator:
    """✅ Enhanced validator dengan YAML loading dan comprehensive validation"""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._default_config = None
        self._config_paths = [
            Path(__file__).parent.parent.parent.parent / 'configs' / 'augmentation_config.yaml',
            Path('configs/augmentation_config.yaml'),
            Path('augmentation_config.yaml')
        ]
    
    def get_default_config(self) -> Dict[str, Any]:
        """Load default config dengan caching dan fallback"""
        if self._default_config is not None:
            return self._default_config
        
        # Try to load dari YAML files
        for config_path in self._config_paths:
            try:
                if config_path.exists():
                    with open(config_path, 'r', encoding='utf-8') as f:
                        self._default_config = yaml.safe_load(f)
                        self.logger.info(f"📋 Default config loaded dari: {config_path}")
                        return self._default_config
            except Exception as e:
                self.logger.debug(f"Failed to load {config_path}: {str(e)}")
                continue
        
        # Fallback ke hardcoded config
        self.logger.warning("⚠️ Menggunakan fallback config - YAML file tidak ditemukan")
        self._default_config = self._get_fallback_config()
        return self._default_config
    
    def validate_and_normalize(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate dan normalize dengan deep merge"""
        default_config = self.get_default_config()
        
        # Validate input config
        validation_errors = self._validate_config_structure(config)
        if validation_errors:
            self.logger.warning(f"⚠️ Config validation warnings: {', '.join(validation_errors)}")
        
        # Deep merge dengan defaults
        normalized = self._deep_merge(default_config, config)
        
        # Post-validation cleanup
        normalized = self._normalize_values(normalized)
        
        return normalized
    
    def load_config_from_file(self, config_path: str) -> Optional[Dict[str, Any]]:
        """Load config dari file tertentu"""
        try:
            path = Path(config_path)
            if not path.exists():
                self.logger.error(f"❌ Config file tidak ditemukan: {config_path}")
                return None
            
            with open(path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                self.logger.info(f"📋 Config loaded dari: {config_path}")
                return config
                
        except Exception as e:
            self.logger.error(f"❌ Error loading config {config_path}: {str(e)}")
            return None
    
    def _validate_config_structure(self, config: Dict[str, Any]) -> list:
        """Validate struktur config dan return warnings"""
        warnings = []
        
        # Check required sections
        required_sections = ['data', 'augmentation', 'preprocessing']
        for section in required_sections:
            if section not in config:
                warnings.append(f"Missing section: {section}")
        
        # Validate augmentation config
        if 'augmentation' in config:
            aug_config = config['augmentation']
            
            # Check types
            if 'types' in aug_config:
                valid_types = ['lighting', 'position', 'combined', 'geometric', 'color', 'noise']
                invalid_types = [t for t in aug_config['types'] if t not in valid_types]
                if invalid_types:
                    warnings.append(f"Invalid augmentation types: {invalid_types}")
            
            # Check numeric ranges
            if 'intensity' in aug_config:
                intensity = aug_config['intensity']
                if not (0.0 <= intensity <= 1.0):
                    warnings.append("Intensity should be between 0.0 and 1.0")
            
            if 'num_variations' in aug_config:
                if aug_config['num_variations'] < 1:
                    warnings.append("num_variations should be >= 1")
        
        # Validate preprocessing config
        if 'preprocessing' in config and 'normalization' in config['preprocessing']:
            norm_config = config['preprocessing']['normalization']
            
            if 'method' in norm_config:
                valid_methods = ['minmax', 'standard', 'imagenet', 'none']
                if norm_config['method'] not in valid_methods:
                    warnings.append(f"Invalid normalization method: {norm_config['method']}")
        
        # Validate validation config
        if 'validation' in config:
            val_config = config['validation']
            
            if 'min_bbox_size' in val_config:
                min_bbox_size = val_config['min_bbox_size']
                if not (0.0 <= min_bbox_size <= 1.0):
                    warnings.append("min_bbox_size should be between 0.0 and 1.0")
            
            if 'min_valid_boxes' in val_config:
                min_valid_boxes = val_config['min_valid_boxes']
                if min_valid_boxes < 0:
                    warnings.append("min_valid_boxes should be >= 0")
        
        return warnings
    
    def _normalize_values(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize dan clamp values ke range yang valid"""
        # Clamp intensity
        if 'augmentation' in config and 'intensity' in config['augmentation']:
            config['augmentation']['intensity'] = max(0.0, min(1.0, config['augmentation']['intensity']))
        
        # Ensure positive values
        if 'augmentation' in config:
            aug_config = config['augmentation']
            if 'num_variations' in aug_config:
                aug_config['num_variations'] = max(1, aug_config['num_variations'])
            if 'target_count' in aug_config:
                aug_config['target_count'] = max(1, aug_config['target_count'])
        
        # Ensure target_size is list of 2 integers
        if 'preprocessing' in config and 'normalization' in config['preprocessing']:
            norm_config = config['preprocessing']['normalization']
            if 'target_size' in norm_config:
                target_size = norm_config['target_size']
                if isinstance(target_size, (list, tuple)) and len(target_size) >= 2:
                    norm_config['target_size'] = [int(target_size[0]), int(target_size[1])]
                else:
                    norm_config['target_size'] = [640, 640]  # Default
        
        # Normalize validation config values
        if 'validation' in config:
            val_config = config['validation']
            if 'min_bbox_size' in val_config:
                val_config['min_bbox_size'] = max(0.0, min(1.0, float(val_config['min_bbox_size'])))
            if 'min_valid_boxes' in val_config:
                val_config['min_valid_boxes'] = max(0, int(val_config['min_valid_boxes']))
        
        return config
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge dengan list handling"""
        import copy
        result = copy.deepcopy(base)
        
        for key, value in override.items():
            if key in result:
                if isinstance(result[key], dict) and isinstance(value, dict):
                    result[key] = self._deep_merge(result[key], value)
                else:
                    result[key] = value
            else:
                result[key] = value
        
        return result
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Hardcoded fallback config"""
        return {
            'data': {
                'dir': 'data',
                'splits': {'train': 'data/train', 'valid': 'data/valid', 'test': 'data/test'},
                'output': {'augmented': 'data/augmented', 'preprocessed': 'data/preprocessed'}
            },
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 500,
                'intensity': 0.7,
                'balance_classes': True,
                'target_split': 'train'
            },
            'preprocessing': {
                'normalization': {
                    'enabled': True,
                    'method': 'minmax',
                    'denormalize': False,
                    'target_size': [640, 640]
                }
            },
            'validation': {
                'enabled': True,
                'min_bbox_size': 0.001,
                'min_valid_boxes': 1
            },
            'balancing': {
                'enabled': True,
                'strategy': 'weighted',
                'layer_weights': {'layer1': 1.0, 'layer2': 0.8, 'layer3': 0.5}
            },
            'file_processing': {'max_workers': 4, 'batch_size': 100},
            'progress': {'enabled': True, 'granular_tracking': True}
        }


# Global validator instance
_validator = ConfigValidator()

def validate_augmentation_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """One-liner validation"""
    return _validator.validate_and_normalize(config)

def get_default_augmentation_config() -> Dict[str, Any]:
    """One-liner default config"""
    return _validator.get_default_config()

def load_config_from_file(config_path: str) -> Optional[Dict[str, Any]]:
    """One-liner file loading"""
    return _validator.load_config_from_file(config_path)

def reload_default_config() -> Dict[str, Any]:
    """Force reload default config"""
    _validator._default_config = None
    return _validator.get_default_config()