"""
File: smartcash/dataset/preprocessor/utils/preprocessing_config.py
Deskripsi: Unified configuration management untuk preprocessing dengan validation dan merging
"""

from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger
from smartcash.dataset.utils.dataset_constants import DEFAULT_IMG_SIZE


class PreprocessingConfig:
    """Unified configuration manager untuk preprocessing dengan validation dan smart merging."""
    
    def __init__(self, base_config: Dict[str, Any], logger=None):
        """Initialize config manager dengan base configuration."""
        self.base_config = base_config
        self.logger = logger or get_logger()
        
        # Extract preprocessing config
        self.preprocessing_config = base_config.get('preprocessing', {})
        self.data_config = base_config.get('data', {})
        
    def extract_ui_config(self, ui_values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract configuration dari UI values dengan normalization.
        
        Args:
            ui_values: Values dari UI components
            
        Returns:
            Dictionary normalized config
        """
        extracted_config = {}
        
        try:
            # Extract img_size dengan validation
            resolution_str = ui_values.get('resolution', '640x640')
            if 'x' in resolution_str:
                width, height = map(int, resolution_str.split('x'))
                extracted_config['img_size'] = [width, height]
            else:
                extracted_config['img_size'] = DEFAULT_IMG_SIZE
            
            # Extract normalization dengan advanced mapping
            normalization = ui_values.get('normalization', 'minmax')
            extracted_config['normalize'] = normalization != 'none'
            extracted_config['normalization_method'] = normalization
            
            # Extract workers dengan bounds checking
            num_workers = ui_values.get('num_workers', 4)
            extracted_config['num_workers'] = max(1, min(16, int(num_workers)))
            
            # Extract split dengan val->valid mapping
            split = ui_values.get('split', 'all')
            extracted_config['split'] = 'valid' if split == 'val' else split
            
            # Extract additional options
            extracted_config['preserve_aspect_ratio'] = ui_values.get('preserve_aspect_ratio', True)
            extracted_config['force_reprocess'] = ui_values.get('force_reprocess', False)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Config extraction error: {str(e)}")
            # Return safe defaults
            extracted_config = self._get_safe_defaults()
        
        return extracted_config
    
    def validate_config_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate configuration parameters dengan comprehensive checks.
        
        Args:
            config: Configuration untuk validation
            
        Returns:
            Dictionary validation result
        """
        validation_result = {'valid': True, 'errors': [], 'warnings': [], 'normalized_config': {}}
        
        # Validate img_size
        img_size = config.get('img_size', DEFAULT_IMG_SIZE)
        if isinstance(img_size, int):
            validation_result['normalized_config']['img_size'] = [img_size, img_size]
        elif isinstance(img_size, (list, tuple)) and len(img_size) == 2:
            if all(isinstance(x, int) and x > 0 for x in img_size):
                validation_result['normalized_config']['img_size'] = list(img_size)
            else:
                validation_result['errors'].append('img_size values harus integer positif')
                validation_result['normalized_config']['img_size'] = DEFAULT_IMG_SIZE
        else:
            validation_result['errors'].append('img_size harus integer atau [width, height]')
            validation_result['normalized_config']['img_size'] = DEFAULT_IMG_SIZE
        
        # Validate normalization
        normalize = config.get('normalize', True)
        if not isinstance(normalize, bool):
            validation_result['warnings'].append('normalize converted to boolean')
            normalize = bool(normalize)
        validation_result['normalized_config']['normalize'] = normalize
        
        # Validate num_workers
        num_workers = config.get('num_workers', 4)
        if not isinstance(num_workers, int) or num_workers < 1:
            validation_result['warnings'].append('num_workers adjusted to valid range')
            num_workers = 4
        elif num_workers > 16:
            validation_result['warnings'].append('num_workers capped at 16')
            num_workers = 16
        validation_result['normalized_config']['num_workers'] = num_workers
        
        # Validate split
        split = config.get('split', 'all')
        valid_splits = ['all', 'train', 'valid', 'test']
        if split not in valid_splits:
            validation_result['warnings'].append(f'split "{split}" normalized to "all"')
            split = 'all'
        validation_result['normalized_config']['split'] = split
        
        # Set overall validity
        validation_result['valid'] = len(validation_result['errors']) == 0
        
        return validation_result
    
    def merge_config_sources(self, *config_sources: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge multiple configuration sources dengan intelligent prioritization.
        
        Args:
            *config_sources: Variable config dictionaries untuk merge
            
        Returns:
            Dictionary merged configuration
        """
        merged_config = {}
        
        # Start with base defaults
        base_defaults = self._get_safe_defaults()
        merged_config.update(base_defaults)
        
        # Merge each source dengan priority order
        for source in config_sources:
            if isinstance(source, dict):
                merged_config.update(source)
        
        # Apply final normalization
        normalized_config = self._normalize_merged_config(merged_config)
        
        return normalized_config
    
    def prepare_processing_config(self, **override_params) -> Dict[str, Any]:
        """
        Prepare final processing configuration dengan override parameters.
        
        Args:
            **override_params: Parameters untuk override
            
        Returns:
            Dictionary final processing config
        """
        # Base configuration
        processing_config = {
            'img_size': self.preprocessing_config.get('img_size', DEFAULT_IMG_SIZE),
            'normalize': self.preprocessing_config.get('normalize', True),
            'preserve_aspect_ratio': self.preprocessing_config.get('preserve_aspect_ratio', True),
            'num_workers': self.preprocessing_config.get('num_workers', 4),
            'output_dir': self.preprocessing_config.get('output_dir', 'data/preprocessed'),
            'file_prefix': self.preprocessing_config.get('file_prefix', 'rp')
        }
        
        # Apply overrides
        for key, value in override_params.items():
            if key in processing_config:
                processing_config[key] = value
        
        # Validate final config
        validation_result = self.validate_config_parameters(processing_config)
        if not validation_result['valid']:
            self.logger.warning(f"⚠️ Config validation issues: {validation_result['errors']}")
        
        return validation_result['normalized_config']
    
    def get_config_summary(self, config: Dict[str, Any]) -> str:
        """Generate human-readable config summary."""
        img_size = config.get('img_size', [640, 640])
        normalize = config.get('normalize', True)
        num_workers = config.get('num_workers', 4)
        split = config.get('split', 'all')
        
        summary_parts = [
            f"🖼️ Size: {img_size[0]}x{img_size[1]}",
            f"🔧 Normalize: {'Yes' if normalize else 'No'}",
            f"⚙️ Workers: {num_workers}",
            f"📂 Split: {split}"
        ]
        
        return " | ".join(summary_parts)
    
    def _get_safe_defaults(self) -> Dict[str, Any]:
        """Get safe default configuration values."""
        return {
            'img_size': DEFAULT_IMG_SIZE,
            'normalize': True,
            'preserve_aspect_ratio': True,
            'num_workers': 4,
            'split': 'all',
            'force_reprocess': False,
            'normalization_method': 'minmax'
        }
    
    def _normalize_merged_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize merged configuration dengan consistency checks."""
        normalized = config.copy()
        
        # Normalize img_size consistency
        img_size = normalized.get('img_size', DEFAULT_IMG_SIZE)
        if isinstance(img_size, int):
            normalized['img_size'] = [img_size, img_size]
        
        # Normalize split consistency (val -> valid)
        split = normalized.get('split', 'all')
        if split == 'val':
            normalized['split'] = 'valid'
        
        # Normalize boolean values
        for bool_key in ['normalize', 'preserve_aspect_ratio', 'force_reprocess']:
            if bool_key in normalized:
                normalized[bool_key] = bool(normalized[bool_key])
        
        return normalized