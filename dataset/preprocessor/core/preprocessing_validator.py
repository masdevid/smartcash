"""
File: smartcash/dataset/preprocessor/core/preprocessing_validator.py
Deskripsi: Validator untuk input preprocessing dengan comprehensive validation checks
"""

from typing import Dict, Any, List, Tuple
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.dataset.utils.path_validator import get_path_validator
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS, DEFAULT_IMG_SIZE


class PreprocessingValidator:
    """Validator untuk semua aspek preprocessing dengan comprehensive checks."""
    
    def __init__(self, config: Dict[str, Any], logger=None):
        """Initialize validator dengan configuration dependencies."""
        self.config = config
        self.logger = logger or get_logger()
        self.path_validator = get_path_validator(logger)
        
    def validate_preprocessing_request(self, split: str, force_reprocess: bool, **kwargs) -> Dict[str, Any]:
        """
        Validate complete preprocessing request dengan comprehensive checks.
        
        Args:
            split: Target split untuk preprocessing
            force_reprocess: Flag reprocess paksa
            **kwargs: Parameter preprocessing tambahan
            
        Returns:
            Dictionary validation result dengan target splits dan status
        """
        validation_result = {
            'valid': True,
            'message': '',
            'issues': [],
            'warnings': [],
            'target_splits': [],
            'config_validation': {},
            'source_validation': {},
            'compatibility_check': {}
        }
        
        try:
            # 1. Validate split request
            split_validation = self._validate_split_request(split)
            if not split_validation['valid']:
                validation_result['valid'] = False
                validation_result['issues'].extend(split_validation['issues'])
            else:
                validation_result['target_splits'] = split_validation['target_splits']
            
            # 2. Validate preprocessing config
            config_validation = self._validate_preprocessing_config(**kwargs)
            validation_result['config_validation'] = config_validation
            if not config_validation['valid']:
                validation_result['issues'].extend(config_validation['issues'])
            
            # 3. Validate source dataset
            source_validation = self.validate_source_dataset()
            validation_result['source_validation'] = source_validation
            if not source_validation['valid']:
                validation_result['valid'] = False
                validation_result['issues'].extend(source_validation['issues'])
            
            # 4. Check preprocessing compatibility
            if validation_result['valid']:
                compatibility = self.check_preprocessing_compatibility(
                    validation_result['target_splits'], force_reprocess
                )
                validation_result['compatibility_check'] = compatibility
                validation_result['warnings'].extend(compatibility['warnings'])
            
            # Final validation message
            if validation_result['valid']:
                target_count = len(validation_result['target_splits'])
                validation_result['message'] = f'Validation berhasil: {target_count} split siap diproses'
            else:
                issue_count = len(validation_result['issues'])
                validation_result['message'] = f'Validation gagal: {issue_count} critical issues'
            
        except Exception as e:
            validation_result['valid'] = False
            validation_result['message'] = f'Error validation: {str(e)}'
            validation_result['issues'].append(f'Unexpected validation error: {str(e)}')
        
        return validation_result
    
    def validate_source_dataset(self) -> Dict[str, Any]:
        """Validate source dataset structure dan availability."""
        data_dir = self.config.get('data', {}).get('dir', 'data')
        validation_result = self.path_validator.validate_dataset_structure(data_dir)
        
        if not validation_result['valid']:
            return {
                'valid': False,
                'issues': [f"Source dataset tidak valid: {data_dir}"],
                'data_dir': data_dir
            }
        
        # Check critical requirements
        critical_issues = [issue for issue in validation_result['issues'] if '❌' in issue]
        if critical_issues:
            return {
                'valid': False,
                'issues': critical_issues,
                'data_dir': data_dir,
                'total_images': validation_result['total_images']
            }
        
        return {
            'valid': True,
            'message': f"Source dataset valid: {validation_result['total_images']} gambar",
            'data_dir': data_dir,
            'total_images': validation_result['total_images'],
            'splits_available': [s for s in DEFAULT_SPLITS if validation_result['splits'][s]['exists']],
            'issues': validation_result['issues']  # Non-critical issues
        }
    
    def check_preprocessing_compatibility(self, target_splits: List[str], force_reprocess: bool) -> Dict[str, Any]:
        """Check compatibility untuk preprocessing operation."""
        compatibility = {
            'compatible': True,
            'warnings': [],
            'existing_data': {},
            'estimated_resources': {}
        }
        
        # Check existing preprocessed data
        preprocessed_dir = self.config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')
        for split in target_splits:
            split_path = Path(preprocessed_dir) / split
            if split_path.exists():
                existing_count = len(list(split_path.glob('**/*.jpg')))
                if existing_count > 0:
                    compatibility['existing_data'][split] = existing_count
                    if not force_reprocess:
                        compatibility['warnings'].append(
                            f"Split {split} sudah memiliki {existing_count} gambar preprocessed"
                        )
        
        # Estimate resource requirements
        source_validation = self.validate_source_dataset()
        if source_validation['valid']:
            total_images = source_validation['total_images']
            compatibility['estimated_resources'] = {
                'processing_time_minutes': (total_images * 0.1) / 60,  # 0.1 sec per image
                'storage_mb': total_images * 0.25,  # ~0.25MB per processed image
                'memory_requirement': 'Low to Medium'
            }
        
        return compatibility
    
    def _validate_split_request(self, split_request: str) -> Dict[str, Any]:
        """Validate dan normalize split request."""
        # Normalize split request
        split_mapping = {'val': 'valid', 'validation': 'valid'}
        normalized = split_mapping.get(split_request.lower(), split_request.lower())
        
        if normalized == 'all':
            return {
                'valid': True,
                'target_splits': DEFAULT_SPLITS,
                'message': 'All splits akan diproses'
            }
        elif normalized in DEFAULT_SPLITS:
            return {
                'valid': True,
                'target_splits': [normalized],
                'message': f'Split {normalized} akan diproses'
            }
        else:
            return {
                'valid': False,
                'issues': [f'Split tidak valid: {split_request}. Valid: {DEFAULT_SPLITS + ["all"]}'],
                'target_splits': []
            }
    
    def _validate_preprocessing_config(self, **kwargs) -> Dict[str, Any]:
        """Validate preprocessing configuration parameters."""
        config_validation = {'valid': True, 'issues': [], 'validated_config': {}}
        
        # Validate img_size
        img_size = kwargs.get('img_size', self.config.get('preprocessing', {}).get('img_size', DEFAULT_IMG_SIZE))
        if isinstance(img_size, int):
            img_size = [img_size, img_size]
        elif not isinstance(img_size, (list, tuple)) or len(img_size) != 2:
            config_validation['issues'].append('img_size harus integer atau [width, height]')
        elif not all(isinstance(x, int) and x > 0 for x in img_size):
            config_validation['issues'].append('img_size harus berupa integer positif')
        
        config_validation['validated_config']['img_size'] = img_size
        
        # Validate normalization
        normalize = kwargs.get('normalize', self.config.get('preprocessing', {}).get('normalize', True))
        if not isinstance(normalize, bool):
            config_validation['issues'].append('normalize harus boolean')
        config_validation['validated_config']['normalize'] = normalize
        
        # Validate num_workers
        num_workers = kwargs.get('num_workers', self.config.get('preprocessing', {}).get('num_workers', 4))
        if not isinstance(num_workers, int) or num_workers < 1 or num_workers > 16:
            config_validation['issues'].append('num_workers harus integer antara 1-16')
        config_validation['validated_config']['num_workers'] = min(max(num_workers, 1), 16)
        
        # Set validity
        config_validation['valid'] = len(config_validation['issues']) == 0
        
        return config_validation
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Dapatkan summary status validator."""
        return {
            'validator_ready': True,
            'path_validator_available': self.path_validator is not None,
            'config_available': bool(self.config),
            'supported_validations': [
                'split_request', 'preprocessing_config', 'source_dataset', 'compatibility_check'
            ]
        }