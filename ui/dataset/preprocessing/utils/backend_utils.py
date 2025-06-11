"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Updated backend utils dengan integrasi API preprocessor yang dikonsolidasi
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

def validate_dataset_ready(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate dataset menggunakan consolidated API"""
    try:
        from smartcash.dataset.preprocessor import validate_dataset
        
        target_split = config.get('preprocessing', {}).get('target_splits', ['train'])[0]
        result = validate_dataset(
            config=config,
            target_split=target_split
        )
        
        if result.get('success', False):
            summary = result.get('summary', {})
            total_images = summary.get('total_images', 0)
            validation_rate = summary.get('validation_rate', '0%')
            return True, f"Dataset valid: {total_images:,} gambar (validation rate: {validation_rate})"
        else:
            return False, result.get('message', 'Validation failed')
            
    except Exception as e:
        get_logger('backend_utils').error(f"Validation error: {str(e)}")
        return False, f"Error validasi: {str(e)}"

def create_backend_preprocessor_with_progress(ui_components: Dict[str, Any]) -> Optional[Any]:
    """Create preprocessing service dengan consolidated API dan progress integration"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import (
            create_integrated_preprocessing_service
        )
        
        service = create_integrated_preprocessing_service(ui_components)
        if not service:
            get_logger('backend_utils').error("Gagal membuat integrated service")
            return None
        
        return service
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating preprocessor: {str(e)}")
        return None

def create_backend_cleanup_service_with_progress(ui_components: Dict[str, Any]):
    """Create cleanup service dengan consolidated API"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import (
            create_integrated_preprocessing_service
        )
        
        # Service yang sama bisa handle cleanup juga
        service = create_integrated_preprocessing_service(ui_components)
        return service
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating cleanup service: {str(e)}")
        return None

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check preprocessed data dengan consolidated API"""
    try:
        from smartcash.dataset.preprocessor import get_preprocessing_status
        
        status = get_preprocessing_status(config=config)
        
        if not status.get('success', False):
            return False, status.get('message', 'Status check failed')
        
        preprocessed_info = status.get('preprocessed_data', {})
        exists = preprocessed_info.get('exists', False)
        
        if not exists:
            return False, "Tidak ada data preprocessed"
        
        # Extract detailed statistics
        total_files = preprocessed_info.get('total_files', 0)
        by_split = preprocessed_info.get('by_split', {})
        
        details = []
        for split, count in by_split.items():
            if count > 0:
                details.append(f"{split}: {count:,} files")
        
        detail_str = f"{total_files:,} total files"
        if details:
            detail_str += f" ({', '.join(details)})"
        
        return True, detail_str
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error checking preprocessed: {str(e)}")
        return False, f"Error: {str(e)}"

def create_backend_checker(config: Dict[str, Any]):
    """Create validation service dengan consolidated API"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import (
            create_integrated_preprocessing_service
        )
        
        service = create_integrated_preprocessing_service({})  # No UI needed for checker
        if not service:
            return None
        
        class ValidationWrapper:
            def __init__(self, service):
                self.service = service
            
            def validate(self) -> Tuple[bool, str]:
                try:
                    result = self.service.validate_dataset_only()
                    success = result.get('success', False)
                    message = result.get('message', 'Validation completed')
                    
                    if success:
                        summary = result.get('summary', {})
                        total_images = summary.get('total_images', 0)
                        validation_rate = summary.get('validation_rate', '0%')
                        return True, f"✅ Dataset valid: {total_images:,} gambar (rate: {validation_rate})"
                    else:
                        return False, f"❌ {message}"
                        
                except Exception as e:
                    return False, f"❌ Validation error: {str(e)}"
        
        return ValidationWrapper(service)
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating checker: {str(e)}")
        return None

def get_preprocessing_samples(ui_components: Dict[str, Any], target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
    """Get dataset samples menggunakan consolidated API"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import (
            create_samples_service
        )
        
        samples_service = create_samples_service(ui_components)
        if not samples_service:
            return {'success': False, 'message': 'Samples service tidak tersedia', 'samples': []}
        
        result = samples_service.get_samples(target_split, max_samples)
        return result
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error getting samples: {str(e)}")
        return {'success': False, 'message': f"Error: {str(e)}", 'samples': []}

def get_system_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get comprehensive system status"""
    try:
        from smartcash.ui.dataset.preprocessing.utils.backend_integration import (
            create_status_service, validate_api_compatibility
        )
        
        # Check API compatibility
        api_compat = validate_api_compatibility()
        
        # Get status service
        status_service = create_status_service(ui_components)
        if not status_service:
            return {
                'success': False,
                'message': 'Status service tidak tersedia',
                'api_compatibility': api_compat
            }
        
        # Get comprehensive status
        status = status_service.get_comprehensive_status()
        status['api_compatibility'] = api_compat
        
        return status
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error getting system status: {str(e)}")
        return {
            'success': False,
            'message': f"Error: {str(e)}",
            'service_ready': False
        }

def _extract_and_enhance_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan enhancement untuk API compatibility"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        ui_config = extract_preprocessing_config(ui_components)
        
        # Enhance untuk API compatibility
        enhanced_config = _enhance_config_for_api(ui_config)
        
        return enhanced_config
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error extracting config: {str(e)}")
        return _get_fallback_config()

def _enhance_config_for_api(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance config untuk compatibility dengan consolidated API"""
    enhanced = ui_config.copy()
    
    # Ensure required structure untuk API
    preprocessing = enhanced.setdefault('preprocessing', {})
    performance = enhanced.setdefault('performance', {})
    data = enhanced.setdefault('data', {})
    
    # Setup data paths dengan proper structure
    base_dir = data.get('dir', 'data')
    target_splits = preprocessing.get('target_splits', ['train', 'valid'])
    
    # Ensure local paths
    data.setdefault('local', {})
    for split in target_splits:
        if split not in data['local']:
            data['local'][split] = f"{base_dir}/{split}"
    
    # API requirements
    preprocessing.setdefault('output_dir', f"{base_dir}/preprocessed")
    preprocessing.setdefault('enabled', True)
    
    # Validation settings
    validation = preprocessing.setdefault('validation', {})
    validation.setdefault('enabled', True)
    validation.setdefault('move_invalid', True)
    validation.setdefault('invalid_dir', f"{base_dir}/invalid")
    
    # Normalization settings
    normalization = preprocessing.setdefault('normalization', {})
    normalization.setdefault('enabled', True)
    normalization.setdefault('method', 'minmax')
    normalization.setdefault('target_size', [640, 640])
    normalization.setdefault('preserve_aspect_ratio', True)
    
    # Performance settings
    performance.setdefault('batch_size', 32)
    performance.setdefault('use_gpu', True)
    
    # File naming untuk API
    enhanced.setdefault('file_naming', {
        'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
        'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'augmented_pattern': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'preserve_uuid': True
    })
    
    return enhanced

def _get_fallback_config() -> Dict[str, Any]:
    """Get fallback config untuk error cases"""
    return {
        'preprocessing': {
            'enabled': True,
            'target_splits': ['train', 'valid'],
            'output_dir': 'data/preprocessed',
            'normalization': {
                'enabled': True,
                'method': 'minmax',
                'target_size': [640, 640],
                'preserve_aspect_ratio': True
            },
            'validation': {
                'enabled': True,
                'move_invalid': True,
                'invalid_dir': 'data/invalid'
            }
        },
        'performance': {
            'batch_size': 32,
            'use_gpu': True
        },
        'data': {
            'dir': 'data',
            'local': {
                'train': 'data/train',
                'valid': 'data/valid'
            }
        },
        'file_naming': {
            'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True
        }
    }

# Backward compatibility exports
_convert_ui_to_backend_config = _extract_and_enhance_config
create_backend_preprocessor = create_backend_preprocessor_with_progress
create_backend_cleanup_service = create_backend_cleanup_service_with_progress

# New API utilities
validate_api_ready = lambda ui_components: get_system_status(ui_components).get('service_ready', False)
get_api_status_message = lambda ui_components: get_system_status(ui_components).get('message', 'Status unknown')
check_api_compatibility = lambda: validate_api_compatibility()['compatible']