"""
File: smartcash/ui/dataset/preprocessing/utils/backend_integration.py
Deskripsi: Service integration dengan smartcash.dataset.preprocessor API yang dikonsolidasi
"""

from typing import Dict, Any, Tuple, Optional, Callable, Union
from pathlib import Path
from smartcash.common.logger import get_logger

def create_integrated_preprocessing_service(ui_components: Dict[str, Any]) -> Optional[Any]:
    """🔧 Create preprocessing service yang terintegrasi dengan API baru"""
    try:
        from smartcash.dataset.preprocessor import preprocess_dataset
        
        config = _extract_backend_config(ui_components)
        progress_callback = _create_ui_progress_callback(ui_components)
        
        class IntegratedPreprocessingService:
            def __init__(self, config, ui_components, progress_callback):
                self.config = config
                self.ui_components = ui_components
                self.progress_callback = progress_callback
            
            def preprocess_dataset(self) -> Dict[str, Any]:
                """Execute preprocessing dengan API baru"""
                try:
                    result = preprocess_dataset(
                        config=self.config,
                        ui_components=self.ui_components,
                        progress_callback=self.progress_callback
                    )
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Preprocessing error: {str(e)}",
                        'stats': {},
                        'processing_time': 0.0
                    }
            
            def validate_dataset_only(self) -> Dict[str, Any]:
                """Validate dataset menggunakan API validation"""
                try:
                    from smartcash.dataset.preprocessor import validate_dataset
                    target_split = self.config.get('preprocessing', {}).get('target_splits', ['train'])[0]
                    
                    result = validate_dataset(
                        config=self.config,
                        target_split=target_split,
                        ui_components=self.ui_components
                    )
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Validation error: {str(e)}",
                        'summary': {}
                    }
            
            def check_preprocessed_exists(self) -> Tuple[bool, int]:
                """Check preprocessed data existence"""
                try:
                    from smartcash.dataset.preprocessor import get_preprocessing_status
                    
                    status = get_preprocessing_status(
                        config=self.config,
                        ui_components=self.ui_components
                    )
                    
                    preprocessed_info = status.get('preprocessed_data', {})
                    exists = preprocessed_info.get('exists', False)
                    count = preprocessed_info.get('total_files', 0)
                    
                    return exists, count
                except Exception:
                    return False, 0
            
            def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
                """Cleanup preprocessed data"""
                try:
                    from smartcash.dataset.preprocessor import cleanup_preprocessed_data
                    
                    result = cleanup_preprocessed_data(
                        config=self.config,
                        target_split=target_split,
                        ui_components=self.ui_components
                    )
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Cleanup error: {str(e)}",
                        'stats': {'files_removed': 0}
                    }
        
        return IntegratedPreprocessingService(config, ui_components, progress_callback)
        
    except Exception as e:
        get_logger('backend_integration').error(f"Error creating service: {str(e)}")
        return None

def _extract_backend_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config yang compatible dengan API baru"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        config = extract_preprocessing_config(ui_components)
        
        # Enhance dengan file_naming requirements
        config.setdefault('file_naming', {
            'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True
        })
        
        return config
    except Exception as e:
        get_logger('backend_integration').warning(f"Config extraction error: {str(e)}")
        return _get_fallback_config()

def _create_ui_progress_callback(ui_components: Dict[str, Any]) -> Callable:
    """Create progress callback yang sesuai dengan API specifications"""
    def enhanced_progress_callback(level: str, current: int, total: int, message: str):
        """Enhanced callback dengan level mapping yang proper"""
        try:
            progress_tracker = ui_components.get('progress_tracker')
            if not progress_tracker:
                return
            
            # Map API level ke UI tracker level
            if level in ['overall', 'primary']:
                if hasattr(progress_tracker, 'update_overall'):
                    progress_tracker.update_overall(current, message)
            elif level in ['step', 'current']:
                if hasattr(progress_tracker, 'update_current'):
                    progress_tracker.update_current(current, message)
            
            # Log milestone progress
            if _is_milestone_progress(current, total):
                from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                log_to_accordion(ui_components, f"📊 {message} ({current}/{total})", "info")
                
        except Exception:
            pass  # Silent fail untuk prevent process interruption
    
    return enhanced_progress_callback

def create_samples_service(ui_components: Dict[str, Any]) -> Optional[Any]:
    """Create service untuk dataset samples"""
    try:
        from smartcash.dataset.preprocessor import get_preprocessing_samples
        
        config = _extract_backend_config(ui_components)
        
        class SamplesService:
            def __init__(self, config, ui_components):
                self.config = config
                self.ui_components = ui_components
            
            def get_samples(self, target_split: str = "train", max_samples: int = 5) -> Dict[str, Any]:
                """Get dataset samples"""
                try:
                    result = get_preprocessing_samples(
                        config=self.config,
                        target_split=target_split,
                        max_samples=max_samples,
                        ui_components=self.ui_components
                    )
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Samples error: {str(e)}",
                        'samples': []
                    }
        
        return SamplesService(config, ui_components)
        
    except Exception as e:
        get_logger('backend_integration').error(f"Error creating samples service: {str(e)}")
        return None

def create_status_service(ui_components: Dict[str, Any]) -> Optional[Any]:
    """Create service untuk system status"""
    try:
        from smartcash.dataset.preprocessor import get_preprocessing_status
        
        config = _extract_backend_config(ui_components)
        
        class StatusService:
            def __init__(self, config, ui_components):
                self.config = config
                self.ui_components = ui_components
            
            def get_comprehensive_status(self) -> Dict[str, Any]:
                """Get comprehensive system status"""
                try:
                    result = get_preprocessing_status(
                        config=self.config,
                        ui_components=self.ui_components
                    )
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Status error: {str(e)}",
                        'service_ready': False
                    }
        
        return StatusService(config, ui_components)
        
    except Exception as e:
        get_logger('backend_integration').error(f"Error creating status service: {str(e)}")
        return None

def validate_api_compatibility() -> Dict[str, Any]:
    """Validate compatibility dengan consolidated API"""
    try:
        from smartcash.dataset.preprocessor import (
            preprocess_dataset, validate_dataset, 
            get_preprocessing_samples, cleanup_preprocessed_data,
            get_preprocessing_status
        )
        
        available_functions = {
            'preprocess_dataset': preprocess_dataset is not None,
            'validate_dataset': validate_dataset is not None,
            'get_preprocessing_samples': get_preprocessing_samples is not None,
            'cleanup_preprocessed_data': cleanup_preprocessed_data is not None,
            'get_preprocessing_status': get_preprocessing_status is not None
        }
        
        all_available = all(available_functions.values())
        
        return {
            'compatible': all_available,
            'available_functions': available_functions,
            'enhanced_features': all_available,
            'message': '✅ Full API compatibility' if all_available else '⚠️ Partial API compatibility'
        }
        
    except ImportError as e:
        return {
            'compatible': False,
            'available_functions': {},
            'enhanced_features': False,
            'message': f'❌ API not available: {str(e)}'
        }

def _get_fallback_config() -> Dict[str, Any]:
    """Fallback config untuk error cases"""
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

def _is_milestone_progress(current: int, total: int) -> bool:
    """Check if progress adalah milestone"""
    if total <= 10:
        return current % 2 == 0 or current == total
    
    milestones = [0, 10, 25, 50, 75, 90, 100]
    progress_pct = (current / total) * 100 if total > 0 else 0
    return any(abs(progress_pct - milestone) < 2 for milestone in milestones) or current == total

# API validation functions
check_preprocessing_compatibility = lambda: validate_api_compatibility()
is_enhanced_api_available = lambda: validate_api_compatibility()['enhanced_features']
get_api_status = lambda: validate_api_compatibility()['message']