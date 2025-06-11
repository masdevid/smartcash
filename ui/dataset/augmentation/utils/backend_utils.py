"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Enhanced backend integration dengan proper callback system dan DRY principles
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path

def validate_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """🔍 Validate dataset menggunakan backend service"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        service = create_preprocessing_service(config)
        result = service.validate_dataset_only()
        
        return result['success'], result['message']
        
    except Exception as e:
        error_msg = f"❌ Error validation: {str(e)}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int]:
    """📊 Check preprocessed data existence"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        service = create_preprocessing_service(config)
        return service.check_preprocessed_exists()
        
    except Exception:
        return False, 0

def create_backend_preprocessor(ui_config: Dict[str, Any], logger=None, progress_callback: Optional[Callable] = None):
    """🏭 Create preprocessor dengan progress callback untuk UI dual tracker"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Convert UI config ke backend format
        backend_config = _convert_ui_to_backend_config(ui_config)
        
        # Create service dengan progress callback
        service = create_preprocessing_service(
            config=backend_config,
            progress_callback=progress_callback
        )
        
        return service
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating preprocessor: {str(e)}")
        return None

def create_backend_checker(config: Dict[str, Any], logger=None):
    """🔍 Create checker service wrapper"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        service = create_preprocessing_service(config)
        
        class CheckerWrapper:
            def __init__(self, service, logger):
                self.service = service
                self.logger = logger
            
            def check_dataset(self) -> Dict[str, Any]:
                """Check dataset dengan service backend"""
                try:
                    result = self.service.validate_dataset_only()
                    return {
                        'success': result['success'],
                        'message': result['message'],
                        'summary': result.get('summary', {})
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Check failed: {str(e)}"
                    }
            
            def validate(self) -> Tuple[bool, str]:
                """Backward compatibility method"""
                result = self.check_dataset()
                return result['success'], result['message']
        
        return CheckerWrapper(service, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating checker: {str(e)}")
        return None

def create_backend_cleanup_service(config: Dict[str, Any], logger=None, ui_components: Optional[Dict[str, Any]] = None):
    """🧹 Create cleanup service dengan UI confirmation integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        service = create_preprocessing_service(config)
        
        class CleanupServiceWrapper:
            def __init__(self, service, logger, ui_components):
                self.service = service
                self.logger = logger
                self.ui_components = ui_components or {}
            
            def cleanup_preprocessed_data(self) -> Dict[str, Any]:
                """Enhanced cleanup dengan UI confirmation"""
                try:
                    # Check existing data
                    has_data, file_count = self.service.check_preprocessed_exists()
                    
                    if not has_data:
                        return {
                            'success': True,
                            'message': "ℹ️ Tidak ada data untuk dibersihkan",
                            'stats': {'files_removed': 0}
                        }
                    
                    # Show confirmation jika ada UI components
                    if not self._confirm_cleanup(file_count):
                        return {
                            'success': False,
                            'message': "🚫 Cleanup dibatalkan",
                            'cancelled': True
                        }
                    
                    # Perform cleanup via backend service
                    result = self.service.cleanup_preprocessed_data()
                    return result
                    
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Error cleanup: {str(e)}"
                    }
            
            def _confirm_cleanup(self, file_count: int) -> bool:
                """🤔 Show cleanup confirmation"""
                try:
                    # Try UI confirmation first
                    if self.ui_components:
                        from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import _show_confirmation_in_area
                        
                        confirmed = [False]  # Use list untuk mutable reference
                        
                        def on_confirm():
                            confirmed[0] = True
                            self._clear_confirmation_area()
                        
                        def on_cancel():
                            confirmed[0] = False
                            self._clear_confirmation_area()
                        
                        _show_confirmation_in_area(
                            ui_components=self.ui_components,
                            title="⚠️ Konfirmasi Cleanup",
                            message=f"Akan menghapus {file_count:,} file preprocessed.\n\n⚠️ Tindakan ini tidak dapat dibatalkan!\n\nLanjutkan?",
                            confirm_text="Ya, Hapus",
                            cancel_text="Batal",
                            on_confirm=on_confirm,
                            on_cancel=on_cancel
                        )
                        
                        # Tunggu user response (simplified)
                        return confirmed[0]
                    
                    # Fallback ke console confirmation
                    response = input(f"⚠️ Akan menghapus {file_count} file. Lanjutkan? (y/N): ").strip().lower()
                    return response in ['y', 'yes', 'ya']
                    
                except Exception:
                    return False
            
            def _clear_confirmation_area(self):
                """Clear confirmation area"""
                try:
                    if 'confirmation_area' in self.ui_components:
                        from IPython.display import clear_output
                        with self.ui_components['confirmation_area']:
                            clear_output(wait=True)
                except Exception:
                    pass
        
        return CleanupServiceWrapper(service, logger, ui_components)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating cleanup service: {str(e)}")
        return None

def _convert_ui_to_backend_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """🔄 Convert UI config ke backend format"""
    preprocessing = ui_config.get('preprocessing', {})
    normalization = preprocessing.get('normalization', {})
    validation = preprocessing.get('validation', {})
    performance = ui_config.get('performance', {})
    
    # Enhanced target_splits handling
    target_splits = preprocessing.get('target_splits', ['train', 'valid'])
    if isinstance(target_splits, str):
        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
    
    # Enhanced target_size handling
    target_size = normalization.get('target_size', [640, 640])
    if isinstance(target_size, list) and len(target_size) >= 2:
        img_size = target_size
    else:
        img_size = [640, 640]
    
    return {
        'data': ui_config.get('data', {'dir': 'data'}),
        'preprocessing': {
            **preprocessing,
            'target_splits': target_splits,
            'normalization': {
                **normalization,
                'target_size': img_size,
                'enabled': normalization.get('enabled', True),
                'method': normalization.get('method', 'minmax'),
                'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True)
            },
            'validation': {
                **validation,
                'enabled': validation.get('enabled', True),
                'move_invalid': validation.get('move_invalid', True),
                'invalid_dir': validation.get('invalid_dir', 'data/invalid')
            },
            'output': {
                'output_dir': preprocessing.get('output_dir', 'data/preprocessed'),
                'create_npy': preprocessing.get('output', {}).get('create_npy', True),
                'organize_by_split': True
            }
        },
        'performance': {
            **performance,
            'batch_size': performance.get('batch_size', 32),
            'threading': performance.get('threading', {
                'io_workers': 8,
                'cpu_workers': None,
                'parallel_threshold': 100,
                'batch_processing': True
            })
        }
    }