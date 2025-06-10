"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Fixed backend integration dengan proper return values dan pola augmentasi
"""

from typing import Dict, Any, Tuple, List, Optional

def validate_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """🔍 Fixed validation dengan proper return format"""
    try:
        # Use enhanced preprocessor validation
        from smartcash.dataset.preprocessor import validate_dataset
        
        # Get target splits dari config
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        all_valid = True
        total_images = 0
        validation_messages = []
        
        # Validate each target split
        for split in target_splits:
            result = validate_dataset(config, split)
            
            if result.get('success', False):
                summary = result.get('summary', {})
                valid_images = summary.get('valid_images', 0)
                total_imgs = summary.get('total_images', 0)
                
                total_images += valid_images
                validation_messages.append(f"✅ {split}: {valid_images}/{total_imgs} valid")
                
                if logger:
                    logger.info(f"✅ {split} validation: {valid_images} gambar valid")
            else:
                all_valid = False
                error_msg = result.get('message', f'Validation failed untuk {split}')
                validation_messages.append(f"❌ {split}: {error_msg}")
                
                if logger:
                    logger.error(f"❌ {split} validation failed: {error_msg}")
        
        if not all_valid:
            return False, f"❌ Validation issues: {'; '.join(validation_messages)}"
        
        if total_images == 0:
            return False, "❌ Tidak ada gambar valid ditemukan di semua splits"
        
        success_msg = f"✅ Dataset ready: {total_images:,} gambar valid dalam {len(target_splits)} splits"
        return True, success_msg
        
    except Exception as e:
        error_msg = f"❌ Error validation: {str(e)}"
        if logger:
            logger.error(error_msg)
        return False, error_msg

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int]:
    """📊 Fixed check dengan proper return format - only return 2 values"""
    try:
        from pathlib import Path
        
        output_dir = Path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed'))
        
        if not output_dir.exists():
            return False, 0
        
        total_files = 0
        
        # Check for both .npy and regular image files
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.npy']
        
        # Get target splits dari config
        target_splits = config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        # Check each target split
        for split in target_splits:
            split_images_dir = output_dir / split / 'images'
            if split_images_dir.exists():
                split_files = []
                for ext in image_extensions:
                    split_files.extend(list(split_images_dir.glob(f'*{ext}')))
                
                split_count = len(split_files)
                if split_count > 0:
                    total_files += split_count
        
        return total_files > 0, total_files
        
    except Exception:
        return False, 0

def create_backend_preprocessor(ui_config: Dict[str, Any], logger=None):
    """🏭 Create enhanced preprocessor dengan dual progress support"""
    try:
        from smartcash.dataset.preprocessor import create_preprocessing_service
        
        # Convert UI config ke backend format
        backend_config = _convert_ui_to_backend_config(ui_config)
        
        # Create service dengan enhanced features
        service = create_preprocessing_service(backend_config)
        
        # Add compatibility methods jika tidak ada
        if not hasattr(service, 'preprocess_dataset'):
            service.preprocess_dataset = service.preprocess_and_visualize
        
        return service
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating preprocessor: {str(e)}")
        return None

def create_backend_checker(config: Dict[str, Any], logger=None):
    """🔍 Create dataset checker dengan proper error handling"""
    try:
        from smartcash.dataset.preprocessor import validate_dataset, get_preprocessing_statistics
        
        class EnhancedChecker:
            def __init__(self, config, logger):
                self.config = config
                self.logger = logger
            
            def check_dataset(self) -> Dict[str, Any]:
                """Enhanced dataset check dengan comprehensive stats"""
                try:
                    # Get target splits
                    target_splits = self.config.get('preprocessing', {}).get('target_splits', ['train', 'valid'])
                    if isinstance(target_splits, str):
                        target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
                    
                    results = {}
                    total_valid = 0
                    
                    for split in target_splits:
                        result = validate_dataset(self.config, split)
                        results[split] = result
                        
                        if result.get('success'):
                            total_valid += result.get('summary', {}).get('valid_images', 0)
                    
                    # Get comprehensive statistics
                    try:
                        stats_result = get_preprocessing_statistics(self.config)
                        stats = stats_result.get('stats', {})
                    except Exception:
                        stats = {}
                    
                    return {
                        'success': True,
                        'message': f"✅ Dataset check completed: {total_valid} valid images",
                        'results': results,
                        'statistics': stats,
                        'total_valid_images': total_valid
                    }
                    
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Check failed: {str(e)}"
                    }
        
        return EnhancedChecker(config, logger)
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating checker: {str(e)}")
        return None

def create_backend_cleanup_service(config: Dict[str, Any], logger=None, ui_components: Optional[Dict[str, Any]] = None):
    """
    Create a cleanup service with enhanced functionality
    
    Args:
        config: Configuration dictionary
        logger: Optional logger instance
        ui_components: Optional UI components for interaction
        
    Returns:
        EnhancedCleanupService instance
    """
    from smartcash.dataset.preprocessor import cleanup_preprocessed_data
    from typing import Optional, Dict, Any, Callable
    
    class EnhancedCleanupService:
        def __init__(self, config, logger, ui_components=None):
            self.config = config
            self.logger = logger
            self.ui_components = ui_components or {}
            
        def show_confirmation_dialog(self, title: str, message: str, 
                                 confirm_text: str, cancel_text: str) -> bool:
            """
            Show confirmation dialog using UI components if available
            
            Args:
                title: Dialog title
                message: Dialog message
                confirm_text: Text for confirm button
                cancel_text: Text for cancel button
                
            Returns:
                bool: True if confirmed, False otherwise
            """
            if not self.ui_components:
                if self.logger:
                    self.logger.warning("UI components not available, skipping confirmation dialog")
                return True
                
            try:
                from smartcash.ui.dataset.preprocessing.handlers.preprocessing_handlers import _show_confirmation_in_area
                import threading
                
                response_event = threading.Event()
                response = None
                
                def on_confirm():
                    nonlocal response
                    response = True
                    response_event.set()
                    
                def on_cancel():
                    nonlocal response
                    response = False
                    response_event.set()
                
                # Show dialog in UI thread
                if 'ipython' in globals():
                    from IPython.display import display
                    display("Please check the UI for confirmation dialog")
                
                _show_confirmation_in_area(
                    ui_components=self.ui_components,
                    title=title,
                    message=message,
                    confirm_text=confirm_text,
                    cancel_text=cancel_text,
                    on_confirm=on_confirm,
                    on_cancel=on_cancel
                )
                
                # Wait for user response (with timeout)
                response_event.wait(timeout=300)  # 5 minutes timeout
                return response if response is not None else False
                
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error showing confirmation dialog: {str(e)}", exc_info=True)
                return True  # Default to True to avoid blocking the operation
                
        def cleanup_preprocessed_data(self) -> Dict[str, Any]:
            """Enhanced cleanup dengan detailed reporting"""
            try:
                # Check existing data sebelum cleanup
                has_data, file_count = check_preprocessed_exists(self.config)
                
                if not has_data:
                    return {
                        'success': True,
                        'message': "ℹ️ Tidak ada preprocessed data untuk dibersihkan",
                        'stats': {'files_removed': 0, 'splits_cleaned': 0}
                    }
                
                # Show confirmation dialog
                confirmed = self.show_confirmation_dialog(
                    title="Konfirmasi Cleanup",
                    message=f"Anda yakin ingin menghapus {file_count} file preprocessed?\n\nTindakan ini tidak dapat dibatalkan.",
                    confirm_text="Ya, Hapus",
                    cancel_text="Batal"
                )
                
                if not confirmed:
                    return {
                        'success': False,
                        'message': 'Cleanup dibatalkan oleh pengguna',
                        'cancelled': True
                    }
                
                # Perform cleanup
                result = cleanup_preprocessed_data(self.config)
                
                if result.get('success'):
                    return {
                        'success': True,
                        'message': f"🧹 Cleanup berhasil: {file_count} file dihapus",
                        'stats': {
                            'files_removed': file_count,
                            'splits_cleaned': len(self.config.get('preprocessing', {}).get('target_splits', ['train', 'valid']))
                        }
                    }
                else:
                    return {
                        'success': False,
                        'message': result.get('message', 'Gagal membersihkan data')
                    }
                    
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
                return {
                    'success': False,
                    'message': f"❌ Terjadi kesalahan saat membersihkan data: {str(e)}"
                }
        
    try:
        # Create and return an instance of the EnhancedCleanupService
        return EnhancedCleanupService(config, logger, ui_components)
    except Exception as e:
        if logger:
            logger.error(f"❌ Error creating cleanup service: {str(e)}")
        return None

def _convert_ui_to_backend_config(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """🔄 Convert UI config ke backend format dengan full compatibility"""
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
    
    # Enhanced backend config yang kompatibel dengan semua features
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
        },
        'cleanup': ui_config.get('cleanup', {}),
        
        # Legacy compatibility keys
        'img_size': img_size,
        'normalize': normalization.get('enabled', True),
        'normalization_method': normalization.get('method', 'minmax'),
        'preserve_aspect_ratio': normalization.get('preserve_aspect_ratio', True),
        'validation_enabled': validation.get('enabled', True),
        'move_invalid': validation.get('move_invalid', True),
        'invalid_dir': validation.get('invalid_dir', 'data/invalid'),
        'output_dir': preprocessing.get('output_dir', 'data/preprocessed'),
        'force_reprocess': preprocessing.get('force_reprocess', False)
    }