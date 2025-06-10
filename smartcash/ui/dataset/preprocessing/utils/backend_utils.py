"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Completed backend utils dengan full Progress Bridge integration dan enhanced service wrappers
"""

from typing import Dict, Any, Optional, Callable, Tuple
from smartcash.common.logger import get_logger

def validate_dataset_ready(config: Dict[str, Any], logger=None) -> Tuple[bool, str]:
    """🔍 Enhanced dataset validation dengan progress callback integration"""
    logger = logger or get_logger('backend_utils')
    
    try:
        # Extract target splits dari config
        preprocessing_config = config.get('preprocessing', {})
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        # Validate basic structure
        from pathlib import Path
        
        missing_splits = []
        total_images = 0
        
        for split in target_splits:
            # Check source directories
            data_config = config.get('data', {})
            split_paths = data_config.get('splits', {})
            
            if split in split_paths:
                base_path = Path(split_paths[split])
            else:
                base_path = Path('data') / split
            
            img_dir = base_path / 'images'
            label_dir = base_path / 'labels'
            
            if not img_dir.exists() or not label_dir.exists():
                missing_splits.append(f"{split} (missing {img_dir} atau {label_dir})")
                continue
            
            # Count images
            image_count = sum(1 for ext in ['.jpg', '.jpeg', '.png'] 
                            for _ in img_dir.glob(f'*{ext}'))
            
            if image_count == 0:
                missing_splits.append(f"{split} (no images)")
            else:
                total_images += image_count
        
        if missing_splits:
            return False, f"Dataset tidak siap: {', '.join(missing_splits)}"
        
        if total_images == 0:
            return False, "Tidak ada gambar ditemukan dalam dataset"
        
        return True, f"Dataset siap: {total_images:,} gambar dari {len(target_splits)} splits"
        
    except Exception as e:
        error_msg = f"Error validasi dataset: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, int]:
    """📊 Enhanced check preprocessed data dengan detailed counting"""
    try:
        from pathlib import Path
        
        preprocessing_config = config.get('preprocessing', {})
        output_dir = Path(preprocessing_config.get('output_dir', 'data/preprocessed'))
        
        if not output_dir.exists():
            return False, 0
        
        # Count files in all splits
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        total_files = 0
        for split in target_splits:
            split_dir = output_dir / split
            if split_dir.exists():
                # Count both images and labels
                for subdir in ['images', 'labels']:
                    dir_path = split_dir / subdir
                    if dir_path.exists():
                        extensions = ['.jpg', '.jpeg', '.png', '.npy'] if subdir == 'images' else ['.txt']
                        for ext in extensions:
                            total_files += len(list(dir_path.glob(f'*{ext}')))
        
        return total_files > 0, total_files
        
    except Exception:
        return False, 0

def create_backend_preprocessor(config: Dict[str, Any], logger=None, progress_callback: Optional[Callable] = None):
    """🏭 Enhanced factory untuk backend preprocessing service dengan Progress Bridge integration"""
    logger = logger or get_logger('backend_utils')
    
    try:
        # Import backend service
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Create service dengan progress callback
        service = create_preprocessing_service(config, progress_callback)
        
        logger.info("🚀 Backend preprocessing service created dengan progress callback")
        return service
        
    except Exception as e:
        error_msg = f"❌ Error creating backend preprocessor: {str(e)}"
        logger.error(error_msg)
        return None

def create_backend_checker(config: Dict[str, Any], logger=None):
    """🔍 Enhanced factory untuk backend validation service"""
    logger = logger or get_logger('backend_utils')
    
    try:
        # Import backend validation
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Create service untuk validation
        service = create_preprocessing_service(config)
        
        # Return wrapper dengan validation methods
        class ValidationServiceWrapper:
            def __init__(self, service):
                self.service = service
            
            def validate(self) -> Tuple[bool, str]:
                """Validate dataset dengan backend service"""
                try:
                    result = self.service.validate_dataset_only()
                    success = result.get('success', False)
                    message = result.get('message', 'Validation completed')
                    
                    if success:
                        summary = result.get('summary', {})
                        total_images = summary.get('total_valid_images', 0)
                        return True, f"✅ Dataset valid: {total_images:,} gambar"
                    else:
                        return False, f"❌ {message}"
                        
                except Exception as e:
                    return False, f"❌ Validation error: {str(e)}"
        
        wrapper = ValidationServiceWrapper(service)
        logger.info("🔍 Backend validation service created")
        return wrapper
        
    except Exception as e:
        error_msg = f"❌ Error creating backend checker: {str(e)}"
        logger.error(error_msg)
        return None

def create_backend_cleanup_service(config: Dict[str, Any], logger=None, ui_components: Optional[Dict[str, Any]] = None):
    """🧹 Enhanced factory untuk backend cleanup service dengan UI confirmation integration"""
    logger = logger or get_logger('backend_utils')
    
    try:
        # Import backend service
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Create service
        service = create_preprocessing_service(config)
        
        # Return wrapper dengan UI confirmation integration
        class CleanupServiceWrapper:
            def __init__(self, service, ui_components=None):
                self.service = service
                self.ui_components = ui_components or {}
            
            def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
                """Cleanup dengan UI confirmation integration"""
                try:
                    # Check files exist first
                    exists, file_count = check_preprocessed_exists(config)
                    if not exists:
                        return {
                            'success': True,
                            'message': "ℹ️ Tidak ada data untuk dibersihkan",
                            'stats': {'files_removed': 0}
                        }
                    
                    # UI confirmation jika ada UI components
                    if self.ui_components and 'confirmation_area' in self.ui_components:
                        confirmed = self._show_ui_confirmation(file_count, target_split or "semua split")
                        if confirmed is False:
                            return {
                                'success': True,
                                'cancelled': True,
                                'message': "🚫 Cleanup dibatalkan oleh user",
                                'stats': {'files_removed': 0}
                            }
                        elif confirmed is None:
                            return {
                                'success': False,
                                'message': "⏰ Timeout waiting for confirmation",
                                'stats': {'files_removed': 0}
                            }
                    
                    # Execute backend cleanup
                    result = self.service.cleanup_preprocessed_data(target_split)
                    return result
                    
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Cleanup error: {str(e)}",
                        'stats': {'files_removed': 0}
                    }
            
            def _show_ui_confirmation(self, files_count: int, target_split: str) -> Optional[bool]:
                """Show UI confirmation untuk cleanup"""
                try:
                    from smartcash.ui.dataset.preprocessing.utils.confirmation_utils import show_cleanup_confirmation
                    return show_cleanup_confirmation(self.ui_components, files_count, target_split)
                except Exception as e:
                    logger.warning(f"⚠️ UI confirmation error: {str(e)}")
                    return True  # Default to proceed jika UI confirmation gagal
        
        wrapper = CleanupServiceWrapper(service, ui_components)
        logger.info("🧹 Backend cleanup service created dengan UI integration")
        return wrapper
        
    except Exception as e:
        error_msg = f"❌ Error creating backend cleanup service: {str(e)}"
        logger.error(error_msg)
        return None

def _convert_ui_to_backend_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """🔄 Enhanced conversion dari UI components ke backend config format"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        # Extract config dari UI dengan validation
        ui_config = extract_preprocessing_config(ui_components)
        
        # Enhanced conversion untuk backend compatibility
        backend_config = _enhance_config_for_backend(ui_config)
        
        return backend_config
        
    except Exception as e:
        logger = get_logger('backend_utils')
        logger.error(f"❌ Error converting UI to backend config: {str(e)}")
        
        # Fallback ke basic config
        return {
            'preprocessing': {
                'enabled': True,
                'target_splits': ['train', 'valid'],
                'normalization': {'method': 'minmax', 'target_size': [640, 640]},
                'validation': {'enabled': True}
            },
            'performance': {'batch_size': 32}
        }

def _enhance_config_for_backend(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """🚀 Enhanced config untuk backend service compatibility"""
    enhanced = ui_config.copy()
    
    # Ensure required sections exist
    preprocessing = enhanced.setdefault('preprocessing', {})
    performance = enhanced.setdefault('performance', {})
    
    # Backend-specific enhancements
    preprocessing.setdefault('output_dir', 'data/preprocessed')
    preprocessing.setdefault('enabled', True)
    
    # Validation enhancements
    validation = preprocessing.setdefault('validation', {})
    validation.setdefault('enabled', True)
    validation.setdefault('move_invalid', True)
    validation.setdefault('fix_issues', False)
    
    # Normalization enhancements
    normalization = preprocessing.setdefault('normalization', {})
    normalization.setdefault('enabled', True)
    normalization.setdefault('method', 'minmax')
    normalization.setdefault('target_size', [640, 640])
    normalization.setdefault('preserve_aspect_ratio', True)
    
    # Performance enhancements
    performance.setdefault('batch_size', 32)
    performance.setdefault('use_gpu', True)
    
    # Threading configuration
    threading = performance.setdefault('threading', {})
    threading.setdefault('io_workers', 8)
    threading.setdefault('parallel_threshold', 100)
    
    return enhanced

def setup_backend_integration(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, Any]:
    """🔗 Enhanced setup untuk complete backend integration dengan Progress Bridge"""
    logger = get_logger('backend_utils')
    
    try:
        # Setup progress callback integration
        from smartcash.ui.dataset.preprocessing.utils.progress_utils import create_dual_progress_callback
        
        if 'progress_callback' not in ui_components:
            progress_callback = create_dual_progress_callback(ui_components)
            ui_components['progress_callback'] = progress_callback
            logger.info("📊 Progress callback created")
        
        # Setup backend service factories dengan progress integration
        ui_components.update({
            'validate_dataset_ready': lambda cfg: validate_dataset_ready(cfg, logger),
            'check_preprocessed_exists': lambda cfg: check_preprocessed_exists(cfg),
            'create_backend_preprocessor': lambda cfg: create_backend_preprocessor(
                cfg, logger, ui_components.get('progress_callback')
            ),
            'create_backend_checker': lambda cfg: create_backend_checker(cfg, logger),
            'create_backend_cleanup_service': lambda cfg: create_backend_cleanup_service(
                cfg, logger, ui_components
            ),
            '_convert_ui_to_backend_config': lambda: _convert_ui_to_backend_config(ui_components)
        })
        
        # Setup button management integration
        from smartcash.ui.dataset.preprocessing.utils.button_manager import create_button_state_manager
        button_manager = create_button_state_manager(ui_components)
        ui_components['button_manager'] = button_manager
        
        # Setup config handler integration
        config_handler = ui_components.get('config_handler')
        if config_handler and hasattr(config_handler, 'set_progress_callback'):
            config_handler.set_progress_callback(ui_components['progress_callback'])
            logger.info("🔄 Config handler progress integration enabled")
        
        logger.info("🔗 Complete backend integration setup completed")
        return ui_components
        
    except Exception as e:
        error_msg = f"❌ Error setting up backend integration: {str(e)}"
        logger.error(error_msg)
        return ui_components

def test_backend_connectivity(ui_components: Dict[str, Any], config: Dict[str, Any]) -> Dict[str, bool]:
    """🧪 Test backend service connectivity dan functionality"""
    logger = get_logger('backend_utils')
    
    results = {
        'validation_service': False,
        'preprocessing_service': False,
        'cleanup_service': False,
        'progress_integration': False
    }
    
    try:
        # Test validation service
        checker = create_backend_checker(config, logger)
        if checker:
            is_valid, msg = checker.validate()
            results['validation_service'] = True
            logger.info(f"✅ Validation service: {msg}")
        
        # Test preprocessing service
        preprocessor = create_backend_preprocessor(config, logger)
        if preprocessor:
            results['preprocessing_service'] = True
            logger.info("✅ Preprocessing service created")
        
        # Test cleanup service
        cleanup = create_backend_cleanup_service(config, logger, ui_components)
        if cleanup:
            results['cleanup_service'] = True
            logger.info("✅ Cleanup service created")
        
        # Test progress integration
        if 'progress_callback' in ui_components:
            try:
                ui_components['progress_callback']('test', 50, 100, 'Test message')
                results['progress_integration'] = True
                logger.info("✅ Progress integration working")
            except Exception:
                logger.warning("⚠️ Progress integration test failed")
        
        all_working = all(results.values())
        status = "✅ All services working" if all_working else "⚠️ Some services have issues"
        logger.info(f"🧪 Backend connectivity test: {status}")
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Backend connectivity test error: {str(e)}")
        return results

# One-liner utilities untuk convenience
create_progress_enabled_preprocessor = lambda config, ui_components: create_backend_preprocessor(config, progress_callback=ui_components.get('progress_callback'))
validate_with_progress = lambda config, ui_components: validate_dataset_ready(config)
check_data_status = lambda config: check_preprocessed_exists(config)
is_backend_ready = lambda ui_components: all(key in ui_components for key in ['progress_callback', 'button_manager'])
get_backend_config = lambda ui_components: _convert_ui_to_backend_config(ui_components)