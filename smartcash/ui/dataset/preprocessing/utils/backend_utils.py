"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Fixed backend utils dengan progress bridge integration sesuai pola augmentation
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

def validate_dataset_ready(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate dataset readiness dengan enhanced checking"""
    try:
        # Extract target splits
        preprocessing_config = config.get('preprocessing', {})
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
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
                missing_splits.append(f"{split} (missing directories)")
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
        get_logger('backend_utils').error(error_msg)
        return False, error_msg

def check_preprocessed_exists(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Check preprocessed data existence dengan detail analysis"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Buat service tanpa progress tracker untuk checking
        service = create_preprocessing_service(config, progress_callback=None)
        
        if not service:
            return False, "Backend service tidak tersedia"
        
        exists, count = service.check_preprocessed_exists()
        
        if not exists:
            return False, "Tidak ada data preprocessed"
        
        # Get detailed count dengan pattern analysis
        detailed_stats = _get_detailed_preprocessed_stats(config)
        message = f"{count:,} total files - {detailed_stats}"
        
        return True, message
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error checking preprocessed: {str(e)}")
        return False, f"Error: {str(e)}"

def create_backend_preprocessor_with_progress(ui_components: Dict[str, Any]) -> Optional[Any]:
    """🔑 KEY FIX: Create backend preprocessor dengan proper progress integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        # Extract config dari UI components
        config = extract_preprocessing_config(ui_components)
        
        # 🎯 CRITICAL: Buat progress callback yang benar-benar terintegrasi
        def integrated_progress_callback(level: str, current: int, total: int, message: str):
            """Progress callback yang terintegrasi dengan UI tracker"""
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if not progress_tracker:
                    return
                
                # Map backend progress ke UI progress tracker
                if level == 'overall':
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(current, message)
                elif level == 'step':
                    if hasattr(progress_tracker, 'update_step'):
                        progress_tracker.update_step(current, message)
                elif level == 'current':
                    if hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(current, message)
                
                # 📝 Log progress ke UI (TANPA logging backend untuk avoid double log)
                from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                if current % max(1, total // 10) == 0 or current == total:  # Log milestone saja
                    log_to_accordion(ui_components, f"🔄 {message} ({current}/{total})", "info")
                    
            except Exception:
                pass  # Silent fail to prevent breaking the process
        
        # Create service dengan integrated progress callback
        service = create_preprocessing_service(config, progress_callback=integrated_progress_callback)
        
        # Store config untuk reference
        if service:
            service._ui_config = config
            service._ui_components = ui_components
        
        return service
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating preprocessor: {str(e)}")
        return None

def create_backend_checker(config: Dict[str, Any]):
    """Create backend validation service dengan UI integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        service = create_preprocessing_service(config, progress_callback=None)  # No progress untuk checker
        
        class ValidationServiceWrapper:
            def __init__(self, service):
                self.service = service
            
            def validate(self) -> Tuple[bool, str]:
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
        
        return ValidationServiceWrapper(service)
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating checker: {str(e)}")
        return None

def create_backend_cleanup_service_with_progress(ui_components: Dict[str, Any]):
    """Create backend cleanup service dengan progress integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        config = extract_preprocessing_config(ui_components)
        
        # 🎯 CRITICAL: Progress callback untuk cleanup
        def cleanup_progress_callback(level: str, current: int, total: int, message: str):
            """Progress callback khusus cleanup operations"""
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(current, message)
                    elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(current, message)
                
                # Log cleanup progress
                from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                if 'menghapus' in message.lower() or 'cleanup' in message.lower():
                    log_to_accordion(ui_components, f"🗑️ {message}", "info")
                    
            except Exception:
                pass
        
        service = create_preprocessing_service(config, progress_callback=cleanup_progress_callback)
        
        class CleanupServiceWrapper:
            def __init__(self, service, ui_components):
                self.service = service
                self.ui_components = ui_components
            
            def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
                try:
                    # Update progress manually untuk cleanup start
                    progress_tracker = self.ui_components.get('progress_tracker')
                    if progress_tracker and hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(10, "Memulai cleanup preprocessed data...")
                    
                    result = self.service.cleanup_preprocessed_data(target_split)
                    
                    # Update progress untuk completion
                    if progress_tracker and hasattr(progress_tracker, 'update_current'):
                        if result.get('success'):
                            progress_tracker.update_current(100, "Cleanup selesai")
                        else:
                            progress_tracker.update_current(0, "Cleanup gagal")
                    
                    return result
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Cleanup error: {str(e)}",
                        'stats': {'files_removed': 0}
                    }
        
        return CleanupServiceWrapper(service, ui_components)
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating cleanup service: {str(e)}")
        return None

def _get_detailed_preprocessed_stats(config: Dict[str, Any]) -> str:
    """Get detailed stats untuk preprocessed files"""
    try:
        preprocessing_config = config.get('preprocessing', {})
        output_dir = Path(preprocessing_config.get('output_dir', 'data/preprocessed'))
        
        if not output_dir.exists():
            return "Folder tidak ditemukan"
        
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        preprocessed_count = 0
        npy_count = 0
        jpg_count = 0
        
        for split in target_splits:
            split_dir = output_dir / split / 'images'
            if split_dir.exists():
                # Count preprocessed files (pre_*.npy dan pre_*.jpg)
                pre_npy = list(split_dir.glob('pre_*.npy'))
                pre_jpg = list(split_dir.glob('pre_*.jpg'))
                
                preprocessed_count += len(pre_npy) + len(pre_jpg)
                npy_count += len(pre_npy)
                jpg_count += len(pre_jpg)
        
        stats_parts = []
        if npy_count > 0:
            stats_parts.append(f"{npy_count:,} .npy files")
        if jpg_count > 0:
            stats_parts.append(f"{jpg_count:,} .jpg files")
        if preprocessed_count > 0:
            stats_parts.append(f"{preprocessed_count:,} total preprocessed")
        
        return ", ".join(stats_parts) if stats_parts else "File pattern tidak dikenali"
        
    except Exception:
        return "Error saat analisis file"

# 🔑 KEY: Compatibility functions untuk existing code
def create_backend_preprocessor(ui_components: Dict[str, Any]):
    """Compatibility wrapper yang menggunakan progress integration"""
    return create_backend_preprocessor_with_progress(ui_components)

def create_backend_cleanup_service(config: Dict[str, Any], ui_components: Optional[Dict[str, Any]] = None):
    """Compatibility wrapper untuk cleanup service"""
    if ui_components:
        return create_backend_cleanup_service_with_progress(ui_components)
    else:
        # Fallback tanpa progress untuk backward compatibility
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        service = create_preprocessing_service(config, progress_callback=None)
        
        class BasicCleanupWrapper:
            def __init__(self, service):
                self.service = service
            
            def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
                try:
                    return self.service.cleanup_preprocessed_data(target_split)
                except Exception as e:
                    return {'success': False, 'message': f"❌ Error: {str(e)}", 'stats': {'files_removed': 0}}
        
        return BasicCleanupWrapper(service)

def _convert_ui_to_backend_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Convert UI components ke backend config format"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        # Extract config dari UI
        ui_config = extract_preprocessing_config(ui_components)
        
        # Enhance untuk backend compatibility
        backend_config = _enhance_config_for_backend(ui_config)
        
        return backend_config
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error converting config: {str(e)}")
        
        # Fallback config
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
    """Enhance config untuk backend service compatibility"""
    enhanced = ui_config.copy()
    
    # Ensure required sections
    preprocessing = enhanced.setdefault('preprocessing', {})
    performance = enhanced.setdefault('performance', {})
    
    # Backend-specific enhancements
    preprocessing.setdefault('output_dir', 'data/preprocessed')
    preprocessing.setdefault('enabled', True)
    
    # Validation enhancements
    validation = preprocessing.setdefault('validation', {})
    validation.setdefault('enabled', True)
    validation.setdefault('move_invalid', True)
    
    # Normalization enhancements
    normalization = preprocessing.setdefault('normalization', {})
    normalization.setdefault('enabled', True)
    normalization.setdefault('method', 'minmax')
    normalization.setdefault('target_size', [640, 640])
    normalization.setdefault('preserve_aspect_ratio', True)
    
    # Performance enhancements
    performance.setdefault('batch_size', 32)
    performance.setdefault('use_gpu', True)
    
    return enhanced