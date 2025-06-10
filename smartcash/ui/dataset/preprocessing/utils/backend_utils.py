"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Fixed backend utils dengan proper progress integration dan path handling yang benar
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

def validate_dataset_ready(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate dataset readiness dengan path handling yang benar"""
    try:
        # Extract target splits
        preprocessing_config = config.get('preprocessing', {})
        target_splits = preprocessing_config.get('target_splits', ['train', 'valid'])
        
        if isinstance(target_splits, str):
            target_splits = [target_splits] if target_splits != 'all' else ['train', 'valid', 'test']
        
        missing_splits = []
        total_images = 0
        
        for split in target_splits:
            # 🔑 KEY: Use proper path resolution
            data_config = config.get('data', {})
            
            # Try config paths first
            if 'local' in data_config and split in data_config['local']:
                base_path = Path(data_config['local'][split])
            else:
                # Default path pattern
                base_dir = data_config.get('dir', 'data')
                base_path = Path(base_dir) / split
            
            img_dir = base_path / 'images'
            label_dir = base_path / 'labels'
            
            if not img_dir.exists() or not label_dir.exists():
                missing_splits.append(f"{split} (missing directories: {base_path})")
                continue
            
            # Count images dengan format yang benar
            image_count = 0
            for ext in ['.jpg', '.jpeg', '.png']:
                image_files = list(img_dir.glob(f'*{ext}'))
                # Filter hanya file dengan format yang benar (raw format expected)
                valid_files = [f for f in image_files 
                             if _is_valid_raw_format(f.name) or _needs_renaming(f.name)]
                image_count += len(valid_files)
            
            if image_count == 0:
                missing_splits.append(f"{split} (no valid images)")
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

def create_backend_preprocessor_with_progress(ui_components: Dict[str, Any]) -> Optional[Any]:
    """🔑 KEY FIX: Create backend preprocessor dengan proper progress integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        # Extract dan enhance config untuk backend
        config = _extract_and_enhance_config(ui_components)
        
        # 🎯 CRITICAL: Create progress callback yang benar-benar working
        def working_progress_callback(level: str, current: int, total: int, message: str):
            """Progress callback yang benar-benar terintegrasi dengan UI tracker"""
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if not progress_tracker:
                    return
                
                # 🔑 KEY: Ensure progress tracker is visible
                if hasattr(progress_tracker, 'show') and hasattr(progress_tracker, 'container'):
                    # Make sure container is visible
                    container = progress_tracker.container
                    if hasattr(container, 'layout'):
                        container.layout.visibility = 'visible'
                        container.layout.display = 'flex'
                
                # Map backend progress ke UI progress tracker dengan percentage
                progress_pct = int((current / total) * 100) if total > 0 else 0
                
                if level == 'overall':
                    if hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(progress_pct, message)
                elif level == 'step':
                    if hasattr(progress_tracker, 'update_step'):
                        progress_tracker.update_step(progress_pct, message)
                elif level == 'current':
                    if hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(progress_pct, message)
                
                # 📝 Log milestone progress ke UI (avoid flooding)
                if _is_progress_milestone(current, total):
                    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
                    log_to_accordion(ui_components, f"🔄 {message} ({current}/{total})", "info")
                    
            except Exception as e:
                # Silent fail to prevent breaking the process
                print(f"Progress callback error: {str(e)}")
        
        # Create service dengan integrated progress callback
        service = create_preprocessing_service(config, progress_callback=working_progress_callback)
        
        # Store enhanced config untuk reference
        if service:
            service._ui_config = config
            service._ui_components = ui_components
        
        return service
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating preprocessor: {str(e)}")
        return None

def create_backend_cleanup_service_with_progress(ui_components: Dict[str, Any]):
    """Create backend cleanup service dengan proper progress integration"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        
        config = _extract_and_enhance_config(ui_components)
        
        # 🎯 CRITICAL: Progress callback untuk cleanup
        def cleanup_progress_callback(level: str, current: int, total: int, message: str):
            """Progress callback khusus cleanup operations"""
            try:
                progress_tracker = ui_components.get('progress_tracker')
                if progress_tracker:
                    # Ensure visibility
                    if hasattr(progress_tracker, 'show') and hasattr(progress_tracker, 'container'):
                        container = progress_tracker.container
                        if hasattr(container, 'layout'):
                            container.layout.visibility = 'visible'
                            container.layout.display = 'flex'
                    
                    progress_pct = int((current / total) * 100) if total > 0 else 0
                    
                    if level == 'overall' and hasattr(progress_tracker, 'update_overall'):
                        progress_tracker.update_overall(progress_pct, message)
                    elif level == 'current' and hasattr(progress_tracker, 'update_current'):
                        progress_tracker.update_current(progress_pct, message)
                
                # Log cleanup progress
                if _is_progress_milestone(current, total):
                    from smartcash.ui.dataset.preprocessing.utils.ui_utils import log_to_accordion
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
                    # Ensure progress tracker is visible
                    progress_tracker = self.ui_components.get('progress_tracker')
                    if progress_tracker:
                        if hasattr(progress_tracker, 'show'):
                            progress_tracker.show()
                        if hasattr(progress_tracker, 'update_current'):
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

def _extract_and_enhance_config(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Extract dan enhance config untuk backend compatibility"""
    try:
        from smartcash.ui.dataset.preprocessing.handlers.config_extractor import extract_preprocessing_config
        
        # Extract config dari UI
        ui_config = extract_preprocessing_config(ui_components)
        
        # 🔑 KEY: Enhance untuk backend dengan proper paths
        enhanced_config = _enhance_config_for_backend(ui_config)
        
        return enhanced_config
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error extracting config: {str(e)}")
        
        # Fallback config dengan proper defaults
        return _get_fallback_config()

def _enhance_config_for_backend(ui_config: Dict[str, Any]) -> Dict[str, Any]:
    """Enhance config untuk backend service compatibility dengan proper paths"""
    enhanced = ui_config.copy()
    
    # Ensure required sections
    preprocessing = enhanced.setdefault('preprocessing', {})
    performance = enhanced.setdefault('performance', {})
    data = enhanced.setdefault('data', {})
    
    # 🔑 KEY: Setup proper data paths
    base_dir = data.get('dir', 'data')
    target_splits = preprocessing.get('target_splits', ['train', 'valid'])
    
    # Setup data paths untuk setiap split
    data.setdefault('local', {})
    for split in target_splits:
        if split not in data['local']:
            data['local'][split] = f"{base_dir}/{split}"
    
    # Backend-specific enhancements
    preprocessing.setdefault('output_dir', f"{base_dir}/preprocessed")
    preprocessing.setdefault('enabled', True)
    
    # Validation enhancements
    validation = preprocessing.setdefault('validation', {})
    validation.setdefault('enabled', True)
    validation.setdefault('move_invalid', True)
    validation.setdefault('invalid_dir', f"{base_dir}/invalid")
    
    # Normalization enhancements
    normalization = preprocessing.setdefault('normalization', {})
    normalization.setdefault('enabled', True)
    normalization.setdefault('method', 'minmax')
    normalization.setdefault('target_size', [640, 640])
    normalization.setdefault('preserve_aspect_ratio', True)
    
    # Performance enhancements
    performance.setdefault('batch_size', 32)
    performance.setdefault('use_gpu', True)
    
    # 🔑 KEY: File naming configuration
    enhanced.setdefault('file_naming', {
        'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
        'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'augmented_pattern': 'aug_rp_{nominal}_{uuid}_{sequence}_{variance}',
        'preserve_uuid': True
    })
    
    return enhanced

def _get_fallback_config() -> Dict[str, Any]:
    """Get fallback config dengan proper structure"""
    return {
        'preprocessing': {
            'enabled': True,
            'target_splits': ['train', 'valid'],
            'output_dir': 'data/preprocessed',
            'normalization': {'method': 'minmax', 'target_size': [640, 640], 'enabled': True},
            'validation': {'enabled': True, 'move_invalid': True, 'invalid_dir': 'data/invalid'}
        },
        'performance': {'batch_size': 32, 'use_gpu': True},
        'data': {
            'dir': 'data',
            'local': {
                'train': 'data/train',
                'valid': 'data/valid',
                'test': 'data/test'
            }
        },
        'file_naming': {
            'raw_pattern': 'rp_{nominal}_{uuid}_{sequence}',
            'preprocessed_pattern': 'pre_rp_{nominal}_{uuid}_{sequence}_{variance}',
            'preserve_uuid': True
        }
    }

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
        txt_count = 0
        
        for split in target_splits:
            split_dir = output_dir / split
            if split_dir.exists():
                # Count preprocessed files
                images_dir = split_dir / 'images'
                labels_dir = split_dir / 'labels'
                
                if images_dir.exists():
                    pre_npy = list(images_dir.glob('pre_*.npy'))
                    npy_count += len(pre_npy)
                    preprocessed_count += len(pre_npy)
                
                if labels_dir.exists():
                    pre_txt = list(labels_dir.glob('pre_*.txt'))
                    txt_count += len(pre_txt)
        
        stats_parts = []
        if npy_count > 0:
            stats_parts.append(f"{npy_count:,} .npy files")
        if txt_count > 0:
            stats_parts.append(f"{txt_count:,} .txt files")
        if preprocessed_count > 0:
            stats_parts.append(f"{preprocessed_count:,} preprocessed images")
        
        return ", ".join(stats_parts) if stats_parts else "File pattern tidak dikenali"
        
    except Exception:
        return "Error saat analisis file"

def _is_progress_milestone(current: int, total: int) -> bool:
    """Check if progress adalah milestone yang perlu di-log"""
    if total <= 10:
        return current % 2 == 0 or current == total
    
    # Log every 10% atau completion
    progress_pct = (current / total) * 100 if total > 0 else 0
    milestones = [0, 10, 25, 50, 75, 90, 100]
    return any(abs(progress_pct - milestone) < 2 for milestone in milestones) or current == total

def _is_valid_raw_format(filename: str) -> bool:
    """Check jika filename sudah dalam format raw yang benar"""
    import re
    # Pattern: rp_nominal_uuid_sequence.ext
    pattern = r'rp_\d{6}_[a-f0-9-]{36}_\d+\.\w+'
    return bool(re.match(pattern, filename, re.IGNORECASE))

def _needs_renaming(filename: str) -> bool:
    """Check jika file perlu di-rename ke format raw"""
    # Jika belum dalam format raw, maka perlu rename
    return not _is_valid_raw_format(filename)

# Backward compatibility exports
_convert_ui_to_backend_config = _extract_and_enhance_config
create_backend_preprocessor = create_backend_preprocessor_with_progress
create_backend_cleanup_service = create_backend_cleanup_service_with_progress