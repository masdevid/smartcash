"""
File: smartcash/ui/dataset/preprocessing/utils/backend_utils.py
Deskripsi: Simplified backend utils dengan clear domain separation
"""

from typing import Dict, Any, Tuple, Optional, Callable
from pathlib import Path
from smartcash.common.logger import get_logger

def validate_dataset_ready(config: Dict[str, Any]) -> Tuple[bool, str]:
    """Validate dataset readiness"""
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
    """Check preprocessed data existence menggunakan backend service"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        service = create_preprocessing_service(config)
        
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
        augmented_count = 0
        npy_count = 0
        
        for split in target_splits:
            split_dir = output_dir / split / 'images'
            if split_dir.exists():
                # Count preprocessed files (pre_*.npy)
                pre_files = list(split_dir.glob('pre_*.npy'))
                preprocessed_count += len(pre_files)
                
                # Count augmented files (aug_*.npy) 
                aug_files = list(split_dir.glob('aug_*.npy'))
                augmented_count += len(aug_files)
                
                # Count all .npy files
                all_npy = list(split_dir.glob('*.npy'))
                npy_count += len(all_npy)
        
        stats_parts = []
        if preprocessed_count > 0:
            stats_parts.append(f"{preprocessed_count:,} preprocessed")
        if augmented_count > 0:
            stats_parts.append(f"{augmented_count:,} augmented")
        if npy_count > 0:
            stats_parts.append(f"{npy_count:,} .npy files")
        
        return ", ".join(stats_parts) if stats_parts else "File pattern tidak dikenali"
        
    except Exception:
        return "Error saat analisis file"

def create_backend_preprocessor(config: Dict[str, Any], progress_callback: Optional[Callable] = None):
    """Create backend preprocessing service"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        return create_preprocessing_service(config, progress_callback)
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating preprocessor: {str(e)}")
        return None

def create_backend_checker(config: Dict[str, Any]):
    """Create backend validation service"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        service = create_preprocessing_service(config)
        
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

def create_backend_cleanup_service(config: Dict[str, Any], ui_components: Optional[Dict[str, Any]] = None):
    """Create backend cleanup service"""
    try:
        from smartcash.dataset.preprocessor.service import create_preprocessing_service
        service = create_preprocessing_service(config)
        
        class CleanupServiceWrapper:
            def __init__(self, service):
                self.service = service
            
            def cleanup_preprocessed_data(self, target_split: str = None) -> Dict[str, Any]:
                try:
                    return self.service.cleanup_preprocessed_data(target_split)
                except Exception as e:
                    return {
                        'success': False,
                        'message': f"❌ Cleanup error: {str(e)}",
                        'stats': {'files_removed': 0}
                    }
        
        return CleanupServiceWrapper(service)
        
    except Exception as e:
        get_logger('backend_utils').error(f"Error creating cleanup service: {str(e)}")
        return None

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