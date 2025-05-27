"""
File: smartcash/dataset/augmentor/utils/core.py
Deskripsi: Updated core utilities dengan split-based directory support dan UUID consistency
"""

import os
import glob
import shutil
import cv2
from pathlib import Path
from typing import Dict, Any, List, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

# =============================================================================
# UPDATED PATH OPERATIONS - Split-aware
# =============================================================================

def resolve_drive_path(path: str) -> str:
    """Smart path resolution dengan prioritas Drive → Content → Local"""
    if not os.path.isabs(path):
        search_bases = ['/content/drive/MyDrive/SmartCash', '/content/drive/MyDrive', '/content/SmartCash', '/content', os.getcwd()]
        for base in search_bases:
            full_path = os.path.join(base, path)
            if os.path.exists(full_path): return full_path
    return path

def build_split_aware_paths(raw_dir: str, aug_dir: str, prep_dir: str, target_split: str = "train") -> Dict[str, str]:
    """Build split-aware paths untuk new directory structure"""
    resolved_raw = resolve_drive_path(raw_dir)
    resolved_aug = resolve_drive_path(aug_dir)
    resolved_prep = resolve_drive_path(prep_dir)
    
    return {
        'raw_dir': resolved_raw,
        'aug_dir': resolved_aug,
        'prep_dir': resolved_prep,
        # Split-specific paths
        'raw_split': f"{resolved_raw}/{target_split}",
        'aug_split': f"{resolved_aug}/{target_split}",
        'prep_split': f"{resolved_prep}/{target_split}",
        # Images/labels paths
        'raw_images': f"{resolved_raw}/{target_split}/images",
        'raw_labels': f"{resolved_raw}/{target_split}/labels",
        'aug_images': f"{resolved_aug}/{target_split}/images",
        'aug_labels': f"{resolved_aug}/{target_split}/labels",
        'prep_images': f"{resolved_prep}/{target_split}/images",
        'prep_labels': f"{resolved_prep}/{target_split}/labels"
    }

def find_split_directories(base_path: str) -> List[str]:
    """Find split directories (train, valid, test) dalam base path"""
    resolved_path = resolve_drive_path(base_path)
    split_dirs = []
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(resolved_path, split)
        if os.path.exists(split_path) and os.path.isdir(split_path):
            split_dirs.append(split_path)
    
    return split_dirs

def smart_find_images_split_aware(base_path: str, target_split: Optional[str] = None, extensions: List[str] = None) -> List[str]:
    """Smart image finder dengan split awareness"""
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    resolved_path = resolve_drive_path(base_path)
    
    # Strategy 1: Split-specific search
    if target_split:
        split_paths = [
            os.path.join(resolved_path, target_split, 'images'),
            os.path.join(resolved_path, target_split),
            os.path.join(resolved_path, 'images', target_split)
        ]
        
        for search_path in split_paths:
            if os.path.exists(search_path):
                for file in os.listdir(search_path):
                    if Path(file).suffix.lower() in extensions:
                        image_files.append(os.path.join(search_path, file))
    
    # Strategy 2: All splits search
    else:
        split_dirs = find_split_directories(resolved_path)
        for split_dir in split_dirs:
            images_dir = os.path.join(split_dir, 'images')
            search_dirs = [images_dir, split_dir] if os.path.exists(images_dir) else [split_dir]
            
            for search_dir in search_dirs:
                try:
                    for file in os.listdir(search_dir):
                        if Path(file).suffix.lower() in extensions:
                            image_files.append(os.path.join(search_dir, file))
                except (PermissionError, OSError):
                    continue
    
    # Strategy 3: Fallback to original method
    if not image_files:
        try:
            for ext in extensions:
                pattern = os.path.join(resolved_path, '**', f'*{ext}')
                image_files.extend(glob.glob(pattern, recursive=True))
        except Exception:
            pass
    
    return list(set(image_files))

# =============================================================================
# UPDATED FILE OPERATIONS - UUID aware
# =============================================================================

def find_augmented_files_split_aware(aug_dir: str, target_split: str = None) -> List[str]:
    """Find augmented files dengan split awareness dan UUID pattern"""
    resolved_dir = resolve_drive_path(aug_dir)
    aug_files = []
    
    if target_split:
        # Search specific split
        search_dirs = [
            os.path.join(resolved_dir, target_split, 'images'),
            os.path.join(resolved_dir, target_split)
        ]
    else:
        # Search all splits
        search_dirs = []
        for split in ['train', 'valid', 'test']:
            split_img_dir = os.path.join(resolved_dir, split, 'images')
            if os.path.exists(split_img_dir):
                search_dirs.append(split_img_dir)
    
    # Find files dengan aug_ prefix (UUID pattern)
    for search_dir in search_dirs:
        if os.path.exists(search_dir):
            for file_path in Path(search_dir).glob('aug_*.jpg'):
                aug_files.append(str(file_path))
    
    return aug_files

def copy_file_with_uuid_preservation(src: str, dst: str) -> bool:
    """Copy file dengan UUID preservation"""
    try:
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(resolve_drive_path(src), resolve_drive_path(dst))
        return True
    except Exception:
        return False

# =============================================================================
# UPDATED DATASET DETECTION - Split aware
# =============================================================================

def detect_split_structure(data_dir: str) -> Dict[str, Any]:
    """Enhanced dataset detection dengan split awareness"""
    resolved_dir = resolve_drive_path(data_dir)
    
    if not os.path.exists(resolved_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
    
    # Detect splits
    available_splits = []
    split_details = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(resolved_dir, split)
        if os.path.exists(split_dir):
            images = smart_find_images_split_aware(resolved_dir, split)
            labels = smart_find_images_split_aware(resolved_dir, split, ['.txt'])
            
            if images or labels:
                available_splits.append(split)
                split_details[split] = {
                    'path': split_dir,
                    'images': len(images),
                    'labels': len(labels),
                    'has_structure': os.path.exists(os.path.join(split_dir, 'images'))
                }
    
    # Overall statistics
    total_images = sum(details['images'] for details in split_details.values())
    total_labels = sum(details['labels'] for details in split_details.values())
    
    return {
        'status': 'success',
        'data_dir': resolved_dir,
        'total_images': total_images,
        'total_labels': total_labels,
        'available_splits': available_splits,
        'split_details': split_details,
        'structure_type': 'split_based' if available_splits else 'flat',
        'recommendations': [
            f'✅ Split structure: {len(available_splits)} splits tersedia' if available_splits else
            '⚠️ Tidak ada split structure - gunakan flat structure',
            f'📊 Total: {total_images} gambar, {total_labels} label',
            f'📁 Splits: {", ".join(available_splits)}' if available_splits else 'No splits detected'
        ]
    }

# =============================================================================
# UPDATED PROGRESS TRACKING - Real-time
# =============================================================================

class ProgressTracker:
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
    
    def progress(self, step: str, current: int, total: int, msg: str = ""):
        """Progress dengan immediate UI updates"""
        if self.comm and hasattr(self.comm, 'progress'):
            percentage = min(100, max(0, int((current / max(1, total)) * 100)))
            self.comm.progress(step, current, total, msg)
        
        if self.comm and hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback(None, step, current, total, msg)
    
    log_info = lambda self, msg: self.comm.log_info(msg) if self.comm else print(f"ℹ️ {msg}")
    log_success = lambda self, msg: self.comm.log_success(msg) if self.comm else print(f"✅ {msg}")
    log_error = lambda self, msg: self.comm.log_error(msg) if self.comm else print(f"❌ {msg}")

# =============================================================================
# UPDATED BATCH PROCESSING - Split aware
# =============================================================================

def process_batch_split_aware(items: List[Any], process_func: Callable, max_workers: int = None, 
                            progress_tracker: ProgressTracker = None, operation_name: str = "processing",
                            split_context: str = None) -> List[Dict[str, Any]]:
    """Batch processing dengan split context"""
    max_workers = max_workers or min(get_optimal_thread_count(), 8)
    results = []
    
    if not items:
        progress_tracker and progress_tracker.log_info("⚠️ Tidak ada item untuk diproses")
        return results
    
    total_items = len(items)
    context_msg = f" untuk split {split_context}" if split_context else ""
    progress_tracker and progress_tracker.log_info(f"🚀 {operation_name}{context_msg}: {total_items} item")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
        
        for completed_count, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                
                # Real-time progress dengan split context
                if progress_tracker:
                    msg_with_context = f"{operation_name}{context_msg}: {completed_count}/{total_items}"
                    progress_tracker.progress("current", completed_count, total_items, msg_with_context)
                
                # Log setiap 10%
                if completed_count % max(1, total_items // 10) == 0:
                    successful = sum(1 for r in results if r.get('status') == 'success')
                    success_rate = (successful / completed_count) * 100
                    progress_tracker and progress_tracker.log_info(
                        f"📊 Progress{context_msg}: {completed_count}/{total_items} ({success_rate:.1f}% berhasil)"
                    )
                    
            except Exception as e:
                results.append({'status': 'error', 'error': str(e)})
    
    # Final summary dengan split context
    successful = sum(1 for r in results if r.get('status') == 'success')
    progress_tracker and progress_tracker.log_success(
        f"✅ {operation_name}{context_msg} selesai: {successful}/{total_items} berhasil"
    )
    
    return results

# =============================================================================
# UPDATED CONFIG EXTRACTION - Split aware
# =============================================================================

def extract_split_aware_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Extract config dengan split awareness"""
    return {
        'raw_dir': resolve_drive_path(config.get('data', {}).get('dir', 'data')),
        'aug_dir': resolve_drive_path(config.get('augmentation', {}).get('output_dir', 'data/augmented')),
        'prep_dir': resolve_drive_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')),
        'num_variations': config.get('augmentation', {}).get('num_variations', 2),
        'target_count': config.get('augmentation', {}).get('target_count', 500),
        'types': config.get('augmentation', {}).get('types', ['combined']),
        'intensity': config.get('augmentation', {}).get('intensity', 0.7),
        'target_split': config.get('augmentation', {}).get('target_split', 'train'),
        'split_aware': True
    }

# =============================================================================
# UPDATED CLEANUP OPERATIONS - Split aware
# =============================================================================

def cleanup_split_aware(aug_dir: str, prep_dir: str = None, target_split: str = None, 
                       progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    """Cleanup dengan split awareness"""
    total_deleted = 0
    errors = []
    
    # Cleanup augmented files
    if target_split:
        # Specific split cleanup
        aug_split_dir = os.path.join(aug_dir, target_split)
        deleted = _cleanup_split_directory(aug_split_dir)
        total_deleted += deleted
        progress_tracker and progress_tracker.progress("overall", 50, 100, 
                                                     f"Cleanup {target_split}: {deleted} files")
    else:
        # All splits cleanup
        for split in ['train', 'valid', 'test']:
            aug_split_dir = os.path.join(aug_dir, split)
            if os.path.exists(aug_split_dir):
                deleted = _cleanup_split_directory(aug_split_dir)
                total_deleted += deleted
    
    # Cleanup preprocessed augmented files jika diminta
    if prep_dir:
        if target_split:
            prep_split_dir = os.path.join(prep_dir, target_split)
            deleted = _cleanup_augmented_from_split(prep_split_dir)
            total_deleted += deleted
        else:
            for split in ['train', 'valid', 'test']:
                prep_split_dir = os.path.join(prep_dir, split)
                if os.path.exists(prep_split_dir):
                    deleted = _cleanup_augmented_from_split(prep_split_dir)
                    total_deleted += deleted
    
    progress_tracker and progress_tracker.progress("overall", 100, 100, 
                                                 f"Split cleanup selesai: {total_deleted} files")
    
    return {
        'status': 'success' if total_deleted > 0 else 'empty',
        'total_deleted': total_deleted,
        'message': f"Split-aware cleanup: {total_deleted} file dihapus",
        'split_aware': True,
        'target_split': target_split,
        'errors': errors
    }

def _cleanup_split_directory(split_dir: str) -> int:
    """Cleanup single split directory"""
    deleted = 0
    split_path = Path(split_dir)
    
    if not split_path.exists():
        return deleted
    
    # Cleanup images dan labels subdirectories
    for subdir in ['images', 'labels']:
        subdir_path = split_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.glob('aug_*.*'):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception:
                    pass
    
    return deleted

def _cleanup_augmented_from_split(split_dir: str) -> int:
    """Cleanup hanya augmented files dari split directory"""
    deleted = 0
    split_path = Path(split_dir)
    
    if not split_path.exists():
        return deleted
    
    # Cleanup aug_ files dari images dan labels
    for subdir in ['images', 'labels']:
        subdir_path = split_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.glob('aug_*.*'):
                try:
                    file_path.unlink()
                    deleted += 1
                except Exception:
                    pass
    
    return deleted

# =============================================================================
# UPDATED FACTORY FUNCTION - Split aware
# =============================================================================

def create_split_aware_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    """Create context dengan split awareness"""
    aug_config = extract_split_aware_config(config)
    progress_tracker = ProgressTracker(communicator)
    target_split = aug_config.get('target_split', 'train')
    
    return {
        'config': aug_config,
        'progress': progress_tracker,
        'paths': build_split_aware_paths(aug_config['raw_dir'], aug_config['aug_dir'], 
                                       aug_config['prep_dir'], target_split),
        'detector': lambda: detect_split_structure(aug_config['raw_dir']),
        'cleaner': lambda split=None: cleanup_split_aware(aug_config['aug_dir'], aug_config['prep_dir'], 
                                                        split or target_split, progress_tracker),
        'comm': communicator,
        'split_aware': True,
        'target_split': target_split
    }

# Backward compatibility dengan split awareness
count_dataset_files_split_aware = lambda data_dir, split=None: (len(smart_find_images_split_aware(data_dir, split)), len(smart_find_images_split_aware(data_dir, split, ['.txt'])))

# One-liner utilities untuk split operations
get_split_path = lambda base_dir, split: os.path.join(resolve_drive_path(base_dir), split)
ensure_split_dirs = lambda base_dir, split: [Path(os.path.join(base_dir, split, subdir)).mkdir(parents=True, exist_ok=True) for subdir in ['images', 'labels']]
list_available_splits = lambda base_dir: [d.name for d in Path(resolve_drive_path(base_dir)).iterdir() if d.is_dir() and d.name in ['train', 'valid', 'test']]

# Update create_context untuk backward compatibility
create_context = create_split_aware_context  # Alias untuk backward compatibility

# Safe execute dengan split context
def safe_execute_split_aware(operation: Callable, fallback_result: Any = None, 
                            split_context: str = None, logger=None) -> Any:
    """Safe execute dengan split context information"""
    try:
        return operation()
    except Exception as e:
        context_msg = f" untuk split {split_context}" if split_context else ""
        error_msg = f"Operation failed{context_msg}: {str(e)}"
        logger and hasattr(logger, 'error') and logger.error(f"❌ {error_msg}")
        return fallback_result

# Enhanced utilities untuk research workflow
def validate_uuid_consistency(file_list: List[str]) -> Dict[str, Any]:
    """Validate UUID consistency across files"""
    from smartcash.common.utils.file_naming_manager import FileNamingManager
    
    naming_manager = FileNamingManager()
    uuid_groups = {}
    inconsistencies = []
    
    for file_path in file_list:
        filename = Path(file_path).name
        parsed = naming_manager.parse_existing_filename(filename)
        
        if parsed:
            uuid_key = parsed.uuid
            if uuid_key not in uuid_groups:
                uuid_groups[uuid_key] = []
            uuid_groups[uuid_key].append({'file': file_path, 'info': parsed})
    
    # Check untuk inconsistencies
    for uuid_key, files in uuid_groups.items():
        nominals = set(f['info'].nominal for f in files)
        if len(nominals) > 1:
            inconsistencies.append({
                'uuid': uuid_key,
                'files': [f['file'] for f in files],
                'nominals': list(nominals)
            })
    
    return {
        'total_files': len(file_list),
        'unique_uuids': len(uuid_groups),
        'inconsistencies': inconsistencies,
        'consistency_rate': (len(uuid_groups) - len(inconsistencies)) / max(len(uuid_groups), 1) * 100
    }

def generate_split_summary(base_dir: str) -> Dict[str, Any]:
    """Generate comprehensive split summary untuk monitoring"""
    split_summary = {'splits': {}, 'totals': {'images': 0, 'labels': 0}}
    
    for split in ['train', 'valid', 'test']:
        split_path = get_split_path(base_dir, split)
        if os.path.exists(split_path):
            images_count = len(smart_find_images_split_aware(base_dir, split))
            labels_count = len(smart_find_images_split_aware(base_dir, split, ['.txt']))
            
            split_summary['splits'][split] = {
                'path': split_path,
                'images': images_count,
                'labels': labels_count,
                'ratio': labels_count / max(images_count, 1)
            }
            
            split_summary['totals']['images'] += images_count
            split_summary['totals']['labels'] += labels_count
    
    split_summary['available_splits'] = list(split_summary['splits'].keys())
    split_summary['split_balance'] = _calculate_split_balance(split_summary['splits'])
    
    return split_summary

def _calculate_split_balance(splits: Dict[str, Dict]) -> str:
    """Calculate balance score untuk splits"""
    if not splits:
        return 'no_data'
    
    image_counts = [split_data['images'] for split_data in splits.values()]
    if not any(image_counts):
        return 'empty'
    
    total_images = sum(image_counts)
    percentages = [(count / total_images) * 100 for count in image_counts if total_images > 0]
    
    if not percentages:
        return 'no_data'
    
    # Calculate coefficient of variation
    mean_pct = sum(percentages) / len(percentages)
    variance = sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)
    cv = (variance ** 0.5) / mean_pct if mean_pct > 0 else 0
    
    if cv < 0.2:
        return 'excellent'
    elif cv < 0.4:
        return 'good'
    elif cv < 0.6:
        return 'moderate'
    else:
        return 'poor'

# Export key functions untuk backward compatibility
__all__ = [
    'resolve_drive_path', 'build_split_aware_paths', 'smart_find_images_split_aware',
    'find_augmented_files_split_aware', 'detect_split_structure', 'ProgressTracker',
    'process_batch_split_aware', 'cleanup_split_aware', 'create_split_aware_context',
    'validate_uuid_consistency', 'generate_split_summary'
]