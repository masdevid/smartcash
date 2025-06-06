"""
File: smartcash/dataset/augmentor/utils/core.py
Deskripsi: Slim core module dengan imports dari SRP modules dan backward compatibility
"""

# Import semua utilities dari SRP modules untuk backward compatibility
from smartcash.dataset.augmentor.utils.path_operations import (
    resolve_drive_path, find_dataset_directories, smart_find_images,
    build_paths, build_split_aware_paths, ensure_dirs, ensure_split_dirs,
    path_exists, get_stem, get_parent, get_split_path, list_available_splits,
    get_best_data_location
)

from smartcash.dataset.augmentor.utils.file_operations import (
    find_images, find_labels, find_aug_files, copy_file, safe_copy_file,
    delete_file, get_file_size, read_image, save_image, validate_image,
    resize_image, safe_read_image, find_augmented_files_split_aware,
    smart_find_images_split_aware, copy_file_with_uuid_preservation
)

from smartcash.dataset.augmentor.utils.bbox_operations import (
    parse_yolo_line, format_yolo_line, validate_bbox, read_yolo_labels,
    save_yolo_labels, load_yolo_labels, save_validated_labels
)

from smartcash.dataset.augmentor.utils.progress_tracker import (
    ProgressTracker, create_progress_tracker
)

from smartcash.dataset.augmentor.utils.batch_processor import (
    process_batch, process_batch_split_aware, create_batch_processor
)

from smartcash.dataset.augmentor.utils.dataset_detector import (
    detect_structure, detect_split_structure, validate_dataset,
    count_dataset_files, count_dataset_files_split_aware
)

from smartcash.dataset.augmentor.utils.cleanup_operations import (
    cleanup_files, cleanup_split_aware, cleanup_augmented_files, cleanup_split_data
)

from smartcash.dataset.augmentor.utils.config_extractor import (
    extract_config, extract_split_aware_config, create_context, create_split_aware_context
)

# =============================================================================
# ADDITIONAL UTILITIES - UUID & Research specific
# =============================================================================

def validate_uuid_consistency(file_list) -> dict:
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
    
    # Check inconsistencies
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

def generate_split_summary(base_dir: str) -> dict:
    """Generate comprehensive split summary"""
    split_summary = {'splits': {}, 'totals': {'images': 0, 'labels': 0}}
    
    for split in ['train', 'valid', 'test']:
        split_path = get_split_path(base_dir, split)
        if path_exists(split_path):
            images_count = len(smart_find_images_split_aware(base_dir, split))
            labels_count = len(smart_find_images_split_aware(base_dir, split, ['.txt']))
            
            split_summary['splits'][split] = {
                'path': split_path, 'images': images_count, 'labels': labels_count,
                'ratio': labels_count / max(images_count, 1)
            }
            
            split_summary['totals']['images'] += images_count
            split_summary['totals']['labels'] += labels_count
    
    split_summary['available_splits'] = list(split_summary['splits'].keys())
    split_summary['split_balance'] = _calculate_split_balance(split_summary['splits'])
    
    return split_summary

def _calculate_split_balance(splits: dict) -> str:
    """Calculate balance score for splits"""
    if not splits:
        return 'no_data'
    
    image_counts = [split_data['images'] for split_data in splits.values()]
    if not any(image_counts):
        return 'empty'
    
    total_images = sum(image_counts)
    percentages = [(count / total_images) * 100 for count in image_counts if total_images > 0]
    
    if not percentages:
        return 'no_data'
    
    mean_pct = sum(percentages) / len(percentages)
    variance = sum((p - mean_pct) ** 2 for p in percentages) / len(percentages)
    cv = (variance ** 0.5) / mean_pct if mean_pct > 0 else 0
    
    return 'excellent' if cv < 0.2 else 'good' if cv < 0.4 else 'moderate' if cv < 0.6 else 'poor'

def safe_execute_split_aware(operation, fallback_result=None, split_context: str = None, logger=None):
    """Safe execute dengan split context"""
    try:
        return operation()
    except Exception as e:
        context_msg = f" untuk split {split_context}" if split_context else ""
        error_msg = f"Operation failed{context_msg}: {str(e)}"
        if logger and hasattr(logger, 'error'):
            logger.error(f"❌ {error_msg}")
        return fallback_result

# =============================================================================
# BACKWARD COMPATIBILITY EXPORTS
# =============================================================================

__all__ = [
    # Path operations
    'resolve_drive_path', 'build_paths', 'build_split_aware_paths', 'smart_find_images',
    'find_augmented_files_split_aware', 'ensure_dirs', 'path_exists', 'get_best_data_location',
    
    # File operations  
    'find_images', 'find_labels', 'copy_file', 'read_image', 'save_image',
    
    # Bbox operations
    'read_yolo_labels', 'save_yolo_labels', 'validate_bbox',
    
    # Processing
    'ProgressTracker', 'process_batch', 'process_batch_split_aware',
    'detect_structure', 'detect_split_structure', 'cleanup_files', 'cleanup_split_aware',
    
    # Context creation
    'create_context', 'create_split_aware_context', 'extract_config',
    
    # Research utilities
    'validate_uuid_consistency', 'generate_split_summary'
]