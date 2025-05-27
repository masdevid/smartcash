"""
File: smartcash/dataset/augmentor/utils/core.py
Deskripsi: Fixed core utilities dengan safe_copy_file yang hilang dan consolidated one-liner operations
"""

import os
import glob
import shutil
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union, Callable
from collections import defaultdict, Counter
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

from smartcash.common.logger import get_logger
from smartcash.common.threadpools import get_optimal_thread_count

# =============================================================================
# PATH OPERATIONS - Replaces paths.py
# =============================================================================

resolve_drive_path = lambda path: next((os.path.join(base, path) for base in [
    '/content/drive/MyDrive/SmartCash', '/content/drive/MyDrive', '/content'
] if os.path.exists(os.path.join(base, path))), path)

build_paths = lambda raw_dir, aug_dir, prep_dir, split="train": {
    'raw_dir': resolve_drive_path(raw_dir), 'aug_dir': resolve_drive_path(aug_dir),
    'prep_dir': resolve_drive_path(prep_dir), 'aug_images': f"{resolve_drive_path(aug_dir)}/images",
    'aug_labels': f"{resolve_drive_path(aug_dir)}/labels", 
    'prep_split': f"{resolve_drive_path(prep_dir)}/{split}"
}

ensure_dirs = lambda *paths: [Path(resolve_drive_path(p)).mkdir(parents=True, exist_ok=True) for p in paths]
path_exists = lambda path: Path(resolve_drive_path(path)).exists()
get_stem = lambda path: Path(path).stem
get_parent = lambda path: str(Path(path).parent)

# =============================================================================
# FILE OPERATIONS - Replaces file.py dengan safe_copy_file yang hilang
# =============================================================================

find_images = lambda dir_path, exts=['.jpg', '.jpeg', '.png']: [
    str(f) for f in Path(resolve_drive_path(dir_path)).rglob('*') 
    if f.suffix.lower() in exts and f.is_file()
]
find_labels = lambda dir_path: find_images(dir_path, ['.txt'])
find_aug_files = lambda dir_path: [str(f) for f in Path(resolve_drive_path(dir_path)).rglob('aug_*.*') if f.is_file()]

copy_file = lambda src, dst: shutil.copy2(resolve_drive_path(src), resolve_drive_path(dst)) if path_exists(src) else False
safe_copy_file = lambda src, dst: safe_execute(lambda: copy_file(src, dst), False)  # ← Added missing function
delete_file = lambda path: Path(resolve_drive_path(path)).unlink() if path_exists(path) else False
get_file_size = lambda path: Path(resolve_drive_path(path)).stat().st_size if path_exists(path) else 0

# =============================================================================
# IMAGE OPERATIONS - Replaces image.py
# =============================================================================

read_image = lambda path: cv2.imread(resolve_drive_path(path)) if path_exists(path) else None
save_image = lambda img, path: cv2.imwrite(resolve_drive_path(path), img) if img is not None else False
validate_image = lambda img: img is not None and img.size > 0
resize_image = lambda img, size: cv2.resize(img, size) if validate_image(img) else None

# =============================================================================
# BBOX OPERATIONS - Replaces bbox.py
# =============================================================================

parse_yolo_line = lambda line: [int(float(parts[0]))] + [float(x) for x in parts[1:]] if (parts := line.strip().split()) and len(parts) >= 5 else []
format_yolo_line = lambda bbox: f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}"
validate_bbox = lambda bbox: len(bbox) >= 5 and isinstance(bbox[0], (int, float)) and all(0 <= x <= 1 for x in bbox[1:5])

def read_yolo_labels(label_path: str) -> List[List[float]]:
    try:
        with open(resolve_drive_path(label_path), 'r') as f:
            return [bbox for line in f if (bbox := parse_yolo_line(line)) and validate_bbox(bbox)]
    except Exception:
        return []

def save_yolo_labels(bboxes: List[List[float]], output_path: str) -> bool:
    try:
        ensure_dirs(get_parent(output_path))
        with open(resolve_drive_path(output_path), 'w') as f:
            [f.write(format_yolo_line(bbox) + '\n') for bbox in bboxes if validate_bbox(bbox)]
        return True
    except Exception:
        return False

# =============================================================================
# DATASET DETECTION - Replaces dataset_detector.py
# =============================================================================

def detect_structure(data_dir: str) -> Dict[str, Any]:
    resolved_dir = resolve_drive_path(data_dir)
    
    if not path_exists(resolved_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
    
    images = find_images(resolved_dir)
    labels = find_labels(resolved_dir)
    has_yolo = path_exists(f"{resolved_dir}/images") and path_exists(f"{resolved_dir}/labels")
    splits = [s for s in ['train', 'valid', 'test'] if path_exists(f"{resolved_dir}/{s}")]
    
    return {
        'status': 'success', 'data_dir': resolved_dir, 'total_images': len(images),
        'total_labels': len(labels), 'structure_type': 'standard_yolo' if has_yolo else 'mixed',
        'splits_detected': splits, 'image_locations': [{'path': resolved_dir, 'count': len(images)}],
        'recommendations': ['✅ Dataset siap untuk augmentasi' if images else '❌ Tidak ada gambar ditemukan']
    }

# =============================================================================
# PROGRESS TRACKING - Consolidates progress patterns
# =============================================================================

class ProgressTracker:
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
    
    progress = lambda self, step, pct, msg="": self.comm and hasattr(self.comm, 'progress') and self.comm.progress(step, pct, 100, msg)
    log_info = lambda self, msg: getattr(self.logger, 'info', print)(f"ℹ️ {msg}")
    log_success = lambda self, msg: getattr(self.logger, 'success', lambda x: print(f"✅ {x}"))(msg)
    log_error = lambda self, msg: getattr(self.logger, 'error', lambda x: print(f"❌ {x}"))(msg)

# =============================================================================
# BATCH PROCESSING - Replaces batch.py
# =============================================================================

def process_batch(items: List[Any], process_func: Callable, max_workers: int = None, 
                 progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
    max_workers = max_workers or min(get_optimal_thread_count(), 8)
    results = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
        
        for completed, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                if progress_tracker and completed % max(1, len(items) // 10) == 0:
                    progress_tracker.progress("current", int(completed/len(items)*100), f"Processed {completed}/{len(items)}")
            except Exception as e:
                results.append({'status': 'error', 'error': str(e)})
    
    return results

# =============================================================================
# CONFIG EXTRACTION - Consolidates config patterns
# =============================================================================

extract_config = lambda config: {
    'raw_dir': resolve_drive_path(config.get('data', {}).get('dir', 'data')),
    'aug_dir': resolve_drive_path(config.get('augmentation', {}).get('output_dir', 'data/augmented')),
    'prep_dir': resolve_drive_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')),
    'num_variations': config.get('augmentation', {}).get('num_variations', 2),
    'target_count': config.get('augmentation', {}).get('target_count', 500),
    'types': config.get('augmentation', {}).get('types', ['combined']),
    'intensity': config.get('augmentation', {}).get('intensity', 0.7)
}

get_best_data_location = lambda: next((path for path in [
    '/content/drive/MyDrive/SmartCash/data', '/content/drive/MyDrive/data', 
    '/content/SmartCash/data', '/content/data', 'data'
] if path_exists(path) or path_exists(get_parent(path))), 'data')

# =============================================================================
# ERROR HANDLING - Consolidates try-catch patterns
# =============================================================================

def safe_execute(operation: Callable, fallback_result: Any = None, logger=None) -> Any:
    try:
        return operation()
    except Exception as e:
        logger and hasattr(logger, 'error') and logger.error(f"❌ Operation failed: {str(e)}")
        return fallback_result

safe_create_dir = lambda path: safe_execute(lambda: Path(resolve_drive_path(path)).mkdir(parents=True, exist_ok=True), False)
safe_read_image = lambda path: safe_execute(lambda: read_image(path), None)

# =============================================================================
# CLEANUP OPERATIONS - Replaces cleaner.py
# =============================================================================

def cleanup_files(aug_dir: str, prep_dir: str = None, progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    total_deleted = 0
    errors = []
    
    aug_files = find_aug_files(aug_dir)
    progress_tracker and progress_tracker.progress("overall", 20, f"Found {len(aug_files)} aug files")
    
    for i, file_path in enumerate(aug_files):
        if delete_file(file_path):
            total_deleted += 1
        else:
            errors.append(f"Failed to delete {file_path}")
        
        if progress_tracker and i % max(1, len(aug_files) // 10) == 0:
            progress_tracker.progress("overall", 20 + int((i/len(aug_files)) * 60), f"Deleting: {i}/{len(aug_files)}")
    
    if prep_dir:
        prep_files = find_aug_files(prep_dir)
        for file_path in prep_files:
            if delete_file(file_path):
                total_deleted += 1
            else:
                errors.append(f"Failed to delete {file_path}")
    
    progress_tracker and progress_tracker.progress("overall", 100, f"Cleanup complete: {total_deleted} files deleted")
    
    return {
        'status': 'success' if total_deleted > 0 else 'empty',
        'total_deleted': total_deleted,
        'message': f"Berhasil menghapus {total_deleted} file" if total_deleted > 0 else "Tidak ada file untuk dihapus",
        'errors': errors
    }

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    aug_config = extract_config(config)
    progress_tracker = ProgressTracker(communicator)
    
    return {
        'config': aug_config,
        'progress': progress_tracker,
        'paths': build_paths(aug_config['raw_dir'], aug_config['aug_dir'], aug_config['prep_dir']),
        'detector': lambda: detect_structure(aug_config['raw_dir']),
        'cleaner': lambda: cleanup_files(aug_config['aug_dir'], aug_config['prep_dir'], progress_tracker)
    }

# Backward compatibility aliases
count_dataset_files = lambda data_dir: (len(find_images(data_dir)), len(find_labels(data_dir)))
validate_dataset = lambda data_dir: detect_structure(data_dir)['total_images'] > 0