"""
File: smartcash/dataset/augmentor/utils/core.py
Deskripsi: Fixed core utilities dengan better error handling dan safe operations
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
# FILE OPERATIONS - Replaces file.py dengan safe_copy_file yang fixed
# =============================================================================

find_images = lambda dir_path, exts=['.jpg', '.jpeg', '.png']: [
    str(f) for f in Path(resolve_drive_path(dir_path)).rglob('*') 
    if f.suffix.lower() in exts and f.is_file()
] if path_exists(dir_path) else []

find_labels = lambda dir_path: find_images(dir_path, ['.txt'])
find_aug_files = lambda dir_path: [str(f) for f in Path(resolve_drive_path(dir_path)).rglob('aug_*.*') if f.is_file()] if path_exists(dir_path) else []

def copy_file(src: str, dst: str) -> bool:
    """Fixed copy file dengan proper error handling"""
    try:
        src_path = resolve_drive_path(src)
        dst_path = resolve_drive_path(dst)
        
        if not os.path.exists(src_path):
            return False
            
        # Ensure destination directory exists
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        shutil.copy2(src_path, dst_path)
        return True
    except Exception:
        return False

safe_copy_file = lambda src, dst: safe_execute(lambda: copy_file(src, dst), False)
delete_file = lambda path: safe_execute(lambda: Path(resolve_drive_path(path)).unlink(), False) if path_exists(path) else False
get_file_size = lambda path: safe_execute(lambda: Path(resolve_drive_path(path)).stat().st_size, 0) if path_exists(path) else 0

# =============================================================================
# IMAGE OPERATIONS - Replaces image.py
# =============================================================================

def read_image(path: str):
    """Fixed read image dengan proper error handling"""
    try:
        resolved_path = resolve_drive_path(path)
        if not os.path.exists(resolved_path):
            return None
        image = cv2.imread(resolved_path)
        return image if image is not None and image.size > 0 else None
    except Exception:
        return None

def save_image(img, path: str) -> bool:
    """Fixed save image dengan proper error handling"""
    try:
        if img is None or img.size == 0:
            return False
        resolved_path = resolve_drive_path(path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        return cv2.imwrite(resolved_path, img)
    except Exception:
        return False

validate_image = lambda img: img is not None and hasattr(img, 'size') and img.size > 0
resize_image = lambda img, size: cv2.resize(img, size) if validate_image(img) else None

# =============================================================================
# BBOX OPERATIONS - Replaces bbox.py
# =============================================================================

def parse_yolo_line(line: str) -> List[float]:
    """Fixed YOLO line parser dengan better validation"""
    try:
        parts = line.strip().split()
        if len(parts) < 5:
            return []
        
        # Convert class_id to int and coordinates to float
        class_id = int(float(parts[0]))
        coords = [float(x) for x in parts[1:5]]
        
        # Validate coordinates are in [0, 1] range
        if all(0 <= x <= 1 for x in coords):
            return [class_id] + coords
        return []
    except (ValueError, IndexError):
        return []

format_yolo_line = lambda bbox: f"{int(bbox[0])} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}" if len(bbox) >= 5 else ""
validate_bbox = lambda bbox: len(bbox) >= 5 and isinstance(bbox[0], (int, float)) and all(0 <= x <= 1 for x in bbox[1:5])

def read_yolo_labels(label_path: str) -> List[List[float]]:
    """Fixed YOLO labels reader dengan better error handling"""
    try:
        resolved_path = resolve_drive_path(label_path)
        if not os.path.exists(resolved_path):
            return []
            
        with open(resolved_path, 'r') as f:
            valid_bboxes = []
            for line in f:
                bbox = parse_yolo_line(line)
                if bbox and validate_bbox(bbox):
                    valid_bboxes.append(bbox)
            return valid_bboxes
    except Exception:
        return []

def save_yolo_labels(bboxes: List[List[float]], output_path: str) -> bool:
    """Fixed YOLO labels saver dengan better error handling"""
    try:
        resolved_path = resolve_drive_path(output_path)
        os.makedirs(os.path.dirname(resolved_path), exist_ok=True)
        
        with open(resolved_path, 'w') as f:
            for bbox in bboxes:
                if validate_bbox(bbox):
                    formatted_line = format_yolo_line(bbox)
                    if formatted_line:
                        f.write(formatted_line + '\n')
        return True
    except Exception:
        return False

# =============================================================================
# DATASET DETECTION - Replaces dataset_detector.py
# =============================================================================

def detect_structure(data_dir: str) -> Dict[str, Any]:
    """Fixed dataset structure detection dengan better error handling"""
    try:
        resolved_dir = resolve_drive_path(data_dir)
        
        if not path_exists(resolved_dir):
            return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
        
        images = find_images(resolved_dir)
        labels = find_labels(resolved_dir)
        has_yolo = path_exists(f"{resolved_dir}/images") and path_exists(f"{resolved_dir}/labels")
        splits = [s for s in ['train', 'valid', 'test'] if path_exists(f"{resolved_dir}/{s}")]
        
        # Better image location detection
        image_locations = []
        if has_yolo:
            image_locations.append({'path': f"{resolved_dir}/images", 'count': len(find_images(f"{resolved_dir}/images"))})
        else:
            image_locations.append({'path': resolved_dir, 'count': len(images)})
        
        return {
            'status': 'success', 
            'data_dir': resolved_dir, 
            'total_images': len(images),
            'total_labels': len(labels), 
            'structure_type': 'standard_yolo' if has_yolo else 'mixed',
            'splits_detected': splits, 
            'image_locations': image_locations,
            'recommendations': ['✅ Dataset siap untuk augmentasi' if images else '❌ Tidak ada gambar ditemukan']
        }
    except Exception as e:
        return {'status': 'error', 'message': f'Error detecting structure: {str(e)}'}

# =============================================================================
# PROGRESS TRACKING - Consolidates progress patterns
# =============================================================================

class ProgressTracker:
    """Fixed progress tracker dengan better error handling"""
    
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
    
    def progress(self, step: str, pct: int, msg: str = ""):
        """Fixed progress method dengan safety checks"""
        try:
            if self.comm and hasattr(self.comm, 'progress'):
                self.comm.progress(step, pct, 100, msg)
        except Exception:
            pass
    
    def log_info(self, msg: str):
        """Fixed log info dengan fallback"""
        try:
            if self.logger and hasattr(self.logger, 'info'):
                self.logger.info(f"ℹ️ {msg}")
            else:
                print(f"ℹ️ {msg}")
        except Exception:
            print(f"ℹ️ {msg}")
    
    def log_success(self, msg: str):
        """Fixed log success dengan fallback"""
        try:
            if self.logger and hasattr(self.logger, 'success'):
                self.logger.success(msg)
            else:
                print(f"✅ {msg}")
        except Exception:
            print(f"✅ {msg}")
    
    def log_error(self, msg: str):
        """Fixed log error dengan fallback"""
        try:
            if self.logger and hasattr(self.logger, 'error'):
                self.logger.error(f"❌ {msg}")
            else:
                print(f"❌ {msg}")
        except Exception:
            print(f"❌ {msg}")

# =============================================================================
# BATCH PROCESSING - Replaces batch.py
# =============================================================================

def process_batch(items: List[Any], process_func: Callable, max_workers: int = None, 
                 progress_tracker: ProgressTracker = None) -> List[Dict[str, Any]]:
    """Fixed batch processing dengan better error handling"""
    if not items:
        return []
        
    max_workers = max_workers or min(get_optimal_thread_count(), 8)
    results = []
    
    try:
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
    except Exception as e:
        # If threading fails, process sequentially
        for item in items:
            try:
                result = process_func(item)
                results.append(result)
            except Exception as e:
                results.append({'status': 'error', 'error': str(e)})
    
    return results

# =============================================================================
# CONFIG EXTRACTION - Consolidates config patterns
# =============================================================================

def extract_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Fixed config extraction dengan better defaults"""
    try:
        return {
            'raw_dir': resolve_drive_path(config.get('data', {}).get('dir', 'data')),
            'aug_dir': resolve_drive_path(config.get('augmentation', {}).get('output_dir', 'data/augmented')),
            'prep_dir': resolve_drive_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')),
            'num_variations': max(1, config.get('augmentation', {}).get('num_variations', 2)),
            'target_count': max(1, config.get('augmentation', {}).get('target_count', 500)),
            'types': config.get('augmentation', {}).get('types', ['combined']),
            'intensity': max(0.1, min(1.0, config.get('augmentation', {}).get('intensity', 0.7)))
        }
    except Exception:
        # Fallback config
        return {
            'raw_dir': 'data', 'aug_dir': 'data/augmented', 'prep_dir': 'data/preprocessed',
            'num_variations': 2, 'target_count': 500, 'types': ['combined'], 'intensity': 0.7
        }

get_best_data_location = lambda: next((path for path in [
    '/content/drive/MyDrive/SmartCash/data', '/content/drive/MyDrive/data', 
    '/content/SmartCash/data', '/content/data', 'data'
] if path_exists(path) or path_exists(get_parent(path))), 'data')

# =============================================================================
# ERROR HANDLING - Consolidates try-catch patterns
# =============================================================================

def safe_execute(operation: Callable, fallback_result: Any = None, logger=None) -> Any:
    """Fixed safe execution dengan better error handling"""
    try:
        return operation()
    except Exception as e:
        if logger and hasattr(logger, 'error'):
            logger.error(f"❌ Operation failed: {str(e)}")
        return fallback_result

safe_create_dir = lambda path: safe_execute(lambda: Path(resolve_drive_path(path)).mkdir(parents=True, exist_ok=True), False)
safe_read_image = lambda path: safe_execute(lambda: read_image(path), None)

# =============================================================================
# CLEANUP OPERATIONS - Replaces cleaner.py
# =============================================================================

def cleanup_files(aug_dir: str, prep_dir: str = None, progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    """Fixed cleanup dengan better error handling"""
    total_deleted = 0
    errors = []
    
    try:
        aug_files = find_aug_files(aug_dir)
        progress_tracker and progress_tracker.progress("overall", 20, f"Found {len(aug_files)} aug files")
        
        for i, file_path in enumerate(aug_files):
            if delete_file(file_path):
                total_deleted += 1
            else:
                errors.append(f"Failed to delete {file_path}")
            
            if progress_tracker and i % max(1, len(aug_files) // 10) == 0:
                progress_tracker.progress("overall", 20 + int((i/len(aug_files)) * 60), f"Deleting: {i}/{len(aug_files)}")
        
        if prep_dir and path_exists(prep_dir):
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
    except Exception as e:
        return {
            'status': 'error',
            'total_deleted': total_deleted,
            'message': f"Error during cleanup: {str(e)}",
            'errors': errors + [str(e)]
        }

# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    """Fixed context creation dengan better error handling"""
    try:
        aug_config = extract_config(config)
        progress_tracker = ProgressTracker(communicator)
        
        return {
            'config': aug_config,
            'progress': progress_tracker,
            'paths': build_paths(aug_config['raw_dir'], aug_config['aug_dir'], aug_config['prep_dir']),
            'detector': lambda: detect_structure(aug_config['raw_dir']),
            'cleaner': lambda: cleanup_files(aug_config['aug_dir'], aug_config['prep_dir'], progress_tracker)
        }
    except Exception as e:
        # Fallback context
        fallback_config = {'raw_dir': 'data', 'aug_dir': 'data/augmented', 'prep_dir': 'data/preprocessed'}
        progress_tracker = ProgressTracker(communicator)
        
        return {
            'config': fallback_config,
            'progress': progress_tracker,
            'paths': build_paths('data', 'data/augmented', 'data/preprocessed'),
            'detector': lambda: {'status': 'error', 'message': 'Context creation failed'},
            'cleaner': lambda: {'status': 'error', 'message': 'Cleaner not available'}
        }

# Backward compatibility aliases
count_dataset_files = lambda data_dir: (len(find_images(data_dir)), len(find_labels(data_dir)))
validate_dataset = lambda data_dir: detect_structure(data_dir)['total_images'] > 0