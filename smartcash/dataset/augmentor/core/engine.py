"""
File: smartcash/dataset/augmentor/utils/core.py
Deskripsi: Fixed core utilities dengan real-time progress tracking untuk resolve stepping updates
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
# ENHANCED PATH OPERATIONS - Unchanged
# =============================================================================

def resolve_drive_path(path: str) -> str:
    """Smart path resolution dengan prioritas Drive → Content → Local"""
    if not os.path.isabs(path):
        search_bases = ['/content/drive/MyDrive/SmartCash', '/content/drive/MyDrive', '/content/SmartCash', '/content', os.getcwd()]
        for base in search_bases:
            full_path = os.path.join(base, path)
            if os.path.exists(full_path): return full_path
    return path

def find_dataset_directories(base_path: str) -> List[str]:
    """Find all possible dataset directories dengan recursive search"""
    resolved_path = resolve_drive_path(base_path)
    yolo_patterns = [resolved_path, os.path.join(resolved_path, 'data'), os.path.join(resolved_path, 'dataset'), os.path.join(resolved_path, 'images')]
    
    # Add split directories
    for split in ['train', 'valid', 'test', 'val']:
        yolo_patterns.extend([os.path.join(resolved_path, split), os.path.join(resolved_path, split, 'images'), os.path.join(resolved_path, 'data', split), os.path.join(resolved_path, 'data', split, 'images')])
    
    return list(set([pattern for pattern in yolo_patterns if os.path.exists(pattern) and os.path.isdir(pattern)]))

def smart_find_images(base_path: str, extensions: List[str] = None) -> List[str]:
    """Smart image finder dengan multiple search strategies"""
    extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    image_files = []
    
    # Strategy 1: Direct directory scan
    for dir_path in find_dataset_directories(base_path):
        try:
            # Scan current directory
            for file in os.listdir(dir_path):
                file_path = os.path.join(dir_path, file)
                if os.path.isfile(file_path) and Path(file).suffix.lower() in extensions:
                    image_files.append(file_path)
            
            # Scan images subdirectory if exists
            images_subdir = os.path.join(dir_path, 'images')
            if os.path.exists(images_subdir):
                for file in os.listdir(images_subdir):
                    file_path = os.path.join(images_subdir, file)
                    if os.path.isfile(file_path) and Path(file).suffix.lower() in extensions:
                        image_files.append(file_path)
        except (PermissionError, OSError):
            continue
    
    # Strategy 2: Recursive glob search for backup
    if not image_files:
        try:
            resolved_path = resolve_drive_path(base_path)
            for ext in extensions:
                pattern = os.path.join(resolved_path, '**', f'*{ext}')
                image_files.extend(glob.glob(pattern, recursive=True))
        except Exception:
            pass
    
    return list(set(image_files))

build_paths = lambda raw_dir, aug_dir, prep_dir, split="train": {'raw_dir': resolve_drive_path(raw_dir), 'aug_dir': resolve_drive_path(aug_dir), 'prep_dir': resolve_drive_path(prep_dir), 'aug_images': f"{resolve_drive_path(aug_dir)}/images", 'aug_labels': f"{resolve_drive_path(aug_dir)}/labels", 'prep_split': f"{resolve_drive_path(prep_dir)}/{split}"}
ensure_dirs = lambda *paths: [Path(resolve_drive_path(p)).mkdir(parents=True, exist_ok=True) for p in paths]
path_exists = lambda path: Path(resolve_drive_path(path)).exists()
get_stem = lambda path: Path(path).stem
get_parent = lambda path: str(Path(path).parent)

# =============================================================================
# ENHANCED FILE OPERATIONS - Unchanged
# =============================================================================

find_images = lambda dir_path, exts=['.jpg', '.jpeg', '.png']: smart_find_images(dir_path, exts)
find_labels = lambda dir_path: smart_find_images(dir_path, ['.txt'])
find_aug_files = lambda dir_path: [str(f) for f in Path(resolve_drive_path(dir_path)).rglob('aug_*.*') if f.is_file()]

copy_file = lambda src, dst: shutil.copy2(resolve_drive_path(src), resolve_drive_path(dst)) if path_exists(src) else False
safe_copy_file = lambda src, dst: safe_execute(lambda: copy_file(src, dst), False)
delete_file = lambda path: Path(resolve_drive_path(path)).unlink() if path_exists(path) else False
get_file_size = lambda path: Path(resolve_drive_path(path)).stat().st_size if path_exists(path) else 0

# =============================================================================
# IMAGE OPERATIONS - Unchanged
# =============================================================================

read_image = lambda path: cv2.imread(resolve_drive_path(path)) if path_exists(path) else None
save_image = lambda img, path: cv2.imwrite(resolve_drive_path(path), img) if img is not None else False
validate_image = lambda img: img is not None and img.size > 0
resize_image = lambda img, size: cv2.resize(img, size) if validate_image(img) else None

# =============================================================================
# BBOX OPERATIONS - Unchanged
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
# ENHANCED DATASET DETECTION - Unchanged
# =============================================================================

def detect_structure(data_dir: str) -> Dict[str, Any]:
    """Enhanced dataset detection dengan smart image finder"""
    resolved_dir = resolve_drive_path(data_dir)
    
    if not path_exists(resolved_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
    
    # Smart image detection
    images = smart_find_images(resolved_dir)
    labels = find_labels(resolved_dir)
    
    # Enhanced structure detection
    has_yolo = path_exists(f"{resolved_dir}/images") and path_exists(f"{resolved_dir}/labels")
    splits = [s for s in ['train', 'valid', 'test', 'val'] if path_exists(f"{resolved_dir}/{s}")]
    
    # Detailed image locations untuk tracking
    image_locations = []
    dataset_dirs = find_dataset_directories(resolved_dir)
    
    for dir_path in dataset_dirs:
        dir_images = [img for img in images if img.startswith(dir_path)]
        if dir_images:
            image_locations.append({'path': dir_path, 'count': len(dir_images), 'has_images_subdir': path_exists(f"{dir_path}/images"), 'sample_files': dir_images[:3]})
    
    return {'status': 'success', 'data_dir': resolved_dir, 'total_images': len(images), 'total_labels': len(labels), 'structure_type': 'standard_yolo' if has_yolo else f'mixed_structure_{len(splits)}_splits' if splits else 'flat_structure', 'splits_detected': splits, 'image_locations': image_locations, 'recommendations': [f'✅ Ditemukan {len(images)} gambar di {len(image_locations)} lokasi' if images else '❌ Tidak ada gambar ditemukan - periksa path dan format file', f'📁 Struktur: {len(image_locations)} direktori dengan gambar', f'🏷️ Labels: {len(labels)} file label ditemukan']}

# =============================================================================
# PROGRESS TRACKING - Fixed dengan Real-time Updates
# =============================================================================

class ProgressTracker:
    def __init__(self, communicator=None):
        self.comm = communicator
        self.logger = getattr(self.comm, 'logger', None) if self.comm else get_logger(__name__)
    
    def progress(self, step: str, current: int, total: int, msg: str = ""):
        """Fixed progress dengan immediate UI updates."""
        if self.comm and hasattr(self.comm, 'progress'):
            percentage = min(100, max(0, int((current / max(1, total)) * 100)))
            self.comm.progress(step, current, total, msg)
        
        # Also update via direct methods untuk backup
        if self.comm and hasattr(self.comm, 'report_progress_with_callback'):
            self.comm.report_progress_with_callback(None, step, current, total, msg)
    
    log_info = lambda self, msg: self.comm.log_info(msg) if self.comm else print(f"ℹ️ {msg}")
    log_success = lambda self, msg: self.comm.log_success(msg) if self.comm else print(f"✅ {msg}")
    log_error = lambda self, msg: self.comm.log_error(msg) if self.comm else print(f"❌ {msg}")

# =============================================================================
# BATCH PROCESSING - Fixed dengan Real-time Progress Updates
# =============================================================================

def process_batch(items: List[Any], process_func: Callable, max_workers: int = None, 
                 progress_tracker: ProgressTracker = None, operation_name: str = "processing") -> List[Dict[str, Any]]:
    """Fixed batch processing dengan real-time progress updates."""
    max_workers = max_workers or min(get_optimal_thread_count(), 8)
    results = []
    
    if not items:
        progress_tracker and progress_tracker.log_info("⚠️ Tidak ada item untuk diproses")
        return results
    
    total_items = len(items)
    progress_tracker and progress_tracker.log_info(f"🚀 Memulai {operation_name}: {total_items} item")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_func, item): i for i, item in enumerate(items)}
        
        # Process results dengan real-time updates
        for completed_count, future in enumerate(as_completed(futures), 1):
            try:
                result = future.result()
                results.append(result)
                
                # Real-time progress update setiap item selesai
                if progress_tracker:
                    progress_tracker.progress("current", completed_count, total_items, 
                                            f"{operation_name}: {completed_count}/{total_items}")
                
                # Log progress setiap 10% atau significant milestones
                if completed_count % max(1, total_items // 10) == 0:
                    successful = sum(1 for r in results if r.get('status') == 'success')
                    success_rate = (successful / completed_count) * 100
                    progress_tracker and progress_tracker.log_info(
                        f"📊 Progress: {completed_count}/{total_items} ({success_rate:.1f}% berhasil)"
                    )
                    
            except Exception as e:
                results.append({'status': 'error', 'error': str(e)})
                completed_count += 1
    
    # Final summary
    successful = sum(1 for r in results if r.get('status') == 'success')
    progress_tracker and progress_tracker.log_success(
        f"✅ {operation_name} selesai: {successful}/{total_items} berhasil"
    )
    
    return results

# =============================================================================
# CONFIG EXTRACTION - Unchanged
# =============================================================================

extract_config = lambda config: {'raw_dir': resolve_drive_path(config.get('data', {}).get('dir', 'data')), 'aug_dir': resolve_drive_path(config.get('augmentation', {}).get('output_dir', 'data/augmented')), 'prep_dir': resolve_drive_path(config.get('preprocessing', {}).get('output_dir', 'data/preprocessed')), 'num_variations': config.get('augmentation', {}).get('num_variations', 2), 'target_count': config.get('augmentation', {}).get('target_count', 500), 'types': config.get('augmentation', {}).get('types', ['combined']), 'intensity': config.get('augmentation', {}).get('intensity', 0.7)}

def get_best_data_location() -> str:
    """Smart data location detection dengan comprehensive search"""
    search_paths = ['/content/drive/MyDrive/SmartCash/data', '/content/drive/MyDrive/data', '/content/SmartCash/data', '/content/data', 'data']
    for path in search_paths:
        if smart_find_images(path): return path
        elif path_exists(path): return path
    return 'data'

# =============================================================================
# ERROR HANDLING - Unchanged
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
# CLEANUP OPERATIONS - Fixed dengan Real-time Progress
# =============================================================================

def cleanup_files(aug_dir: str, prep_dir: str = None, progress_tracker: ProgressTracker = None) -> Dict[str, Any]:
    """Fixed cleanup dengan real-time progress updates."""
    total_deleted = 0
    errors = []
    
    aug_files = find_aug_files(aug_dir)
    total_files = len(aug_files)
    
    if prep_dir:
        prep_files = find_aug_files(prep_dir)
        aug_files.extend(prep_files)
        total_files = len(aug_files)
    
    progress_tracker and progress_tracker.progress("overall", 0, 100, f"Mulai cleanup: {total_files} file")
    
    for i, file_path in enumerate(aug_files):
        if delete_file(file_path):
            total_deleted += 1
        else:
            errors.append(f"Failed to delete {file_path}")
        
        # Real-time progress update
        if progress_tracker:
            current_progress = int((i / max(total_files, 1)) * 100)
            progress_tracker.progress("overall", current_progress, 100, 
                                    f"Cleanup: {i+1}/{total_files} file")
    
    progress_tracker and progress_tracker.progress("overall", 100, 100, f"Cleanup selesai: {total_deleted} file dihapus")
    
    return {'status': 'success' if total_deleted > 0 else 'empty', 'total_deleted': total_deleted, 'message': f"Berhasil menghapus {total_deleted} file" if total_deleted > 0 else "Tidak ada file untuk dihapus", 'errors': errors}

# =============================================================================
# ENHANCED FACTORY FUNCTION
# =============================================================================

def create_context(config: Dict[str, Any], communicator=None) -> Dict[str, Any]:
    """Enhanced context creation dengan communicator integration."""
    aug_config = extract_config(config)
    progress_tracker = ProgressTracker(communicator)
    
    return {
        'config': aug_config,
        'progress': progress_tracker,
        'paths': build_paths(aug_config['raw_dir'], aug_config['aug_dir'], aug_config['prep_dir']),
        'detector': lambda: detect_structure(aug_config['raw_dir']),
        'cleaner': lambda: cleanup_files(aug_config['aug_dir'], aug_config['prep_dir'], progress_tracker),
        'comm': communicator  # Pass communicator untuk service integration
    }

# Backward compatibility aliases
count_dataset_files = lambda data_dir: (len(smart_find_images(data_dir)), len(find_labels(data_dir)))
validate_dataset = lambda data_dir: detect_structure(data_dir)['total_images'] > 0