"""
File: smartcash/dataset/augmentor/processors/__init__.py
Deskripsi: Module processors untuk augmentasi dengan one-liner optimized operations
"""

from .image import ImageProcessor, read_image, save_image, apply_pipeline
from .bbox import BBoxProcessor, parse_yolo, scale_bbox, validate_bbox
from .file import FileProcessor, ensure_dir, copy_file, move_file, find_files
from .batch import BatchProcessor, process_files_sync, process_files_async

__all__ = [
    # Image processing
    'ImageProcessor', 'read_image', 'save_image', 'apply_pipeline',
    
    # BBox processing
    'BBoxProcessor', 'parse_yolo', 'scale_bbox', 'validate_bbox',
    
    # File operations
    'FileProcessor', 'ensure_dir', 'copy_file', 'move_file', 'find_files',
    
    # Batch processing
    'BatchProcessor', 'process_files_sync', 'process_files_async'
]