"""
File: smartcash/dataset/augmentor/utils/dataset_detector.py
Deskripsi: SRP module untuk dataset structure detection dengan comprehensive analysis
"""

from typing import Dict, Any, List
from smartcash.dataset.augmentor.utils.path_operations import resolve_drive_path, path_exists, find_dataset_directories
from smartcash.dataset.augmentor.utils.file_operations import find_images, find_labels

def detect_structure(data_dir: str) -> Dict[str, Any]:
    """Enhanced dataset detection dengan smart image finder"""
    resolved_dir = resolve_drive_path(data_dir)
    
    if not path_exists(resolved_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
    
    # Smart detection
    images = find_images(resolved_dir)
    labels = find_labels(resolved_dir)
    
    # Structure analysis
    has_yolo = path_exists(f"{resolved_dir}/images") and path_exists(f"{resolved_dir}/labels")
    splits = [s for s in ['train', 'valid', 'test', 'val'] if path_exists(f"{resolved_dir}/{s}")]
    
    # Image locations tracking
    image_locations = []
    for dir_path in find_dataset_directories(resolved_dir):
        dir_images = [img for img in images if img.startswith(dir_path)]
        if dir_images:
            image_locations.append({
                'path': dir_path, 'count': len(dir_images),
                'has_images_subdir': path_exists(f"{dir_path}/images"),
                'sample_files': dir_images[:3]
            })
    
    return {
        'status': 'success', 'data_dir': resolved_dir, 'total_images': len(images),
        'total_labels': len(labels), 'splits_detected': splits, 'image_locations': image_locations,
        'structure_type': (
            'standard_yolo' if has_yolo else
            f'mixed_structure_{len(splits)}_splits' if splits else
            'flat_structure'
        ),
        'recommendations': [
            f'✅ Ditemukan {len(images)} gambar di {len(image_locations)} lokasi' if images 
            else '❌ Tidak ada gambar ditemukan - periksa path dan format file',
            f'📁 Struktur: {len(image_locations)} direktori dengan gambar',
            f'🏷️ Labels: {len(labels)} file label ditemukan'
        ]
    }

def detect_split_structure(data_dir: str) -> Dict[str, Any]:
    """Enhanced dataset detection dengan split awareness"""
    resolved_dir = resolve_drive_path(data_dir)
    
    if not path_exists(resolved_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_dir}'}
    
    # Detect splits dengan detail
    available_splits = []
    split_details = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = f"{resolved_dir}/{split}"
        if path_exists(split_dir):
            from smartcash.dataset.augmentor.utils.file_operations import smart_find_images_split_aware
            images = smart_find_images_split_aware(resolved_dir, split)
            labels = smart_find_images_split_aware(resolved_dir, split, ['.txt'])
            
            if images or labels:
                available_splits.append(split)
                split_details[split] = {
                    'path': split_dir, 'images': len(images), 'labels': len(labels),
                    'has_structure': path_exists(f"{split_dir}/images")
                }
    
    # Overall statistics
    total_images = sum(details['images'] for details in split_details.values())
    total_labels = sum(details['labels'] for details in split_details.values())
    
    return {
        'status': 'success', 'data_dir': resolved_dir, 'total_images': total_images,
        'total_labels': total_labels, 'available_splits': available_splits,
        'split_details': split_details,
        'structure_type': 'split_based' if available_splits else 'flat',
        'recommendations': [
            f'✅ Split structure: {len(available_splits)} splits tersedia' if available_splits 
            else '⚠️ Tidak ada split structure - gunakan flat structure',
            f'📊 Total: {total_images} gambar, {total_labels} label',
            f'📁 Splits: {", ".join(available_splits)}' if available_splits else 'No splits detected'
        ]
    }

# One-liner utilities
validate_dataset = lambda data_dir: detect_structure(data_dir)['total_images'] > 0
count_dataset_files = lambda data_dir: (detect_structure(data_dir)['total_images'], detect_structure(data_dir)['total_labels'])
count_dataset_files_split_aware = lambda data_dir, split=None: (len(smart_find_images_split_aware(data_dir, split)), len(smart_find_images_split_aware(data_dir, split, ['.txt'])))

# Import for compatibility
from smartcash.dataset.augmentor.utils.file_operations import smart_find_images_split_aware