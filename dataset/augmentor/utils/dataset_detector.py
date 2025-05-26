"""
File: smartcash/dataset/augmentor/utils/dataset_detector.py
Deskripsi: Fixed dataset structure detector dengan Google Drive path resolution dan smart directory detection yang benar
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict

def detect_dataset_structure(data_dir: str) -> Dict[str, Any]:
    """
    Detect struktur dataset dengan comprehensive analysis dan Google Drive path resolution.
    
    Args:
        data_dir: Path ke directory dataset (bisa local atau Drive)
        
    Returns:
        Dictionary dengan informasi struktur dataset
    """
    # Resolve Google Drive path jika diperlukan
    resolved_data_dir = _resolve_google_drive_path(data_dir)
    
    if not os.path.exists(resolved_data_dir):
        return {'status': 'error', 'message': f'Directory tidak ditemukan: {resolved_data_dir}'}
    
    detection_result = {
        'status': 'success',
        'data_dir': resolved_data_dir,
        'original_path': data_dir,
        'structure_type': 'unknown',
        'image_locations': [],
        'label_locations': [],
        'splits_detected': [],
        'total_images': 0,
        'total_labels': 0,
        'file_extensions': defaultdict(int),
        'recommendations': []
    }
    
    # One-liner detect possible image directories
    possible_image_dirs = _find_image_directories(resolved_data_dir)
    possible_label_dirs = _find_label_directories(resolved_data_dir)
    
    detection_result['image_locations'] = possible_image_dirs
    detection_result['label_locations'] = possible_label_dirs
    
    # Analyze structure type
    structure_analysis = _analyze_structure_type(resolved_data_dir, possible_image_dirs, possible_label_dirs)
    detection_result.update(structure_analysis)
    
    # Count files
    file_counts = _count_dataset_files(possible_image_dirs, possible_label_dirs)
    detection_result.update(file_counts)
    
    # Generate recommendations dengan Drive path consideration
    detection_result['recommendations'] = _generate_recommendations(detection_result)
    
    return detection_result

def _resolve_google_drive_path(data_dir: str) -> str:
    """Resolve Google Drive path dengan prioritas mounting detection."""
    # Check jika sudah absolute path yang valid
    if os.path.exists(data_dir):
        return data_dir
    
    # Try resolve relative to common Colab locations
    colab_base_paths = [
        '/content/drive/MyDrive/SmartCash',
        '/content/drive/MyDrive',
        '/content',
        '/content/SmartCash'
    ]
    
    for base in colab_base_paths:
        # Direct path resolution
        if data_dir.startswith('data'):
            resolved = os.path.join(base, data_dir)
        else:
            resolved = os.path.join(base, 'data') if data_dir == 'data' else os.path.join(base, data_dir)
        
        if os.path.exists(resolved):
            return resolved
    
    # Fallback ke original path
    return data_dir

def _find_image_directories(data_dir: str) -> List[Dict[str, Any]]:
    """One-liner find all directories yang mengandung gambar dengan Google Drive compatibility."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    image_dirs = []
    
    try:
        # Check root directory
        if os.path.exists(data_dir):
            root_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            root_images = [f for f in root_files if Path(f).suffix.lower() in image_extensions]
            if root_images:
                image_dirs.append({
                    'path': data_dir,
                    'type': 'root_mixed',
                    'count': len(root_images),
                    'sample_files': root_images[:3]
                })
        
        # Check standard YOLO structure: data/images
        images_dir = os.path.join(data_dir, 'images')
        if os.path.exists(images_dir):
            img_files = [f for f in os.listdir(images_dir) if os.path.isfile(os.path.join(images_dir, f)) and Path(f).suffix.lower() in image_extensions]
            if img_files:
                image_dirs.append({
                    'path': images_dir,
                    'type': 'standard_yolo',
                    'count': len(img_files),
                    'sample_files': img_files[:3]
                })
        
        # Check subdirectories untuk split structure
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    # Check direct subdirectory
                    try:
                        sub_files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                        sub_images = [f for f in sub_files if Path(f).suffix.lower() in image_extensions]
                        if sub_images:
                            image_dirs.append({
                                'path': item_path,
                                'type': 'split' if item in ['train', 'valid', 'test'] else 'subdirectory',
                                'count': len(sub_images),
                                'sample_files': sub_images[:3]
                            })
                        
                        # Check images subdirectory (train/images, valid/images, etc)
                        images_subdir = os.path.join(item_path, 'images')
                        if os.path.exists(images_subdir):
                            sub_sub_files = [f for f in os.listdir(images_subdir) if os.path.isfile(os.path.join(images_subdir, f))]
                            sub_sub_images = [f for f in sub_sub_files if Path(f).suffix.lower() in image_extensions]
                            if sub_sub_images:
                                image_dirs.append({
                                    'path': images_subdir,
                                    'type': 'split_structured' if item in ['train', 'valid', 'test'] else 'structured',
                                    'count': len(sub_sub_images),
                                    'sample_files': sub_sub_images[:3]
                                })
                    except (PermissionError, OSError):
                        continue
    
    except Exception:
        pass  # Silent fail untuk compatibility
    
    return image_dirs

def _find_label_directories(data_dir: str) -> List[Dict[str, Any]]:
    """One-liner find all directories yang mengandung label dengan Google Drive compatibility."""
    label_extensions = {'.txt', '.xml'}
    label_dirs = []
    
    try:
        # Check root directory
        if os.path.exists(data_dir):
            root_files = [f for f in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, f))]
            root_labels = [f for f in root_files if Path(f).suffix.lower() in label_extensions]
            if root_labels:
                label_dirs.append({
                    'path': data_dir,
                    'type': 'root_mixed',
                    'count': len(root_labels),
                    'sample_files': root_labels[:3]
                })
        
        # Check standard YOLO structure: data/labels
        labels_dir = os.path.join(data_dir, 'labels')
        if os.path.exists(labels_dir):
            lbl_files = [f for f in os.listdir(labels_dir) if os.path.isfile(os.path.join(labels_dir, f)) and Path(f).suffix.lower() in label_extensions]
            if lbl_files:
                label_dirs.append({
                    'path': labels_dir,
                    'type': 'standard_yolo',
                    'count': len(lbl_files),
                    'sample_files': lbl_files[:3]
                })
        
        # Check subdirectories untuk split structure  
        if os.path.exists(data_dir):
            for item in os.listdir(data_dir):
                item_path = os.path.join(data_dir, item)
                if os.path.isdir(item_path):
                    try:
                        # Check direct subdirectory
                        sub_files = [f for f in os.listdir(item_path) if os.path.isfile(os.path.join(item_path, f))]
                        sub_labels = [f for f in sub_files if Path(f).suffix.lower() in label_extensions]
                        if sub_labels:
                            label_dirs.append({
                                'path': item_path,
                                'type': 'split' if item in ['train', 'valid', 'test'] else 'subdirectory',
                                'count': len(sub_labels),
                                'sample_files': sub_labels[:3]
                            })
                        
                        # Check labels subdirectory (train/labels, valid/labels, etc)
                        labels_subdir = os.path.join(item_path, 'labels')
                        if os.path.exists(labels_subdir):
                            sub_sub_files = [f for f in os.listdir(labels_subdir) if os.path.isfile(os.path.join(labels_subdir, f))]
                            sub_sub_labels = [f for f in sub_sub_files if Path(f).suffix.lower() in label_extensions]
                            if sub_sub_labels:
                                label_dirs.append({
                                    'path': labels_subdir,
                                    'type': 'split_structured' if item in ['train', 'valid', 'test'] else 'structured',
                                    'count': len(sub_sub_labels),
                                    'sample_files': sub_sub_labels[:3]
                                })
                    except (PermissionError, OSError):
                        continue
    
    except Exception:
        pass  # Silent fail untuk compatibility
    
    return label_dirs

def _analyze_structure_type(data_dir: str, image_dirs: List[Dict], label_dirs: List[Dict]) -> Dict[str, Any]:
    """Analyze dataset structure type dengan one-liner logic."""
    # One-liner check for standard YOLO structure
    has_standard_yolo = (
        os.path.exists(os.path.join(data_dir, 'images')) and 
        os.path.exists(os.path.join(data_dir, 'labels'))
    )
    
    # One-liner check for split structure
    split_dirs = [d for d in ['train', 'valid', 'test'] if os.path.exists(os.path.join(data_dir, d))]
    has_splits = len(split_dirs) > 0
    
    # One-liner check for structured splits
    has_structured_splits = any(
        os.path.exists(os.path.join(data_dir, split, 'images')) and 
        os.path.exists(os.path.join(data_dir, split, 'labels'))
        for split in split_dirs
    )
    
    # Determine structure type
    if has_standard_yolo and not has_splits:
        structure_type = 'standard_yolo'
    elif has_structured_splits:
        structure_type = 'split_structured'
    elif has_splits:
        structure_type = 'split_mixed'
    elif len(image_dirs) > 0 and len(label_dirs) > 0:
        structure_type = 'mixed'
    elif len(image_dirs) > 0:
        structure_type = 'images_only'
    else:
        structure_type = 'empty'
    
    return {
        'structure_type': structure_type,
        'splits_detected': split_dirs,
        'has_standard_yolo': has_standard_yolo,
        'has_splits': has_splits,
        'has_structured_splits': has_structured_splits
    }

def _count_dataset_files(image_dirs: List[Dict], label_dirs: List[Dict]) -> Dict[str, Any]:
    """One-liner count files in dataset."""
    total_images = sum(img_dir['count'] for img_dir in image_dirs)
    total_labels = sum(lbl_dir['count'] for lbl_dir in label_dirs)
    
    # One-liner extension analysis
    file_extensions = defaultdict(int)
    for img_dir in image_dirs:
        for sample_file in img_dir['sample_files']:
            ext = Path(sample_file).suffix.lower()
            file_extensions[ext] += 1
    
    return {
        'total_images': total_images,
        'total_labels': total_labels,
        'file_extensions': dict(file_extensions)
    }

def _generate_recommendations(detection_result: Dict[str, Any]) -> List[str]:
    """Generate recommendations dengan Google Drive consideration."""
    recommendations = []
    structure_type = detection_result['structure_type']
    total_images = detection_result['total_images']
    total_labels = detection_result['total_labels']
    data_dir = detection_result['data_dir']
    
    # Drive-specific recommendations
    if '/content/drive/MyDrive' in data_dir:
        recommendations.append("✅ Dataset berada di Google Drive - data akan persistent")
    elif '/content' in data_dir and '/drive' not in data_dir:
        recommendations.append("⚠️ Dataset di local Colab - data akan hilang saat session berakhir")
    
    # Structure-specific recommendations
    if structure_type == 'empty':
        recommendations.append("❌ Dataset kosong - tidak ada gambar atau label ditemukan")
    elif structure_type == 'images_only':
        recommendations.append("⚠️ Hanya gambar ditemukan - tidak ada label untuk training")
    elif total_images == 0:
        recommendations.append("❌ Tidak ada gambar ditemukan untuk augmentasi")
    elif total_labels == 0:
        recommendations.append("⚠️ Tidak ada label ditemukan - augmentasi akan berjalan tanpa annotation")
    elif structure_type == 'standard_yolo':
        recommendations.append("✅ Struktur YOLO standard terdeteksi - siap untuk augmentasi")
    elif structure_type == 'split_structured':
        recommendations.append("✅ Struktur split yang terorganisir - pilih split yang akan diaugmentasi")
    elif structure_type == 'mixed':
        recommendations.append("⚠️ Struktur dataset campuran - periksa lokasi file")
    
    # Ratio analysis
    if total_images > 0 and total_labels > 0:
        label_ratio = (total_labels / total_images) * 100
        if label_ratio < 50:
            recommendations.append(f"⚠️ Hanya {label_ratio:.1f}% gambar memiliki label")
        elif label_ratio >= 90:
            recommendations.append(f"✅ {label_ratio:.1f}% gambar memiliki label - dataset siap")
    
    return recommendations

# One-liner utility functions dengan Google Drive support
detect_structure = lambda data_dir: detect_dataset_structure(data_dir)
is_yolo_structure = lambda data_dir: detect_dataset_structure(data_dir)['structure_type'] == 'standard_yolo'
get_image_directories = lambda data_dir: detect_dataset_structure(data_dir)['image_locations']
get_label_directories = lambda data_dir: detect_dataset_structure(data_dir)['label_locations']
count_dataset_files = lambda data_dir: (detect_dataset_structure(data_dir)['total_images'], detect_dataset_structure(data_dir)['total_labels'])
has_valid_dataset = lambda data_dir: detect_dataset_structure(data_dir)['total_images'] > 0
resolve_drive_path = lambda path: _resolve_google_drive_path(path)