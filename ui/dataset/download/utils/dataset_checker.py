"""
File: smartcash/ui/dataset/download/utils/dataset_checker.py
Deskripsi: Utility untuk comprehensive dataset checking dan readiness analysis
"""

from typing import Dict, Any
from pathlib import Path
import os

def check_complete_dataset_status() -> Dict[str, Any]:
    """Check complete dataset status dengan comprehensive analysis."""
    try:
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.constants.paths import get_paths_for_environment
        
        env_manager = get_environment_manager()
        paths = get_paths_for_environment(env_manager.is_colab, env_manager.is_drive_mounted)
        
        # Check semua komponen dataset
        status_components = {
            'final_dataset': _check_final_dataset_structure(paths),
            'downloads_folder': _check_downloads_folder(paths),
            'storage_info': _get_storage_info(env_manager),
            'paths': paths
        }
        
        # Generate overall summary
        status_components['summary'] = _generate_dataset_summary(status_components)
        
        return status_components
        
    except Exception as e:
        return _create_error_status(f"Error checking dataset: {str(e)}")

def get_dataset_readiness_score(dataset_status: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate dataset readiness score untuk training."""
    final_dataset = dataset_status.get('final_dataset', {})
    
    if not final_dataset.get('exists', False):
        return {'overall_score': 0, 'readiness_level': 'Not Ready', 'details': {'reason': 'Dataset not found'}}
    
    # Scoring components dengan weighted calculation
    score_components = {
        'structure': _score_dataset_structure(final_dataset),
        'completeness': _score_dataset_completeness(final_dataset),
        'balance': _score_dataset_balance(final_dataset),
        'size': _score_dataset_size(final_dataset)
    }
    
    # Weighted average (structure=30%, completeness=30%, balance=20%, size=20%)
    weights = {'structure': 0.3, 'completeness': 0.3, 'balance': 0.2, 'size': 0.2}
    overall_score = sum(score_components[comp] * weights[comp] for comp in score_components)
    
    return {
        'overall_score': int(overall_score),
        'readiness_level': _get_readiness_level(overall_score),
        'component_scores': score_components,
        'recommendations': _generate_readiness_recommendations(score_components, overall_score)
    }

def _check_final_dataset_structure(paths: Dict[str, str]) -> Dict[str, Any]:
    """Check final dataset structure di data/{train,valid,test}."""
    data_root = Path(paths.get('data_root', 'data'))
    splits = ['train', 'valid', 'test']
    
    split_info = {}
    total_images = total_labels = 0
    
    for split in splits:
        split_path = data_root / split
        images_path = split_path / 'images'
        labels_path = split_path / 'labels'
        
        # Count files dengan efficient directory traversal
        images_count = len([f for f in images_path.glob('*.jpg')] + [f for f in images_path.glob('*.png')]) if images_path.exists() else 0
        labels_count = len(list(labels_path.glob('*.txt'))) if labels_path.exists() else 0
        
        split_info[split] = {
            'exists': split_path.exists(),
            'images_dir_exists': images_path.exists(),
            'labels_dir_exists': labels_path.exists(),
            'images': images_count,
            'labels': labels_count,
            'path': str(split_path)
        }
        
        total_images += images_count
        total_labels += labels_count
    
    return {
        'exists': total_images > 0,
        'splits': split_info,
        'total_images': total_images,
        'total_labels': total_labels,
        'path': str(data_root)
    }

def _check_downloads_folder(paths: Dict[str, str]) -> Dict[str, Any]:
    """Check downloads folder untuk temporary files."""
    downloads_path = Path(paths.get('downloads', 'data/downloads'))
    
    if not downloads_path.exists():
        return {'exists': False, 'total_files': 0, 'total_size_mb': 0, 'path': str(downloads_path)}
    
    # Count files dan calculate size
    all_files = list(downloads_path.rglob('*'))
    file_count = len([f for f in all_files if f.is_file()])
    
    total_size = sum(f.stat().st_size for f in all_files if f.is_file() and f.exists())
    total_size_mb = round(total_size / (1024 * 1024), 2)
    
    return {
        'exists': True,
        'total_files': file_count,
        'total_size_mb': total_size_mb,
        'path': str(downloads_path)
    }

def _get_storage_info(env_manager) -> Dict[str, Any]:
    """Get comprehensive storage information."""
    if env_manager.is_colab and env_manager.is_drive_mounted:
        return {
            'type': 'Google Drive',
            'persistent': True,
            'path': env_manager.drive_path,
            'recommended': True
        }
    elif env_manager.is_colab:
        return {
            'type': 'Local Storage (Colab)',
            'persistent': False,
            'path': '/content',
            'recommended': False
        }
    else:
        return {
            'type': 'Local Storage',
            'persistent': True,
            'path': os.getcwd(),
            'recommended': True
        }

def _generate_dataset_summary(status_components: Dict[str, Any]) -> Dict[str, Any]:
    """Generate overall dataset summary."""
    final_dataset = status_components['final_dataset']
    downloads_folder = status_components['downloads_folder']
    storage_info = status_components['storage_info']
    
    # Determine overall status
    if final_dataset['exists'] and final_dataset['total_images'] > 0:
        status = 'ready'
        message = f"✅ Dataset siap - {final_dataset['total_images']} gambar tersedia"
    elif downloads_folder['exists'] and downloads_folder['total_files'] > 0:
        status = 'downloaded'
        message = f"📥 Dataset downloaded - perlu organisir ke struktur final"
    else:
        status = 'empty'
        message = "📭 Dataset belum ada - perlu download"
    
    return {
        'status': status,
        'message': message,
        'storage_type': storage_info['type'],
        'storage_persistent': storage_info['persistent'],
        'total_components_checked': 3
    }

def _score_dataset_structure(final_dataset: Dict[str, Any]) -> int:
    """Score dataset structure completeness (0-100)."""
    if not final_dataset.get('exists', False):
        return 0
    
    splits = final_dataset.get('splits', {})
    required_splits = ['train', 'valid']  # test optional
    
    # Check structure completeness
    structure_checks = [
        all(splits.get(split, {}).get('images_dir_exists', False) for split in required_splits),  # Images dirs
        all(splits.get(split, {}).get('labels_dir_exists', False) for split in required_splits),  # Labels dirs
        all(splits.get(split, {}).get('images', 0) > 0 for split in required_splits),           # Has images
        all(splits.get(split, {}).get('labels', 0) > 0 for split in required_splits)            # Has labels
    ]
    
    structure_score = (sum(structure_checks) / len(structure_checks)) * 100
    
    # Bonus untuk test split
    if splits.get('test', {}).get('exists', False):
        structure_score = min(100, structure_score + 10)
    
    return int(structure_score)

def _score_dataset_completeness(final_dataset: Dict[str, Any]) -> int:
    """Score dataset completeness berdasarkan image-label ratio."""
    splits = final_dataset.get('splits', {})
    completeness_scores = []
    
    for split_name, split_info in splits.items():
        images = split_info.get('images', 0)
        labels = split_info.get('labels', 0)
        
        if images == 0:
            completeness_scores.append(0)
        else:
            ratio = min(labels / images, 1.0)  # Cap at 100%
            completeness_scores.append(ratio * 100)
    
    return int(sum(completeness_scores) / len(completeness_scores)) if completeness_scores else 0

def _score_dataset_balance(final_dataset: Dict[str, Any]) -> int:
    """Score dataset balance antar splits."""
    splits = final_dataset.get('splits', {})
    train_images = splits.get('train', {}).get('images', 0)
    valid_images = splits.get('valid', {}).get('images', 0)
    
    if train_images == 0 or valid_images == 0:
        return 0
    
    # Ideal ratio train:valid = 70:20 (3.5:1)
    actual_ratio = train_images / valid_images
    ideal_ratio = 3.5
    
    # Score berdasarkan seberapa dekat dengan ideal ratio
    ratio_score = max(0, 100 - abs(actual_ratio - ideal_ratio) * 20)
    
    return int(ratio_score)

def _score_dataset_size(final_dataset: Dict[str, Any]) -> int:
    """Score dataset size adequacy untuk training."""
    total_images = final_dataset.get('total_images', 0)
    
    # Thresholds untuk YOLOv5 object detection
    if total_images >= 1000:
        return 100  # Excellent
    elif total_images >= 500:
        return 80   # Good
    elif total_images >= 200:
        return 60   # Adequate
    elif total_images >= 100:
        return 40   # Minimal
    elif total_images >= 50:
        return 20   # Very limited
    else:
        return 0    # Insufficient

def _get_readiness_level(score: float) -> str:
    """Get readiness level dari overall score."""
    level_mapping = [
        (90, 'Excellent - Siap Training'),
        (80, 'Good - Siap Training'),
        (70, 'Adequate - Cukup untuk Training'),
        (60, 'Limited - Perlu Improvement'),
        (40, 'Poor - Butuh Augmentasi'),
        (0, 'Insufficient - Perlu Data Lebih')
    ]
    
    return next(level for threshold, level in level_mapping if score >= threshold)

def _generate_readiness_recommendations(component_scores: Dict[str, int], overall_score: float) -> list:
    """Generate actionable recommendations berdasarkan component scores."""
    recommendations = []
    
    # Structure recommendations
    if component_scores['structure'] < 80:
        recommendations.append("🏗️ Perbaiki struktur dataset (pastikan ada train/valid dengan images & labels)")
    
    # Completeness recommendations
    if component_scores['completeness'] < 90:
        recommendations.append("🏷️ Periksa kelengkapan label (beberapa gambar mungkin belum dilabel)")
    
    # Balance recommendations
    if component_scores['balance'] < 70:
        recommendations.append("⚖️ Seimbangkan distribusi data antar train/valid split")
    
    # Size recommendations
    if component_scores['size'] < 60:
        recommendations.append("📈 Tambah jumlah data atau gunakan augmentasi untuk meningkatkan dataset size")
    
    # Overall recommendations
    if overall_score < 60:
        recommendations.append("🚀 Pertimbangkan download dataset tambahan atau augmentasi sebelum training")
    
    return recommendations

def _create_error_status(error_message: str) -> Dict[str, Any]:
    """Create error status response."""
    return {
        'final_dataset': {'exists': False, 'total_images': 0},
        'downloads_folder': {'exists': False, 'total_files': 0},
        'storage_info': {'type': 'Unknown', 'persistent': False},
        'summary': {'status': 'error', 'message': error_message},
        'error': error_message
    }

def quick_dataset_check() -> Dict[str, Any]:
    """Quick dataset check untuk UI status updates."""
    try:
        status = check_complete_dataset_status()
        readiness = get_dataset_readiness_score(status)
        
        return {
            'has_dataset': status['final_dataset']['exists'],
            'total_images': status['final_dataset']['total_images'],
            'readiness_score': readiness['overall_score'],
            'readiness_level': readiness['readiness_level'],
            'storage_type': status['storage_info']['type']
        }
    except Exception:
        return {
            'has_dataset': False, 'total_images': 0, 'readiness_score': 0,
            'readiness_level': 'Unknown', 'storage_type': 'Unknown'
        }