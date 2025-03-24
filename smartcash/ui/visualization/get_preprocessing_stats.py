"""
File: smartcash/ui/visualization/get_preprocessing_stats.py
Deskripsi: Utilitas untuk mendapatkan statistik dataset preprocessing
"""
from typing import Dict, Any
from pathlib import Path
from smartcash.dataset.utils.dataset_constants import DEFAULT_SPLITS

def get_preprocessing_stats(ui_components: Dict[str, Any], preprocessed_dir: str) -> Dict[str, Any]:
    """
    Mendapatkan statistik dataset preprocessing.
    
    Args:
        ui_components: Dictionary komponen UI
        preprocessed_dir: Direktori dataset preprocessed
        
    Returns:
        Dictionary statistik preprocessing
    """
    stats = {
        'splits': {},
        'total': {
            'images': 0,
            'labels': 0
        }
    }
    
    # Cek setiap split
    for split in DEFAULT_SPLITS:
        split_dir = Path(preprocessed_dir) / split
        if not split_dir.exists():
            stats['splits'][split] = {'exists': False, 'images': 0, 'labels': 0}
            continue
            
        # Hitung gambar dan label
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.npy'))) if images_dir.exists() else 0
        num_labels = len(list(labels_dir.glob('*.txt'))) if labels_dir.exists() else 0
        
        # Update statistik
        stats['splits'][split] = {
            'exists': True,
            'images': num_images,
            'labels': num_labels,
            'complete': num_images > 0 and num_labels > 0 and num_images == num_labels
        }
        
        # Update total
        stats['total']['images'] += num_images
        stats['total']['labels'] += num_labels
    
    # Dataset dianggap valid jika minimal ada 1 split dengan data lengkap
    stats['valid'] = any(split_info.get('complete', False) for split_info in stats['splits'].values())
    
    return stats