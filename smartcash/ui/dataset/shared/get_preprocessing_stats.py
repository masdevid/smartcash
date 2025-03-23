
"""
File: smartcash/ui/dataset/shared/get_preprocessing_stats.py
Deskripsi: Helper untuk mendapatkan statistik dataset preprocessed
"""

def get_preprocessing_stats(preprocessed_dir: str) -> Dict[str, Any]:
    """
    Dapatkan statistik dataset preprocessed dengan pendekatan one-liner.
    
    Args:
        preprocessed_dir: Direktori dataset preprocessed
        
    Returns:
        Dictionary statistik dataset
    """
    stats = {
        'splits': {},
        'total': {
            'images': 0,
            'labels': 0
        },
        'valid': False
    }
    
    # Cek setiap split dengan list comprehension
    for split in ['train', 'valid', 'test']:
        split_dir = Path(preprocessed_dir) / split
        if not split_dir.exists():
            stats['splits'][split] = {'exists': False, 'images': 0, 'labels': 0, 'complete': False}
            continue
            
        # Hitung gambar dan label
        images_dir, labels_dir = split_dir / 'images', split_dir / 'labels'
        
        # PERBAIKAN: Cek semua format file yang didukung
        num_images = 0
        if images_dir.exists():
            num_images = len(list(images_dir.glob('*.jpg'))) + len(list(images_dir.glob('*.png'))) + len(list(images_dir.glob('*.npy')))
            
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