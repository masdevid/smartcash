"""
File: smartcash/dataset/services/augmentor/helpers/tracking_helper.py
Deskripsi: Helper untuk tracking progres kelas dalam augmentasi
"""

from typing import Dict, List, Any, Set, Optional, Tuple
import logging
from collections import defaultdict
from pathlib import Path

def track_class_progress(
    augmentation_results: List[Dict[str, Any]], 
    current_class_counts: Dict[str, int],
    target_count: int,
    logger = None
) -> Dict[str, Any]:
    """
    Update tracking progres berdasarkan hasil augmentasi.
    
    Args:
        augmentation_results: Hasil augmentasi batch
        current_class_counts: Jumlah instance kelas saat ini
        target_count: Target jumlah instance per kelas
        logger: Logger untuk logging
        
    Returns:
        Dictionary status tracking dan kelas yang terpenuhi
    """
    # Update copy dari current_class_counts untuk mencegah side effects
    updated_counts = current_class_counts.copy()
    fulfilled_classes = set()
    
    # Track perubahan untuk setiap file hasil augmentasi
    for result in augmentation_results:
        if result.get('status') != 'success':
            continue
            
        # Ambil informasi kelas dari file
        class_id = result.get('class_id')
        all_classes = result.get('all_classes', [class_id] if class_id else [])
        generated = result.get('generated', 0)
        
        # Update count manual untuk semua kelas dalam file
        for cls in all_classes:
            if not cls:
                continue
                
            # Update count
            updated_counts[cls] = updated_counts.get(cls, 0) + generated
            
            # Cek jika sudah mencapai target
            if updated_counts[cls] >= target_count:
                fulfilled_classes.add(cls)
                if logger:
                    logger.info(f"✅ Kelas {cls} telah mencapai target {target_count} instance")
    
    # Versi optimasi menggunakan informasi multi_class_update jika tersedia
    for result in augmentation_results:
        if result.get('status') == 'success' and 'multi_class_update' in result:
            for cls, count in result['multi_class_update'].items():
                # Skip jika kelas tidak valid
                if not cls or cls in fulfilled_classes:
                    continue
                    
                # Update dengan tracking yang lebih akurat
                if cls not in updated_counts:
                    updated_counts[cls] = current_class_counts.get(cls, 0)
                updated_counts[cls] += count
                
                # Cek jika sudah mencapai target
                if updated_counts[cls] >= target_count and cls not in fulfilled_classes:
                    fulfilled_classes.add(cls)
    
    # Return hasil tracking
    return {
        'updated_counts': updated_counts,
        'fulfilled_classes': fulfilled_classes
    }

def prioritize_classes_by_need(
    class_counts: Dict[str, int],
    target_count: int
) -> List[Tuple[str, int]]:
    """
    Prioritaskan kelas berdasarkan kebutuhan augmentasi.
    
    Args:
        class_counts: Jumlah instance kelas saat ini
        target_count: Target jumlah instance per kelas
        
    Returns:
        List tuple (class_id, needed) terurut berdasarkan kebutuhan
    """
    # Hitung kebutuhan untuk tiap kelas dengan one-liner
    class_needs = [(cls, max(0, target_count - count)) 
                 for cls, count in class_counts.items()]
    
    # Urutkan berdasarkan kebutuhan, terbanyak dulu
    sorted_classes = sorted(class_needs, key=lambda x: x[1], reverse=True)
    
    # Filter hanya yang masih membutuhkan augmentasi
    return [(cls, needed) for cls, needed in sorted_classes if needed > 0]

def track_multi_class_distribution(
    label_paths: List[str],
    logger = None
) -> Dict[str, Dict[str, int]]:
    """
    Tracking distribusi multi-class pada file label.
    
    Args:
        label_paths: List path file label
        logger: Logger untuk logging
        
    Returns:
        Dictionary {file_name: {class_id: count}}
    """
    distributions = {}
    
    for label_path in label_paths:
        try:
            # Parse file label
            with open(label_path, 'r') as f:
                lines = f.readlines()
            
            # Count kelas dalam file dengan one-liner
            class_counts = defaultdict(int)
            [class_counts.update({line.split()[0]: class_counts[line.split()[0]] + 1}) 
             for line in lines if len(line.split()) >= 5]
            
            # Simpan distribusi
            file_name = Path(label_path).stem
            distributions[file_name] = dict(class_counts)
            
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat tracking distribusi {label_path}: {str(e)}")
    
    return distributions