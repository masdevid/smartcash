"""
File: smartcash/dataset/utils/file_mapping_utils.py
Deskripsi: Utilitas untuk pemetaan file dan analisis kebutuhan balancing kelas
"""

import os
import random
from pathlib import Path
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from collections import defaultdict

from smartcash.dataset.utils.label_utils import process_label_file

def map_and_analyze_files(
    image_files: List[str], 
    labels_dir: str,
    target_count: int = 1000,
    progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, List[str]], Dict[str, int], Dict[str, int], List[str]]:
    """
    Konsolidasi fungsi mapping, needs calculation, dan selection dengan pendekatan DRY.
    
    Args:
        image_files: List path file gambar
        labels_dir: Direktori label
        target_count: Target jumlah instance per kelas
        progress_callback: Callback untuk progress reporting
        
    Returns:
        Tuple (files_by_class, class_counts, augmentation_needs, selected_files)
    """
    files_by_class, class_counts = defaultdict(list), defaultdict(int)
    
    # Notifikasi mulai pemrosesan
    if progress_callback:
        progress_callback(
            progress=0, total=len(image_files),
            message=f"Menganalisis {len(image_files)} file dataset untuk balancing",
            status="info", step=0
        )
    
    # Proses semua file sekaligus dengan progress reporting
    for i, img_path in enumerate(image_files):
        img_name = Path(img_path).stem
        label_path = str(Path(labels_dir) / f"{img_name}.txt")
        
        # Proses label file untuk mendapatkan kelas
        main_class, all_classes = process_label_file(label_path, True)
        if not main_class: continue
        
        # Update tracking kelas dengan one-liner
        files_by_class[main_class].append(img_path)
        [class_counts.update({cls: class_counts[cls] + 1}) for cls in all_classes]
        
        # Report progres dengan throttling
        if progress_callback and (i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1):
            progress_callback(
                progress=i+1, total=len(image_files),
                message=f"Analisis file ({int((i+1)/len(image_files)*100)}%): {i+1}/{len(image_files)}",
                status="info", step=0
            )
    
    # Calculate augmentation needs dengan one-liner
    augmentation_needs = {cls_id: max(0, target_count - count) 
                       for cls_id, count in class_counts.items()}
    
    # Pilih file untuk augmentasi
    classes_to_augment = [cls_id for cls_id, needed in augmentation_needs.items() if needed > 0]
    selected_files = []
    
    # Proses file selection untuk setiap kelas dengan one-liner where possible
    for i, class_id in enumerate(classes_to_augment):
        needed = augmentation_needs.get(class_id, 0)
        available_files = files_by_class.get(class_id, [])
        
        if available_files and needed > 0:
            # Pilih file secara efisien dengan one-liner
            files_to_augment = (random.sample(available_files, min(len(available_files), needed)) 
                             if len(available_files) > needed else available_files)
            selected_files.extend(files_to_augment)
        
        # Progress reporting untuk class processing
        if progress_callback and (i % max(1, len(classes_to_augment) // 5) == 0 or i == len(classes_to_augment) - 1):
            progress_callback(
                progress=i+1, total=len(classes_to_augment),
                message=f"Pemilihan file ({int((i+1)/len(classes_to_augment)*100)}%): {i+1}/{len(classes_to_augment)} kelas",
                status="info", step=0
            )
    
    # Final progress reporting
    if progress_callback:
        progress_callback(
            message=f"✅ Analisis selesai: {len(selected_files)} file dipilih dari {len(class_counts)} kelas",
            status="info", step=0
        )
    
    return files_by_class, class_counts, augmentation_needs, selected_files

def map_files_to_classes(image_files: List[str], labels_dir: str, progress_callback: Optional[Callable] = None) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Map file ke kelas berdasarkan file label.
    
    Args:
        image_files: List path file gambar
        labels_dir: Path direktori label
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        Tuple (files_by_class, class_counts)
    """
    files_by_class, class_counts, _, _ = map_and_analyze_files(image_files, labels_dir, progress_callback=progress_callback)
    return files_by_class, class_counts

def calculate_augmentation_needs(class_counts: Dict[str, int], target_count: int, progress_callback: Optional[Callable] = None) -> Dict[str, int]:
    """
    Hitung kebutuhan augmentasi untuk setiap kelas.
    
    Args:
        class_counts: Dictionary jumlah instance per kelas
        target_count: Target jumlah instance per kelas
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        Dictionary kebutuhan augmentasi per kelas
    """
    augmentation_needs = {cls_id: max(0, target_count - count) for cls_id, count in class_counts.items()}
    
    # Log ringkasan hasil kebutuhan augmentasi untuk backward compatibility
    if progress_callback:
        classes_needing = sum(1 for needed in augmentation_needs.values() if needed > 0)
        total_needed = sum(augmentation_needs.values())
        progress_callback(
            message=f"📊 Hasil analisis: {classes_needing}/{len(class_counts)} kelas perlu ditambah {total_needed} sampel",
            status="info", step=0
        )
    
    return augmentation_needs

def select_files_for_augmentation(files_by_class: Dict[str, List[str]], augmentation_needs: Dict[str, int], progress_callback: Optional[Callable] = None) -> List[str]:
    """
    Pilih file untuk augmentasi berdasarkan kebutuhan balancing.
    
    Args:
        files_by_class: Dictionary file berdasarkan kelas
        augmentation_needs: Dictionary kebutuhan augmentasi per kelas
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        List file yang akan diaugmentasi
    """
    classes_to_augment = [cls_id for cls_id, needed in augmentation_needs.items() if needed > 0]
    selected_files = []
    
    for class_id in classes_to_augment:
        needed = augmentation_needs.get(class_id, 0)
        available_files = files_by_class.get(class_id, [])
        
        if available_files and needed > 0:
            num_files_to_select = min(len(available_files), needed)
            files_to_augment = random.sample(available_files, num_files_to_select) if len(available_files) > num_files_to_select else available_files
            selected_files.extend(files_to_augment)
    
    # Log ringkasan untuk backward compatibility
    if progress_callback:
        progress_callback(
            message=f"✅ Pemilihan selesai: {len(selected_files)} file dipilih untuk augmentasi",
            status="info", step=0
        )
    
    return selected_files

def select_prioritized_files_for_class(
    class_id: str, 
    files_by_class: Dict[str, List[str]],
    current_counts: Dict[str, int], 
    target_count: int,
    processed_files: Set[str],
    class_data: Dict[str, Any]
) -> List[str]:
    """
    Pilih file untuk augmentasi kelas berdasarkan prioritas.
    
    Args:
        class_id: ID kelas yang akan diprioritaskan
        files_by_class: Dictionary file berdasarkan kelas
        current_counts: Dictionary jumlah instance per kelas saat ini
        target_count: Target jumlah instance per kelas
        processed_files: Set file yang sudah diproses
        class_data: Data kelas (hasil prepare_balancing)
        
    Returns:
        List file yang akan diaugmentasi
    """
    # File yang memiliki kelas ini
    class_candidates = files_by_class.get(class_id, [])
    
    # Hitung kebutuhan berdasarkan target
    current_need = max(0, target_count - current_counts.get(class_id, 0))
    num_files_needed = min(len(class_candidates), current_need)
    
    if num_files_needed <= 0: return []
        
    # Filter file yang belum diproses
    eligible_files = [f for f in class_candidates if f not in processed_files]
    if not eligible_files: return []
    
    # Prioritaskan file berdasarkan kompleksitas kelas
    file_metadata = class_data.get('files_metadata', {})
    
    # Dapatkan kelas yang sudah terpenuhi target
    fulfilled_classes = {cls for cls, count in current_counts.items() if count >= target_count}
    
    # High priority: File dengan kelas yang diperlukan dan tidak memiliki kelas yang sudah terpenuhi
    high_priority = [f for f in eligible_files 
                   if f in file_metadata 
                   and class_id in file_metadata[f].get('classes', set())
                   and not any(cls in fulfilled_classes for cls in file_metadata[f].get('classes', set()))]
    
    # Medium priority: File dengan kelas yang diperlukan tapi memiliki 1-2 kelas yang sudah terpenuhi
    medium_priority = [f for f in eligible_files 
                     if f in file_metadata
                     and class_id in file_metadata[f].get('classes', set())
                     and f not in high_priority
                     and len([cls for cls in file_metadata[f].get('classes', set()) if cls in fulfilled_classes]) <= 2]
    
    # Low priority: Sisanya
    low_priority = [f for f in eligible_files if f not in high_priority and f not in medium_priority]
    
    # Gabungkan berdasarkan prioritas dan batasi jumlah file
    return (high_priority + medium_priority + low_priority)[:num_files_needed]