"""
File: smartcash/dataset/services/augmentor/class_balancer_helper.py
Deskripsi: Fungsi utility untuk membantu class balancer dengan progres tracking yang lebih baik
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import os
import random
from pathlib import Path

def map_files_to_classes(
    image_files: List[str], 
    labels_dir: str,
    progress_callback: Optional[Callable] = None
) -> Tuple[Dict[str, List[str]], Dict[str, int]]:
    """
    Memetakan file gambar ke kelas mereka berdasarkan label YOLOv5.
    
    Args:
        image_files: List path file gambar
        labels_dir: Path direktori label
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        Tuple (files_by_class, class_counts)
    """
    files_by_class = defaultdict(list)
    class_counts = defaultdict(int)
    
    # Notifikasi mulai pemrosesan
    if progress_callback:
        progress_callback(
            progress=0, 
            total=len(image_files),
            message=f"Memetakan {len(image_files)} file ke kelas masing-masing",
            status="info",
            step=0  # Tahap analisis
        )
    
    # Proses setiap file gambar dengan progres tracking
    for i, img_path in enumerate(image_files):
        # Dapatkan file label yang sesuai
        img_name = Path(img_path).stem
        label_path = Path(labels_dir) / f"{img_name}.txt"
        
        if not label_path.exists():
            continue
            
        # Baca file label untuk mendapatkan kelas
        try:
            with open(label_path, 'r') as f:
                label_content = f.readlines()
                
            # Ekstrak class ID dari setiap baris
            for line in label_content:
                parts = line.strip().split()
                if len(parts) >= 5:  # Format YOLO: class_id x y width height
                    class_id = parts[0]
                    files_by_class[class_id].append(img_path)
                    class_counts[class_id] += 1
        except Exception:
            # Skip file yang bermasalah
            continue
        
        # Report progres dengan throttling
        if progress_callback and (i % max(1, len(image_files) // 10) == 0 or i == len(image_files) - 1):
            progress_callback(
                progress=i+1, 
                total=len(image_files),
                message=f"Analisis file ({int((i+1)/len(image_files)*100)}%): {i+1}/{len(image_files)}",
                status="info",
                step=0  # Tahap analisis
            )
    
    # Log ringkasan hasil pemetaan
    if progress_callback:
        progress_callback(
            progress=len(image_files), 
            total=len(image_files),
            message=f"✅ Analisis selesai: {len(files_by_class)} kelas ditemukan dalam {len(image_files)} file",
            status="info",
            step=0  # Tahap analisis
        )
    
    return files_by_class, class_counts

def calculate_augmentation_needs(
    class_counts: Dict[str, int], 
    target_count: int,
    progress_callback: Optional[Callable] = None
) -> Dict[str, int]:
    """
    Menghitung kebutuhan augmentasi per kelas untuk mencapai target balancing.
    
    Args:
        class_counts: Dictionary jumlah instance per kelas
        target_count: Target jumlah instance per kelas
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        Dictionary kebutuhan augmentasi per kelas
    """
    augmentation_needs = {}
    
    # Notifikasi mulai perhitungan
    if progress_callback:
        progress_callback(
            message=f"Menghitung kebutuhan augmentasi untuk {len(class_counts)} kelas (target: {target_count}/kelas)",
            status="info",
            step=0  # Tahap analisis
        )
    
    # Hitung kebutuhan augmentasi per kelas
    for i, (class_id, count) in enumerate(class_counts.items()):
        # Jika jumlah instance lebih kecil dari target, perlu augmentasi
        if count < target_count:
            augmentation_needs[class_id] = target_count - count
        else:
            augmentation_needs[class_id] = 0
            
        # Report progress dengan throttling
        if progress_callback and (i % max(1, len(class_counts) // 5) == 0 or i == len(class_counts) - 1):
            progress_callback(
                progress=i+1, 
                total=len(class_counts),
                message=f"Analisis kebutuhan augmentasi: {i+1}/{len(class_counts)} kelas",
                status="info",
                step=0  # Tahap analisis
            )
    
    # Statistik hasil
    classes_needing_augmentation = sum(1 for needed in augmentation_needs.values() if needed > 0)
    total_needed = sum(augmentation_needs.values())
    
    # Log ringkasan hasil kebutuhan augmentasi
    if progress_callback:
        progress_callback(
            message=f"📊 Hasil analisis: {classes_needing_augmentation}/{len(class_counts)} kelas perlu ditambah {total_needed} sampel",
            status="info",
            step=0  # Tahap analisis
        )
    
    return augmentation_needs

def select_files_for_augmentation(
    files_by_class: Dict[str, List[str]],
    augmentation_needs: Dict[str, int],
    progress_callback: Optional[Callable] = None
) -> List[str]:
    """
    Pilih file untuk augmentasi berdasarkan kebutuhan balancing.
    
    Args:
        files_by_class: Dictionary file per kelas
        augmentation_needs: Dictionary kebutuhan augmentasi per kelas
        progress_callback: Callback untuk melaporkan progres
        
    Returns:
        List file yang dipilih untuk augmentasi
    """
    selected_files = []
    
    # Notifikasi mulai pemilihan file
    if progress_callback:
        progress_callback(
            message=f"Memilih file untuk augmentasi berdasarkan kebutuhan balancing",
            status="info",
            step=0  # Tahap analisis
        )
    
    # Hitung jumlah total kelas yang membutuhkan augmentasi
    classes_to_augment = [cls_id for cls_id, needed in augmentation_needs.items() if needed > 0]
    
    # Proses setiap kelas dengan progres tracking
    for i, class_id in enumerate(classes_to_augment):
        needed = augmentation_needs[class_id]
        available_files = files_by_class.get(class_id, [])
        
        if not available_files or needed <= 0:
            continue
            
        # Jika jumlah file lebih sedikit dari kebutuhan augmentasi
        # kita perlu menggunakan beberapa file lebih dari sekali
        num_files_to_select = min(len(available_files), needed)
        
        # Pilih file secara acak jika ada banyak file
        if len(available_files) > num_files_to_select:
            files_to_augment = random.sample(available_files, num_files_to_select)
        else:
            files_to_augment = available_files
            
        # Tambahkan ke daftar file terpilih
        selected_files.extend(files_to_augment)
        
        # Report progress dengan throttling
        if progress_callback and (i % max(1, len(classes_to_augment) // 5) == 0 or i == len(classes_to_augment) - 1):
            percentage = int((i+1) / len(classes_to_augment) * 100)
            progress_callback(
                progress=i+1, 
                total=len(classes_to_augment),
                message=f"Pemilihan file ({percentage}%): kelas {i+1}/{len(classes_to_augment)}",
                status="info",
                step=0  # Tahap analisis
            )
    
    # Log ringkasan hasil pemilihan file
    if progress_callback:
        progress_callback(
            message=f"✅ Pemilihan selesai: {len(selected_files)} file dipilih untuk augmentasi",
            status="info",
            step=0  # Tahap analisis
        )
    
    return selected_files