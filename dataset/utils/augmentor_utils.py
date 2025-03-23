"""
File: smartcash/dataset/utils/augmentor_utils.py
Deskripsi: Fungsi helper untuk augmentasi dataset yang digunakan oleh augmentation_worker.py dan class_balancer.py
"""

import os
import glob
from pathlib import Path
import random
from typing import Dict, List, Any, Optional, Tuple, Callable
from collections import defaultdict
# Import tambahan yang mungkin diperlukan
import shutil

def get_class_from_label(label_path: str) -> Optional[str]:
    """
    Ekstrak ID kelas utama dari file label YOLOv5.
    
    Args:
        label_path: Path file label
        
    Returns:
        ID kelas utama atau None jika tidak ada
    """
    try:
        if not os.path.exists(label_path):
            return None
            
        # Baca file label
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Cari class ID
        class_ids = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Format YOLOv5: class_id x y width height
                class_ids.append(parts[0])
                
        # Jika ada class ID, ambil yang terkecil (prioritas banknote)
        if class_ids:
            return min(class_ids)
        
        return None
    except Exception:
        return None

def move_files_to_preprocessed(images_output_dir: str, labels_output_dir: str, 
                              output_prefix: str, final_output_dir: str,
                              split: str, logger=None) -> bool:
    """
    Pindahkan file augmentasi ke direktori preprocessed.
    
    Args:
        images_output_dir: Direktori output gambar
        labels_output_dir: Direktori output label
        output_prefix: Prefix file output
        final_output_dir: Direktori target akhir
        split: Split dataset
        logger: Logger untuk logging
        
    Returns:
        Boolean status keberhasilan
    """
    try:
        # Buat direktori target jika belum ada
        os.makedirs(os.path.join(final_output_dir, split, 'images'), exist_ok=True)
        os.makedirs(os.path.join(final_output_dir, split, 'labels'), exist_ok=True)
        
        # Pattern untuk mencari file hasil augmentasi
        pattern = f"{output_prefix}_*.jpg"
        augmented_files = glob.glob(os.path.join(images_output_dir, pattern))
        
        # Log jumlah file yang akan dipindahkan
        if logger:
            logger.info(f"📦 Memindahkan {len(augmented_files)} file augmentasi ke {final_output_dir}/{split}")
            
        # Pindahkan semua file gambar
        for img_file in augmented_files:
            img_name = os.path.basename(img_file)
            label_name = f"{os.path.splitext(img_name)[0]}.txt"
            
            # Path target
            img_target = os.path.join(final_output_dir, split, 'images', img_name)
            label_target = os.path.join(final_output_dir, split, 'labels', label_name)
            
            # Path label
            label_file = os.path.join(labels_output_dir, label_name)
            
            # Pindahkan file gambar dan label
            if os.path.exists(img_file):
                shutil.copy2(img_file, img_target)
                os.remove(img_file)  # Hapus file sumber
            
            if os.path.exists(label_file):
                shutil.copy2(label_file, label_target)
                os.remove(label_file)  # Hapus file sumber
                
        return True
    except Exception as e:
        if logger:
            logger.error(f"❌ Error saat memindahkan file: {str(e)}")
        return False

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

