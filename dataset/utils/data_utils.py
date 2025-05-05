"""
File: smartcash/dataset/utils/data_utils.py
Deskripsi: Utilitas umum untuk manipulasi data dan file dataset
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from pathlib import Path
import os
import shutil
import glob
import time
import logging
import random
import numpy as np
import cv2

def find_image_files(directory: str, patterns: List[str] = None) -> List[str]:
    """
    Cari file gambar dalam direktori berdasarkan pattern.
    
    Args:
        directory: Direktori yang akan dicari
        patterns: List pattern file (default: ['*.jpg', '*.jpeg', '*.png'])
        
    Returns:
        List path file gambar
    """
    patterns = patterns or ['*.jpg', '*.jpeg', '*.png']
    image_files = []
    
    for pattern in patterns:
        image_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    return sorted(image_files)

def find_matching_label(image_path: str, labels_dir: str) -> Optional[str]:
    """
    Cari file label yang sesuai dengan gambar.
    
    Args:
        image_path: Path file gambar
        labels_dir: Direktori label
        
    Returns:
        Path file label atau None jika tidak ditemukan
    """
    img_name = Path(image_path).stem
    label_path = os.path.join(labels_dir, f"{img_name}.txt")
    
    return label_path if os.path.exists(label_path) else None

def load_image(image_path: str, convert_rgb: bool = True) -> np.ndarray:
    """
    Load gambar dengan support untuk berbagai format.
    
    Args:
        image_path: Path gambar
        convert_rgb: Flag untuk konversi ke RGB (dari BGR)
        
    Returns:
        Array numpy berisi gambar
    """
    if str(image_path).endswith('.npy'):
        # Handle numpy array
        img = np.load(str(image_path))
        # Denormalisasi jika perlu
        if img.dtype == np.float32 and img.max() <= 1.0:
            img = (img * 255).astype(np.uint8)
    else:
        # Handle gambar biasa
        img = cv2.imread(str(image_path))
        if convert_rgb and img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    return img

def save_image(image: np.ndarray, save_path: str, is_normalized: bool = False) -> bool:
    """
    Simpan gambar ke file dengan dukungan normalisasi.
    
    Args:
        image: Array numpy berisi gambar
        save_path: Path untuk menyimpan gambar
        is_normalized: Flag apakah gambar dinormalisasi (nilai 0-1)
        
    Returns:
        Boolean yang menunjukkan keberhasilan
    """
    try:
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Denormalisasi jika perlu
        save_image = (image * 255).astype(np.uint8) if is_normalized and image.dtype == np.float32 else image
        
        # Konversi ke BGR jika perlu (untuk cv2.imwrite)
        if len(save_image.shape) == 3 and save_image.shape[2] == 3:
            save_image = cv2.cvtColor(save_image, cv2.COLOR_RGB2BGR)
            
        # Simpan gambar
        return cv2.imwrite(save_path, save_image)
    except Exception:
        return False

def split_dataset(
    image_files: List[str], 
    output_dir: str, 
    split_ratios: Dict[str, float] = None, 
    with_labels: bool = True,
    labels_dir: str = None,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Split dataset menjadi train, validation, dan test.
    
    Args:
        image_files: List file gambar
        output_dir: Direktori output
        split_ratios: Dictionary rasio split (default: {'train': 0.7, 'valid': 0.15, 'test': 0.15})
        with_labels: Flag untuk menyertakan label
        labels_dir: Direktori label (jika with_labels=True)
        random_seed: Seed untuk random
        
    Returns:
        Dictionary file per split
    """
    # Default split ratios
    split_ratios = split_ratios or {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    
    # Validasi split ratios
    total_ratio = sum(split_ratios.values())
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Total split ratio harus mendekati 1.0, didapatkan {total_ratio}")
    
    # Set random seed
    random.seed(random_seed)
    
    # Acak file
    image_files = image_files.copy()
    random.shuffle(image_files)
    
    # Jika with_labels, filter file yang memiliki label
    if with_labels and labels_dir:
        image_files = [f for f in image_files if os.path.exists(os.path.join(labels_dir, f"{Path(f).stem}.txt"))]
    
    # Hitung jumlah file per split
    total_files = len(image_files)
    split_counts = {
        split: int(ratio * total_files) for split, ratio in split_ratios.items()
    }
    
    # Koreksi jika total tidak sama dengan jumlah file
    remaining = total_files - sum(split_counts.values())
    if remaining > 0:
        # Tambahkan sisanya ke split dengan ratio terbesar
        max_split = max(split_ratios.items(), key=lambda x: x[1])[0]
        split_counts[max_split] += remaining
    
    # Split files
    result = {}
    start_idx = 0
    for split, count in split_counts.items():
        end_idx = start_idx + count
        result[split] = image_files[start_idx:end_idx]
        start_idx = end_idx
    
    return result

def organize_dataset(
    split_files: Dict[str, List[str]], 
    output_dir: str, 
    with_labels: bool = True,
    labels_dir: str = None,
    copy_files: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Organize dataset ke struktur yang sesuai.
    
    Args:
        split_files: Dictionary file per split
        output_dir: Direktori output
        with_labels: Flag untuk menyertakan label
        labels_dir: Direktori label (jika with_labels=True)
        copy_files: Flag untuk copy file (jika False, gunakan symlink)
        
    Returns:
        Dictionary statistik per split
    """
    stats = {}
    
    for split, files in split_files.items():
        # Buat direktori
        split_dir = os.path.join(output_dir, split)
        images_dir = os.path.join(split_dir, 'images')
        labels_dir_out = os.path.join(split_dir, 'labels') if with_labels else None
        
        os.makedirs(images_dir, exist_ok=True)
        if labels_dir_out:
            os.makedirs(labels_dir_out, exist_ok=True)
        
        # Copy/symlink files
        copied_images = 0
        copied_labels = 0
        
        for img_path in files:
            # Copy/symlink gambar
            dest_img = os.path.join(images_dir, os.path.basename(img_path))
            if copy_files:
                shutil.copy2(img_path, dest_img)
            else:
                if os.path.exists(dest_img):
                    os.remove(dest_img)
                os.symlink(os.path.abspath(img_path), dest_img)
            copied_images += 1
            
            # Copy/symlink label jika perlu
            if with_labels and labels_dir:
                label_name = f"{Path(img_path).stem}.txt"
                src_label = os.path.join(labels_dir, label_name)
                if os.path.exists(src_label):
                    dest_label = os.path.join(labels_dir_out, label_name)
                    if copy_files:
                        shutil.copy2(src_label, dest_label)
                    else:
                        if os.path.exists(dest_label):
                            os.remove(dest_label)
                        os.symlink(os.path.abspath(src_label), dest_label)
                    copied_labels += 1
        
        # Update stats
        stats[split] = {
            'images': copied_images,
            'labels': copied_labels
        }
    
    return stats