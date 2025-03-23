"""
File: smartcash/dataset/utils/label_utils.py
Deskripsi: Utilitas untuk pemrosesan dan ekstraksi data dari file label YOLO
"""

import os
from typing import Dict, List, Any, Optional, Set, Tuple
from pathlib import Path

from smartcash.dataset.utils.denomination_utils import DENOMINATION_CLASS_MAP

def get_class_from_label(label_path: str) -> Optional[str]:
    """
    Ekstrak ID kelas utama dari file label YOLOv5 dengan prioritas denominasi.
    
    Args:
        label_path: Path ke file label
        
    Returns:
        ID kelas utama atau None jika tidak ditemukan
    """
    try:
        if not os.path.exists(label_path): 
            return None
            
        # Baca file dan ekstrak class_ids dengan one-liner
        with open(label_path, 'r') as f:
            class_ids = [parts[0] for line in f.readlines() 
                      if len(parts := line.strip().split()) >= 5]
                      
        # Jika tidak ada kelas yang valid, return None
        if not class_ids: 
            return None
        
        # Prioritaskan denominasi (kelas yang ada di mapping)
        valid_denomination_ids = [cls for cls in class_ids if cls in DENOMINATION_CLASS_MAP]
        
        # Jika ada valid denomination, ambil yang terkecil (untuk konsistensi)
        if valid_denomination_ids:
            return min(valid_denomination_ids)
            
        # Jika tidak ada valid denomination, ambil yang terkecil dari semua kelas
        return min(class_ids)
    except Exception:
        return None

def process_label_file(label_path: str, collect_all_classes: bool = False) -> Tuple[Optional[str], Set[str]]:
    """
    Proses file label untuk mendapatkan kelas utama dan semua kelas dengan prioritas denominasi.
    
    Args:
        label_path: Path ke file label
        collect_all_classes: Flag untuk mengumpulkan semua kelas
        
    Returns:
        Tuple berisi (kelas utama, set semua kelas) atau (None, set kosong) jika gagal
    """
    try:
        if not os.path.exists(label_path): 
            return None, set()
            
        # Baca file dan ekstrak class_ids dengan one-liner
        with open(label_path, 'r') as f:
            class_ids = [parts[0] for line in f.readlines() 
                      if len(parts := line.strip().split()) >= 5]
        
        if not class_ids: 
            return None, set()
        
        # Prioritaskan denominasi (kelas yang ada di mapping)
        valid_denomination_ids = [cls for cls in class_ids if cls in DENOMINATION_CLASS_MAP]
        
        # Tentukan kelas utama
        main_class = min(valid_denomination_ids) if valid_denomination_ids else min(class_ids)
        
        return (main_class, set(class_ids) if collect_all_classes and class_ids else set())
    except Exception:
        return None, set()

def extract_class_from_label(label_path: Path, config: Dict = None, logger=None) -> Optional[str]:
    """
    Ekstrak kelas dari file label YOLO dengan memilih class ID terkecil (prioritas banknote).
    
    Args:
        label_path: Path ke file label
        config: Konfigurasi aplikasi (opsional)
        logger: Logger untuk mencatat pesan (opsional)
        
    Returns:
        Nama kelas atau None jika tidak ditemukan
    """
    try:
        if not label_path.exists(): 
            return None
            
        with open(label_path, 'r') as f: 
            label_lines = f.readlines()
            
        if not label_lines: 
            return None
        
        # Ekstrak class IDs dari semua baris dengan one-liner
        class_ids = [parts[0] for line in label_lines if (parts := line.strip().split()) and parts]
        
        if not class_ids: 
            return None
        
        # Prioritaskan kelas yang ada di mapping denominasi
        valid_denomination_ids = [cls for cls in class_ids if cls in DENOMINATION_CLASS_MAP]
        
        # Jika ada kelas denominasi yang valid, ambil terkecil
        if valid_denomination_ids:
            min_class_id = min(valid_denomination_ids)
        else:
            # Jika tidak ada, ambil class ID terkecil
            min_class_id = min(class_ids)
            
        # Konversi ke nama kelas jika tersedia di config
        class_names = config.get('data', {}).get('class_names', {}) if config else {}
        return class_names[str(min_class_id)] if class_names and str(min_class_id) in class_names else f"class{min_class_id}"
    except Exception as e:
        if logger: 
            logger.debug(f"⚠️ Gagal ekstrak kelas dari {label_path}: {str(e)}")
        return None

def get_classes_from_label(label_path: str) -> Tuple[Optional[str], Set[str]]:
    """
    Ekstrak ID kelas utama dan semua kelas dari file label YOLOv5 dengan prioritas denominasi.
    
    Args:
        label_path: Path file label
        
    Returns:
        Tuple (ID kelas utama, set semua kelas)
    """
    try:
        if not os.path.exists(label_path):
            return None, set()
            
        # Baca file label
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        # Ekstrak semua class ID
        class_ids = []
        all_classes = set()
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:  # Format YOLOv5: class_id x y width height
                class_id = parts[0]
                class_ids.append(class_id)
                all_classes.add(class_id)
        
        if not class_ids:
            return None, set()
            
        # Prioritaskan denominasi (kelas yang ada di mapping)
        valid_denomination_ids = [cls for cls in class_ids if cls in DENOMINATION_CLASS_MAP]
        
        # Jika ada valid denomination, ambil yang terkecil (untuk konsistensi)
        if valid_denomination_ids:
            main_class = min(valid_denomination_ids)
        else:
            # Jika tidak ada valid denomination, ambil yang terkecil dari semua kelas
            main_class = min(class_ids)
        
        return main_class, all_classes
    except Exception:
        return None, set()