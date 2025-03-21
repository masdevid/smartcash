"""
File: smartcash/dataset/services/augmentor/class_balancer.py
Deskripsi: Komponen untuk melakukan balancing kelas pada dataset deteksi objek
"""

import os
import glob
import random
from collections import defaultdict, Counter
from typing import Dict, List, Any, Tuple, Set, Optional
from pathlib import Path

from smartcash.common.logger import get_logger

class ClassBalancer:
    """
    Komponen untuk balancing kelas dataset dengan fokus pada file yang berisi single class 
    untuk meningkatkan kualitas augmentasi.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi ClassBalancer.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger kustom
        """
        self.config = config or {}
        self.logger = logger or get_logger("class_balancer")
    
    def prepare_balanced_dataset(
        self,
        image_files: List[str],
        labels_dir: str,
        target_count: int = 1000,
        filter_single_class: bool = True
    ) -> Dict[str, Any]:
        """
        Persiapkan dataset dengan jumlah sampel yang seimbang antar kelas.
        Jika kelas memiliki N sampel asli (N < target_count), maka kita butuh (target_count - N) sampel augmentasi.
        
        Args:
            image_files: List path file gambar yang tersedia
            labels_dir: Direktori file label
            target_count: Jumlah target sampel per kelas (asli + augmentasi)
            filter_single_class: Filter hanya file dengan 1 kelas
            
        Returns:
            Dictionary informasi balancing dan files yang terpilih
        """
        # Kelompokkan file berdasarkan kelas
        self.logger.info(f"📊 Memulai proses balancing dataset dengan target {target_count} sampel per kelas")
        
        class_to_files = defaultdict(list)  # class_id -> list of files
        file_to_classes = {}  # file_path -> list of classes
        file_class_count = {}  # file_path -> jumlah kelas
        
        # Analisis semua file untuk mendapatkan distribusi kelas
        for img_path in image_files:
            # Dapatkan file label
            filename = os.path.basename(img_path)
            filename_stem = os.path.splitext(filename)[0]
            label_path = os.path.join(labels_dir, f"{filename_stem}.txt")
            
            if not os.path.exists(label_path):
                continue
                
            # Parse label untuk mendapatkan kelas
            try:
                classes_in_file = set()
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            classes_in_file.add(class_id)
                
                # Jika filter single class dan file memiliki lebih dari 1 kelas, skip
                if filter_single_class and len(classes_in_file) > 1:
                    continue
                
                # Simpan informasi
                file_to_classes[img_path] = list(classes_in_file)
                file_class_count[img_path] = len(classes_in_file)
                
                # Tambahkan file ke setiap kelas yang ada di dalamnya
                for class_id in classes_in_file:
                    class_to_files[class_id].append(img_path)
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        # Hitung jumlah file per kelas
        class_counts = {class_id: len(files) for class_id, files in class_to_files.items()}
        
        # Log distribusi kelas
        self.logger.info(f"🔍 Ditemukan {len(class_counts)} kelas berbeda")
        for class_id, count in sorted(class_counts.items()):
            self.logger.info(f"🏷️ Kelas {class_id}: {count} sampel asli, target: {target_count}")
        
        # Pilih file yang perlu diaugmentasi
        selected_files = set()
        augmentation_needs = {}  # class_id -> jumlah yang perlu diaugmentasi
        
        for class_id, files in class_to_files.items():
            current_count = len(files)
            
            # Hitung kebutuhan augmentasi (jumlah variasi yang diperlukan)
            if current_count < target_count:
                # Jika jumlah sampel asli kurang dari target, kita perlu augmentasi
                needed = target_count - current_count
                augmentation_needs[class_id] = needed
                self.logger.info(f"🎯 Kelas {class_id}: perlu {needed} sampel tambahan")
                
                # Pilih semua file untuk kelas ini untuk diaugmentasi
                # Karena 1 file dapat diaugmentasi menjadi beberapa variasi
                selected_files.update(files)
            else:
                # Jika sudah mencukupi, tidak perlu augmentasi
                augmentation_needs[class_id] = 0
                self.logger.info(f"✅ Kelas {class_id}: sudah mencukupi ({current_count} >= {target_count})")
        
        # Jika tidak ada file terpilih, gunakan strategis lain
        if not selected_files and image_files:
            self.logger.warning("⚠️ Tidak ada file terpilih untuk balancing, menggunakan strategi alternatif")
            
            # Strategi alternatif: ambil beberapa file dari kelas yang paling sedikit
            min_class = min(class_counts, key=class_counts.get) if class_counts else None
            if min_class is not None:
                min_files = class_to_files[min_class]
                selected_files.update(min_files)
                self.logger.info(f"🔄 Strategi alternatif: menggunakan {len(min_files)} file dari kelas {min_class}")
        
        # Konversi hasil menjadi list
        selected_files_list = list(selected_files)
        
        # Log hasil
        self.logger.info(f"✅ Terpilih {len(selected_files_list)} file untuk diaugmentasi dari {len(image_files)} total")
        
        return {
            "status": "success",
            "selected_files": selected_files_list,
            "class_counts": class_counts,
            "augmentation_needs": augmentation_needs,
            "total_classes": len(class_counts),
            "total_files": len(image_files),
            "selected_files_count": len(selected_files_list)
        }