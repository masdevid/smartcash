"""
File: smartcash/dataset/services/augmentor/class_balancer.py
Deskripsi: Komponen untuk melakukan balancing kelas pada dataset deteksi objek dengan perhitungan ulang target sebaran
"""

import os
import glob
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
        Target = 1000 - jumlah label class X di split train.
        
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
        class_instance_counts = defaultdict(int)  # class_id -> total instance count
        
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
                classes_in_file = []
                class_counts = defaultdict(int)
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            classes_in_file.append(class_id)
                            class_counts[class_id] += 1
                            class_instance_counts[class_id] += 1
                
                unique_classes = set(classes_in_file)
                
                # Jika filter single class dan file memiliki lebih dari 1 kelas, skip
                if filter_single_class and len(unique_classes) > 1:
                    continue
                
                # Simpan informasi
                file_to_classes[img_path] = list(unique_classes)
                file_class_count[img_path] = len(unique_classes)
                
                # Tambahkan file ke setiap kelas yang ada di dalamnya
                for class_id in unique_classes:
                    class_to_files[class_id].append((img_path, class_counts[class_id]))
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat membaca label {label_path}: {str(e)}")
        
        # Hitung jumlah file dan instance per kelas
        class_counts = {class_id: len(files) for class_id, files in class_to_files.items()}
        
        # Log distribusi kelas asli
        self.logger.info(f"🔍 Ditemukan {len(class_counts)} kelas berbeda")
        for class_id, count in sorted(class_instance_counts.items()):
            self.logger.info(f"🏷️ Kelas {class_id}: {count} objek dalam {class_counts.get(class_id, 0)} file")
        
        # Pilih file yang perlu diaugmentasi berdasarkan jumlah instance per kelas
        selected_files = set()
        augmentation_needs = {}  # class_id -> jumlah yang perlu diaugmentasi
        
        for class_id, instance_count in class_instance_counts.items():
            # Hitung kebutuhan augmentasi (jumlah instance tambahan yang diperlukan)
            if instance_count < target_count:
                # Jika jumlah objek kurang dari target, kita perlu augmentasi
                needed = target_count - instance_count
                augmentation_needs[class_id] = needed
                self.logger.info(f"🎯 Kelas {class_id}: memiliki {instance_count} objek, perlu {needed} tambahan (target {target_count})")
                
                # Pilih file yang mengandung kelas ini untuk diaugmentasi
                # Urutkan berdasarkan jumlah instance kelas ini dalam file (prioritaskan yang lebih banyak)
                files_with_counts = class_to_files.get(class_id, [])
                # Sort descending berdasarkan jumlah instance
                files_with_counts.sort(key=lambda x: x[1], reverse=True)
                
                # Tambahkan file ke daftar yang akan diaugmentasi
                for file_path, _ in files_with_counts:
                    selected_files.add(file_path)
            else:
                # Jika sudah mencukupi, tidak perlu augmentasi
                augmentation_needs[class_id] = 0
                self.logger.info(f"✅ Kelas {class_id}: memiliki {instance_count} objek, sudah mencukupi (target {target_count})")
        
        # Jika tidak ada file terpilih, gunakan strategi alternatif
        if not selected_files and image_files:
            self.logger.warning("⚠️ Tidak ada file terpilih untuk balancing, menggunakan strategi alternatif")
            
            # Strategi alternatif: ambil beberapa file dari kelas yang paling sedikit instancenya
            min_class_id = min(class_instance_counts, key=class_instance_counts.get) if class_instance_counts else None
            if min_class_id is not None:
                files_with_counts = class_to_files[min_class_id]
                for file_path, _ in files_with_counts:
                    selected_files.add(file_path)
                self.logger.info(f"🔄 Strategi alternatif: menggunakan {len(selected_files)} file dari kelas {min_class_id}")
        
        # Konversi hasil menjadi list
        selected_files_list = list(selected_files)
        
        # Log hasil
        self.logger.info(f"✅ Terpilih {len(selected_files_list)} file untuk diaugmentasi dari {len(image_files)} total")
        
        return {
            "status": "success",
            "selected_files": selected_files_list,
            "class_counts": class_counts,
            "class_instance_counts": dict(class_instance_counts),
            "augmentation_needs": augmentation_needs,
            "total_classes": len(class_counts),
            "total_files": len(image_files),
            "selected_files_count": len(selected_files_list)
        }