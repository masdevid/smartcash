"""
File: smartcash/dataset/services/augmentor/class_balancer.py
Deskripsi: Kelas untuk balancing class dalam dataset dengan pendekatan DRY
"""

from typing import Dict, List, Tuple, Any, Optional, Callable
from collections import defaultdict
import os
import logging
from pathlib import Path

# Import utils yang telah dipindahkan
from smartcash.dataset.utils.augmentor_utils import (
    map_files_to_classes,
    calculate_augmentation_needs,
    select_files_for_augmentation
)

class ClassBalancer:
    """Kelas untuk balancing class dalam dataset"""
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi ClassBalancer.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging
        """
        self.config = config or {}
        self.logger = logger or logging.getLogger(__name__)
    
    def prepare_balanced_dataset(
        self,
        image_files: List[str],
        labels_dir: str,
        target_count: int = 1000,
        progress_callback: Optional[Callable] = None,
        filter_single_class: bool = False
    ) -> Dict[str, Any]:
        """
        Persiapkan dataset yang dibalance berdasarkan kebutuhan augmentasi per kelas.
        
        Args:
            image_files: List path file gambar
            labels_dir: Path direktori label
            target_count: Target jumlah instance per kelas
            progress_callback: Callback untuk melaporkan progres
            filter_single_class: Filter hanya file yang memiliki satu kelas
            
        Returns:
            Dictionary hasil balancing
        """
        # Logging analysis start
        self.logger.info(f"🔍 Menganalisis distribusi kelas untuk balancing (target: {target_count}/kelas)")
        
        # Gunakan fungsi util untuk memetakan file ke kelas
        files_by_class, class_counts = map_files_to_classes(
            image_files, labels_dir, progress_callback
        )
        
        # Hitung kebutuhan augmentasi untuk setiap kelas
        augmentation_needs = calculate_augmentation_needs(
            class_counts, target_count, progress_callback
        )
        
        # Pilih file untuk augmentasi
        selected_files = select_files_for_augmentation(
            files_by_class, augmentation_needs, progress_callback
        )
        
        # Filter file dengan single class jika diminta
        if filter_single_class:
            selected_files = self._filter_single_class_files(selected_files, labels_dir)
            self.logger.info(f"🔍 Filtering untuk single class: {len(selected_files)} file terpilih")
        
        # Generate hasil
        result = {
            'class_counts': class_counts,
            'class_files': files_by_class,
            'augmentation_needs': augmentation_needs,
            'selected_files': selected_files,
            'target_count': target_count,
            'total_classes': len(class_counts),
            'classes_to_augment': sum(1 for v in augmentation_needs.values() if v > 0),
            'total_needed': sum(augmentation_needs.values())
        }
        
        # Log ringkasan balancing
        self.logger.info(f"📊 Ringkasan balancing: {result['classes_to_augment']}/{result['total_classes']} kelas perlu ditambah {result['total_needed']} instance")
        
        return result
    
    def _filter_single_class_files(self, files: List[str], labels_dir: str) -> List[str]:
        """
        Filter file yang hanya memiliki single class.
        
        Args:
            files: List path file
            labels_dir: Path direktori label
            
        Returns:
            List file yang hanya memiliki satu kelas
        """
        single_class_files = []
        
        for file_path in files:
            # Dapatkan path label
            file_name = Path(file_path).stem
            label_path = os.path.join(labels_dir, f"{file_name}.txt")
            
            if not os.path.exists(label_path):
                continue
                
            # Cek apakah file hanya memiliki satu class
            try:
                with open(label_path, 'r') as f:
                    lines = f.readlines()
                
                # Ekstrak semua class ID
                class_ids = set()
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_ids.add(parts[0])
                
                # Jika hanya ada satu class, tambahkan ke daftar
                if len(class_ids) == 1:
                    single_class_files.append(file_path)
            except Exception as e:
                self.logger.debug(f"⚠️ Error saat filter file {file_path}: {str(e)}")
                continue
        
        return single_class_files