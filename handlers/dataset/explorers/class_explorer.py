# File: smartcash/handlers/dataset/explorers/class_explorer.py
# Author: Alfrida Sabar
# Deskripsi: Explorer khusus untuk distribusi kelas dalam dataset

import collections
from pathlib import Path
from typing import Dict, List, Optional, Any
import numpy as np
from tqdm.auto import tqdm

from smartcash.handlers.dataset.explorers.base_explorer import BaseExplorer

class ClassExplorer(BaseExplorer):
    """
    Explorer khusus untuk distribusi kelas dalam dataset.
    Menganalisis keseimbangan, representasi, dan statistik kelas.
    """
    
    def explore(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Dict hasil analisis kelas
        """
        self.logger.info(f"🔍 Analisis distribusi kelas: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file gambar
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_files.extend(list(images_dir.glob(ext)))
        
        if not image_files:
            self.logger.warning(f"⚠️ Tidak ada file gambar di split {split}")
            return {'error': f"Tidak ada file gambar di split {split}"}
        
        # Batasi sampel jika diperlukan
        if 0 < sample_size < len(image_files):
            import random
            image_files = random.sample(image_files, sample_size)
            self.logger.info(f"🔍 Menggunakan {sample_size} sampel untuk analisis kelas")
        
        # Analisis distribusi kelas
        class_balance = self._analyze_class_distribution(labels_dir, image_files)
        
        # Log hasil analisis
        total_objects = class_balance.get('total_objects', 0)
        class_counts = class_balance.get('class_counts', {})
        class_percentages = class_balance.get('class_percentages', {})
        underrepresented = class_balance.get('underrepresented_classes', [])
        overrepresented = class_balance.get('overrepresented_classes', [])
        imbalance_score = class_balance.get('imbalance_score', 0)
        
        self.logger.info(
            f"📊 Hasil analisis kelas '{split}':\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah kelas terdeteksi: {len(class_counts)}\n"
            f"   • Skor ketidakseimbangan: {imbalance_score:.2f}/10"
        )
        
        if underrepresented:
            self.logger.info(f"   • Kelas kurang terwakili: {', '.join(underrepresented[:5])}" + 
                          (f" dan {len(underrepresented) - 5} lainnya" if len(underrepresented) > 5 else ""))
            
        if overrepresented:
            self.logger.info(f"   • Kelas dominan: {', '.join(overrepresented[:5])}" + 
                          (f" dan {len(overrepresented) - 5} lainnya" if len(overrepresented) > 5 else ""))
        
        return class_balance
    
    def get_class_statistics(self, split: str) -> Dict[str, Any]:
        """
        Dapatkan statistik kelas untuk split tertentu.
        Fokus pada jumlah objek per kelas dan gambar per kelas.
        
        Args:
            split: Split dataset ('train', 'valid', 'test')
            
        Returns:
            Dict statistik kelas
        """
        self.logger.info(f"📊 Mengumpulkan statistik kelas untuk split: {split}")
        
        # Tentukan path split
        split_dir = self._get_split_path(split)
        images_dir = split_dir / 'images'
        labels_dir = split_dir / 'labels'
        
        if not (images_dir.exists() and labels_dir.exists()):
            self.logger.warning(f"⚠️ Split {split} tidak ditemukan atau tidak lengkap")
            return {'error': f"Split {split} tidak ditemukan atau tidak lengkap"}
        
        # Cari semua file label
        label_files = list(labels_dir.glob('*.txt'))
        
        # Hitung jumlah objek per kelas
        class_counts = collections.Counter()
        images_per_class = {}
        
        for label_path in tqdm(label_files, desc=f"Menganalisis kelas {split}"):
            try:
                if not label_path.exists():
                    continue
                    
                # Kumpulkan kelas yang ada dalam gambar ini
                classes_in_image = set()
                
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                class_name = self._get_class_name(cls_id)
                                
                                class_counts[class_name] += 1
                                classes_in_image.add(class_name)
                            except (ValueError, IndexError):
                                continue
                
                # Update set gambar per kelas
                for cls_name in classes_in_image:
                    if cls_name not in images_per_class:
                        images_per_class[cls_name] = set()
                    images_per_class[cls_name].add(label_path.stem)
                    
            except Exception as e:
                self.logger.debug(f"⚠️ Error membaca {label_path}: {str(e)}")
        
        # Total objek dan persentase
        total_objects = sum(class_counts.values())
        class_percentages = {}
        
        if total_objects > 0:
            for cls, count in class_counts.items():
                class_percentages[cls] = (count / total_objects) * 100
        
        # Statistik jumlah gambar per kelas
        images_count = {cls: len(images) for cls, images in images_per_class.items()}
        
        return {
            'total_objects': total_objects,
            'class_counts': dict(class_counts),
            'class_percentages': class_percentages,
            'images_per_class': images_count
        }
    
    def _analyze_class_distribution(
        self,
        labels_dir: Path,
        image_files: List[Path]
    ) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            labels_dir: Direktori label
            image_files: List file gambar
            
        Returns:
            Dict berisi statistik distribusi kelas
        """
        self.logger.info("🔍 Menganalisis distribusi kelas...")
        
        # Hitung jumlah objek per kelas
        class_counts = collections.Counter()
        
        for img_path in tqdm(image_files, desc="Menganalisis kelas"):
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            try:
                with open(label_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                cls_id = int(float(parts[0]))
                                class_counts[cls_id] += 1
                            except (ValueError, IndexError):
                                continue
            except Exception:
                # Skip file jika ada error
                continue
        
        # Total objek
        total_objects = sum(class_counts.values())
        
        # Hitung persentase per kelas
        class_percentages = {}
        if total_objects > 0:
            for cls_id, count in class_counts.items():
                class_percentages[self._get_class_name(cls_id)] = (count / total_objects) * 100
        
        # Identifikasi kelas yang tidak seimbang
        if class_counts:
            avg_count = total_objects / len(class_counts)
            
            # Kelas yang kurang terwakili (< 50% dari rata-rata)
            underrepresented = []
            for cls_id, count in class_counts.items():
                if count < avg_count * 0.5:
                    underrepresented.append(self._get_class_name(cls_id))
            
            # Kelas yang terlalu dominan (> 200% dari rata-rata)
            overrepresented = []
            for cls_id, count in class_counts.items():
                if count > avg_count * 2:
                    overrepresented.append(self._get_class_name(cls_id))
            
            # Hitung skor ketidakseimbangan (0-10)
            # 0 = sangat seimbang, 10 = sangat tidak seimbang
            counts = list(class_counts.values())
            if len(counts) > 1:
                max_count = max(counts)
                min_count = min(counts)
                
                if min_count > 0:
                    imbalance_ratio = max_count / min_count
                    imbalance_score = min(10, (imbalance_ratio - 1) / 2)
                else:
                    imbalance_score = 10
            else:
                imbalance_score = 0
                
        else:
            underrepresented = []
            overrepresented = []
            imbalance_score = 0
        
        return {
            'total_objects': total_objects,
            'class_counts': {self._get_class_name(cls_id): count for cls_id, count in class_counts.items()},
            'class_percentages': class_percentages,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented,
            'imbalance_score': imbalance_score
        }