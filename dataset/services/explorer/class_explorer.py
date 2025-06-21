"""
File: smartcash/dataset/services/explorer/class_explorer.py
Deskripsi: Explorer untuk analisis distribusi kelas dalam dataset
"""

import collections
from typing import Dict, Any

from smartcash.dataset.services.explorer.base_explorer import BaseExplorer


class ClassExplorer(BaseExplorer):
    """Explorer khusus untuk analisis distribusi kelas."""
    
    def analyze_distribution(self, split: str, sample_size: int = 0) -> Dict[str, Any]:
        """
        Analisis distribusi kelas dalam dataset.
        
        Args:
            split: Split dataset yang akan dianalisis
            sample_size: Jumlah sampel (0 = semua)
            
        Returns:
            Hasil analisis distribusi kelas
        """
        self.logger.info(f"📊 Analisis distribusi kelas untuk split {split}")
        split_path, images_dir, labels_dir, valid = self._validate_directories(split)
        
        if not valid:
            return {'status': 'error', 'message': f"Direktori tidak lengkap: {split_path}"}
        
        # Dapatkan file gambar valid
        image_files = self._get_valid_files(images_dir, labels_dir, sample_size)
        if not image_files:
            return {'status': 'error', 'message': f"Tidak ada gambar valid ditemukan"}
        
        # Hitung frekuensi kelas
        class_counts = collections.Counter()
        image_class_counts = collections.defaultdict(set)  # Kelas -> set img_stems
        
        for img_path in image_files:
            label_path = labels_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue
            
            # Parse label dan hitung kelas
            bbox_data = self.utils.parse_yolo_label(label_path)
            for box in bbox_data:
                cls_id = box['class_id']
                if 'class_name' in box:
                    class_name = box['class_name']
                else:
                    class_name = self.utils.get_class_name(cls_id)
                
                class_counts[class_name] += 1
                image_class_counts[class_name].add(img_path.stem)
        
        # Statistik distribusi
        total_objects = sum(class_counts.values())
        class_percentages = {}
        images_per_class = {}
        
        if total_objects > 0:
            for cls, count in class_counts.items():
                class_percentages[cls] = (count / total_objects) * 100
                
            for cls, img_stems in image_class_counts.items():
                images_per_class[cls] = len(img_stems)
            
            # Hitung ketidakseimbangan
            if len(class_counts) > 1:
                counts = list(class_counts.values())
                max_count, min_count = max(counts), min(counts)
                imbalance_score = min(10.0, (max_count / max(min_count, 1) - 1) / 2)
                
                # Identifikasi kelas-kelas (under/over-represented)
                avg_count = total_objects / len(class_counts)
                underrepresented = [cls for cls, count in class_counts.items() if count < avg_count * 0.5]
                overrepresented = [cls for cls, count in class_counts.items() if count > avg_count * 2]
            else:
                imbalance_score = 0.0
                underrepresented, overrepresented = [], []
        else:
            class_percentages = {}
            images_per_class = {}
            imbalance_score = 0.0
            underrepresented, overrepresented = [], []
        
        # Log hasil
        self.logger.info(
            f"📊 Distribusi kelas di {split}:\n"
            f"   • Total objek: {total_objects}\n"
            f"   • Jumlah kelas: {len(class_counts)}\n"
            f"   • Skor ketidakseimbangan: {imbalance_score:.2f}/10"
        )
        
        top_classes = sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_classes:
            self.logger.info("   • Top 5 kelas:")
            for cls, count in top_classes:
                percentage = class_percentages.get(cls, 0)
                self.logger.info(f"     - {cls}: {count} ({percentage:.1f}%)")
        
        return {
            'status': 'success',
            'total_objects': total_objects,
            'class_count': len(class_counts),
            'counts': dict(class_counts),
            'percentages': class_percentages,
            'images_per_class': images_per_class,
            'imbalance_score': imbalance_score,
            'underrepresented_classes': underrepresented,
            'overrepresented_classes': overrepresented
        }