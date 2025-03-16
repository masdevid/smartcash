"""
File: smartcash/dataset/services/augmentor/bbox_augmentor.py
Deskripsi: Komponen untuk augmentasi khusus bounding box
"""

import random
import numpy as np
from typing import Dict, List, Tuple, Optional, Any

from smartcash.common.logger import get_logger


class BBoxAugmentor:
    """
    Komponen untuk augmentasi khusus bounding box.
    Bisa melakukan jitter, shift, dan resize pada bbox.
    """
    
    def __init__(self, config: Dict = None, logger=None):
        """
        Inisialisasi BBoxAugmentor.
        
        Args:
            config: Konfigurasi aplikasi (opsional)
            logger: Logger kustom (opsional)
        """
        self.config = config or {}
        self.logger = logger or get_logger("bbox_augmentor")
        
        # Setup parameter augmentasi
        aug_config = self.config.get('augmentation', {}).get('bbox', {})
        self.jitter_factor = aug_config.get('jitter_factor', 0.05)
        self.shift_factor = aug_config.get('shift_factor', 0.1)
        self.resize_factor = aug_config.get('resize_factor', 0.1)
        self.dropout_prob = aug_config.get('dropout_prob', 0.03)
        self.copy_prob = aug_config.get('copy_prob', 0.05)
        self.min_bbox_size = aug_config.get('min_bbox_size', 0.001)
        
        self.logger.info(f"📦 BBoxAugmentor diinisialisasi dengan parameter augmentasi custom")
    
    def augment_bboxes(
        self,
        bboxes: List[List[float]],
        class_ids: List[int],
        p: float = 0.5,
        jitter: bool = True,
        shift: bool = True,
        resize: bool = True,
        dropout: bool = False,
        copy: bool = False
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Augmentasi bounding box.
        
        Args:
            bboxes: List bbox dalam format YOLO [x_center, y_center, width, height]
            class_ids: List class ID yang bersesuaian dengan bbox
            p: Probabilitas augmentasi untuk setiap bbox
            jitter: Apakah melakukan jitter (pergeseran kecil acak)
            shift: Apakah melakukan shift (pergeseran lebih besar)
            resize: Apakah melakukan resize (perubahan ukuran)
            dropout: Apakah menerapkan dropout (menghapus bbox secara acak)
            copy: Apakah menerapkan copy (menggandakan bbox dengan sedikit offset)
            
        Returns:
            Tuple (bbox yang diaugmentasi, class ID yang bersesuaian)
        """
        if not bboxes or random.random() > p:
            return bboxes, class_ids
            
        # Copy input agar tidak mempengaruhi aslinya
        aug_bboxes = [bbox.copy() for bbox in bboxes]
        aug_class_ids = class_ids.copy()
        
        # Dropout
        if dropout and random.random() < self.dropout_prob and len(aug_bboxes) > 1:
            idx_to_drop = random.randint(0, len(aug_bboxes) - 1)
            del aug_bboxes[idx_to_drop]
            del aug_class_ids[idx_to_drop]
            
        # Copy
        if copy and random.random() < self.copy_prob and len(aug_bboxes) < 20:  # Batasi jumlah bbox
            if aug_bboxes:
                idx_to_copy = random.randint(0, len(aug_bboxes) - 1)
                new_bbox = aug_bboxes[idx_to_copy].copy()
                
                # Tambahkan sedikit offset
                offset_x = random.uniform(-0.05, 0.05)
                offset_y = random.uniform(-0.05, 0.05)
                
                new_bbox[0] = max(0, min(1, new_bbox[0] + offset_x))
                new_bbox[1] = max(0, min(1, new_bbox[1] + offset_y))
                
                aug_bboxes.append(new_bbox)
                aug_class_ids.append(aug_class_ids[idx_to_copy])
        
        # Proses setiap bbox
        for i, bbox in enumerate(aug_bboxes):
            x_center, y_center, width, height = bbox
            
            # Jitter: pergeseran kecil acak
            if jitter and random.random() < p:
                x_jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * width
                y_jitter = random.uniform(-self.jitter_factor, self.jitter_factor) * height
                
                bbox[0] = max(0, min(1, x_center + x_jitter))
                bbox[1] = max(0, min(1, y_center + y_jitter))
            
            # Shift: pergeseran lebih besar
            if shift and random.random() < p:
                x_shift = random.uniform(-self.shift_factor, self.shift_factor)
                y_shift = random.uniform(-self.shift_factor, self.shift_factor)
                
                bbox[0] = max(0, min(1, x_center + x_shift))
                bbox[1] = max(0, min(1, y_center + y_shift))
            
            # Resize: perubahan ukuran
            if resize and random.random() < p:
                w_scale = random.uniform(1 - self.resize_factor, 1 + self.resize_factor)
                h_scale = random.uniform(1 - self.resize_factor, 1 + self.resize_factor)
                
                new_width = max(self.min_bbox_size, min(1, width * w_scale))
                new_height = max(self.min_bbox_size, min(1, height * h_scale))
                
                # Pastikan bbox tetap dalam gambar
                x_min = max(0, bbox[0] - new_width / 2)
                y_min = max(0, bbox[1] - new_height / 2)
                x_max = min(1, bbox[0] + new_width / 2)
                y_max = min(1, bbox[1] + new_height / 2)
                
                # Hitung ulang center dan ukuran
                bbox[0] = (x_min + x_max) / 2
                bbox[1] = (y_min + y_max) / 2
                bbox[2] = x_max - x_min
                bbox[3] = y_max - y_min
            
        return aug_bboxes, aug_class_ids
    
    def mixup_bboxes(
        self,
        bboxes1: List[List[float]],
        class_ids1: List[int],
        bboxes2: List[List[float]],
        class_ids2: List[int],
        alpha: float = 0.5
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Gabungkan dua set bbox dengan strategi mixup.
        
        Args:
            bboxes1: Set bbox pertama
            class_ids1: Class ID untuk set pertama
            bboxes2: Set bbox kedua
            class_ids2: Class ID untuk set kedua
            alpha: Parameter mixup (0.5 = equal mix)
            
        Returns:
            Tuple (bbox gabungan, class ID gabungan)
        """
        # Salin dulu
        combined_bboxes = bboxes1.copy()
        combined_class_ids = class_ids1.copy()
        
        # Filter yang terlalu overlap
        for i, bbox2 in enumerate(bboxes2):
            should_add = True
            
            for bbox1 in bboxes1:
                iou = self._calculate_iou(bbox1, bbox2)
                if iou > 0.7:  # Threshold overlap
                    should_add = False
                    break
            
            if should_add:
                combined_bboxes.append(bbox2)
                combined_class_ids.append(class_ids2[i])
        
        return combined_bboxes, combined_class_ids
    
    def mosaic_bboxes(
        self,
        all_bboxes: List[List[List[float]]],
        all_class_ids: List[List[int]],
        grid_size: Tuple[int, int] = (2, 2)
    ) -> Tuple[List[List[float]], List[int]]:
        """
        Gabungkan multiple set bbox dengan strategi mosaic (4-gambar grid).
        
        Args:
            all_bboxes: List set bbox (satu per gambar)
            all_class_ids: List set class ID (satu per gambar)
            grid_size: Ukuran grid (default 2x2)
            
        Returns:
            Tuple (bbox gabungan, class ID gabungan)
        """
        rows, cols = grid_size
        combined_bboxes = []
        combined_class_ids = []
        
        # Jumlah gambar tidak boleh lebih dari grid cells
        n_samples = min(len(all_bboxes), rows * cols)
        
        for idx in range(n_samples):
            row = idx // cols
            col = idx % cols
            
            # Batas sel dalam koordinat normalized
            x_start = col / cols
            y_start = row / rows
            x_end = (col + 1) / cols
            y_end = (row + 1) / rows
            
            cell_width = x_end - x_start
            cell_height = y_end - y_start
            
            # Transformasi bbox ke koordinat baru
            for i, bbox in enumerate(all_bboxes[idx]):
                x_center, y_center, width, height = bbox
                
                # Konversi ke koordinat sel
                new_x_center = x_start + x_center * cell_width
                new_y_center = y_start + y_center * cell_height
                new_width = width * cell_width
                new_height = height * cell_height
                
                # Tambahkan ke hasil
                combined_bboxes.append([new_x_center, new_y_center, new_width, new_height])
                combined_class_ids.append(all_class_ids[idx][i])
        
        return combined_bboxes, combined_class_ids
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Hitung IoU (Intersection over Union) antara dua bbox.
        
        Args:
            bbox1: Bbox pertama dalam format YOLO [x_center, y_center, width, height]
            bbox2: Bbox kedua dalam format YOLO
            
        Returns:
            Nilai IoU
        """
        # Konversi dari format YOLO ke koordinat min, max
        x1_min = bbox1[0] - bbox1[2] / 2
        y1_min = bbox1[1] - bbox1[3] / 2
        x1_max = bbox1[0] + bbox1[2] / 2
        y1_max = bbox1[1] + bbox1[3] / 2
        
        x2_min = bbox2[0] - bbox2[2] / 2
        y2_min = bbox2[1] - bbox2[3] / 2
        x2_max = bbox2[0] + bbox2[2] / 2
        y2_max = bbox2[1] + bbox2[3] / 2
        
        # Hitung area intersection
        x_left = max(x1_min, x2_min)
        y_top = max(y1_min, y2_min)
        x_right = min(x1_max, x2_max)
        y_bottom = min(y1_max, y2_max)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Hitung area masing-masing bbox
        bbox1_area = bbox1[2] * bbox1[3]
        bbox2_area = bbox2[2] * bbox2[3]
        
        # Hitung union
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Hitung IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        
        return iou