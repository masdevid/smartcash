"""
File: smartcash/utils/polygon_metrics.py
Author: Alfrida Sabar
Deskripsi: Utilitas untuk menghitung metrik berbasis polygon seperti IoU (Intersection over Union) untuk evaluasi deteksi objek
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from shapely.geometry import Polygon, box


class PolygonMetrics:
    """Kelas untuk kalkulasi metrik evaluasi berbasis polygon."""
    
    def __init__(self, logger=None):
        """
        Inisialisasi PolygonMetrics.
        
        Args:
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger
    
    def calculate_iou(
        self, 
        box1: Union[List[float], np.ndarray], 
        box2: Union[List[float], np.ndarray]
    ) -> float:
        """
        Hitung IoU (Intersection over Union) antara dua bounding box.
        
        Args:
            box1: Bounding box pertama dalam format [x1, y1, x2, y2]
            box2: Bounding box kedua dalam format [x1, y1, x2, y2]
            
        Returns:
            Nilai IoU (0-1)
        """
        # Pastikan box dalam format yang benar
        if len(box1) != 4 or len(box2) != 4:
            if self.logger:
                self.logger.warning("⚠️ Format box tidak valid, harus [x1, y1, x2, y2]")
            return 0.0
        
        # Ekstrak koordinat
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Cek validitas box
        if x1_1 > x2_1 or y1_1 > y2_1 or x1_2 > x2_2 or y1_2 > y2_2:
            if self.logger:
                self.logger.warning("⚠️ Koordinat box tidak valid (x1 > x2 or y1 > y2)")
            return 0.0
        
        # Hitung koordinat intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        # Cek apakah ada intersection
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        # Hitung area intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Hitung area masing-masing box
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Hitung area union
        union_area = box1_area + box2_area - intersection_area
        
        # Hitung IoU
        iou = intersection_area / union_area
        
        return iou
    
    def calculate_poly_iou(
        self, 
        poly1: List[Tuple[float, float]], 
        poly2: List[Tuple[float, float]]
    ) -> float:
        """
        Hitung IoU (Intersection over Union) antara dua polygon.
        
        Args:
            poly1: Polygon pertama sebagai list koordinat [(x1, y1), (x2, y2), ...]
            poly2: Polygon kedua sebagai list koordinat [(x1, y1), (x2, y2), ...]
            
        Returns:
            Nilai IoU (0-1)
        """
        try:
            # Konversi ke objek Polygon
            polygon1 = Polygon(poly1)
            polygon2 = Polygon(poly2)
            
            # Cek validitas polygon
            if not polygon1.is_valid or not polygon2.is_valid:
                if self.logger:
                    self.logger.warning("⚠️ Polygon tidak valid")
                return 0.0
                
            # Hitung intersection
            intersection = polygon1.intersection(polygon2).area
            
            # Hitung union
            union = polygon1.union(polygon2).area
            
            # Handle division by zero
            if union == 0:
                return 0.0
                
            # Hitung IoU
            iou = intersection / union
            
            return iou
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"❌ Error menghitung Polygon IoU: {str(e)}")
            return 0.0
            
    def box_to_polygon(
        self, 
        box: List[float], 
        format: str = 'xyxy'
    ) -> List[Tuple[float, float]]:
        """
        Konversi bounding box ke format polygon.
        
        Args:
            box: Bounding box dalam format [x1, y1, x2, y2] (xyxy) atau [x, y, w, h] (xywh)
            format: Format box ('xyxy' atau 'xywh')
            
        Returns:
            List koordinat polygon [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        if format == 'xyxy':
            x1, y1, x2, y2 = box
            return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        elif format == 'xywh':
            x, y, w, h = box
            return [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
        else:
            if self.logger:
                self.logger.warning(f"⚠️ Format box tidak didukung: {format}")
            return []
            
    def calculate_mean_iou(
        self, 
        boxes_true: List[List[float]], 
        boxes_pred: List[List[float]], 
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Hitung mIoU (mean Intersection over Union) untuk evaluasi deteksi.
        
        Args:
            boxes_true: List box ground truth [x1, y1, x2, y2]
            boxes_pred: List box prediksi [x1, y1, x2, y2]
            threshold: Threshold IoU untuk true positive
            
        Returns:
            Dictionary berisi mIoU dan metrik terkait
        """
        if not boxes_true or not boxes_pred:
            return {
                'miou': 0.0,
                'tp': 0,
                'fp': len(boxes_pred),
                'fn': len(boxes_true)
            }
            
        # Hitung IoU untuk semua kombinasi box
        ious = np.zeros((len(boxes_true), len(boxes_pred)))
        for i, box_true in enumerate(boxes_true):
            for j, box_pred in enumerate(boxes_pred):
                ious[i, j] = self.calculate_iou(box_true, box_pred)
        
        # Temukan matching dengan IoU tertinggi
        matched_indices = []
        
        # Untuk setiap ground truth, temukan prediksi dengan IoU tertinggi
        for i in range(len(boxes_true)):
            if len(boxes_pred) == 0:
                break
                
            # Temukan indeks prediksi dengan IoU tertinggi
            j = np.argmax(ious[i])
            
            # Jika IoU diatas threshold, anggap sebagai match
            if ious[i, j] >= threshold:
                matched_indices.append((i, j))
                
                # Set IoU ke nol untuk mencegah matching berulang
                ious[i, :] = 0
                ious[:, j] = 0
        
        # Hitung true positives, false positives, false negatives
        tp = len(matched_indices)
        fp = len(boxes_pred) - tp
        fn = len(boxes_true) - tp
        
        # Hitung mean IoU dari matched pairs
        if tp > 0:
            total_iou = sum(self.calculate_iou(boxes_true[i], boxes_pred[j]) for i, j in matched_indices)
            miou = total_iou / tp
        else:
            miou = 0.0
            
        return {
            'miou': miou,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1)
        }