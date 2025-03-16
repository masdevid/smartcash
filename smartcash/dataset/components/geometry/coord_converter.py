"""
File: smartcash/dataset/components/geometry/coord_converter.py
Deskripsi: Utilitas untuk konversi koordinat bounding box antar format yang berbeda
"""

from typing import List, Tuple, Dict, Union, Any
import numpy as np


class CoordinateConverter:
    """
    Kelas utilitas untuk konversi koordinat bounding box antar format yang berbeda.
    
    Format yang didukung:
    - YOLO: [x_center, y_center, width, height] dalam skala relatif [0-1]
    - COCO/Pascal VOC: [x_min, y_min, width, height] dalam piksel
    - Corners: [x_min, y_min, x_max, y_max] dalam piksel
    - Relative Corners: [x_min, y_min, x_max, y_max] dalam skala relatif [0-1]
    """
    
    @staticmethod
    def yolo_to_corners(bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """
        Konversi dari format YOLO ke format corners dalam piksel.
        
        Args:
            bbox: Koordinat dalam format YOLO [x_center, y_center, width, height]
            img_width: Lebar gambar dalam piksel
            img_height: Tinggi gambar dalam piksel
            
        Returns:
            Koordinat dalam format corners [x_min, y_min, x_max, y_max]
        """
        x_center, y_center, width, height = bbox
        
        # Konversi ke piksel
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Hitung corners
        x_min = int(max(0, x_center - width / 2))
        y_min = int(max(0, y_center - height / 2))
        x_max = int(min(img_width, x_center + width / 2))
        y_max = int(min(img_height, y_center + height / 2))
        
        return [x_min, y_min, x_max, y_max]
    
    @staticmethod
    def corners_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        Konversi dari format corners dalam piksel ke format YOLO.
        
        Args:
            bbox: Koordinat dalam format corners [x_min, y_min, x_max, y_max]
            img_width: Lebar gambar dalam piksel
            img_height: Tinggi gambar dalam piksel
            
        Returns:
            Koordinat dalam format YOLO [x_center, y_center, width, height]
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Hitung width dan height
        width = x_max - x_min
        height = y_max - y_min
        
        # Hitung center
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        
        # Normalisasi ke skala relatif [0-1]
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def coco_to_yolo(bbox: List[float], img_width: int, img_height: int) -> List[float]:
        """
        Konversi dari format COCO [x_min, y_min, width, height] ke format YOLO.
        
        Args:
            bbox: Koordinat dalam format COCO [x_min, y_min, width, height]
            img_width: Lebar gambar dalam piksel
            img_height: Tinggi gambar dalam piksel
            
        Returns:
            Koordinat dalam format YOLO [x_center, y_center, width, height]
        """
        x_min, y_min, width, height = bbox
        
        # Hitung center
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        
        # Normalisasi ke skala relatif [0-1]
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def yolo_to_coco(bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """
        Konversi dari format YOLO ke format COCO.
        
        Args:
            bbox: Koordinat dalam format YOLO [x_center, y_center, width, height]
            img_width: Lebar gambar dalam piksel
            img_height: Tinggi gambar dalam piksel
            
        Returns:
            Koordinat dalam format COCO [x_min, y_min, width, height]
        """
        x_center, y_center, width, height = bbox
        
        # Konversi ke piksel
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Hitung corners
        x_min = int(max(0, x_center - width / 2))
        y_min = int(max(0, y_center - height / 2))
        
        # Width dan height dalam piksel
        width = int(width)
        height = int(height)
        
        return [x_min, y_min, width, height]
    
    @staticmethod
    def relative_corners_to_yolo(bbox: List[float]) -> List[float]:
        """
        Konversi dari format corners dalam skala relatif ke format YOLO.
        
        Args:
            bbox: Koordinat dalam format corners relatif [x_min, y_min, x_max, y_max]
            
        Returns:
            Koordinat dalam format YOLO [x_center, y_center, width, height]
        """
        x_min, y_min, x_max, y_max = bbox
        
        # Hitung width dan height relatif
        width = x_max - x_min
        height = y_max - y_min
        
        # Hitung center relatif
        x_center = x_min + width / 2
        y_center = y_min + height / 2
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def yolo_to_relative_corners(bbox: List[float]) -> List[float]:
        """
        Konversi dari format YOLO ke format corners dalam skala relatif.
        
        Args:
            bbox: Koordinat dalam format YOLO [x_center, y_center, width, height]
            
        Returns:
            Koordinat dalam format corners relatif [x_min, y_min, x_max, y_max]
        """
        x_center, y_center, width, height = bbox
        
        # Hitung corners relatif
        x_min = max(0.0, x_center - width / 2)
        y_min = max(0.0, y_center - height / 2)
        x_max = min(1.0, x_center + width / 2)
        y_max = min(1.0, y_center + height / 2)
        
        return [x_min, y_min, x_max, y_max]
    
    @staticmethod
    def clip_bbox(bbox: List[float], img_width: int, img_height: int, format: str = 'yolo') -> List[float]:
        """
        Clip bounding box agar tidak keluar dari gambar.
        
        Args:
            bbox: Koordinat bounding box
            img_width: Lebar gambar dalam piksel
            img_height: Tinggi gambar dalam piksel
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Koordinat bbox yang telah di-clip
        """
        if format == 'yolo':
            # Konversi ke corners
            corners = CoordinateConverter.yolo_to_corners(bbox, img_width, img_height)
            
            # Clip
            x_min = max(0, corners[0])
            y_min = max(0, corners[1])
            x_max = min(img_width, corners[2])
            y_max = min(img_height, corners[3])
            
            # Konversi kembali ke YOLO
            return CoordinateConverter.corners_to_yolo([x_min, y_min, x_max, y_max], img_width, img_height)
            
        elif format == 'coco':
            x_min, y_min, width, height = bbox
            
            # Clip
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            width = min(img_width - x_min, width)
            height = min(img_height - y_min, height)
            
            return [x_min, y_min, width, height]
            
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            
            # Clip
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(img_width, x_max)
            y_max = min(img_height, y_max)
            
            return [x_min, y_min, x_max, y_max]
            
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
    
    @staticmethod
    def scale_bbox(bbox: List[float], scale_x: float, scale_y: float, format: str = 'yolo') -> List[float]:
        """
        Skala bounding box.
        
        Args:
            bbox: Koordinat bounding box
            scale_x: Faktor skala untuk sumbu x
            scale_y: Faktor skala untuk sumbu y
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Koordinat bbox yang telah di-skala
        """
        if format == 'yolo':
            x_center, y_center, width, height = bbox
            
            # Skala width dan height
            width *= scale_x
            height *= scale_y
            
            return [x_center, y_center, width, height]
            
        elif format == 'coco':
            x_min, y_min, width, height = bbox
            
            # Skala
            x_min *= scale_x
            y_min *= scale_y
            width *= scale_x
            height *= scale_y
            
            return [x_min, y_min, width, height]
            
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            
            # Skala
            x_min *= scale_x
            y_min *= scale_y
            x_max *= scale_x
            y_max *= scale_y
            
            return [x_min, y_min, x_max, y_max]
            
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
    
    @staticmethod
    def compute_iou(box1: List[float], box2: List[float], format: str = 'yolo') -> float:
        """
        Hitung Intersection over Union (IoU) antara dua bounding box.
        
        Args:
            box1: Koordinat bbox pertama
            box2: Koordinat bbox kedua
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Nilai IoU
        """
        # Konversi ke format corners relatif
        if format == 'yolo':
            corners1 = CoordinateConverter.yolo_to_relative_corners(box1)
            corners2 = CoordinateConverter.yolo_to_relative_corners(box2)
        elif format == 'coco':
            # Asumsikan dalam skala [0-1]
            x1, y1, w1, h1 = box1
            x2, y2, w2, h2 = box2
            corners1 = [x1, y1, x1 + w1, y1 + h1]
            corners2 = [x2, y2, x2 + w2, y2 + h2]
        elif format == 'corners':
            # Asumsikan dalam skala [0-1]
            corners1 = box1
            corners2 = box2
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
        
        # Hitung area overlap
        x_min_overlap = max(corners1[0], corners2[0])
        y_min_overlap = max(corners1[1], corners2[1])
        x_max_overlap = min(corners1[2], corners2[2])
        y_max_overlap = min(corners1[3], corners2[3])
        
        # Periksa apakah ada overlap
        if x_max_overlap <= x_min_overlap or y_max_overlap <= y_min_overlap:
            return 0.0
        
        # Hitung area overlap
        overlap_area = (x_max_overlap - x_min_overlap) * (y_max_overlap - y_min_overlap)
        
        # Hitung area masing-masing box
        box1_area = (corners1[2] - corners1[0]) * (corners1[3] - corners1[1])
        box2_area = (corners2[2] - corners2[0]) * (corners2[3] - corners2[1])
        
        # Hitung IoU
        iou = overlap_area / (box1_area + box2_area - overlap_area)
        
        return iou