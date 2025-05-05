"""
File: smartcash/dataset/components/geometry/geometry_utils.py
Deskripsi: Utilitas geometri untuk operasi bounding box dan bentuk lainnya
"""

from typing import List, Tuple, Dict, Union, Any
import numpy as np


class GeometryUtils:
    """Kelas utilitas untuk operasi geometri pada deteksi objek."""
    
    @staticmethod
    def compute_area(bbox: List[float], format: str = 'yolo') -> float:
        """
        Hitung area bounding box.
        
        Args:
            bbox: Koordinat bbox
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Area bbox dalam unit persegi atau relatif [0-1]
        """
        if format == 'yolo':
            _, _, width, height = bbox
            return width * height
        
        elif format == 'coco':
            _, _, width, height = bbox
            return width * height
        
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            return (x_max - x_min) * (y_max - y_min)
        
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
    
    @staticmethod
    def compute_center(bbox: List[float], format: str = 'yolo') -> Tuple[float, float]:
        """
        Hitung titik tengah bbox.
        
        Args:
            bbox: Koordinat bbox
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Tuple (x_center, y_center)
        """
        if format == 'yolo':
            x_center, y_center, _, _ = bbox
            return (x_center, y_center)
        
        elif format == 'coco':
            x_min, y_min, width, height = bbox
            return (x_min + width / 2, y_min + height / 2)
        
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            return ((x_min + x_max) / 2, (y_min + y_max) / 2)
        
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
    
    @staticmethod
    def compute_aspect_ratio(bbox: List[float], format: str = 'yolo') -> float:
        """
        Hitung aspect ratio (width/height) bbox.
        
        Args:
            bbox: Koordinat bbox
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Aspect ratio bbox
        """
        if format == 'yolo':
            _, _, width, height = bbox
            return width / height if height > 0 else 0
        
        elif format == 'coco':
            _, _, width, height = bbox
            return width / height if height > 0 else 0
        
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            width = x_max - x_min
            height = y_max - y_min
            return width / height if height > 0 else 0
        
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
    
    @staticmethod
    def compute_distance(bbox1: List[float], bbox2: List[float], format: str = 'yolo') -> float:
        """
        Hitung jarak Euclidean antara pusat dua bbox.
        
        Args:
            bbox1: Koordinat bbox pertama
            bbox2: Koordinat bbox kedua
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Jarak Euclidean
        """
        center1 = GeometryUtils.compute_center(bbox1, format)
        center2 = GeometryUtils.compute_center(bbox2, format)
        
        # Hitung jarak Euclidean
        return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    
    @staticmethod
    def compute_intersection(bbox1: List[float], bbox2: List[float], format: str = 'yolo') -> float:
        """
        Hitung area intersection antara dua bbox.
        
        Args:
            bbox1: Koordinat bbox pertama
            bbox2: Koordinat bbox kedua
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Area intersection
        """
        from smartcash.dataset.components.geometry.coord_converter import CoordinateConverter
        
        # Konversi ke format corners relatif
        if format == 'yolo':
            corners1 = CoordinateConverter.yolo_to_relative_corners(bbox1)
            corners2 = CoordinateConverter.yolo_to_relative_corners(bbox2)
        elif format == 'coco':
            # Asumsikan dalam skala [0-1]
            x1, y1, w1, h1 = bbox1
            x2, y2, w2, h2 = bbox2
            corners1 = [x1, y1, x1 + w1, y1 + h1]
            corners2 = [x2, y2, x2 + w2, y2 + h2]
        elif format == 'corners':
            # Asumsikan dalam skala [0-1]
            corners1 = bbox1
            corners2 = bbox2
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
        return (x_max_overlap - x_min_overlap) * (y_max_overlap - y_min_overlap)
    
    @staticmethod
    def is_inside(point: Tuple[float, float], bbox: List[float], format: str = 'yolo') -> bool:
        """
        Cek apakah titik berada di dalam bbox.
        
        Args:
            point: Koordinat titik (x, y)
            bbox: Koordinat bbox
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            True jika titik di dalam bbox, False jika tidak
        """
        from smartcash.dataset.components.geometry.coord_converter import CoordinateConverter
        
        x, y = point
        
        if format == 'yolo':
            corners = CoordinateConverter.yolo_to_relative_corners(bbox)
        elif format == 'coco':
            # Asumsikan dalam skala [0-1]
            x_min, y_min, width, height = bbox
            corners = [x_min, y_min, x_min + width, y_min + height]
        elif format == 'corners':
            corners = bbox
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")
        
        x_min, y_min, x_max, y_max = corners
        
        return x_min <= x <= x_max and y_min <= y <= y_max
    
    @staticmethod
    def scale_bbox_around_center(bbox: List[float], scale_factor: float, format: str = 'yolo') -> List[float]:
        """
        Skala bbox di sekitar pusat dengan faktor tertentu.
        
        Args:
            bbox: Koordinat bbox
            scale_factor: Faktor skala
            format: Format bbox ('yolo', 'coco', 'corners')
            
        Returns:
            Koordinat bbox yang telah diskala
        """
        if format == 'yolo':
            x_center, y_center, width, height = bbox
            
            # Skala width dan height
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            return [x_center, y_center, new_width, new_height]
            
        elif format == 'coco':
            x_min, y_min, width, height = bbox
            center_x = x_min + width / 2
            center_y = y_min + height / 2
            
            # Skala width dan height
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            # Hitung new x_min, y_min
            new_x_min = center_x - new_width / 2
            new_y_min = center_y - new_height / 2
            
            return [new_x_min, new_y_min, new_width, new_height]
            
        elif format == 'corners':
            x_min, y_min, x_max, y_max = bbox
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            # Skala width dan height
            width = x_max - x_min
            height = y_max - y_min
            new_width = width * scale_factor
            new_height = height * scale_factor
            
            # Hitung new corners
            new_x_min = center_x - new_width / 2
            new_y_min = center_y - new_height / 2
            new_x_max = center_x + new_width / 2
            new_y_max = center_y + new_height / 2
            
            return [new_x_min, new_y_min, new_x_max, new_y_max]
            
        else:
            raise ValueError(f"Format '{format}' tidak dikenali.")