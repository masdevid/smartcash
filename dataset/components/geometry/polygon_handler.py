"""
File: smartcash/dataset/components/geometry/polygon_handler.py
Deskripsi: Utilitas untuk manipulasi dan konversi polygon ke format bbox
"""

from typing import List, Tuple, Dict, Union, Any
import numpy as np

from smartcash.dataset.components.geometry.coord_converter import CoordinateConverter


class PolygonHandler:
    """Kelas utilitas untuk penanganan polygon dan operasi konversi."""
    
    @staticmethod
    def polygon_to_bbox(polygon: List[Tuple[float, float]], format: str = 'yolo', 
                        img_width: int = None, img_height: int = None) -> List[float]:
        """
        Konversi polygon ke format bounding box.
        
        Args:
            polygon: List koordinat polygon [(x1, y1), (x2, y2), ...]
            format: Format output bbox ('yolo', 'coco', 'corners')
            img_width: Lebar gambar dalam piksel (diperlukan untuk format 'yolo')
            img_height: Tinggi gambar dalam piksel (diperlukan untuk format 'yolo')
            
        Returns:
            Koordinat bounding box dalam format yang ditentukan
        """
        if not polygon:
            raise ValueError("Polygon tidak boleh kosong")
        
        # Ekstrak koordinat x dan y
        x_coords = [p[0] for p in polygon]
        y_coords = [p[1] for p in polygon]
        
        # Hitung bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        if format == 'corners':
            return [x_min, y_min, x_max, y_max]
        
        elif format == 'coco':
            width = x_max - x_min
            height = y_max - y_min
            return [x_min, y_min, width, height]
        
        elif format == 'yolo':
            if img_width is None or img_height is None:
                raise ValueError("img_width dan img_height diperlukan untuk format YOLO")
            
            # Normalisasi ke [0, 1]
            x_min_rel = x_min / img_width
            y_min_rel = y_min / img_height
            x_max_rel = x_max / img_width
            y_max_rel = y_max / img_height
            
            # Konversi ke format YOLO
            return CoordinateConverter.relative_corners_to_yolo([x_min_rel, y_min_rel, x_max_rel, y_max_rel])
        
        else:
            raise ValueError(f"Format '{format}' tidak dikenali")
    
    @staticmethod
    def bbox_to_polygon(bbox: List[float], format: str = 'yolo', 
                        img_width: int = None, img_height: int = None, 
                        num_points: int = 4) -> List[Tuple[float, float]]:
        """
        Konversi bbox ke polygon dengan jumlah titik tertentu.
        
        Args:
            bbox: Koordinat bounding box
            format: Format bbox ('yolo', 'coco', 'corners')
            img_width: Lebar gambar dalam piksel (diperlukan untuk format 'yolo')
            img_height: Tinggi gambar dalam piksel (diperlukan untuk format 'yolo')
            num_points: Jumlah titik polygon (min 4)
            
        Returns:
            List koordinat polygon [(x1, y1), (x2, y2), ...]
        """
        if num_points < 4:
            raise ValueError("Jumlah titik harus minimal 4")
        
        # Konversi ke format corners
        if format == 'yolo':
            if img_width is None or img_height is None:
                raise ValueError("img_width dan img_height diperlukan untuk format YOLO")
            
            corners = CoordinateConverter.yolo_to_corners(bbox, img_width, img_height)
            
        elif format == 'coco':
            x_min, y_min, width, height = bbox
            corners = [x_min, y_min, x_min + width, y_min + height]
            
        elif format == 'corners':
            corners = bbox
            
        else:
            raise ValueError(f"Format '{format}' tidak dikenali")
        
        # Ekstrak corners
        x_min, y_min, x_max, y_max = corners
        
        if num_points == 4:
            # Return kotak sederhana
            return [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]
        
        # Jika jumlah titik > 4, buat polygon yang mengikuti bentuk kotak
        polygon = []
        
        # Distribusikan titik di setiap sisi kotak
        points_per_side = num_points // 4
        extra_points = num_points % 4
        
        # Atas
        for i in range(points_per_side + (1 if extra_points > 0 else 0)):
            t = i / (points_per_side + (1 if extra_points > 0 else 0))
            polygon.append((x_min + t * (x_max - x_min), y_min))
        
        # Kanan
        for i in range(points_per_side + (1 if extra_points > 1 else 0)):
            t = i / (points_per_side + (1 if extra_points > 1 else 0))
            polygon.append((x_max, y_min + t * (y_max - y_min)))
        
        # Bawah
        for i in range(points_per_side + (1 if extra_points > 2 else 0)):
            t = i / (points_per_side + (1 if extra_points > 2 else 0))
            polygon.append((x_max - t * (x_max - x_min), y_max))
        
        # Kiri
        for i in range(points_per_side):
            t = i / points_per_side
            polygon.append((x_min, y_max - t * (y_max - y_min)))
        
        return polygon
    
    @staticmethod
    def simplify_polygon(polygon: List[Tuple[float, float]], tolerance: float = 1.0) -> List[Tuple[float, float]]:
        """
        Sederhanakan polygon dengan mengurangi jumlah titik.
        
        Args:
            polygon: List koordinat polygon [(x1, y1), (x2, y2), ...]
            tolerance: Toleransi penyederhanaan (semakin besar, semakin sederhana)
            
        Returns:
            Polygon yang disederhanakan
        """
        try:
            # Gunakan shapely untuk Ramer-Douglas-Peucker algorithm
            from shapely.geometry import LineString
            
            if len(polygon) <= 4:
                return polygon
                
            line = LineString(polygon)
            simplified = line.simplify(tolerance, preserve_topology=True)
            
            # Pastikan polygon tertutup
            coords = list(simplified.coords)
            if coords[0] != coords[-1]:
                coords.append(coords[0])
                
            return coords
        except ImportError:
            # Fallback jika shapely tidak tersedia
            if len(polygon) <= 4:
                return polygon
            
            # Sederhana: ambil setiap n titik
            n = max(1, int(len(polygon) * tolerance / 10))
            simplified = [polygon[i] for i in range(0, len(polygon), n)]
            
            # Pastikan polygon tertutup
            if simplified[0] != simplified[-1]:
                simplified.append(simplified[0])
                
            return simplified
    
    @staticmethod
    def compute_polygon_area(polygon: List[Tuple[float, float]]) -> float:
        """
        Hitung luas polygon menggunakan rumus shoelace.
        
        Args:
            polygon: List koordinat polygon [(x1, y1), (x2, y2), ...]
            
        Returns:
            Luas polygon
        """
        if len(polygon) < 3:
            return 0.0
            
        # Hitung menggunakan rumus shoelace
        n = len(polygon)
        area = 0.0
        
        for i in range(n):
            j = (i + 1) % n
            area += polygon[i][0] * polygon[j][1]
            area -= polygon[j][0] * polygon[i][1]
            
        area = abs(area) / 2.0
        return area
    
    @staticmethod
    def is_point_inside_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """
        Periksa apakah titik berada di dalam polygon menggunakan ray casting algorithm.
        
        Args:
            point: Koordinat titik (x, y)
            polygon: List koordinat polygon [(x1, y1), (x2, y2), ...]
            
        Returns:
            True jika titik di dalam polygon, False jika tidak
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
            
        return inside
    
    @staticmethod
    def polygon_to_mask(polygon: List[Tuple[float, float]], shape: Tuple[int, int]) -> np.ndarray:
        """
        Konversi polygon ke mask biner.
        
        Args:
            polygon: List koordinat polygon [(x1, y1), (x2, y2), ...]
            shape: Bentuk (height, width) mask yang akan dibuat
            
        Returns:
            Mask biner dengan nilai 1 di dalam polygon dan 0 di luar
        """
        try:
            import cv2
            
            height, width = shape
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Konversi polygon ke format yang sesuai untuk cv2.fillPoly
            polygon_np = np.array([polygon], dtype=np.int32)
            
            # Isi polygon
            cv2.fillPoly(mask, polygon_np, 1)
            
            return mask
            
        except ImportError:
            # Fallback jika cv2 tidak tersedia (sangat lambat untuk gambar besar)
            height, width = shape
            mask = np.zeros((height, width), dtype=np.uint8)
            
            # Untuk setiap piksel, periksa apakah di dalam polygon
            for y in range(height):
                for x in range(width):
                    if PolygonHandler.is_point_inside_polygon((x, y), polygon):
                        mask[y, x] = 1
                        
            return mask