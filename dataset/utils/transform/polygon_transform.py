"""
File: smartcash/dataset/utils/transform/polygon_transform.py
Deskripsi: Transformasi dan konversi format koordinat polygon untuk segmentasi
"""

import numpy as np
import cv2
from typing import List, Tuple, Dict, Union, Any


class PolygonTransformer:
    """Utilitas untuk transformasi dan konversi format koordinat polygon."""
    
    @staticmethod
    def normalize_polygon(
        polygon: List[float], 
        img_width: int, 
        img_height: int
    ) -> List[float]:
        """
        Normalisasi koordinat polygon ke range [0, 1].
        
        Args:
            polygon: List koordinat polygon dalam format [x1, y1, x2, y2, ...]
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Polygon dinormalisasi
        """
        normalized = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = polygon[i] / img_width
                y = polygon[i + 1] / img_height
                normalized.extend([x, y])
        return normalized
    
    @staticmethod
    def         denormalize_polygon(
        polygon: List[float], 
        img_width: int, 
        img_height: int
    ) -> List[int]:
        """
        Denormalisasi koordinat polygon dari range [0, 1] ke piksel.
        
        Args:
            polygon: List koordinat polygon dalam format [x1, y1, x2, y2, ...] (nilai 0-1)
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Polygon dengan koordinat piksel
        """
        denormalized = []
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x = int(polygon[i] * img_width)
                y = int(polygon[i + 1] * img_height)
                denormalized.extend([x, y])
        return denormalized
    
    @staticmethod
    def polygon_to_binary_mask(
        polygon: List[float], 
        img_width: int, 
        img_height: int, 
        normalized: bool = True
    ) -> np.ndarray:
        """
        Konversi polygon ke binary mask.
        
        Args:
            polygon: List koordinat polygon dalam format [x1, y1, x2, y2, ...]
            img_width: Lebar gambar
            img_height: Tinggi gambar
            normalized: Apakah polygon dalam format ternormalisasi (0-1)
            
        Returns:
            Binary mask sebagai array numpy
        """
        # Denormalisasi polygon jika perlu
        points = []
        if normalized:
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    x = int(polygon[i] * img_width)
                    y = int(polygon[i + 1] * img_height)
                    points.append([x, y])
        else:
            for i in range(0, len(polygon), 2):
                if i + 1 < len(polygon):
                    points.append([int(polygon[i]), int(polygon[i + 1])])
                    
        # Buat mask kosong
        mask = np.zeros((img_height, img_width), dtype=np.uint8)
        
        # Gambar polygon ke mask
        points = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 1)
        
        return mask
    
    @staticmethod
    def binary_mask_to_polygon(
        mask: np.ndarray, 
        epsilon: float = 0.005, 
        normalize: bool = True,
        min_area: int = 5
    ) -> List[List[float]]:
        """
        Konversi binary mask ke polygon.
        
        Args:
            mask: Binary mask sebagai array numpy
            epsilon: Parameter untuk penyederhanaan kurva
            normalize: Apakah menormalisasi output (0-1)
            min_area: Area minimum contour (dalam piksel)
            
        Returns:
            List of polygons, masing-masing dalam format [x1, y1, x2, y2, ...]
        """
        # Pastikan mask adalah binary
        binary_mask = (mask > 0).astype(np.uint8)
        
        # Temukan contour
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contour berdasarkan area minimum
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= min_area]
        
        # Dapatkan dimensi gambar
        img_height, img_width = mask.shape
        
        polygons = []
        for contour in contours:
            # Sederhanakan contour dengan Ramer-Douglas-Peucker
            epsilon_value = epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon_value, True)
            
            # Konversi ke format [x1, y1, x2, y2, ...]
            flattened = []
            for point in approx:
                x, y = point[0]
                if normalize:
                    x = x / img_width
                    y = y / img_height
                flattened.extend([x, y])
                
            polygons.append(flattened)
            
        return polygons
    
    @staticmethod
    def find_bbox_from_polygon(
        polygon: List[float], 
        img_width: int = 1, 
        img_height: int = 1, 
        normalized: bool = True
    ) -> List[float]:
        """
        Temukan bounding box dari polygon.
        
        Args:
            polygon: List koordinat polygon dalam format [x1, y1, x2, y2, ...]
            img_width: Lebar gambar (untuk denormalisasi)
            img_height: Tinggi gambar (untuk denormalisasi)
            normalized: Apakah polygon dalam format ternormalisasi (0-1)
            
        Returns:
            Bounding box dalam format [x, y, width, height] (COCO)
        """
        # Extract x and y coordinates
        x_coords = []
        y_coords = []
        
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x, y = polygon[i], polygon[i + 1]
                
                # Denormalisasi jika perlu
                if normalized:
                    x = x * img_width
                    y = y * img_height
                    
                x_coords.append(x)
                y_coords.append(y)
        
        # Find the bounding box
        x_min = min(x_coords)
        y_min = min(y_coords)
        x_max = max(x_coords)
        y_max = max(y_coords)
        
        # Convert to [x, y, width, height] format
        width = x_max - x_min
        height = y_max - y_min
        
        # Normalisasi kembali jika input normalized
        if normalized:
            x_min /= img_width
            y_min /= img_height
            width /= img_width
            height /= img_height
            
        return [x_min, y_min, width, height]
    
    @staticmethod
    def simplify_polygon(
        polygon: List[float], 
        epsilon: float = 0.005, 
        img_width: int = 1, 
        img_height: int = 1,
        normalized: bool = True
    ) -> List[float]:
        """
        Sederhanakan polygon dengan algoritma Ramer-Douglas-Peucker.
        
        Args:
            polygon: List koordinat polygon dalam format [x1, y1, x2, y2, ...]
            epsilon: Parameter epsilon untuk penyederhanaan (relatif thd perimeter)
            img_width: Lebar gambar (untuk denormalisasi)
            img_height: Tinggi gambar (untuk denormalisasi)
            normalized: Apakah polygon dalam format ternormalisasi (0-1)
            
        Returns:
            Polygon yang disederhanakan
        """
        # Konversi ke format points array untuk OpenCV
        points = []
        
        for i in range(0, len(polygon), 2):
            if i + 1 < len(polygon):
                x, y = polygon[i], polygon[i + 1]
                
                # Denormalisasi jika perlu
                if normalized:
                    x = x * img_width
                    y = y * img_height
                    
                points.append([[int(x), int(y)]])
        
        # Konversi ke format numpy array
        contour = np.array(points, dtype=np.int32)
        
        # Hitung perimeter
        perimeter = cv2.arcLength(contour, True)
        
        # Sederhanakan dengan RDP algorithm
        epsilon_value = epsilon * perimeter
        approx = cv2.approxPolyDP(contour, epsilon_value, True)
        
        # Konversi kembali ke format [x1, y1, x2, y2, ...]
        simplified = []
        for point in approx:
            x, y = point[0]
            
            # Normalisasi kembali jika input normalized
            if normalized:
                x = x / img_width
                y = y / img_height
                
            simplified.extend([float(x), float(y)])
            
        return simplified