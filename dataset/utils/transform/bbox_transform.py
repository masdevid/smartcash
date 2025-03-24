"""
File: smartcash/dataset/utils/transform/bbox_transform.py
Deskripsi: Transformasi dan konversi format koordinat bounding box
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Any


class BBoxTransformer:
    """Utilitas untuk transformasi dan konversi format bounding box."""
    
    @staticmethod
    def yolo_to_xyxy(bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """
        Konversi format YOLO (centerX, centerY, width, height) ke format XYXY (x1, y1, x2, y2).
        
        Args:
            bbox: Bounding box dalam format YOLO [x_center, y_center, width, height] (nilai 0-1)
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Bounding box dalam format XYXY [x1, y1, x2, y2] (nilai piksel)
        """
        x_center, y_center, width, height = bbox
        
        # Konversi relatif ke absolut
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Konversi ke XYXY
        x1 = int(max(0, x_center - width / 2))
        y1 = int(max(0, y_center - height / 2))
        x2 = int(min(img_width, x_center + width / 2))
        y2 = int(min(img_height, y_center + height / 2))
        
        return [x1, y1, x2, y2]
    
    @staticmethod
    def xyxy_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        Konversi format XYXY (x1, y1, x2, y2) ke format YOLO (centerX, centerY, width, height).
        
        Args:
            bbox: Bounding box dalam format XYXY [x1, y1, x2, y2] (nilai piksel)
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Bounding box dalam format YOLO [x_center, y_center, width, height] (nilai 0-1)
        """
        x1, y1, x2, y2 = bbox
        
        # Konversi ke center, width, height
        width = x2 - x1
        height = y2 - y1
        x_center = x1 + width / 2
        y_center = y1 + height / 2
        
        # Konversi absolut ke relatif
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def yolo_to_coco(bbox: List[float], img_width: int, img_height: int) -> List[int]:
        """
        Konversi format YOLO (centerX, centerY, width, height) ke format COCO (x, y, width, height).
        
        Args:
            bbox: Bounding box dalam format YOLO [x_center, y_center, width, height] (nilai 0-1)
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Bounding box dalam format COCO [x, y, width, height] (nilai piksel)
        """
        x_center, y_center, width, height = bbox
        
        # Konversi relatif ke absolut
        x_center *= img_width
        y_center *= img_height
        width *= img_width
        height *= img_height
        
        # Konversi ke COCO
        x = int(max(0, x_center - width / 2))
        y = int(max(0, y_center - height / 2))
        
        return [x, y, int(width), int(height)]
    
    @staticmethod
    def coco_to_yolo(bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        Konversi format COCO (x, y, width, height) ke format YOLO (centerX, centerY, width, height).
        
        Args:
            bbox: Bounding box dalam format COCO [x, y, width, height] (nilai piksel)
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Bounding box dalam format YOLO [x_center, y_center, width, height] (nilai 0-1)
        """
        x, y, width, height = bbox
        
        # Konversi ke center, width, height
        x_center = x + width / 2
        y_center = y + height / 2
        
        # Konversi absolut ke relatif
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        return [x_center, y_center, width, height]
    
    @staticmethod
    def clip_bbox(bbox: List[float], format: str = 'yolo') -> List[float]:
        """
        Clip bounding box agar tetap dalam range yang valid.
        
        Args:
            bbox: Bounding box
            format: Format bbox ('yolo', 'coco', 'xyxy')
            
        Returns:
            Bounding box yang di-clip
        """
        if format == 'yolo':
            # Format YOLO: [x_center, y_center, width, height] (nilai 0-1)
            x_center, y_center, width, height = bbox
            
            # Clip koordinat pusat dan dimensi agar tetap dalam range [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            
            # Pastikan width dan height positif dan tidak melebihi batas
            width = max(0.001, min(0.999, width))
            height = max(0.001, min(0.999, height))
            
            # Pastikan bbox dalam frame
            if x_center - width / 2 < 0:
                width = 2 * x_center
            if x_center + width / 2 > 1:
                width = 2 * (1 - x_center)
            if y_center - height / 2 < 0:
                height = 2 * y_center
            if y_center + height / 2 > 1:
                height = 2 * (1 - y_center)
                
            return [x_center, y_center, width, height]
            
        elif format == 'coco' or format == 'xyxy':
            # Implementasikan jika diperlukan
            # Untuk COCO dan XYXY biasanya perlu info img_width dan img_height
            return bbox
            
        else:
            raise ValueError(f"Format bbox tidak dikenal: {format}")
    
    @staticmethod
    def scale_bbox(bbox: List[float], scale_factor: float, format: str = 'yolo') -> List[float]:
        """
        Skala ukuran bounding box.
        
        Args:
            bbox: Bounding box
            scale_factor: Faktor skala
            format: Format bbox ('yolo', 'coco', 'xyxy')
            
        Returns:
            Bounding box yang di-skala
        """
        if format == 'yolo':
            # Format YOLO: [x_center, y_center, width, height]
            x_center, y_center, width, height = bbox
            
            # Skala width dan height
            width *= scale_factor
            height *= scale_factor
            
            # Clip kembali ke range yang valid
            return BBoxTransformer.clip_bbox([x_center, y_center, width, height], format)
            
        elif format == 'coco':
            # Format COCO: [x, y, width, height]
            x, y, width, height = bbox
            
            # Hitung center
            x_center = x + width / 2
            y_center = y + height / 2
            
            # Skala width dan height
            width *= scale_factor
            height *= scale_factor
            
            # Hitung koordinat baru
            x = x_center - width / 2
            y = y_center - height / 2
            
            return [x, y, width, height]
            
        elif format == 'xyxy':
            # Format XYXY: [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            
            # Hitung center
            x_center = (x1 + x2) / 2
            y_center = (y1 + y2) / 2
            
            # Hitung width dan height
            width = x2 - x1
            height = y2 - y1
            
            # Skala width dan height
            width *= scale_factor
            height *= scale_factor
            
            # Hitung koordinat baru
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            return [x1, y1, x2, y2]
            
        else:
            raise ValueError(f"Format bbox tidak dikenal: {format}")
    
    @staticmethod
    def iou(boxA: List[float], boxB: List[float], format: str = 'xyxy') -> float:
        """
        Hitung Intersection over Union (IoU) antara dua bounding box.
        
        Args:
            boxA: Bounding box pertama
            boxB: Bounding box kedua
            format: Format bbox ('yolo', 'coco', 'xyxy')
            
        Returns:
            Nilai IoU (0-1)
        """
        if format == 'yolo' or format == 'coco':
            # Implementasikan konversi jika diperlukan
            # Untuk contoh ini, berasumsi kedua bbox sudah dalam format XYXY
            pass
        
        # Implementasi IoU untuk format XYXY
        x1_A, y1_A, x2_A, y2_A = boxA
        x1_B, y1_B, x2_B, y2_B = boxB
        
        # Hitung koordinat intersection
        x1_I = max(x1_A, x1_B)
        y1_I = max(y1_A, y1_B)
        x2_I = min(x2_A, x2_B)
        y2_I = min(y2_A, y2_B)
        
        # Hitung area intersection
        if x2_I < x1_I or y2_I < y1_I:
            return 0.0  # Tidak ada intersection
            
        intersection_area = (x2_I - x1_I) * (y2_I - y1_I)
        
        # Hitung area masing-masing bbox
        boxA_area = (x2_A - x1_A) * (y2_A - y1_A)
        boxB_area = (x2_B - x1_B) * (y2_B - y1_B)
        
        # Hitung IoU
        union_area = boxA_area + boxB_area - intersection_area
        
        if union_area <= 0:
            return 0.0
            
        return intersection_area / union_area