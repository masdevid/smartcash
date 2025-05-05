"""
File: smartcash/dataset/services/augmentor/bbox_augmentor.py
Deskripsi: Augmentor untuk bounding box dengan tracking class dan optimasi one-liner
"""

import cv2
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import albumentations as A

class BBoxAugmentor:
    """Kelas untuk augmentasi bounding box dengan tracking class ID."""
    
    def __init__(self, config: Dict = None, logger = None):
        """
        Inisialisasi BBoxAugmentor.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Logger untuk logging
        """
        self.config = config or {}
        self.logger = logger
        
    def transform_bboxes(
        self, 
        original_image: np.ndarray, 
        transformed_image: np.ndarray,
        original_bboxes: List[str],
        transforms: List[Dict[str, Any]] = None
    ) -> List[str]:
        """
        Transform bounding box untuk menyesuaikan dengan transformasi gambar.
        
        Args:
            original_image: Gambar asli
            transformed_image: Gambar hasil transformasi
            original_bboxes: List string bounding box format YOLO
            transforms: List transformasi yang diaplikasikan
            
        Returns:
            List string bounding box hasil transformasi
        """
        try:
            # Jika tidak ada bbox, return empty list
            if not original_bboxes: return []
            
            # Jika transformasi ke dimensi yang sama, gunakan bbox asli
            if not self._should_transform(original_image, transformed_image, transforms):
                return original_bboxes
            
            # Parse original bboxes dengan multi-class tracking
            orig_height, orig_width = original_image.shape[:2]
            parsed_bboxes = [self._parse_yolo_bbox(bbox_str, orig_width, orig_height) for bbox_str in original_bboxes]
            
            # Filter invalid bboxes dengan one-liner
            valid_bboxes = [bbox for bbox in parsed_bboxes if bbox is not None]
            if not valid_bboxes: return []
            
            # Extract components dengan one-liner untuk optimasi
            classes, bboxes_xyxy = zip(*[(bbox[0], bbox[1]) for bbox in valid_bboxes])
            
            # Transform bboxes berdasarkan aspect ratio
            dest_height, dest_width = transformed_image.shape[:2]
            transformed_bboxes = self._transform_bboxes_by_ratio(bboxes_xyxy, classes, orig_width, orig_height, dest_width, dest_height)
            
            # Format hasil ke YOLO format dengan tracking class
            return [f"{cls} {x1/dest_width:.6f} {y1/dest_height:.6f} {(x2-x1)/dest_width:.6f} {(y2-y1)/dest_height:.6f}" 
                   for cls, (x1, y1, x2, y2) in zip(classes, transformed_bboxes)]
        except Exception as e:
            if self.logger: self.logger.warning(f"⚠️ Error saat transform bbox: {str(e)}")
            return []
    
    def _should_transform(
        self, 
        original_image: np.ndarray, 
        transformed_image: np.ndarray,
        transforms: List[Dict[str, Any]] = None
    ) -> bool:
        """
        Cek apakah perlu melakukan transformasi bbox.
        
        Args:
            original_image: Gambar asli
            transformed_image: Gambar hasil transformasi
            transforms: List transformasi yang diaplikasikan
            
        Returns:
            Boolean apakah perlu transformasi
        """
        # Jika dimensi berubah, transform
        if original_image.shape[:2] != transformed_image.shape[:2]:
            return True
            
        # Jika ada transformasi yang memerlukan update bbox
        if transforms:
            for t in transforms:
                applied = t.get('applied', False)
                if applied and t.get('name') in ['ShiftScaleRotate', 'Perspective', 'Affine']:
                    return True
        return False
    
    def _parse_yolo_bbox(self, bbox_str: str, img_width: int, img_height: int) -> Optional[Tuple[str, Tuple[float, float, float, float]]]:
        """
        Parse string bounding box YOLO ke format absolut dengan tracking class ID.
        
        Args:
            bbox_str: String bounding box YOLO
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Tuple (class_id, (x1, y1, x2, y2))
        """
        try:
            # Parse YOLO format dengan class ID dan normalisasi
            parts = bbox_str.split()
            if len(parts) < 5: return None
            
            # Extract class dan koordinat dengan one-liner
            class_id, x_center, y_center, width, height = parts[0], float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            
            # Convert koordinat YOLO ke absolute dengan one-liner
            x1, y1 = (x_center - width/2) * img_width, (y_center - height/2) * img_height
            x2, y2 = (x_center + width/2) * img_width, (y_center + height/2) * img_height
            
            # Validasi koordinat dengan one-liner
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_width, x2), min(img_height, y2)
            
            return (class_id, (x1, y1, x2, y2))
        except (ValueError, IndexError):
            return None
            
    def _transform_bboxes_by_ratio(
        self, 
        bboxes_xyxy: List[Tuple[float, float, float, float]],
        classes: List[str],
        src_width: int, 
        src_height: int,
        dest_width: int,
        dest_height: int
    ) -> List[Tuple[float, float, float, float]]:
        """
        Transform bounding box berdasarkan rasio perubahan dimensi dengan tracking class ID.
        
        Args:
            bboxes_xyxy: List bbox format (x1, y1, x2, y2)
            classes: List class ID
            src_width: Lebar gambar asli
            src_height: Tinggi gambar asli
            dest_width: Lebar gambar tujuan
            dest_height: Tinggi gambar tujuan
            
        Returns:
            List bbox yang sudah ditransformasi dengan format (x1, y1, x2, y2)
        """
        # Kalkulasi rasio transformasi dengan one-liner
        width_ratio, height_ratio = dest_width / src_width, dest_height / src_height
        
        # Transform koordinat bbox dengan one-liner
        transformed = [(x1 * width_ratio, y1 * height_ratio, x2 * width_ratio, y2 * height_ratio) 
                     for x1, y1, x2, y2 in bboxes_xyxy]
        
        # Validasi koordinat bbox dalam rentang gambar tujuan dengan one-liner
        validated = [(max(0, x1), max(0, y1), min(dest_width, x2), min(dest_height, y2)) 
                   for x1, y1, x2, y2 in transformed]
        
        return validated