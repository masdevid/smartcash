"""
File: smartcash/detection/services/postprocessing/bbox_refiner.py
Deskripsi: Perbaikan bounding box pada hasil deteksi.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any
import time

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection


class BBoxRefiner:
    """Perbaikan bounding box pada hasil deteksi"""
    
    def __init__(self, 
                clip_boxes: bool = True,
                expand_factor: float = 0.0,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi BBox Refiner
        
        Args:
            clip_boxes: Flag untuk memastikan bbox tetap dalam batasan gambar (0-1)
            expand_factor: Faktor ekspansi untuk bbox (0.0 = tanpa ekspansi)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.clip_boxes = clip_boxes
        self.expand_factor = expand_factor
        self.logger = logger or get_logger("BBoxRefiner")
    
    def process(self, 
               detections: List[Detection],
               image_width: Optional[int] = None,
               image_height: Optional[int] = None,
               specific_classes: Optional[List[int]] = None) -> List[Detection]:
        """
        Perbaiki bounding box pada list deteksi
        
        Args:
            detections: List hasil deteksi untuk diperbaiki
            image_width: Lebar gambar untuk perbaikan bbox absolut (opsional)
            image_height: Tinggi gambar untuk perbaikan bbox absolut (opsional)
            specific_classes: List kelas yang akan diperbaiki (None=semua)
            
        Returns:
            List deteksi dengan bbox yang diperbaiki
        """
        if not detections:
            return []
        
        start_time = time.time()
        
        # Salin deteksi agar tidak mengubah input asli
        refined_detections = []
        
        for detection in detections:
            # Skip jika kelas tidak dalam specific_classes (jika ditentukan)
            if specific_classes is not None and detection.class_id not in specific_classes:
                refined_detections.append(detection)
                continue
                
            # Salin deteksi
            refined = Detection(
                bbox=detection.bbox.copy(),
                confidence=detection.confidence,
                class_id=detection.class_id,
                class_name=detection.class_name
            )
            
            # Expand bbox jika expand_factor > 0
            if self.expand_factor > 0:
                refined.bbox = self._expand_bbox(refined.bbox, self.expand_factor)
            
            # Clip bbox jika diperlukan
            if self.clip_boxes:
                refined.bbox = self._clip_bbox(refined.bbox)
            
            # Jika dimensi gambar disediakan, perbaiki bbox absolut
            if image_width is not None and image_height is not None:
                refined.bbox = self._fix_absolute_bbox(refined.bbox, image_width, image_height)
            
            refined_detections.append(refined)
        
        refine_time = time.time() - start_time
        self.logger.debug(f"✓ BBox Refiner: {len(detections)} bbox diperbaiki dalam {refine_time:.4f}s")
        
        return refined_detections
    
    def _expand_bbox(self, bbox: np.ndarray, factor: float) -> np.ndarray:
        """
        Ekspansi bounding box dengan faktor tertentu
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            factor: Faktor ekspansi
            
        Returns:
            Bounding box yang sudah diekspansi
        """
        x1, y1, x2, y2 = bbox
        
        # Hitung lebar dan tinggi
        width = x2 - x1
        height = y2 - y1
        
        # Hitung ekspansi
        x_expand = width * factor / 2
        y_expand = height * factor / 2
        
        # Terapkan ekspansi
        x1 = x1 - x_expand
        y1 = y1 - y_expand
        x2 = x2 + x_expand
        y2 = y2 + y_expand
        
        return np.array([x1, y1, x2, y2])
    
    def _clip_bbox(self, bbox: np.ndarray) -> np.ndarray:
        """
        Clip bounding box ke range [0, 1]
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Bounding box yang sudah di-clip
        """
        return np.clip(bbox, 0.0, 1.0)
    
    def _fix_absolute_bbox(self, bbox: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
        """
        Perbaiki bounding box absolut (dalam piksel)
        
        Args:
            bbox: Bounding box [x1, y1, x2, y2]
            img_width: Lebar gambar
            img_height: Tinggi gambar
            
        Returns:
            Bounding box yang sudah diperbaiki
        """
        x1, y1, x2, y2 = bbox
        
        # Convert ke koordinat absolut
        x1_abs = x1 * img_width
        y1_abs = y1 * img_height
        x2_abs = x2 * img_width
        y2_abs = y2 * img_height
        
        # Clip ke batasan gambar
        x1_abs = max(0, min(img_width, x1_abs))
        y1_abs = max(0, min(img_height, y1_abs))
        x2_abs = max(0, min(img_width, x2_abs))
        y2_abs = max(0, min(img_height, y2_abs))
        
        # Convert kembali ke koordinat relatif
        x1_rel = x1_abs / img_width
        y1_rel = y1_abs / img_height
        x2_rel = x2_abs / img_width
        y2_rel = y2_abs / img_height
        
        return np.array([x1_rel, y1_rel, x2_rel, y2_rel])