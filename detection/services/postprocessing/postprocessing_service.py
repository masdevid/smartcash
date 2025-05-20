"""
File: smartcash/detection/services/postprocessing/postprocessing_service.py
Description: Layanan untuk postprocessing hasil deteksi yang menggunakan komponen dari domain model.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection

# Import NMSProcessor dari domain model
from smartcash.model.services.postprocessing.nms_processor import NMSProcessor as ModelNMSProcessor
from smartcash.detection.services.postprocessing.confidence_filter import ConfidenceFilter
from smartcash.detection.services.postprocessing.bbox_refiner import BBoxRefiner


class PostprocessingService:
    """Layanan untuk postprocessing hasil deteksi"""
    
    def __init__(self, 
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Postprocessing Service
        
        Args:
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.logger = logger or get_logger()
        
        # Gunakan NMSProcessor dari domain model
        self.nms_processor = ModelNMSProcessor(logger=self.logger)
        self.confidence_filter = ConfidenceFilter(logger=self.logger)
        self.bbox_refiner = BBoxRefiner(logger=self.logger)
    
    def process(self,
               detections: List[Detection],
               conf_threshold: float = 0.25,
               iou_threshold: float = 0.45,
               refine_boxes: bool = True,
               class_specific_nms: bool = True,
               max_detections: Optional[int] = None) -> List[Detection]:
        """
        Terapkan postprocessing pada hasil deteksi
        
        Args:
            detections: List hasil deteksi
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            refine_boxes: Flag untuk memperbaiki bounding box
            class_specific_nms: Flag untuk NMS per kelas
            max_detections: Jumlah maksimum deteksi yang dipertahankan
            
        Returns:
            List hasil deteksi setelah postprocessing
        """
        start_time = time.time()
        
        # Filter berdasarkan confidence
        filtered = self.confidence_filter.process(
            detections=detections,
            global_threshold=conf_threshold
        )
        
        if not filtered:
            return []
        
        # Terapkan NMS
        nms_results = self.nms_processor.process(
            detections=filtered,
            iou_threshold=iou_threshold,
            conf_threshold=conf_threshold,
            class_specific=class_specific_nms,
            max_detections=max_detections
        )
        
        if not nms_results or not refine_boxes:
            return nms_results
        
        # Perbaiki bounding box jika diperlukan
        refined = self.bbox_refiner.process(
            detections=nms_results
        )
        
        total_time = time.time() - start_time
        self.logger.debug(f"✨ Postprocessing: {len(detections)} -> {len(refined)} deteksi dalam {total_time:.4f}s")
        
        return refined