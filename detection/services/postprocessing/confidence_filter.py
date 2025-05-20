"""
File: smartcash/detection/services/postprocessing/confidence_filter.py
Description: Filter deteksi berdasarkan confidence threshold.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection


class ConfidenceFilter:
    """Filter deteksi berdasarkan confidence threshold"""
    
    def __init__(self, 
                default_threshold: float = 0.25,
                class_thresholds: Optional[Dict[int, float]] = None,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Confidence Filter
        
        Args:
            default_threshold: Threshold confidence default
            class_thresholds: Dictionary threshold per kelas (ID kelas -> threshold)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.default_threshold = default_threshold
        self.class_thresholds = class_thresholds or {}
        self.logger = logger or get_logger()
    
    def process(self, 
               detections: List[Detection],
               global_threshold: Optional[float] = None) -> List[Detection]:
        """
        Filter deteksi berdasarkan confidence threshold
        
        Args:
            detections: List hasil deteksi yang akan difilter
            global_threshold: Override threshold untuk semua kelas (opsional)
            
        Returns:
            List deteksi yang lolos filter
        """
        if not detections:
            return []
        
        start_time = time.time()
        
        # Gunakan global_threshold jika disediakan
        main_threshold = global_threshold if global_threshold is not None else self.default_threshold
        
        # Filter deteksi
        filtered = []
        for detection in detections:
            # Tentukan threshold untuk kelas ini
            if global_threshold is not None:
                # Jika global_threshold disediakan, gunakan itu
                threshold = global_threshold
            else:
                # Jika tidak, coba dapatkan dari class_thresholds atau gunakan default
                threshold = self.class_thresholds.get(detection.class_id, self.default_threshold)
            
            # Tambahkan deteksi ke hasil jika confidence >= threshold
            if detection.confidence >= threshold:
                filtered.append(detection)
        
        filter_time = time.time() - start_time
        self.logger.debug(f"✓ Filter: {len(detections)} -> {len(filtered)} deteksi dalam {filter_time:.4f}s")
        
        return filtered
    
    def set_threshold(self, class_id: int, threshold: float):
        """
        Set threshold confidence untuk kelas tertentu
        
        Args:
            class_id: ID kelas
            threshold: Nilai threshold
        """
        self.class_thresholds[class_id] = threshold
    
    def get_threshold(self, class_id: int) -> float:
        """
        Dapatkan threshold confidence untuk kelas tertentu
        
        Args:
            class_id: ID kelas
            
        Returns:
            Nilai threshold untuk kelas
        """
        return self.class_thresholds.get(class_id, self.default_threshold)
    
    def reset_thresholds(self):
        """Reset semua threshold kelas ke default"""
        self.class_thresholds = {}