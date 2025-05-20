"""
File: smartcash/detection/handlers/detection_handler.py
Deskripsi: Handler untuk deteksi objek pada gambar tunggal.
"""

import os
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

from smartcash.common.io import ensure_dir
from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common import Detection, ImageType, DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD


class DetectionHandler:
    """Handler untuk mengelola proses deteksi pada gambar tunggal."""
    
    def __init__(self, 
                 inference_service, 
                 postprocessing_service=None, 
                 logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Detection Handler
        
        Args:
            inference_service: Layanan inferensi model
            postprocessing_service: Layanan postprocessing hasil deteksi (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.inference_service = inference_service
        self.postprocessing_service = postprocessing_service
        self.logger = logger or get_logger()
    
    def detect(self, 
               image: ImageType, 
               conf_threshold: float = DEFAULT_CONF_THRESHOLD,
               iou_threshold: float = DEFAULT_IOU_THRESHOLD,
               apply_postprocessing: bool = True,
               return_visualization: bool = False) -> Union[List[Detection], Tuple[List[Detection], np.ndarray]]:
        """
        Deteksi objek pada gambar
        
        Args:
            image: Gambar yang akan dideteksi (ndarray, path, atau PIL Image)
            conf_threshold: Threshold minimum confidence untuk deteksi
            iou_threshold: Threshold IoU untuk NMS
            apply_postprocessing: Flag untuk menerapkan postprocessing (NMS, dll)
            return_visualization: Flag untuk mengembalikan visualisasi hasil deteksi
            
        Returns:
            List deteksi, atau tuple (list deteksi, gambar tervisualisasi) jika return_visualization=True
        """
        self.logger.info(f"🔍 Mendeteksi objek pada gambar dengan conf_threshold={conf_threshold:.2f}, iou_threshold={iou_threshold:.2f}")
        
        # Lakukan inferensi menggunakan inference service
        detections = self.inference_service.infer(
            image=image, 
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Terapkan postprocessing jika diminta dan service tersedia
        if apply_postprocessing and self.postprocessing_service:
            detections = self.postprocessing_service.process(
                detections=detections,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            
        count = len(detections)
        self.logger.debug(f"✅ Deteksi selesai: {count} objek terdeteksi")
        
        # Kembalikan hasil sesuai flag return_visualization
        if return_visualization:
            # Jika inference_service memiliki method visualize, gunakan itu
            if hasattr(self.inference_service, 'visualize'):
                visualization = self.inference_service.visualize(image, detections)
                return detections, visualization
            else:
                self.logger.warning("⚠️ Visualisasi diminta tetapi service tidak mendukung visualisasi")
                return detections, None
        
        return detections
    
    def save_result(self, 
                   detections: List[Detection], 
                   output_path: str, 
                   image: Optional[ImageType] = None,
                   save_visualization: bool = True,
                   format: str = 'json') -> Dict[str, str]:
        """
        Simpan hasil deteksi ke file
        
        Args:
            detections: List deteksi
            output_path: Path tempat menyimpan hasil
            image: Gambar original (untuk visualisasi)
            save_visualization: Flag untuk menyimpan visualisasi hasil
            format: Format penyimpanan ('json', 'txt', 'csv')
            
        Returns:
            Dictionary berisi path file hasil ('data' dan 'visualization')
        """
        # Pastikan direktori output ada
        ensure_dir(os.path.dirname(output_path))
        
        result_paths = {}
        
        # Simpan data deteksi dalam format yang diminta
        if format.lower() == 'json':
            import json
            data_path = f"{output_path}.json"
            with open(data_path, 'w') as f:
                json.dump([d.__dict__ for d in detections], f, indent=2)
            result_paths['data'] = data_path
        
        elif format.lower() == 'txt':
            data_path = f"{output_path}.txt"
            with open(data_path, 'w') as f:
                for detection in detections:
                    f.write(f"{detection.class_id} {detection.confidence:.4f} "
                           f"{detection.bbox[0]:.4f} {detection.bbox[1]:.4f} "
                           f"{detection.bbox[2]:.4f} {detection.bbox[3]:.4f}\n")
            result_paths['data'] = data_path
            
        elif format.lower() == 'csv':
            import csv
            data_path = f"{output_path}.csv"
            with open(data_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
                for d in detections:
                    writer.writerow([
                        d.class_id, 
                        d.class_name, 
                        f"{d.confidence:.4f}",
                        f"{d.bbox[0]:.4f}", 
                        f"{d.bbox[1]:.4f}", 
                        f"{d.bbox[2]:.4f}", 
                        f"{d.bbox[3]:.4f}"
                    ])
            result_paths['data'] = data_path
        
        # Simpan visualisasi jika diminta dan gambar tersedia
        if save_visualization and image is not None and hasattr(self.inference_service, 'visualize'):
            viz_path = f"{output_path}_visualization.jpg"
            visualization = self.inference_service.visualize(image, detections)
            
            # Simpan visualisasi sebagai gambar
            if isinstance(visualization, np.ndarray):
                Image.fromarray(visualization).save(viz_path)
                result_paths['visualization'] = viz_path
        
        self.logger.info(f"💾 Hasil deteksi disimpan ke {result_paths}")
        return result_paths