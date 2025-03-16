"""
File: smartcash/detection/services/inference/inference_service.py
Deskripsi: Layanan inferensi model untuk deteksi objek.
"""

import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union, Any

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection, ImageType
from smartcash.common.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD, ModelFormat
from smartcash.common.utils import format_time


class InferenceService:
    """Layanan inferensi model untuk deteksi objek"""
    
    def __init__(self,
                prediction_service,
                postprocessing_service=None,
                accelerator=None,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Inference Service
        
        Args:
            prediction_service: Layanan prediksi dari domain model
            postprocessing_service: Layanan postprocessing hasil (opsional)
            accelerator: Akselerator hardware untuk inferensi (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.prediction_service = prediction_service
        self.postprocessing_service = postprocessing_service
        self.accelerator = accelerator
        self.logger = logger or get_logger("InferenceService")
        
        # Aktifkan akselerator jika tersedia
        if self.accelerator:
            self.accelerator.setup()
    
    def infer(self, 
             image: ImageType, 
             conf_threshold: float = DEFAULT_CONF_THRESHOLD,
             iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[Detection]:
        """
        Melakukan inferensi model pada gambar
        
        Args:
            image: Gambar (path, numpy array, atau PIL Image)
            conf_threshold: Threshold minimum confidence untuk deteksi
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi
        """
        start_time = time.time()
        
        # Lakukan prediksi menggunakan prediction service dari domain model
        # Return raw detections (belum diproses dengan NMS)
        raw_detections = self.prediction_service.predict(
            images=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            return_annotated=False
        )
        
        # Terapkan postprocessing jika service tersedia
        detections = raw_detections
        if self.postprocessing_service:
            detections = self.postprocessing_service.process(
                detections=raw_detections,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
        
        inference_time = time.time() - start_time
        
        # Log hasil inferensi
        self.logger.debug(f"✨ Inferensi selesai dalam {inference_time:.4f}s: {len(detections)} objek terdeteksi")
        
        return detections
    
    def batch_infer(self,
                  images: List[ImageType],
                  conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                  iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[List[Detection]]:
        """
        Melakukan inferensi model pada batch gambar
        
        Args:
            images: List gambar (path, numpy array, atau PIL Image)
            conf_threshold: Threshold minimum confidence untuk deteksi
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi untuk setiap gambar
        """
        start_time = time.time()
        
        # Lakukan prediksi batch menggunakan prediction service
        batch_detections = self.prediction_service.predict(
            images=images,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            return_annotated=False
        )
        
        # Terapkan postprocessing untuk setiap gambar jika service tersedia
        if self.postprocessing_service:
            processed_batch = []
            for detections in batch_detections:
                processed = self.postprocessing_service.process(
                    detections=detections,
                    conf_threshold=conf_threshold,
                    iou_threshold=iou_threshold
                )
                processed_batch.append(processed)
            batch_detections = processed_batch
        
        inference_time = time.time() - start_time
        total_detections = sum(len(detections) for detections in batch_detections)
        
        # Log hasil inferensi
        self.logger.debug(f"✨ Batch inferensi selesai dalam {inference_time:.4f}s: "
                        f"{total_detections} objek terdeteksi pada {len(images)} gambar")
        
        return batch_detections
    
    def visualize(self, image: ImageType, detections: List[Detection]) -> np.ndarray:
        """
        Visualisasikan hasil deteksi pada gambar
        
        Args:
            image: Gambar original
            detections: List hasil deteksi
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Gunakan visualisasi dari prediction service
        return self.prediction_service.visualize_predictions(
            image=image,
            detections=detections,
            conf_threshold=0.0  # Gunakan 0.0 agar tidak ada filtering tambahan
        )
    
    def optimize_model(self, target_format: ModelFormat = None, **kwargs):
        """
        Optimalkan model untuk inferensi
        
        Args:
            target_format: Format target untuk optimasi (ONNX, TensorRT, dll.)
            **kwargs: Parameter tambahan untuk optimasi
        """
        if not hasattr(self.prediction_service, 'optimize_model'):
            self.logger.warning("⚠️ Prediction service tidak mendukung optimasi model")
            return False
        
        # Coba optimalkan model
        try:
            self.logger.info(f"🔧 Mengoptimalkan model ke format {target_format}")
            result = self.prediction_service.optimize_model(target_format, **kwargs)
            if result:
                self.logger.info(f"✅ Optimasi model berhasil")
            else:
                self.logger.warning(f"⚠️ Optimasi model tidak berhasil")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Error saat mengoptimalkan model: {str(e)}")
            return False
    
    def get_model_info(self) -> Dict:
        """
        Dapatkan informasi model yang digunakan
        
        Returns:
            Dictionary berisi informasi model
        """
        if hasattr(self.prediction_service, 'get_model_info'):
            return self.prediction_service.get_model_info()
        else:
            # Coba dapatkan informasi dasar dari model
            model_info = {
                "available": True,
                "timestamp": time.time()
            }
            
            # Coba dapatkan model dari prediction_service
            if hasattr(self.prediction_service, 'model'):
                model = self.prediction_service.model
                
                # Coba dapatkan informasi dari model
                if hasattr(model, 'name'):
                    model_info['name'] = model.name
                    
                # Coba dapatkan metadata jika ada
                if hasattr(model, 'metadata'):
                    model_info['metadata'] = model.metadata
            
            return model_info