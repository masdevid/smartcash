"""
smartcash/detection/detector.py
Koordinator utama proses deteksi.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import time
import numpy as np

from smartcash.common.logger import SmartCashLogger, get_logger
from smartcash.common.types import Detection, ImageType
from smartcash.common.constants import DEFAULT_CONF_THRESHOLD, DEFAULT_IOU_THRESHOLD

from smartcash.detection.services.inference import InferenceService
from smartcash.detection.services.postprocessing import PostprocessingService
from smartcash.detection.services.visualization_adapter import DetectionVisualizationAdapter


class Detector:
    """Koordinator utama proses deteksi"""
    
    def __init__(self,
                model_manager=None,
                prediction_service=None,
                inference_service=None,
                postprocessing_service=None,
                visualization_adapter=None,
                logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Detector
        
        Args:
            model_manager: Manager model dari domain model (diperlukan jika prediction_service tidak disediakan)
            prediction_service: Layanan prediksi dari domain model (diperlukan jika inference_service tidak disediakan)
            inference_service: Layanan inferensi custom (opsional)
            postprocessing_service: Layanan postprocessing custom (opsional)
            visualization_adapter: Adapter visualisasi custom (opsional)
            logger: Logger untuk mencatat aktivitas (opsional)
        """
        self.logger = logger or get_logger("Detector")
        
        # Setup inference service
        if inference_service is not None:
            self.inference_service = inference_service
        elif prediction_service is not None:
            self.inference_service = InferenceService(
                prediction_service=prediction_service,
                logger=self.logger
            )
        elif model_manager is not None:
            # Buat prediction service dari model manager
            self.inference_service = InferenceService(
                prediction_service=model_manager.get_prediction_service(),
                logger=self.logger
            )
        else:
            raise ValueError("Salah satu dari model_manager, prediction_service, atau inference_service harus disediakan")
        
        # Setup postprocessing service
        self.postprocessing_service = postprocessing_service or PostprocessingService(
            logger=self.logger
        )
        
        # Setup visualization adapter
        self.visualization_adapter = visualization_adapter
    
    def detect(self,
              image: ImageType,
              conf_threshold: float = DEFAULT_CONF_THRESHOLD,
              iou_threshold: float = DEFAULT_IOU_THRESHOLD,
              with_visualization: bool = False) -> Union[List[Detection], Tuple[List[Detection], np.ndarray]]:
        """
        Deteksi objek pada gambar
        
        Args:
            image: Gambar (path, numpy array, atau PIL Image)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            with_visualization: Flag untuk mengembalikan visualisasi hasil
            
        Returns:
            List deteksi, atau tuple (list deteksi, gambar tervisualisasi) jika with_visualization=True
        """
        start_time = time.time()
        
        # Lakukan inferensi
        detections = self.inference_service.infer(
            image=image,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Terapkan postprocessing
        processed_detections = self.postprocessing_service.process(
            detections=detections,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        detection_time = time.time() - start_time
        self.logger.debug(f"🔍 Deteksi selesai: {len(processed_detections)} objek terdeteksi dalam {detection_time:.4f}s")
        
        # Kembalikan dengan visualisasi jika diminta
        if with_visualization:
            # Gunakan visualization adapter jika tersedia
            if self.visualization_adapter is not None:
                visualization = self.visualization_adapter.visualize_detection(
                    image=image,
                    detections=processed_detections,
                    conf_threshold=conf_threshold
                )
            else:
                # Fallback ke visualisasi dari inference service
                visualization = self.inference_service.visualize(
                    image=image,
                    detections=processed_detections
                )
                
            return processed_detections, visualization
            
        return processed_detections
    
    def detect_multilayer(self,
                        image: ImageType,
                        threshold: Dict[str, float] = None) -> Dict[str, List[Detection]]:
        """
        Deteksi objek multilayer pada gambar
        
        Args:
            image: Gambar (path, numpy array, atau PIL Image)
            threshold: Dictionary threshold per layer (nama layer -> threshold)
            
        Returns:
            Dictionary hasil deteksi per layer
        """
        # Set default threshold jika tidak disediakan
        if threshold is None:
            threshold = {
                "banknote": DEFAULT_CONF_THRESHOLD,
                "nominal": DEFAULT_CONF_THRESHOLD,
                "security": DEFAULT_CONF_THRESHOLD
            }
        
        start_time = time.time()
        
        # Lakukan inferensi standar
        all_detections = self.detect(image)
        
        # Kelompokkan deteksi berdasarkan layer
        from ..common.layer_config import get_layer_config
        layer_config = get_layer_config()
        
        result = {}
        for layer_name in layer_config.get_layer_names():
            # Filter deteksi untuk layer ini
            layer_detections = []
            
            for detection in all_detections:
                # Dapatkan layer untuk class_id ini
                detection_layer = layer_config.get_layer_for_class_id(detection.class_id)
                
                # Filter berdasarkan layer dan threshold
                if detection_layer == layer_name and detection.confidence >= threshold.get(layer_name, DEFAULT_CONF_THRESHOLD):
                    layer_detections.append(detection)
            
            result[layer_name] = layer_detections
        
        detection_time = time.time() - start_time
        
        # Log hasil
        layer_counts = ", ".join([f"{layer}: {len(dets)}" for layer, dets in result.items()])
        self.logger.debug(f"🔍 Deteksi multilayer selesai dalam {detection_time:.4f}s: {layer_counts}")
        
        return result
    
    def detect_batch(self,
                   images: List[ImageType],
                   conf_threshold: float = DEFAULT_CONF_THRESHOLD,
                   iou_threshold: float = DEFAULT_IOU_THRESHOLD) -> List[List[Detection]]:
        """
        Deteksi objek pada batch gambar
        
        Args:
            images: List gambar (path, numpy array, atau PIL Image)
            conf_threshold: Threshold confidence minimum
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List hasil deteksi untuk setiap gambar
        """
        start_time = time.time()
        
        # Lakukan batch inference
        batch_detections = self.inference_service.batch_infer(
            images=images,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
        
        # Terapkan postprocessing untuk setiap set deteksi
        processed_batch = []
        for detections in batch_detections:
            processed = self.postprocessing_service.process(
                detections=detections,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold
            )
            processed_batch.append(processed)
        
        batch_time = time.time() - start_time
        total_detections = sum(len(dets) for dets in processed_batch)
        
        self.logger.debug(f"🔍 Batch deteksi selesai: {total_detections} objek terdeteksi "
                        f"pada {len(images)} gambar dalam {batch_time:.4f}s")
        
        return processed_batch
    
    def visualize(self,
                 image: ImageType,
                 detections: List[Detection],
                 conf_threshold: float = 0.0,
                 show_labels: bool = True,
                 show_conf: bool = True,
                 filename: Optional[str] = None) -> np.ndarray:
        """
        Visualisasikan hasil deteksi pada gambar
        
        Args:
            image: Gambar yang akan divisualisasikan
            detections: List hasil deteksi
            conf_threshold: Threshold minimum confidence untuk visualisasi
            show_labels: Flag untuk menampilkan label kelas
            show_conf: Flag untuk menampilkan nilai confidence
            filename: Nama file untuk menyimpan hasil (opsional)
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Gunakan visualization adapter jika tersedia
        if self.visualization_adapter is not None:
            return self.visualization_adapter.visualize_detection(
                image=image,
                detections=detections,
                filename=filename,
                conf_threshold=conf_threshold,
                show_labels=show_labels,
                show_conf=show_conf
            )
        
        # Fallback ke visualisasi dari inference service
        return self.inference_service.visualize(
            image=image,
            detections=detections
        )