"""
File: smartcash/model/services/prediction/core_prediction_service.py
Deskripsi: Layanan utama untuk prediksi model deteksi objek
"""

import os
import torch
import numpy as np
import cv2
from typing import Dict, List, Optional, Tuple, Union, Any
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.utils.preprocessing_model_utils import ModelPreprocessor
from smartcash.model.services.postprocessing.nms_processor import NMSProcessor
from smartcash.model.visualization.detection_visualizer import DetectionVisualizer


class PredictionService:
    """
    Layanan untuk melakukan prediksi menggunakan model deteksi objek.
    Mendukung prediksi tunggal dan batch dengan berbagai format input.
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[Dict[str, Any]] = None,
        logger: Optional[Any] = None
    ):
        """
        Inisialisasi service prediksi.
        
        Args:
            model: Model untuk prediksi
            config: Konfigurasi prediksi
            logger: Logger untuk mencatat aktivitas
        """
        self.logger = logger or get_logger()
        self.model = model
        
        # Default config
        self.default_config = {
            'conf_threshold': 0.25,      # Threshold confidence
            'iou_threshold': 0.45,       # Threshold IoU untuk NMS
            'img_size': (640, 640),      # Ukuran gambar input
            'max_detections': 100,       # Jumlah max deteksi
            'device': None,              # Auto-detect if None
            'return_annotated': True,    # Return gambar teranotasi
            'mean': (0.485, 0.456, 0.406),  # ImageNet mean
            'std': (0.229, 0.224, 0.225),   # ImageNet std
            'pad_to_square': True        # Pad gambar menjadi persegi
        }
        
        # Merge konfigurasi
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Setup device
        if self.config['device'] is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(self.config['device'])
        
        # Setup model
        self.model.to(self.device)
        self.model.eval()
        
        # Setup preprocessor untuk preprocessing on-the-fly
        self.model_preprocessor = ModelPreprocessor(
            img_size=self.config.get('img_size', (640, 640)),
            mean=self.config.get('mean', (0.485, 0.456, 0.406)),
            std=self.config.get('std', (0.229, 0.224, 0.225)),
            pad_to_square=self.config.get('pad_to_square', True)
        )
        
        # Setup NMS processor
        self.nms_processor = NMSProcessor()
        
        # Setup visualizer
        self.visualizer = DetectionVisualizer()
        
        self.logger.info(f"🔮 PredictionService diinisialisasi: device={self.device}")
    
    def predict(
        self,
        images: Union[np.ndarray, List[np.ndarray], str, List[str]],
        return_annotated: Optional[bool] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Prediksi objek dalam gambar.
        
        Args:
            images: Gambar input (numpy array, path, atau list)
            return_annotated: Apakah mengembalikan gambar teranotasi
            conf_threshold: Threshold confidence (override config)
            iou_threshold: Threshold IoU untuk NMS (override config)
            
        Returns:
            Dictionary hasil deteksi
        """
        # Gunakan parameter dari argumen atau default
        conf_threshold = conf_threshold or self.config['conf_threshold']
        iou_threshold = iou_threshold or self.config['iou_threshold']
        return_annotated = return_annotated if return_annotated is not None else self.config['return_annotated']
        
        # Load gambar jika input berupa string path
        if isinstance(images, str):
            # Single image path
            orig_images = [cv2.imread(images)]
            image_paths = [images]
        elif isinstance(images, list) and all(isinstance(i, str) for i in images):
            # List of image paths
            orig_images = [cv2.imread(img_path) for img_path in images]
            image_paths = images
        elif isinstance(images, np.ndarray) and len(images.shape) == 3:
            # Single numpy array
            orig_images = [images.copy()]
            image_paths = ['unknown']
        elif isinstance(images, list) and all(isinstance(i, np.ndarray) for i in images):
            # List of numpy arrays
            orig_images = [img.copy() for img in images]
            image_paths = ['unknown'] * len(images)
        else:
            raise ValueError(f"Format input tidak valid: {type(images)}")
        
        # Filter gambar yang berhasil dimuat
        valid_images = []
        valid_paths = []
        
        for i, img in enumerate(orig_images):
            if img is not None:
                valid_images.append(img)
                valid_paths.append(image_paths[i])
            else:
                self.logger.warning(f"⚠️ Gagal memuat gambar: {image_paths[i]}")
        
        if not valid_images:
            return {'detections': [], 'annotated_images': []}
        
        # Preprocess gambar
        preprocessed = self._preprocess_images(valid_images)
        
        # Run inference
        with torch.no_grad():
            predictions = self.model(preprocessed)
        
        # Postprocess predictions
        detections = self._postprocess_predictions(
            predictions,
            valid_images,
            conf_threshold,
            iou_threshold
        )
        
        # Buat hasil teranotasi jika diminta
        annotated_images = []
        if return_annotated:
            for i, (img, dets) in enumerate(zip(valid_images, detections)):
                # Anotasi gambar
                annotated = self.visualizer.visualize_detection(
                    img, 
                    dets, 
                    conf_threshold=conf_threshold
                )
                annotated_images.append(annotated)
        
        # Format response
        result = {
            'detections': detections,
            'paths': valid_paths
        }
        
        if return_annotated:
            result['annotated_images'] = annotated_images
        
        return result
    
    def _preprocess_images(self, images: List[np.ndarray]) -> torch.Tensor:
        """
        Preprocess gambar untuk inferensi. Selalu menggunakan preprocessing 
        on-the-fly untuk konsistensi untuk prediksi.
        
        Args:
            images: List gambar untuk diproses
            
        Returns:
            Tensor batch gambar yang sudah dipreprocessing
        """
        if isinstance(images, (list, tuple)):
            # Batch preprocessing
            return self.model_preprocessor.preprocess_batch(images)
        else:
            # Single image preprocessing
            return self.model_preprocessor.preprocess_image(images)
    
    def _postprocess_predictions(
        self,
        predictions: Dict[str, torch.Tensor],
        original_images: List[np.ndarray],
        conf_threshold: float,
        iou_threshold: float
    ) -> List[List[Dict[str, Any]]]:
        """
        Postprocess hasil prediksi model.
        
        Args:
            predictions: Output dari model
            original_images: Gambar asli
            conf_threshold: Threshold confidence
            iou_threshold: Threshold IoU untuk NMS
            
        Returns:
            List deteksi per gambar
        """
        # Batched results
        batch_detections = []
        
        # Multi-layer detection
        if isinstance(predictions, dict):
            # Untuk setiap gambar dalam batch
            for img_idx in range(len(original_images)):
                img_detections = []
                
                # Untuk setiap layer (banknote, nominal, security)
                for layer_name, layer_preds in predictions.items():
                    # Skip jika layer tidak memiliki prediksi untuk gambar ini
                    if img_idx >= len(layer_preds):
                        continue
                    
                    # Ambil prediksi untuk gambar ini
                    img_pred = layer_preds[img_idx]
                    
                    # Hanya ambil prediksi di atas threshold
                    confident_mask = img_pred[:, 4] >= conf_threshold
                    img_pred = img_pred[confident_mask]
                    
                    if len(img_pred) == 0:
                        continue
                    
                    # Apply NMS
                    img_pred = self.nms_processor.process(
                        img_pred,
                        iou_threshold=iou_threshold
                    )
                    
                    # Konversi ke format yang lebih mudah digunakan
                    for det in img_pred:
                        x1, y1, x2, y2, conf, cls_id = det
                        
                        # Skala koordinat ke ukuran gambar asli
                        orig_h, orig_w = original_images[img_idx].shape[:2]
                        img_h, img_w = self.config['img_size']
                        
                        # Skala
                        x1 = int(x1 * orig_w / img_w)
                        y1 = int(y1 * orig_h / img_h)
                        x2 = int(x2 * orig_w / img_w)
                        y2 = int(y2 * orig_h / img_h)
                        
                        # Tambahkan ke hasil
                        img_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': f"{layer_name}_{int(cls_id)}",  # Format: layer_classid
                            'layer': layer_name
                        })
                
                # Sort berdasarkan confidence (descending)
                img_detections = sorted(img_detections, key=lambda x: x['confidence'], reverse=True)
                
                # Limit jumlah deteksi
                img_detections = img_detections[:self.config['max_detections']]
                
                batch_detections.append(img_detections)
        else:
            # Single layer detection (backward compatibility)
            for img_idx in range(len(original_images)):
                img_pred = predictions[img_idx]
                img_detections = []
                
                # Threshold and NMS
                confident_mask = img_pred[:, 4] >= conf_threshold
                img_pred = img_pred[confident_mask]
                
                if len(img_pred) > 0:
                    img_pred = self.nms_processor.process(
                        img_pred,
                        iou_threshold=iou_threshold
                    )
                    
                    # Konversi ke format yang lebih mudah digunakan
                    for det in img_pred:
                        x1, y1, x2, y2, conf, cls_id = det
                        
                        # Skala koordinat ke ukuran gambar asli
                        orig_h, orig_w = original_images[img_idx].shape[:2]
                        img_h, img_w = self.config['img_size']
                        
                        # Skala
                        x1 = int(x1 * orig_w / img_w)
                        y1 = int(y1 * orig_h / img_h)
                        x2 = int(x2 * orig_w / img_w)
                        y2 = int(y2 * orig_h / img_h)
                        
                        # Tambahkan ke hasil
                        img_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'confidence': float(conf),
                            'class_id': int(cls_id),
                            'class_name': str(int(cls_id)),
                            'layer': 'default'
                        })
                
                # Sort berdasarkan confidence (descending)
                img_detections = sorted(img_detections, key=lambda x: x['confidence'], reverse=True)
                
                # Limit jumlah deteksi
                img_detections = img_detections[:self.config['max_detections']]
                
                batch_detections.append(img_detections)
        
        return batch_detections
    
    def predict_from_files(
        self,
        image_paths: Union[str, List[str]],
        return_annotated: Optional[bool] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Prediksi objek dari file gambar.
        
        Args:
            image_paths: Path gambar atau list path
            return_annotated: Apakah mengembalikan gambar teranotasi
            conf_threshold: Threshold confidence (override config)
            iou_threshold: Threshold IoU untuk NMS (override config)
            
        Returns:
            Dictionary hasil deteksi
        """
        return self.predict(
            images=image_paths,
            return_annotated=return_annotated,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold
        )
    
    def visualize_predictions(
        self,
        image: Union[np.ndarray, str],
        detections: List[Dict[str, Any]],
        conf_threshold: Optional[float] = None,
        output_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Visualisasikan hasil prediksi pada gambar.
        
        Args:
            image: Gambar input (numpy array atau path)
            detections: Hasil deteksi
            conf_threshold: Threshold confidence
            output_path: Path untuk menyimpan hasil visualisasi
            
        Returns:
            Gambar dengan visualisasi deteksi
        """
        # Load gambar jika input berupa string path
        if isinstance(image, str):
            image = cv2.imread(image)
        
        # Gunakan parameter atau default
        conf_threshold = conf_threshold or self.config['conf_threshold']
        
        # Buat visualisasi
        result = self.visualizer.visualize_detection(
            image,
            detections,
            filename=Path(output_path).name if output_path else None,
            conf_threshold=conf_threshold
        )
        
        return result