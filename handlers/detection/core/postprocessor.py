# File: smartcash/handlers/detection/core/postprocessor.py
# Author: Alfrida Sabar
# Deskripsi: Postprocessor untuk hasil deteksi mata uang

from typing import Dict, Any, List, Union, Tuple, Optional
import torch
import numpy as np

from smartcash.utils.logger import get_logger
from smartcash.utils.coordinate_utils import CoordinateUtils

class DetectionPostprocessor:
    """
    Postprocessor hasil deteksi objek mata uang.
    Mengubah hasil mentah dari model menjadi format yang lebih mudah digunakan.
    """
    
    def __init__(
        self,
        config: Dict[str, Any],
        logger = None
    ):
        """
        Inisialisasi postprocessor.
        
        Args:
            config: Konfigurasi
            logger: Logger kustom (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("postprocessor")
        
        # Dapatkan mapping class_id ke nama kelas dan layer
        self.class_names = {}
        self.class_layers = {}
        
        # Inisialisasi mapping dari konfigurasi
        self._init_class_mappings()
        
    def _init_class_mappings(self):
        """Inisialisasi mapping class ID ke nama kelas dan layer dari konfigurasi."""
        layers_config = self.config.get('layers', {})
        
        # Iterasi untuk setiap layer (banknote, nominal, security)
        for layer_name, layer_config in layers_config.items():
            classes = layer_config.get('classes', [])
            class_ids = layer_config.get('class_ids', [])
            
            # Pastikan jumlah class dan class_id sama
            if len(classes) != len(class_ids):
                self.logger.warning(f"⚠️ Jumlah kelas dan class_id tidak sama untuk layer {layer_name}")
                continue
                
            # Mapping class_id ke nama kelas dan layer
            for class_name, class_id in zip(classes, class_ids):
                self.class_names[class_id] = class_name
                self.class_layers[class_id] = layer_name
    
    def process(
        self,
        detection_result: Dict[str, Any],
        original_shape: Optional[Tuple[int, int]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Postprocess hasil deteksi.
        
        Args:
            detection_result: Hasil dari detector
            original_shape: Shape gambar original (h, w) untuk rescaling
            **kwargs: Parameter tambahan
            
        Returns:
            Dictionary hasil deteksi terformat
        """
        # Ekstrak prediksi dan parameter dari hasil detector
        predictions = detection_result.get('predictions')
        conf_threshold = detection_result.get('conf_threshold')
        inference_time = detection_result.get('inference_time')
        
        # Dapatkan deteksi terformat
        detections = self._format_detections(predictions, conf_threshold)
        
        # Rescale koordinat ke ukuran gambar original jika diperlukan
        if original_shape is not None:
            detections = self._rescale_detections(detections, original_shape)
        
        # Kelompokkan deteksi berdasarkan layer
        detections_by_layer = self._group_by_layer(detections)
        
        # Siapkan hasil akhir
        result = {
            'detections': detections,
            'detections_by_layer': detections_by_layer,
            'inference_time': inference_time,
            'num_detections': len(detections)
        }
        
        return result
    
    def _format_detections(
        self,
        predictions: torch.Tensor,
        conf_threshold: float
    ) -> List[Dict[str, Any]]:
        """
        Format prediksi model menjadi list objek deteksi.
        
        Args:
            predictions: Tensor prediksi dari model (bs, num_preds, 5+num_classes)
            conf_threshold: Threshold konfidiensi
            
        Returns:
            List objek deteksi
        """
        # Pastikan predictions adalah tensor
        if not isinstance(predictions, torch.Tensor):
            predictions = torch.tensor(predictions)
            
        # Hanya gunakan elemen pertama jika ada batch dimension
        if len(predictions.shape) == 3:
            predictions = predictions[0]
            
        # Filter berdasarkan threshold
        mask = predictions[:, 4] > conf_threshold
        predictions = predictions[mask]
        
        # Jika tidak ada deteksi, kembalikan list kosong
        if len(predictions) == 0:
            return []
            
        # Format deteksi
        detections = []
        
        for pred in predictions:
            # Dapatkan class id dengan konfidiensi tertinggi
            class_scores = pred[5:]
            class_id = int(torch.argmax(class_scores).item())
            class_conf = float(class_scores[class_id].item())
            
            # Hanya proses jika konfidiensi melebihi threshold
            if class_conf < conf_threshold:
                continue
                
            # Ambil nilai box dan confidence
            box = pred[:4].tolist()  # [x, y, w, h] normalized
            box_conf = float(pred[4].item())
            
            # Dapatkan nama kelas dan layer
            class_name = self.class_names.get(class_id, f"class_{class_id}")
            layer_name = self.class_layers.get(class_id, "unknown")
            
            # Buat objek deteksi
            detection = {
                'bbox': box,  # [x, y, w, h] normalized
                'class_id': class_id,
                'class_name': class_name,
                'layer': layer_name,
                'confidence': float(box_conf * class_conf),
                'box_confidence': float(box_conf),
                'class_confidence': float(class_conf)
            }
            
            detections.append(detection)
            
        return detections
    
    def _rescale_detections(
        self,
        detections: List[Dict[str, Any]],
        original_shape: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Rescale koordinat deteksi ke ukuran gambar original.
        
        Args:
            detections: List deteksi dengan koordinat normalisasi
            original_shape: Shape gambar original (h, w)
            
        Returns:
            List deteksi dengan koordinat rescale
        """
        # Gunakan koordinat utils untuk denormalisasi
        original_h, original_w = original_shape
        
        for detection in detections:
            bbox_norm = detection['bbox']  # [x, y, w, h] normalized
            
            # Konversi ke piksel
            x_center = bbox_norm[0] * original_w
            y_center = bbox_norm[1] * original_h
            width = bbox_norm[2] * original_w
            height = bbox_norm[3] * original_h
            
            # Simpan koordinat denormalisasi
            detection['bbox_pixels'] = [x_center, y_center, width, height]
            
            # Tambahkan koordinat xmin, ymin, xmax, ymax untuk kemudahan
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            detection['bbox_xyxy'] = [xmin, ymin, xmax, ymax]
            
        return detections
    
    def _group_by_layer(
        self,
        detections: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Kelompokkan deteksi berdasarkan layer.
        
        Args:
            detections: List deteksi
            
        Returns:
            Dictionary deteksi berdasarkan layer
        """
        result = {}
        
        for detection in detections:
            layer = detection.get('layer', 'unknown')
            if layer not in result:
                result[layer] = []
            result[layer].append(detection)
            
        return result