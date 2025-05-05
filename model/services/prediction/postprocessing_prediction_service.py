"""
File: smartcash/model/services/prediction/postprocessing_prediction_service.py
Deskripsi: Modul untuk layanan post-proses prediksi
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

from smartcash.common.layer_config import get_layer_config
from smartcash.model.utils.metrics import non_max_suppression

def process_detections(
    predictions: torch.Tensor,
    orig_size: Tuple[int, int],
    model_size: Tuple[int, int] = (640, 640),
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45
) -> List[Dict]:
    """
    Proses prediksi raw dari model menjadi deteksi terstruktur.
    
    Meliputi:
    - Pemfilteran berdasarkan confidence threshold
    - Non-maximum suppression (NMS)
    - Penskalan ke ukuran gambar asli
    - Konversi ke format dict yang mudah digunakan
    
    Args:
        predictions: Tensor prediksi dari model (format [n_pred, 6] dengan x1,y1,x2,y2,conf,cls)
        orig_size: Ukuran gambar asli (width, height)
        model_size: Ukuran input model (width, height)
        conf_threshold: Threshold confidence untuk deteksi
        iou_threshold: Threshold IoU untuk NMS
        
    Returns:
        List dari deteksi dalam format dict
    """
    layer_config = get_layer_config()
    class_map = layer_config.get_class_map()
    
    # Jika tidak ada prediksi, return kosong
    if predictions is None or len(predictions) == 0:
        return []
    
    # Filter berdasarkan threshold confidence
    conf_mask = predictions[:, 4] >= conf_threshold
    predictions = predictions[conf_mask]
    
    # Jika tidak ada prediksi setelah filter, return kosong
    if len(predictions) == 0:
        return []
    
    # Non-maximum suppression per kelas
    nms_result = non_max_suppression(
        predictions, 
        conf_threshold, 
        iou_threshold,
        classes=None  # Semua kelas
    )
    
    # Format hasil ke dict
    detections = []
    
    for det in nms_result:
        if det is None or len(det) == 0:
            continue
            
        # Scale koordinat box ke ukuran asli
        scale_w = orig_size[0] / model_size[0]
        scale_h = orig_size[1] / model_size[1]
        
        for *xyxy, conf, cls_id in det:
            # Convert ke integer untuk display
            x1, y1, x2, y2 = [int(coord.item()) for coord in xyxy]
            
            # Scale box ke ukuran asli
            x1, x2 = int(x1 * scale_w), int(x2 * scale_w)
            y1, y2 = int(y1 * scale_h), int(y2 * scale_h)
            
            # Format hasil
            cls_id = int(cls_id.item())
            confidence = float(conf.item())
            
            # Extract metadata tambahan
            layer_name = layer_config.get_layer_for_class_id(cls_id)
            class_name = class_map.get(cls_id, f"class_{cls_id}")
            
            # Buat deteksi dalam format dict
            detection = {
                'bbox': [x1, y1, x2, y2],
                'confidence': confidence,
                'class_id': cls_id,
                'class_name': class_name,
                'layer': layer_name
            }
            
            detections.append(detection)
    
    return detections
