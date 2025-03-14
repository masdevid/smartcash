"""
File: smartcash/model/services/prediction/postprocessing.py
Deskripsi: Utilitas pascapemrosesan untuk deteksi mata uang SmartCash
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Union, Optional

from smartcash.common.layer_config import get_layer_config


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


def non_max_suppression(
    predictions: torch.Tensor,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300
) -> List[torch.Tensor]:
    """
    Implementasi Non-Maximum Suppression untuk deteksi.
    
    Args:
        predictions: Tensor prediksi dari model (format [n_pred, 6] dengan x1,y1,x2,y2,conf,cls)
        conf_threshold: Threshold confidence
        iou_threshold: Threshold IoU
        classes: Filter untuk kelas spesifik
        max_det: Jumlah maksimum deteksi
        
    Returns:
        List tensor hasil NMS
    """
    # Pastikan init output dan device
    bs = predictions.shape[0]  # batch size
    nc = predictions.shape[1] - 5  # jumlah kelas
    device = predictions.device
    
    # Converts xyxy format tensor_k -> list_k
    xyxy2xywh = lambda x: torch.cat((
        (x[0] + x[2])/2, (x[1] + x[3])/2, 
        x[2] - x[0], x[3] - x[1]
    ))
    
    output = [torch.zeros((0, 6), device=device)] * bs
    
    # Per gambar
    for batch_idx, pred in enumerate(predictions):  # gambar i
        pred = pred.unsqueeze(0)
        
        # Filter confidence
        pred = pred[pred[..., 4] > conf_threshold]
        
        # Jika tidak ada deteksi
        if not pred.shape[0]:
            continue
            
        # Filter berdasarkan kelas
        if classes is not None:
            pred = pred[(pred[..., 5:] == torch.tensor(classes, device=device)).any(1)]
        
        # Jika masih tidak ada deteksi
        if not pred.shape[0]:
            continue
        
        # Filter berdasarkan confidence terkuat
        scores = pred[:, 4]
        _, indices = scores.sort(0, descending=True)
        indices = indices[:max_det]
        pred = pred[indices]
        
        # Compute IoU matrix
        boxes = pred[:, :4]
        scores = pred[:, 4]
        classes = pred[:, 5]
        
        # Batching untuk NMS
        keep = torchvision_nms(boxes, scores, iou_threshold)
        
        if keep.shape[0]:
            pred = pred[keep]
            output[batch_idx] = pred
    
    return output


def torchvision_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    NMS menggunakan torchvision ops jika tersedia, fallback ke implementasi manual.
    
    Args:
        boxes: Tensor box [N, 4] format xyxy
        scores: Tensor scores [N]
        iou_threshold: IoU threshold
        
    Returns:
        Tensor indices boxes yang dipertahankan
    """
    try:
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_threshold)
    except (ImportError, AttributeError, AssertionError) as e:
        # Fallback ke NMS manual
        return manual_nms(boxes, scores, iou_threshold)


def manual_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    Implementasi manual NMS saat torchvision tidak tersedia.
    
    Args:
        boxes: Tensor box [N, 4] format xyxy
        scores: Tensor scores [N]
        iou_threshold: IoU threshold
        
    Returns:
        Tensor indices boxes yang dipertahankan
    """
    keep = []
    
    # Sort berdasarkan scores (descending)
    order = torch.argsort(scores, descending=True)
    
    while order.numel() > 0:
        # Pilih box dengan score tertinggi
        i = order[0].item()
        keep.append(i)
        
        # Hitung IoU dengan semua box lainnya
        if order.numel() == 1:
            break
            
        xx1 = torch.max(boxes[i, 0], boxes[order[1:], 0])
        yy1 = torch.max(boxes[i, 1], boxes[order[1:], 1])
        xx2 = torch.min(boxes[i, 2], boxes[order[1:], 2])
        yy2 = torch.min(boxes[i, 3], boxes[order[1:], 3])
        
        # Intersection area
        inter_area = torch.clamp(xx2 - xx1, min=0) * torch.clamp(yy2 - yy1, min=0)
        
        # Union area
        box_area = (boxes[order[1:], 2] - boxes[order[1:], 0]) * (boxes[order[1:], 3] - boxes[order[1:], 1])
        area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
        union_area = area_i + box_area - inter_area
        
        # IoU
        iou = inter_area / union_area
        
        # Eliminasi box dengan IoU > threshold
        mask = iou <= iou_threshold
        if not mask.any():
            break
            
        order = order[1:][mask]
    
    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


def xywh2xyxy(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Konversi bounding box dari format xywh ke xyxy.
    
    Args:
        x: Tensor atau array dengan format [x_center, y_center, width, height]
        
    Returns:
        Tensor atau array dengan format [x_min, y_min, x_max, y_max]
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x_min
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y_min
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x_max
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y_max
    else:
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # x_min
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # y_min
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # x_max
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # y_max
    
    return y


def xyxy2xywh(x: Union[torch.Tensor, np.ndarray]) -> Union[torch.Tensor, np.ndarray]:
    """
    Konversi bounding box dari format xyxy ke xywh.
    
    Args:
        x: Tensor atau array dengan format [x_min, y_min, x_max, y_max]
        
    Returns:
        Tensor atau array dengan format [x_center, y_center, width, height]
    """
    if isinstance(x, torch.Tensor):
        y = x.clone()
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
        y[:, 2] = x[:, 2] - x[:, 0]        # width
        y[:, 3] = x[:, 3] - x[:, 1]        # height
    else:
        y = np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center
        y[:, 2] = x[:, 2] - x[:, 0]        # width
        y[:, 3] = x[:, 3] - x[:, 1]        # height
    
    return y