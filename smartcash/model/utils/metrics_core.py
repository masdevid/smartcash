"""
File: smartcash/model/utils/metrics_core.py
Deskripsi: Fungsi-fungsi dasar untuk manipulasi bounding box dan perhitungan IoU
"""

import torch
import numpy as np
from typing import Union


def box_iou(box1, box2):
    """
    Hitung Intersection over Union (IoU) antara dua set bounding boxes.
    
    Args:
        box1: Tensor koordinat box 1 dalam format xyxy
        box2: Tensor koordinat box 2 dalam format xyxy
        
    Returns:
        Tensor IoU
    """
    # Hitung area boxes
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Hitung intersection
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])
    
    # Clamp untuk menghindari nilai negatif (tidak overlap)
    w = (inter_x2 - inter_x1).clamp(min=0)
    h = (inter_y2 - inter_y1).clamp(min=0)
    
    # Hitung area intersection dan union
    inter_area = w * h
    union_area = area1 + area2 - inter_area
    
    # Hitung IoU
    iou = inter_area / union_area
    
    return iou


def xywh2xyxy(x):
    """
    Konversi format box dari center-xywh ke vertex-xyxy.
    
    Args:
        x: Format xywh (center_x, center_y, width, height)
        
    Returns:
        Format xyxy (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    
    # Konversi dari center format ke corners
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    
    return y


def xyxy2xywh(x):
    """
    Konversi format box dari vertex-xyxy ke center-xywh.
    
    Args:
        x: Format xyxy (top_left_x, top_left_y, bottom_right_x, bottom_right_y)
        
    Returns:
        Format xywh (center_x, center_y, width, height)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    
    # Konversi dari corners ke center format
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # center x
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # center y
    y[..., 2] = x[..., 2] - x[..., 0]        # width
    y[..., 3] = x[..., 3] - x[..., 1]        # height
    
    return y


def compute_ap(recall, precision):
    """
    Hitung Average Precision berdasarkan kurva precision-recall.
    
    Args:
        recall: Array recall
        precision: Array precision
        
    Returns:
        Average Precision
    """
    # Tambahkan sentinel values untuk memudahkan perhitungan
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    
    # Compute envelope kurva precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Cari titik-titik dimana recall berubah
    i = np.where(mrec[1:] != mrec[:-1])[0]
    
    # Hitung area di bawah kurva precision-recall (AP)
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap


def bbox_ioa(box1, box2, eps=1e-7):
    """
    Hitung intersection over area (IoA).
    
    Args:
        box1: Format xyxy (n, 4)
        box2: Format xyxy (n, 4)
        eps: Epsilon untuk menghindari pembagian dengan nol
        
    Returns:
        Tensor IoA
    """
    # Intersection
    inter_x1 = torch.max(box1[0], box2[0])
    inter_y1 = torch.max(box1[1], box2[1])
    inter_x2 = torch.min(box1[2], box2[2])
    inter_y2 = torch.min(box1[3], box2[3])
    
    # Clamp untuk menghindari nilai negatif (tidak overlap)
    w = (inter_x2 - inter_x1).clamp(min=0)
    h = (inter_y2 - inter_y1).clamp(min=0)
    
    # Intersection area
    inter_area = w * h
    
    # Area box1
    area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    
    # IoA = intersection / area box1
    ioa = inter_area / (area + eps)
    
    return ioa