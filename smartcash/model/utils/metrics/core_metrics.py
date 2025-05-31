"""
File: smartcash/model/utils/metrics/core_metrics.py
Deskripsi: Implementasi fungsi metrik dasar untuk deteksi objek
"""

import torch
import math
from typing import Tuple, Union, List

def box_area(box: torch.Tensor) -> torch.Tensor:
    """
    Menghitung area dari bounding box dalam format xyxy.
    
    Args:
        box: Tensor dengan shape (..., 4) dalam format xyxy
        
    Returns:
        Tensor dengan shape (...) berisi area untuk setiap box
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Menghitung Intersection over Union (IoU) antara dua set bounding boxes.
    
    Args:
        box1: Tensor dengan shape (n, 4) dalam format xyxy
        box2: Tensor dengan shape (m, 4) dalam format xyxy
        
    Returns:
        Tensor dengan shape (n, m) berisi IoU untuk setiap pasangan box
    """
    # Pastikan input dalam format yang benar
    assert box1.shape[-1] == 4, f"❌ Box1 harus memiliki dimensi terakhir 4, tetapi memiliki shape {box1.shape}"
    assert box2.shape[-1] == 4, f"❌ Box2 harus memiliki dimensi terakhir 4, tetapi memiliki shape {box2.shape}"
    
    # Hitung area untuk setiap box
    area1 = box_area(box1)  # (n,)
    area2 = box_area(box2)  # (m,)
    
    # Broadcast untuk mendapatkan koordinat intersection
    # box1: (n, 4) -> (n, 1, 4)
    # box2: (m, 4) -> (1, m, 4)
    box1 = box1[:, None, :]  # (n, 1, 4)
    box2 = box2[None, :, :]  # (1, m, 4)
    
    # Hitung koordinat intersection
    top_left = torch.maximum(box1[..., :2], box2[..., :2])  # (n, m, 2)
    bottom_right = torch.minimum(box1[..., 2:], box2[..., 2:])  # (n, m, 2)
    
    # Hitung area intersection
    wh = (bottom_right - top_left).clamp(min=0)  # (n, m, 2)
    intersection = wh[..., 0] * wh[..., 1]  # (n, m)
    
    # Hitung IoU: intersection / union
    # union = area1 + area2 - intersection
    union = area1[:, None] + area2[None, :] - intersection
    
    # Hindari division by zero
    eps = torch.finfo(torch.float32).eps
    iou = intersection / (union + eps)
    
    return iou

def bbox_iou(box1: torch.Tensor, box2: torch.Tensor, xywh: bool = True, GIoU: bool = False, DIoU: bool = False, CIoU: bool = False, eps: float = 1e-7) -> torch.Tensor:
    """
    Menghitung IoU dengan opsi untuk GIoU, DIoU, atau CIoU.
    
    Args:
        box1: Tensor dengan shape (n, 4)
        box2: Tensor dengan shape (n, 4) atau (1, 4) untuk broadcast
        xywh: Jika True, input dalam format xywh, jika False, format xyxy
        GIoU: Jika True, hitung Generalized IoU
        DIoU: Jika True, hitung Distance IoU
        CIoU: Jika True, hitung Complete IoU
        eps: Epsilon untuk stabilitas numerik
        
    Returns:
        Tensor dengan shape (n,) berisi IoU/GIoU/DIoU/CIoU untuk setiap pasangan box
    """
    # Konversi xywh ke xyxy jika diperlukan
    if xywh:
        (x1, y1, w1, h1), (x2, y2, w2, h2) = box1.chunk(4, -1), box2.chunk(4, -1)
        box1_xyxy = torch.cat([x1 - w1 / 2, y1 - h1 / 2, x1 + w1 / 2, y1 + h1 / 2], -1)
        box2_xyxy = torch.cat([x2 - w2 / 2, y2 - h2 / 2, x2 + w2 / 2, y2 + h2 / 2], -1)
    else:
        box1_xyxy, box2_xyxy = box1, box2
    
    # Hitung IoU
    b1_x1, b1_y1, b1_x2, b1_y2 = box1_xyxy.unbind(-1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2_xyxy.unbind(-1)
    
    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)
    
    # Union area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1
    union = w1 * h1 + w2 * h2 - inter + eps
    
    # IoU
    iou = inter / union
    
    if GIoU or DIoU or CIoU:
        # Convex hull (smallest enclosing box)
        c_x1, c_y1 = torch.min(b1_x1, b2_x1), torch.min(b1_y1, b2_y1)
        c_x2, c_y2 = torch.max(b1_x2, b2_x2), torch.max(b1_y2, b2_y2)
        c_w, c_h = c_x2 - c_x1, c_y2 - c_y1
        
        if GIoU:  # Generalized IoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = c_w * c_h + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
        
        if DIoU or CIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            # Center distance
            c1 = (b1_x1 + b1_x2) / 2, (b1_y1 + b1_y2) / 2
            c2 = (b2_x1 + b2_x2) / 2, (b2_y1 + b2_y2) / 2
            d = sum((c1[i] - c2[i]) ** 2 for i in range(2))  # center distance squared
            
            # Diagonal length of the smallest enclosing box
            c_diag = c_w ** 2 + c_h ** 2 + eps
            
            if DIoU:
                return iou - d / c_diag  # DIoU
            
            if CIoU:  # CIoU = IoU - (center_dist²/c_diag² + α*v²)
                # Aspect ratio consistency
                v = (4 / (math.pi ** 2)) * torch.pow(
                    torch.atan(w2 / (h2 + eps)) - torch.atan(w1 / (h1 + eps)), 2
                )
                with torch.no_grad():
                    alpha = v / (1 - iou + v + eps)
                return iou - (d / c_diag + alpha * v)  # CIoU
    
    return iou

def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float = 0.45) -> torch.Tensor:
    """
    Melakukan Non-Maximum Suppression pada bounding boxes.
    
    Args:
        boxes: Tensor dengan shape (n, 4) dalam format xyxy
        scores: Tensor dengan shape (n,) berisi confidence scores
        iou_threshold: Threshold IoU untuk menentukan overlap
        
    Returns:
        Tensor berisi indeks dari boxes yang dipertahankan
    """
    # Jika tidak ada boxes, return empty tensor
    if len(boxes) == 0:
        return torch.zeros(0, dtype=torch.int64, device=boxes.device)
    
    # Konversi ke CPU jika di GPU untuk operasi yang lebih efisien
    cpu_device = torch.device("cpu")
    boxes_cpu = boxes.to(cpu_device) if boxes.is_cuda else boxes
    scores_cpu = scores.to(cpu_device) if scores.is_cuda else scores
    
    # Urutkan scores dari tinggi ke rendah
    _, order = scores_cpu.sort(descending=True)
    
    keep = []
    while order.numel() > 0:
        # Pilih box dengan score tertinggi
        i = order[0].item()
        keep.append(i)
        
        # Jika hanya tersisa satu box, keluar dari loop
        if order.numel() == 1:
            break
            
        # Hitung IoU antara box terpilih dan semua box lainnya
        ious = box_iou(boxes_cpu[i:i+1], boxes_cpu[order[1:]])
        
        # Identifikasi boxes dengan IoU di bawah threshold
        mask = ious[0] <= iou_threshold
        order = order[1:][mask]
    
    # Konversi list ke tensor
    keep = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    
    return keep
