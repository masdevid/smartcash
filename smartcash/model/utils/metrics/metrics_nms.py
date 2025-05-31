"""
File: smartcash/model/utils/metrics/metrics_nms.py
Deskripsi: Implementasi Non-Maximum Suppression untuk post-processing hasil deteksi
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional, Union

def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300,
    multi_label: bool = False
) -> List[torch.Tensor]:
    """
    Melakukan Non-Maximum Suppression (NMS) pada prediksi dari model deteksi.
    
    Args:
        prediction: Tensor output dari model dengan shape (batch_size, num_boxes, 5+num_classes)
                   Format: [x, y, w, h, conf, class_1, class_2, ...]
        conf_thres: Confidence threshold untuk memfilter deteksi
        iou_thres: IoU threshold untuk NMS
        classes: Filter deteksi hanya untuk kelas tertentu
        max_det: Jumlah maksimum deteksi per gambar
        multi_label: Apakah satu box dapat memiliki multiple labels
        
    Returns:
        List[torch.Tensor]: List deteksi untuk setiap gambar dalam batch
                           Format: [x1, y1, x2, y2, conf, class_id]
    """
    # Inisialisasi
    bs = prediction.shape[0]  # batch size
    nc = prediction.shape[2] - 5  # jumlah kelas
    xc = prediction[..., 4] > conf_thres  # kandidat dengan confidence > threshold
    
    # Settings
    max_wh = 7680  # (pixels) maksimum width dan height
    max_nms = 30000  # maksimum box per kelas
    redundant = True  # require redundant detections
    merge = False  # gunakan weighted merging untuk NMS
    
    output = [torch.zeros((0, 6), device=prediction.device)] * bs
    
    # Per image
    for xi, x in enumerate(prediction):  # gambar index, gambar inference
        # Terapkan confidence threshold
        x = x[xc[xi]]  # confidence
        
        # Jika tidak ada box, lanjut ke gambar berikutnya
        if not x.shape[0]:
            continue
            
        # Hitung scores = conf * class probability
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf
        
        # Box (center x, center y, width, height) ke (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        
        # Deteksi multi-label atau single label
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # hanya kelas dengan probabilitas tertinggi
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
        
        # Filter berdasarkan kelas
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]
            
        # Cek jumlah box
        n = x.shape[0]  # jumlah box
        if not n:  # tidak ada box
            continue
        elif n > max_nms:  # excess boxes
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence
            
        # Batched NMS
        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision_nms(boxes, scores, iou_thres)  # NMS
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
            
        output[xi] = x[i]
        
    return output

def torchvision_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """
    Implementasi NMS menggunakan torchvision.ops.nms jika tersedia, atau fallback ke implementasi manual.
    
    Args:
        boxes: Bounding boxes dalam format xyxy
        scores: Confidence scores
        iou_thres: IoU threshold
        
    Returns:
        torch.Tensor: Indeks dari box yang dipertahankan
    """
    try:
        # Coba gunakan torchvision.ops.nms jika tersedia
        import torchvision
        return torchvision.ops.nms(boxes, scores, iou_thres)
    except (ImportError, AttributeError):
        # Fallback ke implementasi manual
        return manual_nms(boxes, scores, iou_thres)

def manual_nms(boxes: torch.Tensor, scores: torch.Tensor, iou_thres: float) -> torch.Tensor:
    """
    Implementasi manual NMS untuk fallback jika torchvision tidak tersedia.
    
    Args:
        boxes: Bounding boxes dalam format xyxy
        scores: Confidence scores
        iou_thres: IoU threshold
        
    Returns:
        torch.Tensor: Indeks dari box yang dipertahankan
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
        mask = ious[0] <= iou_thres
        order = order[1:][mask]
    
    # Konversi list ke tensor dan kembalikan ke device asli
    keep = torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    
    return keep

def box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Menghitung IoU antara dua set bounding boxes.
    
    Args:
        box1: Tensor dengan shape (n, 4) dalam format xyxy
        box2: Tensor dengan shape (m, 4) dalam format xyxy
        
    Returns:
        Tensor dengan shape (n, m) berisi IoU untuk setiap pasangan box
    """
    # Hitung area untuk setiap box
    area1 = box_area(box1)  # (n,)
    area2 = box_area(box2)  # (m,)
    
    # Broadcast untuk mendapatkan koordinat intersection
    # box1: (n, 4) -> (n, 1, 4)
    # box2: (m, 4) -> (1, m, 4)
    box1 = box1[:, None, :]  # (n, 1, 4)
    box2 = box2[None, :, :]  # (1, m, 4)
    
    # Hitung koordinat intersection
    top_left = torch.max(box1[..., :2], box2[..., :2])  # (n, m, 2)
    bottom_right = torch.min(box1[..., 2:], box2[..., 2:])  # (n, m, 2)
    
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

def box_area(box: torch.Tensor) -> torch.Tensor:
    """
    Menghitung area dari bounding box dalam format xyxy.
    
    Args:
        box: Tensor dengan shape (..., 4) dalam format xyxy
        
    Returns:
        Tensor dengan shape (...) berisi area untuk setiap box
    """
    return (box[..., 2] - box[..., 0]) * (box[..., 3] - box[..., 1])

def xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    """
    Konversi bounding box dari format xywh ke xyxy.
    
    Args:
        x: Tensor dengan shape (..., 4) dalam format (x_center, y_center, width, height)
        
    Returns:
        Tensor dengan shape (..., 4) dalam format (x1, y1, x2, y2)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # x1 = x_center - width/2
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # y1 = y_center - height/2
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # x2 = x_center + width/2
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # y2 = y_center + height/2
    return y

def xyxy2xywh(x: torch.Tensor) -> torch.Tensor:
    """
    Konversi bounding box dari format xyxy ke xywh.
    
    Args:
        x: Tensor dengan shape (..., 4) dalam format (x1, y1, x2, y2)
        
    Returns:
        Tensor dengan shape (..., 4) dalam format (x_center, y_center, width, height)
    """
    y = x.clone() if isinstance(x, torch.Tensor) else torch.tensor(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x_center = (x1 + x2) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y_center = (y1 + y2) / 2
    y[:, 2] = x[:, 2] - x[:, 0]         # width = x2 - x1
    y[:, 3] = x[:, 3] - x[:, 1]         # height = y2 - y1
    return y
