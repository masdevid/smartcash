"""
File: smartcash/model/utils/metrics_nms.py
Deskripsi: Implementasi Non-Maximum Suppression untuk deteksi objek
"""

import torch
from typing import List, Optional

from smartcash.model.utils.metrics.metrics_core import box_iou, xywh2xyxy


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float = 0.45,
    classes: Optional[List[int]] = None,
    max_det: int = 300
) -> List[torch.Tensor]:
    """
    Lakukan Non-Maximum Suppression (NMS) pada prediksi deteksi.
    
    Args:
        prediction: Tensor prediksi dari model (batch, num_detections, 6)
                   Format tiap deteksi: [x1, y1, x2, y2, confidence, class]
        conf_thres: Threshold minimum confidence
        iou_thres: Threshold IoU untuk NMS
        classes: Filter berdasarkan kelas tertentu
        max_det: Jumlah maksimum deteksi per gambar
        
    Returns:
        List deteksi yang sudah melewati NMS untuk setiap gambar
    """
    # Variabel untuk output akhir
    batch_size = prediction.shape[0]
    output = [None] * batch_size
    
    # Proses setiap gambar dalam batch
    for i in range(batch_size):
        # Ambil prediksi untuk gambar i
        pred = prediction[i]
        
        # Filter berdasarkan confidence threshold
        mask = pred[:, 4] > conf_thres
        pred = pred[mask]
        
        # Skip jika tidak ada deteksi yang lolos confidence threshold
        if not pred.shape[0]:
            output[i] = torch.zeros((0, 6), device=prediction.device)
            continue
            
        # Filter berdasarkan class jika specified
        if classes is not None:
            mask = torch.stack([pred[:, 5] == c for c in classes], dim=0).any(dim=0)
            pred = pred[mask]
            
            # Skip jika tidak ada deteksi yang sesuai dengan class
            if not pred.shape[0]:
                output[i] = torch.zeros((0, 6), device=prediction.device)
                continue
        
        # Lakukan NMS
        boxes = pred[:, :4]
        scores = pred[:, 4]
        
        # Gunakan torchvision NMS
        try:
            from torchvision.ops import nms
            keep = nms(boxes, scores, iou_thres)
        except ImportError:
            # Implementasi NMS manual jika torchvision tidak tersedia
            keep = []
            while boxes.shape[0]:
                # Ambil box dengan confidence tertinggi
                max_idx = scores.argmax()
                keep.append(max_idx.item())
                
                # Hapus box tersebut dari daftar
                if boxes.shape[0] == 1:
                    break
                    
                # Hitung IoU antara box dengan confidence tertinggi dan box lainnya
                ious = box_iou(boxes[max_idx], boxes)
                mask = ious <= iou_thres
                mask[max_idx] = 0  # Hapus box yang dipilih
                
                # Filter box berdasarkan IoU
                boxes = boxes[mask]
                scores = scores[mask]
        
        # Batasi deteksi maksimum
        keep = keep[:max_det]
        
        # Update output
        output[i] = pred[keep]
    
    return output


def apply_classic_nms(boxes, scores, iou_threshold=0.5):
    """
    Implementasi classic NMS (Non-Maximum Suppression) menggunakan pendekatan greedy.
    
    Args:
        boxes: Tensor bounding boxes dalam format xyxy
        scores: Tensor confidence scores
        iou_threshold: Threshold IoU untuk suppression
        
    Returns:
        List indeks box yang dipertahankan
    """
    # Urutkan box berdasarkan score (descending)
    order = torch.argsort(scores, descending=True)
    keep = []
    
    while order.numel() > 0:
        # Pilih box dengan score tertinggi
        i = order[0].item()
        keep.append(i)
        
        # Jika hanya ada satu box tersisa, selesai
        if order.numel() == 1:
            break
        
        # Hitung IoU antara box terpilih dengan semua box lainnya
        ious = box_iou(boxes[i:i+1], boxes[order[1:]])
        
        # Ambil box yang IoU-nya di bawah threshold
        mask = ious <= iou_threshold
        order = order[1:][mask]
    
    return keep


def soft_nms(boxes, scores, sigma=0.5, score_threshold=0.001, method='gaussian'):
    """
    Implementasi Soft-NMS untuk mengurangi false negatives dalam deteksi.
    
    Args:
        boxes: Tensor bounding boxes dalam format xyxy
        scores: Tensor confidence scores
        sigma: Parameter untuk penurunan weight pada Gaussian method
        score_threshold: Threshold minimum untuk mempertahankan box
        method: Metode penurunan score ('linear' atau 'gaussian')
        
    Returns:
        List indeks box yang dipertahankan
    """
    # Konversi ke numpy untuk mempermudah manipulasi
    boxes_np = boxes.cpu().numpy()
    scores_np = scores.cpu().numpy()
    
    # Urutkan box berdasarkan score (descending)
    indices = scores_np.argsort()[::-1]
    boxes_sorted = boxes_np[indices]
    scores_sorted = scores_np[indices]
    
    # List untuk menyimpan box yang lolos
    keep = []
    updated_scores = scores_sorted.copy()
    
    for i in range(len(boxes_sorted)):
        # Current box
        current_box = boxes_sorted[i]
        current_score = updated_scores[i]
        
        # Skip jika score sudah di bawah threshold
        if current_score < score_threshold:
            continue
            
        # Tambahkan box saat ini ke list yang dipertahankan
        keep.append(indices[i])
        
        # Update score box lainnya
        for j in range(i+1, len(boxes_sorted)):
            # Skip jika score sudah di bawah threshold
            if updated_scores[j] < score_threshold:
                continue
                
            # Hitung IoU
            box1 = current_box
            box2 = boxes_sorted[j]
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])
            
            # Area intersection
            w = max(0, x2 - x1)
            h = max(0, y2 - y1)
            intersection = w * h
            
            # Area boxes
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
            
            # IoU
            iou = intersection / (area1 + area2 - intersection)
            
            # Update score berdasarkan metode
            if method == 'linear':
                # Linear: score * (1 - iou)
                updated_scores[j] *= (1.0 - iou)
            else:
                # Gaussian: score * exp(-(iou^2)/sigma)
                updated_scores[j] *= torch.exp(-(iou * iou) / sigma)
    
    return keep