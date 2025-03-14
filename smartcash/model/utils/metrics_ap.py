"""
File: smartcash/model/utils/metrics_ap.py
Deskripsi: Fungsi-fungsi untuk menghitung Average Precision dan metrik performa
"""

import numpy as np
from typing import Tuple

from smartcash.model.utils.metrics_core import compute_ap


def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    eps: float = 1e-16
) -> Tuple:
    """
    Hitung Average Precision (AP) per kelas.
    
    Args:
        tp: Array true positives
        conf: Array confidence scores
        pred_cls: Array predicted class indices
        target_cls: Array target class indices
        eps: Epsilon untuk menghindari pembagian dengan nol
        
    Returns:
        Tuple (precision, recall, AP, f1, unique_classes)
    """
    # Sort berdasarkan confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Temukan kelas unik
    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(unique_classes)
    
    # Inisialisasi arrays
    ap = np.zeros((nc))
    precision = np.zeros((nc))
    recall = np.zeros((nc))
    f1 = np.zeros((nc))
    
    # Hitung AP untuk setiap kelas
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_p = i.sum()  # Jumlah prediksi untuk kelas ini
        n_gt = (target_cls == c).sum()  # Jumlah ground truth
        
        # Skip jika tidak ada deteksi atau ground truth
        if n_p == 0 or n_gt == 0:
            continue
        
        # Hitung FP dan TP kumulatif
        fpc = (1 - tp[i]).cumsum()
        tpc = tp[i].cumsum()
        
        # Hitung recall dan precision
        recall_curve = tpc / (n_gt + eps)
        precision_curve = tpc / (tpc + fpc)
        
        # Simpan nilai recall dan precision terakhir
        recall[ci] = recall_curve[-1]
        precision[ci] = precision_curve[-1]
        
        # Hitung AP menggunakan metode VOC2007 (11-point interpolation)
        ap[ci] = compute_ap(recall_curve, precision_curve)
        
        # Hitung F1 score
        f1[ci] = 2 * precision[ci] * recall[ci] / (precision[ci] + recall[ci] + eps)
    
    return precision, recall, ap, f1, unique_classes.astype(int)


def mean_average_precision(
    detections: np.ndarray,
    targets: np.ndarray,
    iou_thresholds: np.ndarray = np.linspace(0.5, 0.95, 10),
    class_names: dict = None
) -> dict:
    """
    Hitung mAP berdasarkan berbagai threshold IoU (COCO-style).
    
    Args:
        detections: Array deteksi [N, 7] (batch_idx, x1, y1, x2, y2, conf, class)
        targets: Array targets [M, 6] (batch_idx, class, x, y, w, h)
        iou_thresholds: Array threshold IoU untuk evaluasi
        class_names: Dict mapping class_id ke nama kelas
        
    Returns:
        Dictionary metrik mAP
    """
    # Konversi targets dari xywh ke xyxy
    targets_xyxy = targets.copy()
    targets_xyxy[:, 2:] = xywh2xyxy(targets[:, 2:6])
    
    # Inisialisasi dicts untuk statistik
    stats = {}
    ap_per_class_per_iou = {}
    
    # Hitung AP pada setiap threshold IoU
    for iou_threshold in iou_thresholds:
        # Inisialisasi arrays untuk TP, confidence, dan class
        tp = np.zeros((len(detections)))
        conf = detections[:, 5]
        pred_cls = detections[:, 6]
        
        # Identifikasi true positives
        for i, detection in enumerate(detections):
            batch_idx = int(detection[0])
            pred_bbox = detection[1:5]
            pred_class = int(detection[6])
            
            # Filter target berdasarkan batch_idx dan class
            mask = (targets[:, 0] == batch_idx) & (targets[:, 1] == pred_class)
            target_bboxes = targets_xyxy[mask, 2:6]
            
            # Skip jika tidak ada target untuk kelas ini di batch ini
            if len(target_bboxes) == 0:
                continue
                
            # Hitung IoU dengan semua target
            ious = box_iou(pred_bbox, target_bboxes)
            max_iou = ious.max()
            max_iou_idx = ious.argmax()
            
            # Mark sebagai TP jika IoU > threshold dan belum dideteksi
            if max_iou > iou_threshold:
                tp[i] = 1
        
        # Hitung precision, recall, AP per kelas
        precision, recall, ap, f1, unique_classes = ap_per_class(
            tp, conf, pred_cls, targets[:, 1]
        )
        
        # Simpan ke dict
        ap_per_class_per_iou[iou_threshold] = {
            'precision': precision,
            'recall': recall,
            'ap': ap,
            'f1': f1,
            'unique_classes': unique_classes
        }
    
    # Rata-rata AP pada semua kelas dan IoU thresholds
    all_ap = []
    for iou_threshold, metrics in ap_per_class_per_iou.items():
        all_ap.extend(metrics['ap'])
    
    # Hitung mAP overall
    map_all = np.mean(all_ap) if len(all_ap) > 0 else 0
    
    # Hitung mAP untuk IoU=0.5
    map_50 = np.mean(ap_per_class_per_iou[0.5]['ap']) if 0.5 in ap_per_class_per_iou else 0
    
    # Hitung mAP untuk IoU=0.5:0.95
    map_per_iou = [np.mean(ap_per_class_per_iou[iou]['ap']) for iou in iou_thresholds if iou in ap_per_class_per_iou]
    map_50_95 = np.mean(map_per_iou) if map_per_iou else 0
    
    # Hitung AP per kelas pada IoU=0.5
    ap_per_class = {}
    if 0.5 in ap_per_class_per_iou:
        for i, cls_id in enumerate(ap_per_class_per_iou[0.5]['unique_classes']):
            class_name = class_names[cls_id] if class_names and cls_id in class_names else f'class_{cls_id}'
            ap_per_class[class_name] = ap_per_class_per_iou[0.5]['ap'][i]
    
    # Hitung hasil akhir
    stats = {
        'mAP@0.5': map_50,
        'mAP@0.5:0.95': map_50_95,
        'mAP_all': map_all,
        'AP_per_class': ap_per_class
    }
    
    return stats


def precision_recall_curve(tp, conf, pred_cls, target_cls, eps=1e-16):
    """
    Hitung kurva precision-recall untuk visualisasi.
    
    Args:
        tp: Array true positives
        conf: Array confidence scores
        pred_cls: Array predicted class indices
        target_cls: Array target class indices
        eps: Epsilon untuk menghindari pembagian dengan nol
        
    Returns:
        Dict berisi kurva precision-recall per kelas
    """
    # Sort berdasarkan confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]
    
    # Temukan kelas unik
    unique_classes = np.unique(target_cls)
    
    # Dict untuk menyimpan hasil
    curves = {}
    
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Jumlah ground truth
        n_p = i.sum()  # Jumlah prediksi
        
        if n_p == 0 or n_gt == 0:
            continue
            
        # Accumulate FPs and TPs
        fpc = (1 - tp[i]).cumsum()
        tpc = tp[i].cumsum()
        
        # Recall
        recall = tpc / (n_gt + eps)
        
        # Precision
        precision = tpc / (tpc + fpc)
        
        # Simpan ke dict
        curves[int(c)] = {
            'precision': precision,
            'recall': recall,
            'confidence': conf[i],
            'f1': 2 * precision * recall / (precision + recall + eps),
            'n_predictions': n_p,
            'n_ground_truth': n_gt
        }
    
    return curves