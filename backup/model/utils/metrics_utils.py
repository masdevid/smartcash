"""
File: smartcash/model/utils/metrics_utils.py
Deskripsi: Utilitas untuk menghitung metrik evaluasi deteksi objek seperti mAP, precision, dan recall
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from collections import defaultdict

def calculate_iou(box1, box2):
    """
    Menghitung Intersection over Union (IoU) antara dua bounding box
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU score
    """
    # Koordinat intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    # Luas intersection
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Luas masing-masing box
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # Luas union
    union = box1_area + box2_area - intersection
    
    # IoU
    iou = intersection / union if union > 0 else 0
    
    return iou

def calculate_ap(recalls, precisions):
    """
    Menghitung Average Precision (AP) menggunakan metode 11-point interpolation
    
    Args:
        recalls: List nilai recall
        precisions: List nilai precision
        
    Returns:
        AP score
    """
    # Tambahkan titik awal dan akhir
    mrec = [0] + recalls + [1]
    mpre = [0] + precisions + [0]
    
    # Interpolasi precision
    for i in range(len(mpre) - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    
    # Hitung AP dengan 11-point interpolation
    ap = 0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(np.array(mrec) >= t) == 0:
            p = 0
        else:
            p = np.max(np.array(mpre)[np.array(mrec) >= t])
        ap += p / 11
    
    return ap

def calculate_detection_metrics(
    predictions: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    class_names: List[str],
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Menghitung metrik evaluasi untuk deteksi objek
    
    Args:
        predictions: List prediksi model, setiap item berisi 'boxes', 'scores', 'classes', 'image_path'
        ground_truth: Dictionary ground truth labels
        class_names: List nama kelas
        iou_threshold: Threshold IoU untuk menentukan true positive
        
    Returns:
        Dict berisi metrik evaluasi (mAP, precision, recall per kelas)
    """
    # Cek apakah ground truth tersedia
    if not ground_truth.get('available', False):
        return {
            'map': 0,
            'class_metrics': {},
            'overall_metrics': {
                'precision': 0,
                'recall': 0,
                'f1_score': 0
            },
            'warning': 'Ground truth tidak tersedia, metrik tidak dapat dihitung'
        }
    
    # Inisialisasi metrik per kelas
    class_metrics = {}
    for class_name in class_names:
        class_metrics[class_name] = {
            'true_positives': [],
            'false_positives': [],
            'gt_count': 0,
            'pred_count': 0,
            'confidence_scores': []
        }
    
    # Proses setiap prediksi
    for pred in predictions:
        image_path = pred.get('image_path', '')
        pred_boxes = pred.get('boxes', [])
        pred_scores = pred.get('scores', [])
        pred_classes = pred.get('classes', [])
        
        # Skip jika tidak ada prediksi
        if not pred_boxes or len(pred_boxes) == 0:
            continue
        
        # Dapatkan ground truth untuk gambar ini
        gt_data = ground_truth.get('labels', {}).get(image_path, None)
        
        # Jika tidak ada ground truth untuk gambar ini, semua prediksi adalah false positive
        if gt_data is None:
            for i, cls_idx in enumerate(pred_classes):
                class_name = class_names[cls_idx]
                class_metrics[class_name]['false_positives'].append(1)
                class_metrics[class_name]['true_positives'].append(0)
                class_metrics[class_name]['confidence_scores'].append(pred_scores[i])
                class_metrics[class_name]['pred_count'] += 1
            continue
        
        # Dapatkan ground truth boxes dan classes
        gt_boxes = gt_data.get('boxes', [])
        gt_classes = gt_data.get('classes', [])
        
        # Tambahkan ke gt_count
        for cls_idx in gt_classes:
            if cls_idx < len(class_names):
                class_name = class_names[cls_idx]
                class_metrics[class_name]['gt_count'] += 1
        
        # Tandai ground truth yang sudah di-match
        matched_gt = [False] * len(gt_boxes)
        
        # Untuk setiap prediksi, cari ground truth yang cocok
        for i, (box, score, cls_idx) in enumerate(zip(pred_boxes, pred_scores, pred_classes)):
            if cls_idx >= len(class_names):
                continue
                
            class_name = class_names[cls_idx]
            class_metrics[class_name]['pred_count'] += 1
            class_metrics[class_name]['confidence_scores'].append(score)
            
            # Cari ground truth dengan IoU tertinggi
            max_iou = -1
            max_idx = -1
            
            for j, (gt_box, gt_cls) in enumerate(zip(gt_boxes, gt_classes)):
                # Skip jika kelas berbeda atau sudah di-match
                if gt_cls != cls_idx or matched_gt[j]:
                    continue
                
                # Hitung IoU
                iou = calculate_iou(box, gt_box)
                
                # Update jika IoU lebih tinggi
                if iou > max_iou:
                    max_iou = iou
                    max_idx = j
            
            # Jika ada match dengan IoU > threshold, true positive
            if max_iou >= iou_threshold and max_idx >= 0:
                class_metrics[class_name]['true_positives'].append(1)
                class_metrics[class_name]['false_positives'].append(0)
                matched_gt[max_idx] = True
            else:
                # Jika tidak, false positive
                class_metrics[class_name]['true_positives'].append(0)
                class_metrics[class_name]['false_positives'].append(1)
    
    # Hitung precision, recall, AP untuk setiap kelas
    mean_ap = 0
    valid_classes = 0
    
    for class_name in class_names:
        metrics = class_metrics[class_name]
        
        # Skip jika tidak ada ground truth atau prediksi
        if metrics['gt_count'] == 0 and metrics['pred_count'] == 0:
            metrics['precision'] = 0
            metrics['recall'] = 0
            metrics['ap'] = 0
            continue
        
        # Urutkan berdasarkan confidence score
        indices = np.argsort(-np.array(metrics['confidence_scores']))
        tp = np.array(metrics['true_positives'])[indices]
        fp = np.array(metrics['false_positives'])[indices]
        
        # Hitung cumulative TP dan FP
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        # Hitung precision dan recall
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-10)
        recalls = tp_cumsum / (metrics['gt_count'] + 1e-10)
        
        # Hitung AP
        ap = calculate_ap(recalls.tolist(), precisions.tolist())
        
        # Simpan metrik
        metrics['precision'] = precisions[-1] if len(precisions) > 0 else 0
        metrics['recall'] = recalls[-1] if len(recalls) > 0 else 0
        metrics['ap'] = ap
        
        # Tambahkan ke mAP jika ada ground truth
        if metrics['gt_count'] > 0:
            mean_ap += ap
            valid_classes += 1
    
    # Hitung mAP
    mean_ap = mean_ap / valid_classes if valid_classes > 0 else 0
    
    # Hitung metrik overall
    total_tp = sum([sum(metrics['true_positives']) for metrics in class_metrics.values()])
    total_fp = sum([sum(metrics['false_positives']) for metrics in class_metrics.values()])
    total_gt = sum([metrics['gt_count'] for metrics in class_metrics.values()])
    
    overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    overall_recall = total_tp / total_gt if total_gt > 0 else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0
    
    # Hasil akhir
    return {
        'map': mean_ap,
        'class_metrics': class_metrics,
        'overall_metrics': {
            'precision': overall_precision,
            'recall': overall_recall,
            'f1_score': overall_f1,
            'total_gt': total_gt,
            'total_predictions': total_tp + total_fp,
            'true_positives': total_tp,
            'false_positives': total_fp
        }
    }
