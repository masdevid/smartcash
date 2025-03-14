"""
File: smartcash/model/utils/metrics/metrics_calculator.py
Deskripsi: Kelas untuk menghitung dan melacak metrik evaluasi model
"""

import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

from smartcash.model.utils.metrics.core_metrics import box_iou, xywh2xyxy


class MetricsCalculator:
    """Kelas untuk menghitung metrik evaluasi pada model deteksi objek."""
    
    def __init__(self, num_classes: int = 80, iou_thres: float = 0.5):
        """
        Inisialisasi MetricsCalculator.
        
        Args:
            num_classes: Jumlah kelas untuk deteksi
            iou_thres: Threshold IoU untuk menentukan true positive
        """
        self.num_classes = num_classes
        self.iou_thres = iou_thres
        self.reset()
    
    def reset(self) -> None:
        """Reset statistik evaluasi."""
        self.stats = {}
        self.class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'fn': 0, 'count': 0})
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        self.processed_batches = 0
    
    def process_batch(
        self,
        detections: torch.Tensor,
        targets: torch.Tensor,
        class_map: Optional[Dict[int, str]] = None
    ) -> Dict:
        """
        Proses satu batch deteksi dan update statistik.
        
        Args:
            detections: Tensor deteksi [N, 6] (xyxy, conf, cls)
            targets: Tensor target [M, 6] (batch_idx, cls, xywh)
            class_map: Dictionary mapping id kelas ke nama kelas
            
        Returns:
            Statistik batch
        """
        batch_stats = {
            'true_positives': 0,
            'false_positives': 0,
            'false_negatives': 0,
            'precision': 0,
            'recall': 0,
            'f1': 0
        }
        
        # Kasus khusus: tidak ada deteksi dan tidak ada target
        if detections.shape[0] == 0 and targets.shape[0] == 0:
            self.processed_batches += 1
            return batch_stats
        
        # Kasus khusus: ada target tapi tidak ada deteksi (semua false negatives)
        if detections.shape[0] == 0 and targets.shape[0] > 0:
            batch_stats['false_negatives'] = targets.shape[0]
            
            # Update statistik per kelas
            for i in range(targets.shape[0]):
                cls = int(targets[i, 1])
                self.class_stats[cls]['fn'] += 1
                self.class_stats[cls]['count'] += 1
                
            self.processed_batches += 1
            return batch_stats
        
        # Kasus khusus: ada deteksi tapi tidak ada target (semua false positives)
        if detections.shape[0] > 0 and targets.shape[0] == 0:
            batch_stats['false_positives'] = detections.shape[0]
            
            # Update statistik per kelas
            for i in range(detections.shape[0]):
                cls = int(detections[i, 5])
                self.class_stats[cls]['fp'] += 1
                
            self.processed_batches += 1
            return batch_stats
        
        # Konversi target dari xywh ke xyxy untuk perhitungan IoU
        targets_xyxy = targets.clone()
        if targets.shape[0] > 0:
            targets_xyxy[:, 2:] = xywh2xyxy(targets[:, 2:])
        
        # Matching deteksi dengan target
        matched_indices = []
        unmatched_detections = list(range(detections.shape[0]))
        unmatched_targets = list(range(targets.shape[0]))
        
        # Hitung IoU untuk setiap pasangan deteksi-target
        for target_idx in range(targets.shape[0]):
            target_cls = int(targets[target_idx, 1])
            best_iou = self.iou_thres
            best_detection_idx = -1
            
            for detection_idx in unmatched_detections:
                detection_cls = int(detections[detection_idx, 5])
                
                # Filter berdasarkan kelas
                if detection_cls != target_cls:
                    continue
                
                # Hitung IoU
                iou = box_iou(
                    detections[detection_idx, :4].unsqueeze(0),
                    targets_xyxy[target_idx, 2:].unsqueeze(0)
                ).squeeze()
                
                # Update jika IoU lebih baik
                if iou > best_iou:
                    best_iou = iou
                    best_detection_idx = detection_idx
            
            # Jika ditemukan match yang baik
            if best_detection_idx != -1:
                matched_indices.append((best_detection_idx, target_idx))
                unmatched_detections.remove(best_detection_idx)
                unmatched_targets.remove(target_idx)
        
        # Hitung statistik batch
        batch_stats['true_positives'] = len(matched_indices)
        batch_stats['false_positives'] = len(unmatched_detections)
        batch_stats['false_negatives'] = len(unmatched_targets)
        
        # Update confusion matrix dan statistik per kelas
        for det_idx, tgt_idx in matched_indices:
            det_cls = int(detections[det_idx, 5])
            tgt_cls = int(targets[tgt_idx, 1])
            self.confusion_matrix[tgt_cls, det_cls] += 1
            self.class_stats[tgt_cls]['tp'] += 1
            self.class_stats[tgt_cls]['count'] += 1
        
        # Update statistik untuk unmatched detections (false positives)
        for det_idx in unmatched_detections:
            det_cls = int(detections[det_idx, 5])
            self.class_stats[det_cls]['fp'] += 1
            
        # Update statistik untuk unmatched targets (false negatives)
        for tgt_idx in unmatched_targets:
            tgt_cls = int(targets[tgt_idx, 1])
            self.class_stats[tgt_cls]['fn'] += 1
            self.class_stats[tgt_cls]['count'] += 1
        
        # Hitung precision, recall, dan F1 untuk batch
        tp = batch_stats['true_positives']
        fp = batch_stats['false_positives']
        fn = batch_stats['false_negatives']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        batch_stats['precision'] = precision
        batch_stats['recall'] = recall
        batch_stats['f1'] = f1
        
        self.processed_batches += 1
        return batch_stats
    
    def calculate_metrics(self) -> Dict:
        """
        Hitung metrik final setelah memproses semua batch.
        
        Returns:
            Dictionary berisi metrik evaluasi
        """
        if self.processed_batches == 0:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'mAP': 0,
                'processed_batches': 0
            }
        
        # Hitung metrik global
        true_positives = sum(stats['tp'] for stats in self.class_stats.values())
        false_positives = sum(stats['fp'] for stats in self.class_stats.values())
        false_negatives = sum(stats['fn'] for stats in self.class_stats.values())
        
        # Hitung precision, recall, dan F1 global
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # Hitung metrik per kelas
        class_metrics = {}
        for cls, stats in self.class_stats.items():
            tp = stats['tp']
            fp = stats['fp']
            fn = stats['fn']
            count = stats['count']
            
            # Precision, recall, dan F1 per kelas
            cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            cls_f1 = 2 * cls_precision * cls_recall / (cls_precision + cls_recall) if (cls_precision + cls_recall) > 0 else 0
            
            class_metrics[cls] = {
                'precision': cls_precision,
                'recall': cls_recall,
                'f1': cls_f1,
                'count': count
            }
        
        # Hitung mAP (mean Average Precision)
        # mAP sederhana: rata-rata precision per kelas
        ap_per_class = [metrics['precision'] for cls, metrics in class_metrics.items() if metrics['count'] > 0]
        mAP = sum(ap_per_class) / len(ap_per_class) if ap_per_class else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'mAP': mAP,
            'class_metrics': class_metrics,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'processed_batches': self.processed_batches,
            'confusion_matrix': self.confusion_matrix
        }
    
    def get_confusion_matrix(self, normalized: bool = False) -> np.ndarray:
        """
        Dapatkan confusion matrix dari hasil evaluasi.
        
        Args:
            normalized: Apakah mengembalikan matrix yang dinormalisasi
            
        Returns:
            Confusion matrix
        """
        if normalized:
            # Normalisasi per baris (untuk setiap ground truth class)
            row_sums = self.confusion_matrix.sum(axis=1, keepdims=True)
            normalized_matrix = np.zeros_like(self.confusion_matrix, dtype=float)
            
            # Hindari pembagian dengan nol
            valid_rows = row_sums > 0
            normalized_matrix[valid_rows.flatten()] = (
                self.confusion_matrix[valid_rows.flatten()] / row_sums[valid_rows]
            )
            
            return normalized_matrix
        else:
            return self.confusion_matrix
    
    def get_class_metrics(self, class_id: int) -> Dict:
        """
        Dapatkan metrik untuk kelas tertentu.
        
        Args:
            class_id: ID kelas
            
        Returns:
            Dictionary metrik untuk kelas tersebut
        """
        if class_id not in self.class_stats:
            return {
                'precision': 0,
                'recall': 0,
                'f1': 0,
                'count': 0
            }
        
        stats = self.class_stats[class_id]
        tp = stats['tp']
        fp = stats['fp']
        fn = stats['fn']
        count = stats['count']
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'count': count
        }