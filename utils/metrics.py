# File: utils/metrics.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi perhitungan metrik evaluasi untuk model deteksi

import torch
import numpy as np
from typing import Dict
from collections import defaultdict
import time

class MetricsCalculator:
    """Kalkulasi metrik evaluasi untuk model deteksi mata uang"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        """Reset semua metrik"""
        self.true_positives = defaultdict(int)
        self.false_positives = defaultdict(int)
        self.false_negatives = defaultdict(int)
        self.inference_times = []
        
    def update(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> Dict:
        """
        Update metrik dengan batch prediksi baru
        Args:
            predictions: Prediksi model [batch_size, num_pred, 6] (x,y,w,h,conf,cls)
            targets: Ground truth [batch_size, num_targets, 6]
        Returns:
            Dict metrik untuk batch ini
        """
        batch_metrics = {}
        
        # Measure inference time
        start_time = time.perf_counter()
        
        # Calculate IoU matrix antara prediksi dan target
        ious = self._calculate_iou_matrix(predictions, targets)
        
        # Update confusion matrix berdasarkan IoU threshold
        self._update_confusion_matrix(predictions, targets, ious)
        
        # Calculate batch metrics
        batch_metrics.update(self._calculate_batch_metrics(predictions, targets))
        
        # Record inference time
        inference_time = (time.perf_counter() - start_time) * 1000  # ke ms
        self.inference_times.append(inference_time)
        batch_metrics['inference_time'] = inference_time
        
        return batch_metrics
        
    def compute(self) -> Dict:
        """
        Hitung metrik final
        Returns:
            Dict berisi semua metrik evaluasi
        """
        metrics = {}
        
        # Precision & Recall per kelas
        for cls in self.true_positives.keys():
            tp = self.true_positives[cls]
            fp = self.false_positives[cls]
            fn = self.false_negatives[cls]
            
            # Avoid division by zero
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics[f'precision_cls_{cls}'] = precision
            metrics[f'recall_cls_{cls}'] = recall
            
        # Overall metrics
        total_tp = sum(self.true_positives.values())
        total_fp = sum(self.false_positives.values())
        total_fn = sum(self.false_negatives.values())
        
        metrics.update({
            'precision': total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0,
            'recall': total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0,
            'accuracy': total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0,
            'f1': self._calculate_f1(metrics['precision'], metrics['recall']),
            'mAP': self._calculate_map(),
            'inference_time': np.mean(self.inference_times)
        })
        
        return metrics
        
    def _calculate_iou_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """Hitung IoU matrix antara predictions dan targets"""
        # Implementation of IoU calculation
        # Returns matrix of shape [num_pred, num_targets]
        pass
        
    def _update_confusion_matrix(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        ious: torch.Tensor,
        iou_threshold: float = 0.5
    ) -> None:
        """Update confusion matrix berdasarkan IoU threshold"""
        # Match predictions dengan targets
        matches = ious > iou_threshold
        
        # Update metrics
        for pred_idx, target_idx in zip(*matches.nonzero()):
            pred_cls = predictions[pred_idx, 5]
            target_cls = targets[target_idx, 5]
            
            if pred_cls == target_cls:
                self.true_positives[pred_cls.item()] += 1
            else:
                self.false_positives[pred_cls.item()] += 1
                self.false_negatives[target_cls.item()] += 1
                
    def _calculate_f1(
        self,
        precision: float,
        recall: float
    ) -> float:
        """Hitung F1 score dari precision dan recall"""
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
    def _calculate_map(self) -> float:
        """Calculate mean Average Precision"""
        aps = []
        
        for cls in self.true_positives.keys():
            # Calculate AP untuk setiap kelas
            ap = self._calculate_average_precision(cls)
            aps.append(ap)
            
        return np.mean(aps) if aps else 0
        
    def _calculate_average_precision(
        self,
        cls: int
    ) -> float:
        """Calculate Average Precision untuk kelas tertentu"""
        # Implementation of AP calculation per class
        pass