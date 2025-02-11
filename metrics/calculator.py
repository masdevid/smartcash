# File: src/metrics/calculator.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi metrik evaluasi lengkap dengan cross-validation

import torch
import numpy as np
from sklearn.model_selection import KFold
from scipy import stats
from dataclasses import dataclass
from typing import Dict, List, Optional
from utils.logging import ColoredLogger

@dataclass
class MetricResult:
    mAP: float
    precision: float
    recall: float
    f1_score: float
    feature_quality: Optional[float] = None
    inference_time: Optional[float] = None

class MetricsCalculator:
    def __init__(self, num_classes=7, iou_thresholds=[0.5, 0.75, 0.9]):
        self.logger = ColoredLogger('MetricsCalc')
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.cv = KFold(n_splits=5, shuffle=True)
        
    def compute_metrics(self, pred: torch.Tensor, target: torch.Tensor,
                       features: Optional[torch.Tensor] = None) -> MetricResult:
        """Hitung metrik dengan cross-validation"""
        metrics = []
        
        # Split data untuk cross-validation
        splits = self.cv.split(pred)
        for train_idx, val_idx in splits:
            fold_metrics = self._compute_fold_metrics(
                pred[train_idx], target[train_idx],
                pred[val_idx], target[val_idx]
            )
            metrics.append(fold_metrics)
            
        # Aggregate hasil cross-validation
        mean_metrics = self._aggregate_metrics(metrics)
        
        # Tambahkan feature quality jika tersedia
        if features is not None:
            mean_metrics.feature_quality = self._compute_feature_quality(features)
            
        return mean_metrics
    
    def _compute_fold_metrics(self, train_pred, train_target, 
                            val_pred, val_target) -> MetricResult:
        """Hitung metrik untuk satu fold"""
        results = {}
        
        # Hitung mAP
        maps = []
        for iou_thresh in self.iou_thresholds:
            ap = self._compute_average_precision(
                val_pred, val_target, iou_thresh
            )
            maps.append(ap)
        results['mAP'] = np.mean(maps)
        
        # Hitung precision, recall, F1
        matched = self._match_predictions(val_pred, val_target)
        tp = np.sum(matched)
        fp = len(val_pred) - tp
        fn = len(val_target) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return MetricResult(
            mAP=results['mAP'],
            precision=precision,
            recall=recall,
            f1_score=f1
        )
    
    def _compute_average_precision(self, pred, target, iou_thresh):
        """Hitung Average Precision untuk threshold IoU tertentu"""
        # Sort predictions by confidence
        conf_sort = torch.argsort(pred[..., 4], descending=True)
        pred = pred[conf_sort]
        
        # Calculate precision and recall points
        precisions, recalls = [], []
        for i in range(len(pred)):
            matched = self._match_predictions(pred[:i+1], target, iou_thresh)
            tp = np.sum(matched)
            precision = tp / (i + 1)
            recall = tp / len(target) if len(target) > 0 else 0
            precisions.append(precision)
            recalls.append(recall)
            
        # Compute AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11.0
            
        return ap
    
    def _compute_feature_quality(self, features: torch.Tensor) -> float:
        """Evaluasi kualitas feature maps"""
        # Normalize features
        features = features / (features.norm(dim=1, keepdim=True) + 1e-8)
        
        # Compute activation statistics
        mean_activation = features.mean().item()
        std_activation = features.std().item()
        sparsity = (features == 0).float().mean().item()
        
        # Combine metrics
        quality_score = (mean_activation + std_activation) * (1 - sparsity)
        return quality_score
    
    def _match_predictions(self, pred, target, iou_thresh=0.5):
        """Match predictions dengan ground truth"""
        if len(pred) == 0 or len(target) == 0:
            return []
            
        # Calculate IoU matrix
        ious = torch.zeros((len(pred), len(target)))
        for i, p in enumerate(pred):
            for j, t in enumerate(target):
                ious[i, j] = self._calculate_iou(p[:4], t[:4])
                
        # Match predictions
        matched = []
        matched_targets = set()
        
        for pred_idx in range(len(pred)):
            best_iou = 0
            best_target = None
            
            for tgt_idx in range(len(target)):
                if tgt_idx in matched_targets:
                    continue
                    
                if ious[pred_idx, tgt_idx] > best_iou:
                    best_iou = ious[pred_idx, tgt_idx]
                    best_target = tgt_idx
                    
            if best_iou > iou_thresh:
                matched.append(True)
                matched_targets.add(best_target)
            else:
                matched.append(False)
                
        return matched
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
    
    def _aggregate_metrics(self, metrics: List[MetricResult]) -> MetricResult:
        """Aggregate metrics dari multiple folds"""
        agg_metrics = {}
        for field in MetricResult.__dataclass_fields__:
            values = [getattr(m, field) for m in metrics if getattr(m, field) is not None]
            if values:
                agg_metrics[field] = float(np.mean(values))
            else:
                agg_metrics[field] = None
                
        return MetricResult(**agg_metrics)