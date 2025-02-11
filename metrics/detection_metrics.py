# File: src/metrics/detection_metrics.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi metrik evaluasi komprehensif untuk SmartCash Detector

import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy import stats
from utils.logging import ColoredLogger

@dataclass
class DetectionMetrics:
    """Kelas untuk menyimpan hasil metrik deteksi"""
    precision: float
    recall: float
    f1_score: float
    map: float
    inference_time: float
    feature_quality: float
    
class MetricsCalculator:
    def __init__(self, num_classes: int = 7, iou_thresholds: List[float] = None):
        self.logger = ColoredLogger('MetricsCalculator')
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds or [0.5, 0.75, 0.9]
        self.reset_stats()
        
    def reset_stats(self):
        """Reset statistik untuk evaluasi baru"""
        self.predictions = {
            thresh: {cls: [] for cls in range(self.num_classes)}
            for thresh in self.iou_thresholds
        }
        self.ground_truths = {cls: [] for cls in range(self.num_classes)}
        self.inference_times = []
        self.feature_qualities = []
        
    def update(self, 
              predictions: torch.Tensor, 
              targets: torch.Tensor, 
              inference_time: float,
              feature_quality: float = None):
        """Update statistik dengan batch prediksi baru"""
        for thresh in self.iou_thresholds:
            for cls in range(self.num_classes):
                # Filter prediksi dan ground truth berdasarkan kelas
                cls_preds = predictions[predictions[:, 5] == cls]
                cls_targets = targets[targets[:, 4] == cls]
                
                # Hitung IoU dan update statistik
                matched_preds = self._match_predictions(
                    cls_preds, cls_targets, thresh
                )
                self.predictions[thresh][cls].extend(matched_preds)
                
        # Update statistik tambahan
        self.inference_times.append(inference_time)
        if feature_quality is not None:
            self.feature_qualities.append(feature_quality)
            
    def _calculate_iou(self, box1: torch.Tensor, box2: torch.Tensor) -> float:
        """Hitung Intersection over Union antara dua bounding box"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        return intersection / (union + 1e-6)
        
    def _match_predictions(self, 
                         predictions: torch.Tensor, 
                         targets: torch.Tensor, 
                         iou_thresh: float) -> List[bool]:
        """Match predictions dengan ground truth menggunakan IoU"""
        if len(predictions) == 0 or len(targets) == 0:
            return [False] * len(predictions)
            
        ious = torch.zeros((len(predictions), len(targets)))
        for i, pred in enumerate(predictions):
            for j, target in enumerate(targets):
                ious[i, j] = self._calculate_iou(pred[:4], target[:4])
                
        # Match predictions dengan IoU tertinggi
        matched = []
        for pred_ious in ious:
            best_iou = pred_ious.max().item()
            matched.append(best_iou > iou_thresh)
            
        return matched
        
    def compute_metrics(self) -> Dict[str, DetectionMetrics]:
        """Hitung metrik final untuk setiap threshold IoU"""
        results = {}
        
        for thresh in self.iou_thresholds:
            # Inisialisasi metrik per threshold
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Aggregate metrics across classes
            for cls in range(self.num_classes):
                preds = np.array(self.predictions[thresh][cls])
                tp = np.sum(preds)
                fp = len(preds) - tp
                fn = len(self.ground_truths[cls]) - tp
                
                total_tp += tp
                total_fp += fp
                total_fn += fn
                
            # Calculate final metrics
            precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            results[thresh] = DetectionMetrics(
                precision=precision,
                recall=recall,
                f1_score=f1,
                map=self._compute_map(thresh),
                inference_time=np.mean(self.inference_times),
                feature_quality=np.mean(self.feature_qualities) if self.feature_qualities else 0.0
            )
            
        return results
        
    def _compute_map(self, iou_thresh: float) -> float:
        """Hitung mean Average Precision untuk threshold IoU tertentu"""
        aps = []
        
        for cls in range(self.num_classes):
            # Get predictions and ground truths for class
            preds = np.array(self.predictions[iou_thresh][cls])
            num_gt = len(self.ground_truths[cls])
            
            if num_gt == 0:
                continue
                
            # Calculate precision and recall points
            tp = np.cumsum(preds)
            fp = np.cumsum(~preds)
            recalls = tp / num_gt
            precisions = tp / (tp + fp)
            
            # Calculate AP using 11-point interpolation
            ap = 0
            for t in np.arange(0, 1.1, 0.1):
                if np.sum(recalls >= t) == 0:
                    p = 0
                else:
                    p = np.max(precisions[recalls >= t])
                ap += p / 11.0
                
            aps.append(ap)
            
        return np.mean(aps) if aps else 0.0
        
    def compare_models(self, other_metrics: Dict[str, DetectionMetrics]) -> Dict:
        """Lakukan analisis komparatif antara dua model"""
        comparisons = {}
        
        for thresh in self.iou_thresholds:
            base_metrics = self.compute_metrics()[thresh]
            other_model_metrics = other_metrics[thresh]
            
            # Calculate relative improvements
            improvements = {
                'precision': (other_model_metrics.precision - base_metrics.precision) / base_metrics.precision * 100,
                'recall': (other_model_metrics.recall - base_metrics.recall) / base_metrics.recall * 100,
                'f1_score': (other_model_metrics.f1_score - base_metrics.f1_score) / base_metrics.f1_score * 100,
                'map': (other_model_metrics.map - base_metrics.map) / base_metrics.map * 100,
                'inference_speed': (base_metrics.inference_time - other_model_metrics.inference_time) / base_metrics.inference_time * 100
            }
            
            # Perform statistical significance tests
            t_stat, p_value = stats.ttest_ind(
                self.inference_times,
                other_model_metrics.inference_times
            )
            
            comparisons[thresh] = {
                'improvements': improvements,
                'statistical_tests': {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
            }
            
        return comparisons

    def log_metrics(self, scenario_name: str, metrics: DetectionMetrics):
        """Log metrik dengan format yang mudah dibaca"""
        self.logger.info(f"📊 Hasil Evaluasi - {scenario_name}")
        self.logger.metric("Performance", {
            "mAP": f"{metrics.map:.4f}",
            "F1": f"{metrics.f1_score:.4f}",
            "Inference": f"{metrics.inference_time:.3f}ms"
        })