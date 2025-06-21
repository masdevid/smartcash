"""
File: smartcash/model/training/metrics_tracker.py
Deskripsi: Real-time metrics tracking untuk training dan validation dengan mAP calculation
"""

import torch
import numpy as np
import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import json
from pathlib import Path

class APCalculator:
    """Calculator untuk Average Precision dengan COCO-style evaluation"""
    
    def __init__(self, iou_thresholds: List[float] = None, class_names: List[str] = None):
        self.iou_thresholds = iou_thresholds or [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
        self.class_names = class_names or [f"class_{i}" for i in range(7)]
        self.reset()
    
    def reset(self) -> None:
        """Reset accumulated data"""
        self.predictions = []  # List of [conf, class_pred, x1, y1, x2, y2]
        self.targets = []      # List of [class_true, x1, y1, x2, y2]
        self.image_ids = []    # Image IDs untuk matching
    
    def add_batch(self, pred_boxes: torch.Tensor, pred_scores: torch.Tensor, 
                  pred_classes: torch.Tensor, true_boxes: torch.Tensor, 
                  true_classes: torch.Tensor, image_ids: List[int]) -> None:
        """Add batch predictions dan ground truth"""
        batch_size = len(image_ids)
        
        for i in range(batch_size):
            img_id = image_ids[i]
            
            # Add predictions untuk image ini
            if len(pred_boxes[i]) > 0:
                for j in range(len(pred_boxes[i])):
                    self.predictions.append([
                        pred_scores[i][j].item(),
                        pred_classes[i][j].item(),
                        pred_boxes[i][j][0].item(),
                        pred_boxes[i][j][1].item(),
                        pred_boxes[i][j][2].item(),
                        pred_boxes[i][j][3].item(),
                        img_id
                    ])
            
            # Add ground truth untuk image ini
            if len(true_boxes[i]) > 0:
                for j in range(len(true_boxes[i])):
                    self.targets.append([
                        true_classes[i][j].item(),
                        true_boxes[i][j][0].item(),
                        true_boxes[i][j][1].item(),
                        true_boxes[i][j][2].item(),
                        true_boxes[i][j][3].item(),
                        img_id
                    ])
    
    def compute_ap(self, class_id: int, iou_threshold: float = 0.5) -> float:
        """Compute Average Precision untuk specific class dan IoU threshold"""
        # Filter predictions dan targets untuk class ini
        class_preds = [p for p in self.predictions if p[1] == class_id]
        class_targets = [t for t in self.targets if t[0] == class_id]
        
        if len(class_targets) == 0:
            return 0.0
        
        if len(class_preds) == 0:
            return 0.0
        
        # Sort predictions by confidence (descending)
        class_preds.sort(key=lambda x: x[0], reverse=True)
        
        # Track matched targets
        num_targets = len(class_targets)
        matched = [False] * num_targets
        
        # Calculate precision dan recall
        tp = []  # True positives
        fp = []  # False positives
        
        for pred in class_preds:
            pred_box = pred[2:6]
            img_id = pred[6]
            
            # Find best matching target
            best_iou = 0.0
            best_idx = -1
            
            for i, target in enumerate(class_targets):
                if target[5] != img_id or matched[i]:
                    continue
                
                target_box = target[1:5]
                iou = self._calculate_iou(pred_box, target_box)
                
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            
            # Check if match meets IoU threshold
            if best_iou >= iou_threshold and best_idx >= 0:
                tp.append(1)
                fp.append(0)
                matched[best_idx] = True
            else:
                tp.append(0)
                fp.append(1)
        
        # Calculate cumulative precision dan recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_targets
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-16)
        
        # Calculate AP menggunakan 11-point interpolation
        ap = 0.0
        for recall_threshold in np.linspace(0, 1, 11):
            precision_at_recall = 0.0
            for i in range(len(recalls)):
                if recalls[i] >= recall_threshold:
                    precision_at_recall = max(precision_at_recall, precisions[i])
            ap += precision_at_recall / 11
        
        return ap
    
    def compute_map(self, iou_threshold: float = 0.5) -> Tuple[float, Dict[int, float]]:
        """Compute mean Average Precision across all classes"""
        class_aps = {}
        
        for class_id in range(len(self.class_names)):
            ap = self.compute_ap(class_id, iou_threshold)
            class_aps[class_id] = ap
        
        # Calculate mAP
        valid_aps = [ap for ap in class_aps.values() if ap > 0]
        map_score = np.mean(valid_aps) if valid_aps else 0.0
        
        return map_score, class_aps
    
    def compute_map50_95(self) -> float:
        """Compute mAP@0.5:0.95 (COCO metric)"""
        maps = []
        for iou_thresh in self.iou_thresholds:
            map_score, _ = self.compute_map(iou_thresh)
            maps.append(map_score)
        
        return np.mean(maps)
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate IoU between two boxes [x1, y1, x2, y2]"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0

class MetricsTracker:
    """Real-time metrics tracking untuk training dan validation"""
    
    def __init__(self, config: Dict[str, Any], class_names: List[str] = None):
        self.config = config
        self.class_names = class_names or [f"Rp{v}K" for v in [1, 2, 5, 10, 20, 50, 100]]
        
        # Metrics storage
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        self.best_metrics = {}
        self.current_metrics = {}
        
        # AP calculator
        self.ap_calculator = APCalculator(class_names=self.class_names)
        
        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # History untuk plotting
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_map50': [],
            'val_map50_95': [],
            'learning_rate': []
        }
    
    def start_epoch(self) -> None:
        """Mark start dari epoch baru"""
        self.epoch_start_time = time.time()
        self.ap_calculator.reset()
    
    def start_batch(self) -> None:
        """Mark start dari batch baru"""
        self.batch_start_time = time.time()
    
    def update_train_metrics(self, loss_dict: Dict[str, torch.Tensor], 
                           learning_rate: float = None) -> None:
        """Update training metrics untuk current batch"""
        # Convert tensors ke scalars
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.train_metrics[key].append(value.item())
            else:
                self.train_metrics[key].append(value)
        
        if learning_rate is not None:
            self.train_metrics['learning_rate'].append(learning_rate)
        
        # Calculate batch time
        if self.batch_start_time:
            batch_time = time.time() - self.batch_start_time
            self.train_metrics['batch_time'].append(batch_time)
    
    def update_val_metrics(self, loss_dict: Dict[str, torch.Tensor], 
                          predictions: Dict[str, Any] = None) -> None:
        """Update validation metrics"""
        # Update loss metrics
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                self.val_metrics[key].append(value.item())
            else:
                self.val_metrics[key].append(value)
        
        # Add predictions untuk mAP calculation
        if predictions:
            self.ap_calculator.add_batch(
                pred_boxes=predictions['boxes'],
                pred_scores=predictions['scores'],
                pred_classes=predictions['classes'],
                true_boxes=predictions['true_boxes'],
                true_classes=predictions['true_classes'],
                image_ids=predictions['image_ids']
            )
    
    def compute_epoch_metrics(self, epoch: int) -> Dict[str, float]:
        """Compute metrics untuk completed epoch"""
        metrics = {}
        
        # Training metrics (average dari batches)
        if self.train_metrics['total_loss']:
            metrics['train_loss'] = np.mean(self.train_metrics['total_loss'])
            metrics['train_box_loss'] = np.mean(self.train_metrics.get('box_loss', [0]))
            metrics['train_obj_loss'] = np.mean(self.train_metrics.get('obj_loss', [0]))
            metrics['train_cls_loss'] = np.mean(self.train_metrics.get('cls_loss', [0]))
        
        # Validation metrics
        if self.val_metrics['total_loss']:
            metrics['val_loss'] = np.mean(self.val_metrics['total_loss'])
            metrics['val_box_loss'] = np.mean(self.val_metrics.get('box_loss', [0]))
            metrics['val_obj_loss'] = np.mean(self.val_metrics.get('obj_loss', [0]))
            metrics['val_cls_loss'] = np.mean(self.val_metrics.get('cls_loss', [0]))
        
        # mAP metrics
        val_config = self.config.get('training', {}).get('validation', {})
        if val_config.get('compute_map', True):
            map50, class_aps = self.ap_calculator.compute_map(0.5)
            map50_95 = self.ap_calculator.compute_map50_95()
            
            metrics['val_map50'] = map50
            metrics['val_map50_95'] = map50_95
            
            # Per-class mAP
            for class_id, ap in class_aps.items():
                class_name = self.class_names[class_id] if class_id < len(self.class_names) else f"class_{class_id}"
                metrics[f'val_ap_{class_name}'] = ap
        
        # Timing metrics
        if self.epoch_start_time:
            metrics['epoch_time'] = time.time() - self.epoch_start_time
        
        if self.train_metrics.get('batch_time'):
            metrics['avg_batch_time'] = np.mean(self.train_metrics['batch_time'])
        
        # Learning rate
        if self.train_metrics.get('learning_rate'):
            metrics['learning_rate'] = self.train_metrics['learning_rate'][-1]
        
        # Update current metrics
        self.current_metrics = metrics
        
        # Update history
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['val_loss'].append(metrics.get('val_loss', 0))
        self.history['val_map50'].append(metrics.get('val_map50', 0))
        self.history['val_map50_95'].append(metrics.get('val_map50_95', 0))
        self.history['learning_rate'].append(metrics.get('learning_rate', 0))
        
        # Reset untuk epoch selanjutnya
        self._reset_epoch_metrics()
        
        return metrics
    
    def is_best_model(self, metric_name: str = 'val_map50', mode: str = 'max') -> bool:
        """Check apakah current model adalah yang terbaik"""
        if metric_name not in self.current_metrics:
            return False
        
        current_value = self.current_metrics[metric_name]
        
        if metric_name not in self.best_metrics:
            self.best_metrics[metric_name] = current_value
            return True
        
        best_value = self.best_metrics[metric_name]
        
        if mode == 'max':
            is_better = current_value > best_value
        else:  # mode == 'min'
            is_better = current_value < best_value
        
        if is_better:
            self.best_metrics[metric_name] = current_value
            return True
        
        return False
    
    def get_current_metrics(self) -> Dict[str, float]:
        """Get current epoch metrics"""
        return self.current_metrics.copy()
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best recorded metrics"""
        return self.best_metrics.copy()
    
    def get_metrics_summary(self) -> str:
        """Get formatted metrics summary"""
        if not self.current_metrics:
            return "No metrics available"
        
        summary_parts = []
        
        # Loss metrics
        train_loss = self.current_metrics.get('train_loss', 0)
        val_loss = self.current_metrics.get('val_loss', 0)
        summary_parts.append(f"Loss - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
        
        # mAP metrics
        map50 = self.current_metrics.get('val_map50', 0)
        map50_95 = self.current_metrics.get('val_map50_95', 0)
        summary_parts.append(f"mAP - @0.5: {map50:.3f}, @0.5:0.95: {map50_95:.3f}")
        
        # Best metrics
        best_map = self.best_metrics.get('val_map50', 0)
        summary_parts.append(f"Best mAP@0.5: {best_map:.3f}")
        
        return " | ".join(summary_parts)
    
    def save_metrics(self, save_path: str) -> None:
        """Save metrics history ke file"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        metrics_data = {
            'history': self.history,
            'current_metrics': self.current_metrics,
            'best_metrics': self.best_metrics,
            'class_names': self.class_names
        }
        
        with open(save_path, 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def _reset_epoch_metrics(self) -> None:
        """Reset metrics untuk epoch selanjutnya"""
        self.train_metrics.clear()
        self.val_metrics.clear()
        self.ap_calculator.reset()
        self.epoch_start_time = None
        self.batch_start_time = None

# Convenience functions
def create_metrics_tracker(config: Dict[str, Any], class_names: List[str] = None) -> MetricsTracker:
    """Factory function untuk create metrics tracker"""
    return MetricsTracker(config, class_names)

def calculate_map(predictions: List, targets: List, num_classes: int = 7, 
                 iou_threshold: float = 0.5) -> Tuple[float, Dict[int, float]]:
    """One-liner untuk calculate mAP"""
    calculator = APCalculator()
    # Add predictions dan targets
    for pred, target in zip(predictions, targets):
        # Implementation untuk convert predictions/targets ke format yang sesuai
        pass
    return calculator.compute_map(iou_threshold)