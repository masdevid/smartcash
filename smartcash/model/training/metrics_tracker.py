"""
File: smartcash/model/training/metrics_tracker.py
Deskripsi: Real-time metrics tracking untuk training dan validation
Focus: Classification metrics only (accuracy, precision, recall, f1)
"""

import torch
import numpy as np
import time
from typing import Dict, List, Optional, Any
from collections import defaultdict
import json
from pathlib import Path

from smartcash.common.logger import get_logger

logger = get_logger(__name__)

# APCalculator class completely removed - focusing on classification metrics only

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
        
        # mAP calculation removed - focusing on classification metrics only
        
        # Timing
        self.epoch_start_time = None
        self.batch_start_time = None
        
        # History untuk plotting (classification metrics only)
        self.history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_f1': [],
            'learning_rate': []
        }
    
    def start_epoch(self) -> None:
        """Mark start dari epoch baru"""
        self.epoch_start_time = time.time()
    
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
        
        # mAP computation completely removed - focusing on classification metrics only
        
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
        
        # Update history for plotting (classification metrics only)
        self.history['epoch'].append(epoch)
        self.history['train_loss'].append(metrics.get('train_loss', 0))
        self.history['val_loss'].append(metrics.get('val_loss', 0))
        self.history['val_accuracy'].append(metrics.get('val_accuracy', 0))
        self.history['val_f1'].append(metrics.get('val_f1', 0))
        self.history['learning_rate'].append(metrics.get('learning_rate', 0))
        
        self._reset_epoch_metrics()
        
        return metrics
    
    def is_best_model(self, metric_name: str = 'val_accuracy', mode: str = 'max') -> bool:
        """Check apakah current model adalah yang terbaik"""
        logger.debug(f"🔍 is_best_model check: metric={metric_name}, mode={mode}")
        logger.debug(f"🔍 Available current_metrics: {list(self.current_metrics.keys())}")
        
        if metric_name not in self.current_metrics:
            logger.debug(f"🔍 Metric {metric_name} not found in current_metrics")
            return False
        
        current_value = self.current_metrics[metric_name]
        logger.debug(f"🔍 Current {metric_name} value: {current_value}")
        
        if metric_name not in self.best_metrics:
            logger.debug(f"🔍 First time seeing {metric_name}, setting as best")
            self.best_metrics[metric_name] = current_value
            return True
        
        best_value = self.best_metrics[metric_name]
        logger.debug(f"🔍 Previous best {metric_name} value: {best_value}")
        
        if mode == 'max':
            is_better = current_value > best_value
        else:  # mode == 'min'
            is_better = current_value < best_value
        
        logger.debug(f"🔍 Is {current_value} better than {best_value} (mode={mode})? {is_better}")
        
        if is_better:
            logger.debug(f"🔍 New best {metric_name}: {current_value} (was {best_value})")
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
        # mAP calculator removed
        self.epoch_start_time = None
        self.batch_start_time = None

# Convenience functions
def create_metrics_tracker(config: Dict[str, Any], class_names: List[str] = None) -> MetricsTracker:
    """Factory function untuk create metrics tracker"""
    return MetricsTracker(config, class_names)

# calculate_map function removed - focusing on classification metrics only