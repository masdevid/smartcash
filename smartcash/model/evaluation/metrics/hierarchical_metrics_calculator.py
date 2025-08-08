"""
Hierarchical denomination classification metrics calculator.
Handles SmartCash multi-layer denomination evaluation logic.
"""

import numpy as np
from typing import Dict, List, Any
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from smartcash.common.logger import get_logger


class HierarchicalMetricsCalculator:
    """Calculate metrics using hierarchical denomination classification logic"""
    
    def __init__(self, num_classes: int = 17, confidence_threshold: float = 0.005, 
                 iou_threshold: float = 0.3):
        self.logger = get_logger('hierarchical_metrics')
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
    
    def calculate_denomination_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """📊 Calculate denomination classification metrics using hierarchical logic"""
        try:
            # Extract predicted and true classes for denomination classification
            y_pred = []
            y_true = []
            
            # Match predictions with ground truths
            for pred, gt in zip(predictions, ground_truths):
                gt_annotations = gt.get('annotations', [])
                pred_detections = pred.get('detections', [])
                
                # For each ground truth, find the best matching prediction
                for gt_ann in gt_annotations:
                    gt_class = int(gt_ann['class_id'])
                    gt_bbox = gt_ann['bbox']
                    
                    # Find best matching prediction based on IoU and hierarchical correctness
                    best_pred_class = -1  # No detection class
                    best_iou = 0.0
                    
                    for pred_det in pred_detections:
                        pred_class = int(pred_det['class_id'])
                        pred_bbox = pred_det['bbox']
                        pred_conf = pred_det.get('confidence', 0.0)
                        
                        # Only consider predictions with reasonable confidence
                        if pred_conf >= self.confidence_threshold:
                            # Calculate IoU
                            iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                            
                            if iou > best_iou and iou >= self.iou_threshold:
                                best_iou = iou
                                best_pred_class = pred_class
                    
                    y_true.append(gt_class)
                    y_pred.append(best_pred_class if best_pred_class != -1 else self.num_classes)  # Use num_classes for no detection
            
            if not y_true or not y_pred:
                return self._get_empty_denomination_metrics()
            
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate metrics for ALL ground truth samples (include missed detections)
            # Use all hierarchical classes (0-16) for denomination evaluation
            valid_gt_mask = (y_true < self.num_classes)
            
            if not np.any(valid_gt_mask):
                return self._get_empty_denomination_metrics()
            
            y_true_valid = y_true[valid_gt_mask]
            y_pred_valid = y_pred[valid_gt_mask]
            
            # Convert "no detection" predictions back to a special class for metrics
            y_pred_valid = np.where(y_pred_valid == self.num_classes, self.num_classes, y_pred_valid)
            
            # Calculate hierarchical accuracy considering denomination relationships
            hierarchical_correct = self._count_hierarchical_matches(y_true_valid, y_pred_valid)
            overall_accuracy = hierarchical_correct / len(y_true_valid) if len(y_true_valid) > 0 else 0.0
            
            # Calculate precision/recall/f1 only for detected classes (exclude "no detection" from averaging)
            detected_mask = y_pred_valid < self.num_classes
            if np.any(detected_mask):
                y_true_detected = y_true_valid[detected_mask]
                y_pred_detected = y_pred_valid[detected_mask]
                
                precision_detected = precision_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
                recall_detected = recall_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
                f1_detected = f1_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
            else:
                # No detections made
                precision_detected = 0.0
                recall_detected = 0.0
                f1_detected = 0.0
            
            # Calculate denomination-specific metrics (include extra class for "no detection")
            all_classes = list(range(self.num_classes + 1))  # Classes 0-16 + class 17 for "no detection"
            
            metrics = {
                'accuracy': overall_accuracy,
                'precision': precision_detected,
                'recall': recall_detected,
                'f1_score': f1_detected,
                'confusion_matrix': confusion_matrix(y_true_valid, y_pred_valid, labels=all_classes).tolist(),
                'total_samples': len(y_true_valid),
                'detected_samples': int(np.sum(detected_mask)) if np.any(detected_mask) else 0,
                'missed_samples': int(np.sum(y_pred_valid == self.num_classes))
            }
            
            # Calculate per-class metrics for all denomination classes
            per_class_precision = precision_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(self.num_classes)))
            per_class_recall = recall_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(self.num_classes)))
            per_class_f1 = f1_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(self.num_classes)))
            
            for i in range(self.num_classes):
                metrics[f'precision_class_{i}'] = float(per_class_precision[i])
                metrics[f'recall_class_{i}'] = float(per_class_recall[i])
                metrics[f'f1_class_{i}'] = float(per_class_f1[i])
            
            self.logger.info(f"📊 Denomination classification details:")
            self.logger.info(f"    Total samples: {metrics['total_samples']}")
            self.logger.info(f"    Detected samples: {metrics['detected_samples']}")
            self.logger.info(f"    Missed samples: {metrics['missed_samples']}")
            self.logger.info(f"    Unique predictions: {np.unique(y_pred_valid)}")
            self.logger.info(f"    Unique ground truth: {np.unique(y_true_valid)}")
            self.logger.info(f"📊 Denomination metrics: accuracy={metrics['accuracy']:.3f}, precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1_score']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating denomination metrics: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_empty_denomination_metrics()
    
    def _count_hierarchical_matches(self, y_true: np.ndarray, y_pred: np.ndarray) -> int:
        """Count hierarchical matches based on denomination classification rules"""
        correct_count = 0
        
        for true_class, pred_class in zip(y_true, y_pred):
            # No detection case
            if pred_class >= self.num_classes:
                continue
            
            # Exact match is always correct
            if true_class == pred_class:
                correct_count += 1
                continue
            
            # Hierarchical matching rules for SmartCash denomination system
            if self._is_hierarchical_match(true_class, pred_class):
                correct_count += 1
        
        return correct_count
    
    def _is_hierarchical_match(self, true_class: int, pred_class: int) -> bool:
        """Check if prediction is hierarchically correct for the ground truth"""
        
        # Define hierarchical relationships based on SmartCash training logic
        # Layer 1 (0-6): 1, 5, 10, 20, 50, 100, 200 peso denominations
        # Layer 2 (7-13): Detailed classifications of the same denominations
        # Layer 3 (14-16): Condition/quality classifications
        
        true_layer = self._get_class_layer(true_class)
        pred_layer = self._get_class_layer(pred_class)
        
        # Same denomination across layers should be considered correct
        # For example: true=0 (1 peso, layer 1) and pred=7 (1 peso detailed, layer 2)
        if true_layer == 1 and pred_layer == 2:
            # Layer 1 class 0 maps to Layer 2 class 7, class 1 to 8, etc.
            return true_class == (pred_class - 7)
        elif true_layer == 2 and pred_layer == 1:
            # Layer 2 class 7 maps to Layer 1 class 0, etc.
            return pred_class == (true_class - 7)
        elif true_layer == 1 and pred_layer == 3:
            # Layer 1 to Layer 3: class 0 maps to 14, class 1 to 15, class 2 to 16
            return pred_class == min(true_class + 14, 16)
        elif true_layer == 3 and pred_layer == 1:
            # Layer 3 to Layer 1: class 14 maps to 0, 15 to 1, 16 to 2
            return true_class >= 14 and pred_class == min(true_class - 14, 6)
        
        # No hierarchical relationship found
        return False
    
    def _get_class_layer(self, class_id: int) -> int:
        """Get the layer number for a given class ID"""
        if 0 <= class_id <= 6:
            return 1
        elif 7 <= class_id <= 13:
            return 2
        elif 14 <= class_id <= 16:
            return 3
        else:
            return 0  # Unknown/no detection
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes in YOLO format [x_center, y_center, width, height]"""
        def yolo_to_xyxy(bbox):
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = yolo_to_xyxy(bbox1)
        box2_xyxy = yolo_to_xyxy(bbox2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box2_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_empty_denomination_metrics(self) -> Dict[str, Any]:
        """Get empty denomination metrics structure"""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': [[0] * self.num_classes for _ in range(self.num_classes)]
        }
        
        # Add per-class metrics
        for i in range(self.num_classes):
            metrics[f'precision_class_{i}'] = 0.0
            metrics[f'recall_class_{i}'] = 0.0
            metrics[f'f1_class_{i}'] = 0.0
        
        return metrics


def create_hierarchical_metrics_calculator(num_classes: int = 17, **kwargs) -> HierarchicalMetricsCalculator:
    """Factory function to create hierarchical metrics calculator"""
    return HierarchicalMetricsCalculator(num_classes=num_classes, **kwargs)