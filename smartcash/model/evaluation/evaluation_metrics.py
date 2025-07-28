"""
File: smartcash/model/evaluation/evaluation_metrics.py
Deskripsi: Evaluation-specific metrics calculation untuk research scenarios
"""

import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict
import time

from smartcash.common.logger import get_logger

class EvaluationMetrics:
    """Comprehensive metrics calculation untuk evaluation phase"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = get_logger('evaluation_metrics')
        
        # Safe extraction of metrics config
        try:
            eval_config = self.config.get('evaluation', {})
            if isinstance(eval_config, dict):
                metrics_config = eval_config.get('metrics', {})
                if isinstance(metrics_config, dict):
                    self.metrics_config = metrics_config
                else:
                    self.logger.warning(f"Metrics config is not a dict (got {type(metrics_config).__name__}), using default")
                    self.metrics_config = {}
            else:
                self.logger.warning(f"Evaluation config is not a dict (got {type(eval_config).__name__}), using default")
                self.metrics_config = {}
        except Exception as e:
            self.logger.warning(f"Error accessing metrics config: {e}")
            self.metrics_config = {}
        
        # Metrics storage
        self.predictions = []
        self.ground_truths = []
        self.inference_times = []
        
    def compute_map(self, predictions: List[Dict], ground_truths: List[Dict], 
                   iou_thresholds: List[float] = None) -> Dict[str, float]:
        """🎯 Calculate Mean Average Precision"""
        if iou_thresholds is None:
            iou_thresholds = self.metrics_config.get('map', {}).get('iou_thresholds', [0.5, 0.75])
        
        map_results = {}
        
        for iou_thresh in iou_thresholds:
            # Calculate AP untuk each class
            class_aps = {}
            all_classes = set()
            
            # Collect all classes with safe access
            for gt in ground_truths:
                for ann in gt.get('annotations', []):
                    if isinstance(ann, dict) and 'class_id' in ann:
                        try:
                            class_id = ann['class_id']
                            if isinstance(class_id, (int, float)):
                                all_classes.add(int(class_id))
                        except (ValueError, TypeError):
                            continue  # Skip invalid class_ids
            
            for class_id in all_classes:
                ap = self._calculate_ap_for_class(predictions, ground_truths, class_id, iou_thresh)
                class_aps[class_id] = ap
            
            # Calculate mAP
            map_value = np.mean(list(class_aps.values())) if class_aps else 0.0
            map_results[f'mAP@{iou_thresh}'] = map_value
            map_results[f'class_aps@{iou_thresh}'] = class_aps
        
        # Overall mAP (average across IoU thresholds)
        map_results['mAP'] = np.mean([map_results[f'mAP@{thresh}'] for thresh in iou_thresholds])
        
        self.logger.info(f"🎯 mAP calculated: {map_results['mAP']:.3f}")
        return map_results
    
    def compute_accuracy(self, predictions: List[Dict], ground_truths: List[Dict], 
                        confidence_threshold: float = None, iou_threshold: float = None) -> Dict[str, float]:
        """✅ Calculate detection accuracy"""
        if confidence_threshold is None:
            confidence_threshold = self.metrics_config.get('accuracy', {}).get('confidence_threshold', 0.25)
        if iou_threshold is None:
            iou_threshold = self.metrics_config.get('accuracy', {}).get('iou_threshold', 0.5)
        
        total_gt = 0
        correct_detections = 0
        
        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.get('annotations', [])
            pred_boxes = pred.get('detections', [])
            
            # Filter predictions by confidence
            filtered_preds = [p for p in pred_boxes if p.get('confidence', 0) >= confidence_threshold]
            
            total_gt += len(gt_boxes)
            
            # Match predictions dengan ground truth
            matched_gt = set()
            for pred_box in filtered_preds:
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    # Check class match dan IoU with safe access
                    pred_class = pred_box.get('class_id')
                    gt_class = gt_box.get('class_id')
                    
                    if (isinstance(pred_class, (int, float)) and isinstance(gt_class, (int, float)) and
                        pred_class == gt_class and 'bbox' in pred_box and 'bbox' in gt_box and
                        self._calculate_iou(pred_box['bbox'], gt_box['bbox']) >= iou_threshold):
                        correct_detections += 1
                        matched_gt.add(gt_idx)
                        break
        
        accuracy = correct_detections / total_gt if total_gt > 0 else 0.0
        
        result = {
            'accuracy': accuracy,
            'correct_detections': correct_detections,
            'total_ground_truth': total_gt,
            'confidence_threshold': confidence_threshold,
            'iou_threshold': iou_threshold
        }
        
        self.logger.info(f"✅ Accuracy: {accuracy:.3f} ({correct_detections}/{total_gt})")
        return result
    
    def compute_precision(self, predictions: List[Dict], ground_truths: List[Dict], 
                         per_class: bool = None) -> Dict[str, float]:
        """🎯 Calculate precision metrics"""
        if per_class is None:
            per_class = self.metrics_config.get('precision_recall', {}).get('per_class', True)
        
        confidence_threshold = self.metrics_config.get('precision_recall', {}).get('confidence_threshold', 0.25)
        
        # Collect predictions dan ground truth per class
        class_stats = defaultdict(lambda: {'tp': 0, 'fp': 0, 'total_pred': 0})
        
        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.get('annotations', [])
            pred_boxes = [p for p in pred.get('detections', []) if p.get('confidence', 0) >= confidence_threshold]
            
            # Track matched ground truths untuk avoid double counting
            matched_gt = set()
            
            for pred_box in pred_boxes:
                class_id = pred_box.get('class_id')
                if not isinstance(class_id, (int, float)):
                    continue
                
                class_id = int(class_id)
                class_stats[class_id]['total_pred'] += 1
                
                # Find best matching ground truth
                best_iou = 0
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if (gt_idx not in matched_gt and 
                        gt_box.get('class_id') == class_id):
                        iou = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gt_idx
                
                # Check if prediction is true positive
                if best_iou >= 0.5 and best_gt_idx != -1:
                    class_stats[class_id]['tp'] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    class_stats[class_id]['fp'] += 1
        
        # Calculate precision per class
        precision_results = {}
        precisions = []
        
        for class_id, stats in class_stats.items():
            if stats['total_pred'] > 0:
                precision = stats['tp'] / stats['total_pred']
            else:
                precision = 0.0
            
            precision_results[f'precision_class_{class_id}'] = precision
            precisions.append(precision)
        
        # Overall precision
        precision_results['precision'] = np.mean(precisions) if precisions else 0.0
        
        self.logger.info(f"🎯 Precision: {precision_results['precision']:.3f}")
        return precision_results
    
    def compute_recall(self, predictions: List[Dict], ground_truths: List[Dict], 
                      per_class: bool = None) -> Dict[str, float]:
        """🔍 Calculate recall metrics"""
        if per_class is None:
            per_class = self.metrics_config.get('precision_recall', {}).get('per_class', True)
        
        confidence_threshold = self.metrics_config.get('precision_recall', {}).get('confidence_threshold', 0.25)
        
        # Collect ground truth dan matches per class
        class_stats = defaultdict(lambda: {'tp': 0, 'total_gt': 0})
        
        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.get('annotations', [])
            pred_boxes = [p for p in pred.get('detections', []) if p.get('confidence', 0) >= confidence_threshold]
            
            # Track ground truth matches
            for gt_box in gt_boxes:
                class_id = gt_box.get('class_id')
                class_stats[class_id]['total_gt'] += 1
                
                # Check if ground truth has matching prediction
                best_iou = 0
                for pred_box in pred_boxes:
                    if pred_box.get('class_id') == class_id:
                        iou = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
                        best_iou = max(best_iou, iou)
                
                if best_iou >= 0.5:
                    class_stats[class_id]['tp'] += 1
        
        # Calculate recall per class
        recall_results = {}
        recalls = []
        
        for class_id, stats in class_stats.items():
            if stats['total_gt'] > 0:
                recall = stats['tp'] / stats['total_gt']
            else:
                recall = 0.0
            
            recall_results[f'recall_class_{class_id}'] = recall
            recalls.append(recall)
        
        # Overall recall
        recall_results['recall'] = np.mean(recalls) if recalls else 0.0
        
        self.logger.info(f"🔍 Recall: {recall_results['recall']:.3f}")
        return recall_results
    
    def compute_f1_score(self, predictions: List[Dict], ground_truths: List[Dict], 
                        per_class: bool = None, beta: float = None) -> Dict[str, float]:
        """⚖️ Calculate F1 score"""
        if per_class is None:
            per_class = self.metrics_config.get('f1_score', {}).get('per_class', True)
        if beta is None:
            beta = self.metrics_config.get('f1_score', {}).get('beta', 1.0)
        
        # Calculate precision dan recall
        precision_results = self.compute_precision(predictions, ground_truths, per_class)
        recall_results = self.compute_recall(predictions, ground_truths, per_class)
        
        f1_results = {}
        
        # Calculate F1 per class
        all_classes = set()
        for key in precision_results.keys():
            if key.startswith('precision_class_'):
                class_id = key.replace('precision_class_', '')
                all_classes.add(class_id)
        
        f1_scores = []
        for class_id in all_classes:
            precision = precision_results.get(f'precision_class_{class_id}', 0)
            recall = recall_results.get(f'recall_class_{class_id}', 0)
            
            if precision + recall > 0:
                f1 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
            else:
                f1 = 0.0
            
            f1_results[f'f1_class_{class_id}'] = f1
            f1_scores.append(f1)
        
        # Overall F1
        overall_precision = precision_results.get('precision', 0)
        overall_recall = recall_results.get('recall', 0)
        
        if overall_precision + overall_recall > 0:
            f1_results['f1_score'] = (1 + beta**2) * (overall_precision * overall_recall) / ((beta**2 * overall_precision) + overall_recall)
        else:
            f1_results['f1_score'] = 0.0
        
        self.logger.info(f"⚖️ F1 Score: {f1_results['f1_score']:.3f}")
        return f1_results
    
    def compute_inference_time(self, inference_times: List[float], 
                             batch_sizes: List[int] = None) -> Dict[str, float]:
        """⏱️ Calculate inference timing metrics"""
        if not inference_times:
            return {'avg_inference_time': 0.0, 'total_samples': 0}
        
        timing_results = {
            'avg_inference_time': np.mean(inference_times),
            'median_inference_time': np.median(inference_times),
            'min_inference_time': np.min(inference_times),
            'max_inference_time': np.max(inference_times),
            'std_inference_time': np.std(inference_times),
            'total_samples': len(inference_times),
            'total_time': np.sum(inference_times)
        }
        
        # Calculate FPS
        timing_results['fps'] = 1.0 / timing_results['avg_inference_time'] if timing_results['avg_inference_time'] > 0 else 0.0
        
        # Batch size analysis jika tersedia
        if batch_sizes and len(batch_sizes) == len(inference_times):
            batch_analysis = defaultdict(list)
            for batch_size, inf_time in zip(batch_sizes, inference_times):
                batch_analysis[batch_size].append(inf_time)
            
            timing_results['batch_analysis'] = {
                str(bs): {
                    'avg_time': np.mean(times),
                    'samples': len(times),
                    'fps': len(times) / np.sum(times) if np.sum(times) > 0 else 0.0
                }
                for bs, times in batch_analysis.items()
            }
        
        self.logger.info(f"⏱️ Inference Time: {timing_results['avg_inference_time']:.3f}s ({timing_results['fps']:.1f} FPS)")
        return timing_results
    
    def generate_confusion_matrix(self, predictions: List[Dict], ground_truths: List[Dict], 
                                 num_classes: int = 7) -> np.ndarray:
        """📊 Generate confusion matrix"""
        confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        confidence_threshold = self.metrics_config.get('precision_recall', {}).get('confidence_threshold', 0.25)
        
        for pred, gt in zip(predictions, ground_truths):
            gt_boxes = gt.get('annotations', [])
            pred_boxes = [p for p in pred.get('detections', []) if p.get('confidence', 0) >= confidence_threshold]
            
            # Match predictions dengan ground truth
            matched_gt = set()
            
            for pred_box in pred_boxes:
                pred_class = pred_box.get('class_id', 0)
                if pred_class >= num_classes:
                    continue
                
                best_iou = 0
                best_gt_class = None
                best_gt_idx = -1
                
                for gt_idx, gt_box in enumerate(gt_boxes):
                    if gt_idx in matched_gt:
                        continue
                    
                    gt_class = gt_box.get('class_id', 0)
                    if gt_class >= num_classes:
                        continue
                    
                    iou = self._calculate_iou(pred_box['bbox'], gt_box['bbox'])
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_class = gt_class
                        best_gt_idx = gt_idx
                
                # Update confusion matrix
                if best_iou >= 0.5 and best_gt_class is not None:
                    confusion_matrix[best_gt_class][pred_class] += 1
                    matched_gt.add(best_gt_idx)
                else:
                    # False positive - pred without good match, use background class or handle differently
                    pass
        
        self.logger.info(f"📊 Confusion matrix generated ({num_classes}x{num_classes})")
        return confusion_matrix
    
    def _flatten_multi_layer_predictions(self, predictions: List[Dict]) -> List[Dict]:
        """🔄 Flatten multi-layer predictions from latest model architecture for evaluation"""
        flattened = []
        
        for pred in predictions:
            detections = pred.get('detections', [])
            
            # If detections already flattened, use as-is
            if isinstance(detections, list) and all(isinstance(d, dict) for d in detections):
                flattened.append(pred)
                continue
            
            # If detections are layered (dict with layer_1, layer_2, etc.), flatten them
            if isinstance(detections, dict):
                all_detections = []
                for layer_name, layer_detections in detections.items():
                    if isinstance(layer_detections, list):
                        # Add layer info to each detection for traceability
                        for detection in layer_detections:
                            if isinstance(detection, dict):
                                detection_copy = detection.copy()
                                detection_copy['source_layer'] = layer_name
                                all_detections.append(detection_copy)
                
                # Create flattened prediction
                flattened_pred = pred.copy()
                flattened_pred['detections'] = all_detections
                flattened.append(flattened_pred)
            else:
                # Unknown format, use as-is
                flattened.append(pred)
        
        return flattened
    
    def get_metrics_summary(self, predictions: List[Dict], ground_truths: List[Dict], 
                           inference_times: List[float] = None) -> Dict[str, Any]:
        """📋 Generate comprehensive metrics summary for evaluation scenarios"""
        summary = {}
        
        # Flatten multi-layer predictions for evaluation if needed
        flattened_predictions = self._flatten_multi_layer_predictions(predictions)
        
        # Calculate all metrics
        if self.metrics_config.get('map', {}).get('enabled', True):
            summary.update(self.compute_map(flattened_predictions, ground_truths))
        
        if self.metrics_config.get('accuracy', {}).get('enabled', True):
            summary.update(self.compute_accuracy(predictions, ground_truths))
        
        if self.metrics_config.get('precision_recall', {}).get('enabled', True):
            summary.update(self.compute_precision(predictions, ground_truths))
            summary.update(self.compute_recall(predictions, ground_truths))
        
        if self.metrics_config.get('f1_score', {}).get('enabled', True):
            summary.update(self.compute_f1_score(predictions, ground_truths))
        
        if (self.metrics_config.get('inference_time', {}).get('enabled', True) and 
            inference_times):
            summary.update(self.compute_inference_time(inference_times))
        
        # Generate confusion matrix
        if self.config.get('analysis', {}).get('class_analysis', {}).get('compute_confusion_matrix', True):
            summary['confusion_matrix'] = self.generate_confusion_matrix(predictions, ground_truths)
        
        self.logger.info("📋 Comprehensive metrics summary generated")
        return summary
    
    def _calculate_ap_for_class(self, predictions: List[Dict], ground_truths: List[Dict], 
                               class_id: int, iou_threshold: float) -> float:
        """🎯 Calculate AP untuk specific class"""
        # Collect detections dan ground truths untuk class ini
        detections = []
        num_gt = 0
        
        for img_idx, (pred, gt) in enumerate(zip(predictions, ground_truths)):
            # Ground truth count
            gt_boxes = [ann for ann in gt.get('annotations', []) if ann.get('class_id') == class_id]
            num_gt += len(gt_boxes)
            
            # Predictions untuk class ini
            pred_boxes = [det for det in pred.get('detections', []) if det.get('class_id') == class_id]
            
            for det in pred_boxes:
                detections.append({
                    'confidence': det.get('confidence', 0),
                    'bbox': det['bbox'],
                    'img_idx': img_idx,
                    'matched': False
                })
        
        if num_gt == 0:
            return 0.0
        
        # Sort detections by confidence (descending)
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate precision/recall
        tp = np.zeros(len(detections))
        fp = np.zeros(len(detections))
        
        for det_idx, detection in enumerate(detections):
            img_idx = detection['img_idx']
            gt_boxes = [ann for ann in ground_truths[img_idx].get('annotations', []) 
                       if ann.get('class_id') == class_id]
            
            best_iou = 0
            best_gt_idx = -1
            
            for gt_idx, gt_box in enumerate(gt_boxes):
                iou = self._calculate_iou(detection['bbox'], gt_box['bbox'])
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gt_idx
            
            if best_iou >= iou_threshold:
                # Check if this GT was already matched
                gt_key = f"{img_idx}_{best_gt_idx}"
                if not hasattr(self, '_matched_gt'):
                    self._matched_gt = set()
                
                if gt_key not in self._matched_gt:
                    tp[det_idx] = 1
                    self._matched_gt.add(gt_key)
                else:
                    fp[det_idx] = 1
            else:
                fp[det_idx] = 1
        
        # Calculate precision dan recall curves
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        recalls = tp_cumsum / num_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-8)
        
        # Calculate AP using 11-point interpolation
        ap = 0
        for t in np.arange(0, 1.1, 0.1):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        # Clear matched GT untuk next class
        if hasattr(self, '_matched_gt'):
            delattr(self, '_matched_gt')
        
        return ap
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """📐 Calculate IoU between two bounding boxes"""
        # Convert YOLO format (x_center, y_center, width, height) to (x1, y1, x2, y2)
        def yolo_to_xyxy(box):
            x_center, y_center, width, height = box
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = yolo_to_xyxy(box1)
        box2_xyxy = yolo_to_xyxy(box2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box1_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


# Factory functions
def create_evaluation_metrics(config: Dict[str, Any] = None) -> EvaluationMetrics:
    """🏭 Factory untuk EvaluationMetrics"""
    return EvaluationMetrics(config)

def calculate_comprehensive_metrics(predictions: List[Dict], ground_truths: List[Dict], 
                                  inference_times: List[float] = None, 
                                  config: Dict[str, Any] = None) -> Dict[str, Any]:
    """📊 One-liner untuk calculate comprehensive metrics"""
    metrics = create_evaluation_metrics(config)
    return metrics.get_metrics_summary(predictions, ground_truths, inference_times)