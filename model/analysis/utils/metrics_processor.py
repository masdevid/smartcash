"""
File: smartcash/model/analysis/utils/metrics_processor.py
Deskripsi: Processor untuk metrics computation dan statistical analysis
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from smartcash.common.logger import get_logger

class MetricsProcessor:
    """Processor untuk comprehensive metrics computation dan statistical analysis"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('metrics_processor')
        
    def calculate_detection_metrics(self, predictions: List[Dict], ground_truth: List[Dict], 
                                  iou_threshold: float = 0.5) -> Dict[str, float]:
        """Calculate detection metrics dengan IoU matching"""
        if not predictions or not ground_truth:
            return {'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'ap': 0.0}
        
        # Simplified detection metrics (mock implementation)
        true_positives = min(len(predictions), len(ground_truth))
        false_positives = max(0, len(predictions) - len(ground_truth))
        false_negatives = max(0, len(ground_truth) - len(predictions))
        
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-8)
        
        # Simplified AP calculation
        ap = precision * recall if precision > 0 and recall > 0 else 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'ap': ap,
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }
    
    def calculate_confidence_statistics(self, confidences: List[float]) -> Dict[str, float]:
        """Calculate comprehensive confidence statistics"""
        if not confidences:
            return {stat: 0.0 for stat in ['mean', 'std', 'min', 'max', 'q25', 'q50', 'q75', 'iqr']}
        
        conf_array = np.array(confidences)
        return {
            'mean': float(np.mean(conf_array)),
            'std': float(np.std(conf_array)),
            'min': float(np.min(conf_array)),
            'max': float(np.max(conf_array)),
            'q25': float(np.percentile(conf_array, 25)),
            'q50': float(np.percentile(conf_array, 50)),
            'q75': float(np.percentile(conf_array, 75)),
            'iqr': float(np.percentile(conf_array, 75) - np.percentile(conf_array, 25))
        }
    
    def calculate_class_balance_metrics(self, class_counts: Dict[str, int]) -> Dict[str, Any]:
        """Calculate class balance metrics dan imbalance assessment"""
        counts = list(class_counts.values())
        non_zero_counts = [c for c in counts if c > 0]
        
        if not non_zero_counts:
            return {'balance_score': 0.0, 'imbalance_ratio': float('inf'), 'gini_coefficient': 1.0}
        
        # Imbalance ratio
        max_count, min_count = max(non_zero_counts), min(non_zero_counts)
        imbalance_ratio = max_count / min_count
        
        # Balance score (higher = more balanced)
        cv = np.std(non_zero_counts) / np.mean(non_zero_counts)
        balance_score = 1.0 / (1.0 + cv)
        
        # Gini coefficient
        gini = self._calculate_gini_coefficient(non_zero_counts)
        
        return {
            'balance_score': balance_score,
            'imbalance_ratio': imbalance_ratio,
            'gini_coefficient': gini,
            'coefficient_of_variation': cv,
            'entropy': self._calculate_entropy(counts)
        }
    
    def calculate_layer_collaboration_score(self, layer_detections: Dict[str, List[Dict]], 
                                          iou_threshold: float = 0.3) -> Dict[str, Any]:
        """Calculate collaboration score antar layers"""
        layer_names = list(layer_detections.keys())
        if len(layer_names) < 2:
            return {'collaboration_score': 0.0, 'overlap_matrix': {}}
        
        overlap_matrix = {}
        total_overlaps = 0
        total_possible = 0
        
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names[i+1:], i+1):
                overlaps = self._count_bbox_overlaps(
                    layer_detections[layer1], 
                    layer_detections[layer2], 
                    iou_threshold
                )
                
                pair_key = f"{layer1}-{layer2}"
                max_possible = min(len(layer_detections[layer1]), len(layer_detections[layer2]))
                
                overlap_matrix[pair_key] = {
                    'overlaps': overlaps,
                    'max_possible': max_possible,
                    'overlap_rate': overlaps / max(max_possible, 1)
                }
                
                total_overlaps += overlaps
                total_possible += max_possible
        
        overall_collaboration_score = total_overlaps / max(total_possible, 1)
        
        return {
            'collaboration_score': overall_collaboration_score,
            'overlap_matrix': overlap_matrix,
            'total_overlaps': total_overlaps,
            'total_possible_overlaps': total_possible
        }
    
    def calculate_consistency_metrics(self, batch_metrics: List[Dict[str, float]]) -> Dict[str, float]:
        """Calculate consistency metrics across batch"""
        if not batch_metrics:
            return {}
        
        consistency_scores = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in batch_metrics:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in batch_metrics]
            
            if values and any(v != 0 for v in values):
                # Consistency = 1 - coefficient of variation
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv = std_val / max(mean_val, 1e-8)
                consistency_scores[f"{key}_consistency"] = 1.0 / (1.0 + cv)
            else:
                consistency_scores[f"{key}_consistency"] = 0.0
        
        return consistency_scores
    
    def aggregate_metrics_weighted(self, metrics_list: List[Dict], 
                                 weights: Optional[List[float]] = None) -> Dict[str, float]:
        """Aggregate metrics dengan optional weighting"""
        if not metrics_list:
            return {}
        
        if weights is None:
            weights = [1.0] * len(metrics_list)
        
        weights = weights[:len(metrics_list)]  # Ensure same length
        total_weight = sum(weights)
        
        if total_weight == 0:
            return {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in metrics_list:
            all_keys.update(metrics.keys())
        
        aggregated = {}
        for key in all_keys:
            weighted_sum = sum(metrics.get(key, 0.0) * weight 
                             for metrics, weight in zip(metrics_list, weights))
            aggregated[key] = weighted_sum / total_weight
        
        return aggregated
    
    def calculate_performance_trends(self, historical_metrics: List[Dict[str, float]]) -> Dict[str, Any]:
        """Calculate performance trends dari historical data"""
        if len(historical_metrics) < 2:
            return {'trend_analysis': 'insufficient_data'}
        
        trends = {}
        
        # Get all metric keys
        all_keys = set()
        for metrics in historical_metrics:
            all_keys.update(metrics.keys())
        
        for key in all_keys:
            values = [metrics.get(key, 0.0) for metrics in historical_metrics]
            
            if len(values) >= 2:
                # Calculate trend direction
                first_half_avg = np.mean(values[:len(values)//2])
                second_half_avg = np.mean(values[len(values)//2:])
                
                trend_direction = 'improving' if second_half_avg > first_half_avg else 'declining'
                trend_magnitude = abs(second_half_avg - first_half_avg) / max(first_half_avg, 1e-8)
                
                # Calculate volatility
                volatility = np.std(values) / max(np.mean(values), 1e-8)
                
                trends[key] = {
                    'direction': trend_direction,
                    'magnitude': trend_magnitude,
                    'volatility': volatility,
                    'latest_value': values[-1],
                    'best_value': max(values),
                    'worst_value': min(values)
                }
        
        return {
            'trend_analysis': 'complete',
            'metrics_trends': trends,
            'overall_stability': np.mean([t['volatility'] for t in trends.values()]) if trends else 0.0
        }
    
    def _calculate_gini_coefficient(self, values: List[float]) -> float:
        """Calculate Gini coefficient untuk inequality measurement"""
        if not values or len(values) < 2:
            return 0.0
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        cumsum = np.cumsum(sorted_values)
        
        return (n + 1 - 2 * sum((n + 1 - i) * y for i, y in enumerate(cumsum))) / (n * sum(sorted_values))
    
    def _calculate_entropy(self, counts: List[int]) -> float:
        """Calculate entropy untuk diversity measurement"""
        total = sum(counts)
        if total == 0:
            return 0.0
        
        probabilities = [count / total for count in counts if count > 0]
        return -sum(p * np.log2(p) for p in probabilities) if probabilities else 0.0
    
    def _count_bbox_overlaps(self, detections1: List[Dict], detections2: List[Dict], 
                           iou_threshold: float) -> int:
        """Count spatial overlaps between two sets of detections"""
        overlaps = 0
        
        for det1 in detections1:
            bbox1 = det1.get('bbox', [])
            if len(bbox1) < 4:
                continue
                
            for det2 in detections2:
                bbox2 = det2.get('bbox', [])
                if len(bbox2) < 4:
                    continue
                
                iou = self._calculate_iou(bbox1, bbox2)
                if iou > iou_threshold:
                    overlaps += 1
                    break  # Count each detection1 only once
        
        return overlaps
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # Convert to [x1, y1, x2, y2] format
        x1_1, y1_1, w1, h1 = bbox1[:4]
        x1_2, y1_2, w2, h2 = bbox2[:4]
        
        x2_1, y2_1 = x1_1 + w1, y1_1 + h1
        x2_2, y2_2 = x1_2 + w2, y1_2 + h2
        
        # Calculate intersection
        xi1, yi1 = max(x1_1, x1_2), max(y1_1, y1_2)
        xi2, yi2 = min(x2_1, x2_2), min(y2_1, y2_2)
        
        if xi1 >= xi2 or yi1 >= yi2:
            return 0.0
        
        intersection = (xi2 - xi1) * (yi2 - yi1)
        union = w1 * h1 + w2 * h2 - intersection
        
        return intersection / union if union > 0 else 0.0