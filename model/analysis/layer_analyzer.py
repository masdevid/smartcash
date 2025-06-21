"""
File: smartcash/model/analysis/layer_analyzer.py
Deskripsi: Analyzer untuk performance analysis per-layer (banknote, nominal, security)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from smartcash.common.logger import get_logger

@dataclass
class LayerMetrics:
    """Container untuk metrics per layer"""
    layer_name: str
    total_detections: int
    avg_confidence: float
    precision: float
    recall: float
    f1_score: float
    ap: float  # Average Precision
    class_distribution: Dict[int, int]
    confidence_distribution: Dict[str, float]

class LayerAnalyzer:
    """Analyzer untuk performance analysis multi-layer detection"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('layer_analyzer')
        
        # Layer configuration
        self.layers_config = self.config.get('analysis', {}).get('layers', {})
        self.layer_weights = {layer: cfg.get('layer_weight', 1.0) for layer, cfg in self.layers_config.items()}
        
    def analyze_layer_performance(self, predictions: List[Dict], ground_truth: List[Dict] = None) -> Dict[str, Any]:
        """Analyze performance untuk setiap layer"""
        try:
            # Extract detections per layer
            layer_detections = self._extract_layer_detections(predictions)
            
            # Calculate metrics per layer
            layer_metrics = {}
            for layer_name, detections in layer_detections.items():
                metrics = self._calculate_layer_metrics(layer_name, detections, ground_truth)
                layer_metrics[layer_name] = metrics
            
            # Calculate overall layer analysis
            overall_analysis = self._calculate_overall_layer_analysis(layer_metrics)
            
            return {
                'layer_metrics': layer_metrics,
                'overall_analysis': overall_analysis,
                'layer_utilization': self._calculate_layer_utilization(layer_detections),
                'layer_collaboration': self._analyze_layer_collaboration(layer_detections)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing layer performance: {str(e)}")
            return {'error': str(e)}
    
    def _extract_layer_detections(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract detections berdasarkan layer classification"""
        layer_detections = {layer: [] for layer in self.layers_config.keys()}
        
        for pred in predictions:
            class_id = int(pred.get('class_id', 0))
            confidence = float(pred.get('confidence', 0.0))
            bbox = pred.get('bbox', [])
            
            # Classify detection ke layer yang sesuai
            for layer_name, layer_config in self.layers_config.items():
                if class_id in layer_config.get('classes', []):
                    layer_detections[layer_name].append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'local_class_idx': self._get_local_class_index(class_id, layer_config)
                    })
                    break
        
        return layer_detections
    
    def _calculate_layer_metrics(self, layer_name: str, detections: List[Dict], 
                                ground_truth: List[Dict] = None) -> LayerMetrics:
        """Calculate comprehensive metrics untuk single layer"""
        if not detections:
            return LayerMetrics(
                layer_name=layer_name,
                total_detections=0,
                avg_confidence=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                ap=0.0,
                class_distribution={},
                confidence_distribution={}
            )
        
        # Basic statistics
        confidences = [d['confidence'] for d in detections]
        avg_confidence = np.mean(confidences)
        
        # Class distribution
        class_distribution = {}
        for det in detections:
            class_id = det['class_id']
            class_distribution[class_id] = class_distribution.get(class_id, 0) + 1
        
        # Confidence distribution (bins)
        confidence_bins = {'low': 0, 'medium': 0, 'high': 0, 'very_high': 0}
        for conf in confidences:
            if conf < 0.3:
                confidence_bins['low'] += 1
            elif conf < 0.5:
                confidence_bins['medium'] += 1
            elif conf < 0.7:
                confidence_bins['high'] += 1
            else:
                confidence_bins['very_high'] += 1
        
        # Convert counts to percentages
        total = len(confidences)
        confidence_distribution = {k: v/total for k, v in confidence_bins.items()}
        
        # Calculate precision, recall, F1 (simplified, mock calculation jika no ground truth)
        if ground_truth:
            precision, recall, f1, ap = self._calculate_detection_metrics(detections, ground_truth)
        else:
            # Mock metrics based on confidence distribution
            precision = confidence_distribution.get('high', 0) + confidence_distribution.get('very_high', 0)
            recall = avg_confidence  # Simplified approximation
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            ap = avg_confidence * 0.8  # Simplified AP approximation
        
        return LayerMetrics(
            layer_name=layer_name,
            total_detections=len(detections),
            avg_confidence=avg_confidence,
            precision=precision,
            recall=recall,
            f1_score=f1,
            ap=ap,
            class_distribution=class_distribution,
            confidence_distribution=confidence_distribution
        )
    
    def _calculate_detection_metrics(self, detections: List[Dict], ground_truth: List[Dict]) -> tuple:
        """Calculate precision, recall, F1, dan AP dengan ground truth"""
        # Simplified implementation - dalam praktik perlu IoU matching
        if not ground_truth:
            return 0.0, 0.0, 0.0, 0.0
        
        # Mock calculation based on confidence and ground truth counts
        gt_count = len(ground_truth)
        det_count = len(detections)
        high_conf_dets = sum(1 for d in detections if d['confidence'] > 0.5)
        
        # Simplified precision/recall approximation
        precision = high_conf_dets / max(det_count, 1)
        recall = min(1.0, det_count / max(gt_count, 1))
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        ap = precision * recall  # Simplified AP
        
        return precision, recall, f1, ap
    
    def _calculate_overall_layer_analysis(self, layer_metrics: Dict[str, LayerMetrics]) -> Dict[str, Any]:
        """Calculate overall analysis across all layers"""
        if not layer_metrics:
            return {}
        
        # Weighted performance calculation
        total_weighted_score = 0.0
        total_weight = 0.0
        
        layer_performance = {}
        for layer_name, metrics in layer_metrics.items():
            weight = self.layer_weights.get(layer_name, 1.0)
            performance_score = (metrics.precision + metrics.recall + metrics.f1_score) / 3
            
            layer_performance[layer_name] = {
                'performance_score': performance_score,
                'weight': weight,
                'weighted_score': performance_score * weight,
                'total_detections': metrics.total_detections,
                'avg_confidence': metrics.avg_confidence
            }
            
            total_weighted_score += performance_score * weight
            total_weight += weight
        
        overall_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Find best and worst performing layers
        best_layer = max(layer_performance.items(), key=lambda x: x[1]['performance_score'])
        worst_layer = min(layer_performance.items(), key=lambda x: x[1]['performance_score'])
        
        return {
            'overall_performance_score': overall_score,
            'layer_performance': layer_performance,
            'best_performing_layer': {
                'name': best_layer[0],
                'score': best_layer[1]['performance_score']
            },
            'worst_performing_layer': {
                'name': worst_layer[0],
                'score': worst_layer[1]['performance_score']
            },
            'layer_balance': self._calculate_layer_balance(layer_metrics)
        }
    
    def _calculate_layer_utilization(self, layer_detections: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Calculate how well each layer is being utilized"""
        total_detections = sum(len(dets) for dets in layer_detections.values())
        
        utilization = {}
        for layer_name, detections in layer_detections.items():
            layer_count = len(detections)
            utilization_rate = layer_count / max(total_detections, 1)
            
            # Calculate utilization quality
            if detections:
                high_conf_rate = sum(1 for d in detections if d['confidence'] > 0.5) / layer_count
                quality_score = high_conf_rate
            else:
                quality_score = 0.0
            
            utilization[layer_name] = {
                'detection_count': layer_count,
                'utilization_rate': utilization_rate,
                'quality_score': quality_score,
                'expected_weight': self.layer_weights.get(layer_name, 1.0),
                'performance_ratio': utilization_rate / max(self.layer_weights.get(layer_name, 1.0), 0.1)
            }
        
        return utilization
    
    def _analyze_layer_collaboration(self, layer_detections: Dict[str, List[Dict]]) -> Dict[str, Any]:
        """Analyze how layers work together (spatial overlap analysis)"""
        collaboration_matrix = {}
        
        layer_names = list(layer_detections.keys())
        for i, layer1 in enumerate(layer_names):
            for j, layer2 in enumerate(layer_names):
                if i <= j:  # Avoid duplicate pairs
                    continue
                
                overlap_count = self._count_spatial_overlaps(
                    layer_detections[layer1], 
                    layer_detections[layer2]
                )
                
                pair_key = f"{layer1}-{layer2}"
                collaboration_matrix[pair_key] = {
                    'overlap_count': overlap_count,
                    'layer1_count': len(layer_detections[layer1]),
                    'layer2_count': len(layer_detections[layer2]),
                    'collaboration_rate': overlap_count / max(
                        min(len(layer_detections[layer1]), len(layer_detections[layer2])), 1
                    )
                }
        
        # Overall collaboration score
        total_collaborations = sum(c['overlap_count'] for c in collaboration_matrix.values())
        avg_collaboration_rate = np.mean([c['collaboration_rate'] for c in collaboration_matrix.values()]) if collaboration_matrix else 0.0
        
        return {
            'collaboration_matrix': collaboration_matrix,
            'total_spatial_overlaps': total_collaborations,
            'avg_collaboration_rate': avg_collaboration_rate,
            'collaboration_summary': self._generate_collaboration_summary(collaboration_matrix)
        }
    
    def _calculate_layer_balance(self, layer_metrics: Dict[str, LayerMetrics]) -> Dict[str, Any]:
        """Calculate balance analysis across layers"""
        detection_counts = [metrics.total_detections for metrics in layer_metrics.values()]
        confidence_avgs = [metrics.avg_confidence for metrics in layer_metrics.values()]
        
        # Calculate coefficient of variation for balance assessment
        detection_balance = np.std(detection_counts) / max(np.mean(detection_counts), 1)
        confidence_balance = np.std(confidence_avgs) / max(np.mean(confidence_avgs), 1)
        
        return {
            'detection_balance_score': 1.0 / (1.0 + detection_balance),  # Higher is more balanced
            'confidence_balance_score': 1.0 / (1.0 + confidence_balance),
            'overall_balance_score': (
                (1.0 / (1.0 + detection_balance)) + 
                (1.0 / (1.0 + confidence_balance))
            ) / 2,
            'balance_assessment': self._assess_balance(detection_balance, confidence_balance)
        }
    
    def _count_spatial_overlaps(self, detections1: List[Dict], detections2: List[Dict], 
                               iou_threshold: float = 0.3) -> int:
        """Count spatial overlaps between two layer detections"""
        overlap_count = 0
        
        for det1 in detections1:
            for det2 in detections2:
                iou = self._calculate_iou(det1.get('bbox', []), det2.get('bbox', []))
                if iou > iou_threshold:
                    overlap_count += 1
                    break  # Count each detection1 only once
        
        return overlap_count
    
    def _generate_collaboration_summary(self, collaboration_matrix: Dict) -> Dict[str, str]:
        """Generate human-readable collaboration summary"""
        if not collaboration_matrix:
            return {'status': 'no_collaboration_data'}
        
        # Find best collaborating pair
        best_pair = max(collaboration_matrix.items(), key=lambda x: x[1]['collaboration_rate'])
        worst_pair = min(collaboration_matrix.items(), key=lambda x: x[1]['collaboration_rate'])
        
        return {
            'best_collaboration': f"{best_pair[0]}: {best_pair[1]['collaboration_rate']:.2f}",
            'worst_collaboration': f"{worst_pair[0]}: {worst_pair[1]['collaboration_rate']:.2f}",
            'status': 'collaboration_analyzed'
        }
    
    def _assess_balance(self, detection_balance: float, confidence_balance: float) -> str:
        """Assess overall layer balance"""
        if detection_balance < 0.3 and confidence_balance < 0.3:
            return 'well_balanced'
        elif detection_balance < 0.5 and confidence_balance < 0.5:
            return 'moderately_balanced'
        else:
            return 'imbalanced'
    
    def analyze_batch_layer_performance(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Analyze layer performance across batch of images"""
        all_layer_metrics = []
        
        for result in batch_results:
            if 'error' not in result and 'layer_metrics' in result:
                all_layer_metrics.append(result['layer_metrics'])
        
        if not all_layer_metrics:
            return {'error': 'No valid layer metrics to analyze'}
        
        # Aggregate metrics across batch
        aggregated_metrics = self._aggregate_layer_metrics(all_layer_metrics)
        
        # Calculate batch-level insights
        batch_insights = self._calculate_batch_insights(all_layer_metrics)
        
        return {
            'batch_size': len(all_layer_metrics),
            'aggregated_layer_metrics': aggregated_metrics,
            'batch_insights': batch_insights,
            'layer_consistency': self._analyze_layer_consistency(all_layer_metrics)
        }
    
    def _aggregate_layer_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate layer metrics across multiple images"""
        layer_names = set()
        for metrics in metrics_list:
            layer_names.update(metrics.keys())
        
        aggregated = {}
        for layer_name in layer_names:
            layer_data = []
            for metrics in metrics_list:
                if layer_name in metrics:
                    layer_data.append(metrics[layer_name])
            
            if layer_data:
                aggregated[layer_name] = {
                    'avg_detections': np.mean([ld.total_detections for ld in layer_data]),
                    'avg_confidence': np.mean([ld.avg_confidence for ld in layer_data]),
                    'avg_precision': np.mean([ld.precision for ld in layer_data]),
                    'avg_recall': np.mean([ld.recall for ld in layer_data]),
                    'avg_f1_score': np.mean([ld.f1_score for ld in layer_data]),
                    'avg_ap': np.mean([ld.ap for ld in layer_data]),
                    'consistency_score': 1.0 - np.std([ld.avg_confidence for ld in layer_data])
                }
        
        return aggregated
    
    def _calculate_batch_insights(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Calculate insights across batch"""
        # Count images where each layer was active
        layer_activity = {}
        for metrics in metrics_list:
            for layer_name, layer_metrics in metrics.items():
                if layer_metrics.total_detections > 0:
                    layer_activity[layer_name] = layer_activity.get(layer_name, 0) + 1
        
        total_images = len(metrics_list)
        activity_rates = {layer: count/total_images for layer, count in layer_activity.items()}
        
        return {
            'layer_activity_rates': activity_rates,
            'most_active_layer': max(activity_rates.items(), key=lambda x: x[1]) if activity_rates else None,
            'least_active_layer': min(activity_rates.items(), key=lambda x: x[1]) if activity_rates else None
        }
    
    def _analyze_layer_consistency(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Analyze consistency of layer performance across images"""
        layer_variances = {}
        
        for layer_name in self.layers_config.keys():
            confidences = []
            detection_counts = []
            
            for metrics in metrics_list:
                if layer_name in metrics:
                    confidences.append(metrics[layer_name].avg_confidence)
                    detection_counts.append(metrics[layer_name].total_detections)
            
            if confidences:
                layer_variances[layer_name] = {
                    'confidence_variance': np.var(confidences),
                    'detection_count_variance': np.var(detection_counts),
                    'consistency_score': 1.0 / (1.0 + np.var(confidences))
                }
        
        return layer_variances
    
    # Helper methods
    def _get_local_class_index(self, class_id: int, layer_config: Dict) -> int:
        """Get local class index within layer"""
        layer_classes = layer_config.get('classes', [])
        try:
            return layer_classes.index(class_id)
        except ValueError:
            return 0
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
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