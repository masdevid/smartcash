"""
File: smartcash/model/analysis/currency_analyzer.py
Deskripsi: Analyzer untuk deteksi denominasi currency dengan strategi multi-layer
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from smartcash.common.logger import get_logger

@dataclass
class CurrencyDetection:
    """Container untuk hasil deteksi currency dengan confidence strategy"""
    denomination: str
    confidence: float
    primary_class: int
    boost_confidence: float = 0.0
    security_validated: bool = False
    bbox: List[float] = None
    detection_strategy: str = 'primary'

class CurrencyAnalyzer:
    """Analyzer untuk deteksi denominasi currency dengan multi-layer validation"""
    
    def __init__(self, config: Dict[str, Any] = None, logger=None):
        self.config = config or {}
        self.logger = logger or get_logger('currency_analyzer')
        
        # Currency analysis config
        currency_config = self.config.get('analysis', {}).get('currency', {})
        self.primary_layer = currency_config.get('primary_layer', 'banknote')
        self.boost_layer = currency_config.get('confidence_boost_layer', 'nominal')
        self.validation_layer = currency_config.get('validation_layer', 'security')
        self.denomination_classes = currency_config.get('denomination_classes', list(range(7)))
        self.confidence_threshold = currency_config.get('confidence_threshold', 0.3)
        self.iou_threshold = currency_config.get('iou_threshold', 0.5)
        
        # Layer config mapping
        layers_config = self.config.get('analysis', {}).get('layers', {})
        self.layer_mapping = {layer: config for layer, config in layers_config.items()}
        
    def analyze_currency_detections(self, predictions: List[Dict], image_id: str = None) -> Dict[str, Any]:
        """Analisis deteksi currency dengan strategi multi-layer"""
        try:
            # Extract detections per layer
            layer_detections = self._extract_layer_detections(predictions)
            
            # Apply currency detection strategy
            currency_detections = self._apply_currency_strategy(layer_detections)
            
            # Calculate currency metrics
            metrics = self._calculate_currency_metrics(currency_detections, layer_detections)
            
            return {
                'image_id': image_id,
                'currency_detections': currency_detections,
                'layer_detections': layer_detections,
                'metrics': metrics,
                'detection_summary': self._generate_detection_summary(currency_detections)
            }
            
        except Exception as e:
            self.logger.error(f"❌ Error analyzing currency detections: {str(e)}")
            return {'error': str(e), 'image_id': image_id}
    
    def _extract_layer_detections(self, predictions: List[Dict]) -> Dict[str, List[Dict]]:
        """Extract detections berdasarkan layer classification"""
        layer_detections = {layer: [] for layer in self.layer_mapping.keys()}
        
        for pred in predictions:
            class_id = int(pred.get('class_id', 0))
            confidence = float(pred.get('confidence', 0.0))
            bbox = pred.get('bbox', [])
            
            # Determine layer dari class_id
            for layer, layer_config in self.layer_mapping.items():
                if class_id in layer_config.get('classes', []):
                    layer_detections[layer].append({
                        'class_id': class_id,
                        'confidence': confidence,
                        'bbox': bbox,
                        'denomination_idx': self._get_denomination_index(class_id, layer)
                    })
                    break
        
        return layer_detections
    
    def _apply_currency_strategy(self, layer_detections: Dict[str, List[Dict]]) -> List[CurrencyDetection]:
        """Apply multi-layer currency detection strategy"""
        currency_detections = []
        primary_detections = layer_detections.get(self.primary_layer, [])
        
        for primary_det in primary_detections:
            if primary_det['confidence'] < self.confidence_threshold:
                continue
                
            # Primary detection
            denomination = self._get_denomination_name(primary_det['denomination_idx'])
            detection = CurrencyDetection(
                denomination=denomination,
                confidence=primary_det['confidence'],
                primary_class=primary_det['class_id'],
                bbox=primary_det['bbox'],
                detection_strategy='primary'
            )
            
            # Confidence boost strategy
            boost_confidence = self._find_boost_confidence(primary_det, layer_detections)
            if boost_confidence > 0:
                detection.boost_confidence = boost_confidence
                detection.confidence = min(1.0, detection.confidence + boost_confidence * 0.2)
                detection.detection_strategy = 'boosted'
            
            # Security validation
            security_validated = self._validate_security(primary_det, layer_detections)
            detection.security_validated = security_validated
            if security_validated:
                detection.detection_strategy = 'validated'
            
            currency_detections.append(detection)
        
        # Fallback strategy: nominal-only detections
        fallback_detections = self._apply_fallback_strategy(layer_detections, currency_detections)
        currency_detections.extend(fallback_detections)
        
        return currency_detections
    
    def _find_boost_confidence(self, primary_det: Dict, layer_detections: Dict) -> float:
        """Find confidence boost dari nominal layer"""
        boost_detections = layer_detections.get(self.boost_layer, [])
        primary_bbox = primary_det['bbox']
        
        for boost_det in boost_detections:
            if boost_det['denomination_idx'] == primary_det['denomination_idx']:
                # Check spatial overlap
                iou = self._calculate_iou(primary_bbox, boost_det['bbox'])
                if iou > self.iou_threshold:
                    return boost_det['confidence']
        
        return 0.0
    
    def _validate_security(self, primary_det: Dict, layer_detections: Dict) -> bool:
        """Validate menggunakan security layer"""
        security_detections = layer_detections.get(self.validation_layer, [])
        primary_bbox = primary_det['bbox']
        
        for sec_det in security_detections:
            if sec_det['confidence'] > self.confidence_threshold:
                iou = self._calculate_iou(primary_bbox, sec_det['bbox'])
                if iou > self.iou_threshold:
                    return True
        
        return False
    
    def _apply_fallback_strategy(self, layer_detections: Dict, existing_detections: List) -> List[CurrencyDetection]:
        """Apply fallback strategy untuk nominal-only detections"""
        fallback_detections = []
        boost_detections = layer_detections.get(self.boost_layer, [])
        existing_bboxes = [det.bbox for det in existing_detections]
        
        for boost_det in boost_detections:
            if boost_det['confidence'] < self.confidence_threshold * 1.5:  # Higher threshold for fallback
                continue
                
            # Check jika sudah ada detection di area yang sama
            overlap_found = False
            for existing_bbox in existing_bboxes:
                if self._calculate_iou(boost_det['bbox'], existing_bbox) > self.iou_threshold:
                    overlap_found = True
                    break
            
            if not overlap_found:
                denomination = self._get_denomination_name(boost_det['denomination_idx'])
                fallback_detections.append(CurrencyDetection(
                    denomination=denomination,
                    confidence=boost_det['confidence'],
                    primary_class=boost_det['class_id'],
                    bbox=boost_det['bbox'],
                    detection_strategy='fallback'
                ))
        
        return fallback_detections
    
    def _calculate_currency_metrics(self, currency_detections: List[CurrencyDetection], 
                                  layer_detections: Dict) -> Dict[str, Any]:
        """Calculate comprehensive currency metrics"""
        total_detections = len(currency_detections)
        
        # Strategy distribution
        strategy_counts = {}
        for det in currency_detections:
            strategy_counts[det.detection_strategy] = strategy_counts.get(det.detection_strategy, 0) + 1
        
        # Denomination distribution
        denomination_counts = {}
        for det in currency_detections:
            denomination_counts[det.denomination] = denomination_counts.get(det.denomination, 0) + 1
        
        # Confidence statistics
        confidences = [det.confidence for det in currency_detections]
        conf_stats = {
            'mean': np.mean(confidences) if confidences else 0.0,
            'std': np.std(confidences) if confidences else 0.0,
            'min': np.min(confidences) if confidences else 0.0,
            'max': np.max(confidences) if confidences else 0.0
        }
        
        # Layer utilization
        layer_utilization = {}
        for layer, detections in layer_detections.items():
            layer_utilization[layer] = {
                'count': len(detections),
                'avg_confidence': np.mean([d['confidence'] for d in detections]) if detections else 0.0
            }
        
        return {
            'total_currency_detections': total_detections,
            'strategy_distribution': strategy_counts,
            'denomination_distribution': denomination_counts,
            'confidence_statistics': conf_stats,
            'layer_utilization': layer_utilization,
            'validation_rate': sum(1 for d in currency_detections if d.security_validated) / max(total_detections, 1),
            'boost_rate': sum(1 for d in currency_detections if d.boost_confidence > 0) / max(total_detections, 1)
        }
    
    def _generate_detection_summary(self, currency_detections: List[CurrencyDetection]) -> Dict[str, Any]:
        """Generate human-readable detection summary"""
        if not currency_detections:
            return {'status': 'no_currency_detected', 'message': 'Tidak ada currency terdeteksi'}
        
        primary_detections = [d for d in currency_detections if d.detection_strategy in ['primary', 'boosted', 'validated']]
        fallback_detections = [d for d in currency_detections if d.detection_strategy == 'fallback']
        
        # Get highest confidence detection
        best_detection = max(currency_detections, key=lambda x: x.confidence)
        
        summary = {
            'status': 'currency_detected',
            'total_detections': len(currency_detections),
            'primary_detections': len(primary_detections),
            'fallback_detections': len(fallback_detections),
            'best_detection': {
                'denomination': best_detection.denomination,
                'confidence': best_detection.confidence,
                'strategy': best_detection.detection_strategy,
                'validated': best_detection.security_validated
            },
            'denominations_found': list(set(d.denomination for d in currency_detections))
        }
        
        # Generate message
        if len(primary_detections) > 0:
            summary['message'] = f"Terdeteksi {len(primary_detections)} currency utama, best: {best_detection.denomination}"
        else:
            summary['message'] = f"Terdeteksi {len(fallback_detections)} currency fallback saja"
        
        return summary
    
    def analyze_batch_results(self, batch_results: List[Dict]) -> Dict[str, Any]:
        """Analyze batch currency detection results"""
        all_metrics = []
        all_detections = []
        
        for result in batch_results:
            if 'error' not in result:
                all_metrics.append(result['metrics'])
                all_detections.extend(result['currency_detections'])
        
        if not all_metrics:
            return {'error': 'No valid results to analyze'}
        
        # Aggregate metrics
        aggregated = self._aggregate_batch_metrics(all_metrics)
        
        # Overall statistics
        total_images = len(batch_results)
        successful_images = len(all_metrics)
        images_with_currency = sum(1 for m in all_metrics if m['total_currency_detections'] > 0)
        
        return {
            'batch_summary': {
                'total_images': total_images,
                'successful_analysis': successful_images,
                'images_with_currency': images_with_currency,
                'success_rate': successful_images / total_images,
                'detection_rate': images_with_currency / max(successful_images, 1)
            },
            'aggregated_metrics': aggregated,
            'total_currency_detections': len(all_detections)
        }
    
    def _aggregate_batch_metrics(self, metrics_list: List[Dict]) -> Dict[str, Any]:
        """Aggregate metrics dari multiple images"""
        # Strategy distribution
        all_strategies = {}
        all_denominations = {}
        all_confidences = []
        
        for metrics in metrics_list:
            # Aggregate strategy counts
            for strategy, count in metrics.get('strategy_distribution', {}).items():
                all_strategies[strategy] = all_strategies.get(strategy, 0) + count
            
            # Aggregate denomination counts
            for denom, count in metrics.get('denomination_distribution', {}).items():
                all_denominations[denom] = all_denominations.get(denom, 0) + count
        
        return {
            'strategy_distribution': all_strategies,
            'denomination_distribution': all_denominations,
            'avg_validation_rate': np.mean([m.get('validation_rate', 0) for m in metrics_list]),
            'avg_boost_rate': np.mean([m.get('boost_rate', 0) for m in metrics_list])
        }
    
    # Helper methods
    def _get_denomination_index(self, class_id: int, layer: str) -> int:
        """Get denomination index dari class_id dan layer"""
        layer_classes = self.layer_mapping.get(layer, {}).get('classes', [])
        try:
            return layer_classes.index(class_id)
        except ValueError:
            return 0
    
    def _get_denomination_name(self, denomination_idx: int) -> str:
        """Get denomination name dari index"""
        denominations = ['Rp1K', 'Rp2K', 'Rp5K', 'Rp10K', 'Rp20K', 'Rp50K', 'Rp100K']
        return denominations[denomination_idx] if 0 <= denomination_idx < len(denominations) else 'Unknown'
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU antara dua bounding boxes"""
        if not bbox1 or not bbox2 or len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        # Convert to [x1, y1, x2, y2] format jika perlu
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