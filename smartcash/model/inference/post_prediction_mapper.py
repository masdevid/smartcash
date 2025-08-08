"""
Post-Prediction Mapper for SmartCash YOLOv5
Maps 17 training classes to 7 denomination classes for inference
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass

from smartcash.common.logger import SmartCashLogger


@dataclass
class Detection:
    """Detection result structure"""
    bbox: torch.Tensor  # [x1, y1, x2, y2]
    confidence: float
    class_id: int
    denomination: Optional[int] = None
    supporting_evidence: Optional[Dict] = None


class PostPredictionMapper:
    """
    Maps 17-class YOLOv5 predictions to 7 denomination classes
    
    Training Classes (17):
    - 0-6: Main denominations (001, 002, 005, 010, 020, 050, 100)
    - 7-13: Denomination-specific features 
    - 14-16: Authenticity/security features
    
    Inference Output (7):
    - 0-6: Final denomination predictions with confidence adjustment
    """
    
    def __init__(self, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize mapper
        
        Args:
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for spatial matching
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        self.logger = SmartCashLogger(__name__)
        
        # Class mapping configuration
        self.class_mapping = self._create_class_mapping()
        self.denomination_names = ['001', '002', '005', '010', '020', '050', '100']
        
        self.logger.info(f"✅ PostPredictionMapper initialized with {len(self.denomination_names)} denominations")
    
    def _create_class_mapping(self) -> Dict[str, Dict]:
        """Create comprehensive class mapping configuration"""
        return {
            # Layer 1: Main denominations (direct mapping)
            'layer_1': {
                'class_range': (0, 6),
                'mapping_type': 'direct',
                'confidence_weight': 1.0,
                'description': 'Main denomination detection'
            },
            
            # Layer 2: Denomination-specific features (supporting evidence)
            'layer_2': {
                'class_range': (7, 13),
                'mapping_type': 'supporting',
                'confidence_weight': 0.15,
                'description': 'Denomination-specific visual cues'
            },
            
            # Layer 3: Authenticity features (confidence modulation)
            'layer_3': {
                'class_range': (14, 16),
                'mapping_type': 'authenticity',
                'confidence_weight': 0.2,
                'description': 'Security and authenticity features'
            }
        }
    
    def map_predictions(
        self, 
        predictions: torch.Tensor, 
        image_shape: Tuple[int, int] = None
    ) -> List[Dict[str, Union[torch.Tensor, float, int]]]:
        """
        Map 17-class predictions to 7 denominations
        
        Args:
            predictions: YOLOv5 output tensor [batch, detections, 6] or [detections, 6]
                        Format: [x1, y1, x2, y2, conf, class_id]
            image_shape: Original image shape (height, width)
            
        Returns:
            List of denomination detection results
        """
        if predictions is None or predictions.numel() == 0:
            return []
        
        # Handle batch dimension
        if predictions.dim() == 3:
            # Multiple images - process each separately
            results = []
            for batch_pred in predictions:
                batch_result = self._map_single_image(batch_pred, image_shape)
                results.append(batch_result)
            return results
        else:
            # Single image
            return self._map_single_image(predictions, image_shape)
    
    def _map_single_image(
        self, 
        predictions: torch.Tensor,
        image_shape: Tuple[int, int] = None
    ) -> Dict[str, Union[torch.Tensor, List[Dict]]]:
        """
        Map predictions for a single image
        
        Args:
            predictions: Tensor [N, 6] with detections
            image_shape: Original image shape
            
        Returns:
            Dictionary with denomination results
        """
        if predictions.numel() == 0:
            return self._empty_result()
        
        # Filter by confidence
        valid_mask = predictions[:, 4] >= self.confidence_threshold
        if not valid_mask.any():
            return self._empty_result()
        
        valid_predictions = predictions[valid_mask]
        
        # Convert to Detection objects
        detections = self._tensor_to_detections(valid_predictions)
        
        # Group by layer type
        layer_groups = self._group_by_layers(detections)
        
        # Process denominations
        denomination_results = self._process_denominations(layer_groups)
        
        # Create final result
        final_result = self._create_final_result(denomination_results)
        
        return final_result
    
    def _tensor_to_detections(self, predictions: torch.Tensor) -> List[Detection]:
        """Convert prediction tensor to Detection objects"""
        detections = []
        
        for pred in predictions:
            bbox = pred[:4]  # [x1, y1, x2, y2]
            confidence = pred[4].item()
            class_id = int(pred[5].item())
            
            detection = Detection(
                bbox=bbox,
                confidence=confidence,
                class_id=class_id
            )
            
            detections.append(detection)
        
        return detections
    
    def _group_by_layers(self, detections: List[Detection]) -> Dict[str, List[Detection]]:
        """Group detections by layer type"""
        layer_groups = {
            'layer_1': [],  # Main denominations (0-6)
            'layer_2': [],  # Denomination features (7-13) 
            'layer_3': []   # Authenticity features (14-16)
        }
        
        for detection in detections:
            cls = detection.class_id
            
            if 0 <= cls <= 6:
                layer_groups['layer_1'].append(detection)
            elif 7 <= cls <= 13:
                layer_groups['layer_2'].append(detection)
            elif 14 <= cls <= 16:
                layer_groups['layer_3'].append(detection)
                
        return layer_groups
    
    def _process_denominations(
        self, 
        layer_groups: Dict[str, List[Detection]]
    ) -> Dict[int, Dict]:
        """
        Process denominations with confidence adjustment
        
        Args:
            layer_groups: Detections grouped by layer
            
        Returns:
            Dictionary mapping denomination_id to result info
        """
        denomination_results = {}
        
        # Process each main denomination detection (Layer 1)
        for detection in layer_groups['layer_1']:
            denom_id = detection.class_id  # Direct mapping for 0-6
            
            # Find supporting evidence
            supporting_layer2 = self._find_supporting_evidence(
                detection, layer_groups['layer_2'], denom_id + 7  # Layer 2 classes are offset by 7
            )
            
            supporting_layer3 = self._find_supporting_evidence(
                detection, layer_groups['layer_3'], None  # Layer 3 is general authenticity
            )
            
            # Calculate adjusted confidence
            adjusted_confidence = self._calculate_adjusted_confidence(
                detection.confidence,
                supporting_layer2,
                supporting_layer3
            )
            
            # Store result (keep best detection per denomination)
            if denom_id not in denomination_results or adjusted_confidence > denomination_results[denom_id]['confidence']:
                denomination_results[denom_id] = {
                    'detection': detection,
                    'confidence': adjusted_confidence,
                    'supporting_layer2': supporting_layer2,
                    'supporting_layer3': supporting_layer3,
                    'denomination_id': denom_id,
                    'denomination_name': self.denomination_names[denom_id]
                }
        
        return denomination_results
    
    def _find_supporting_evidence(
        self,
        primary_detection: Detection,
        candidate_detections: List[Detection],
        target_class: Optional[int] = None
    ) -> List[Detection]:
        """
        Find supporting evidence for a primary detection
        
        Args:
            primary_detection: Main denomination detection
            candidate_detections: List of potential supporting detections
            target_class: Specific class to look for (None for any)
            
        Returns:
            List of supporting detections
        """
        supporting = []
        
        for candidate in candidate_detections:
            # Check class match if specified
            if target_class is not None and candidate.class_id != target_class:
                continue
            
            # Check spatial overlap
            iou = self._calculate_iou(primary_detection.bbox, candidate.bbox)
            
            if iou >= self.iou_threshold:
                supporting.append(candidate)
        
        return supporting
    
    def _calculate_iou(self, bbox1: torch.Tensor, bbox2: torch.Tensor) -> float:
        """Calculate IoU between two bounding boxes"""
        # Convert to [x1, y1, x2, y2] if needed
        box1 = bbox1.clone()
        box2 = bbox2.clone()
        
        # Calculate intersection
        x1 = torch.max(box1[0], box2[0])
        y1 = torch.max(box1[1], box2[1])
        x2 = torch.min(box1[2], box2[2])
        y2 = torch.min(box1[3], box2[3])
        
        # Check if there's intersection
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        # Calculate areas
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return (intersection / union).item()
    
    def _calculate_adjusted_confidence(
        self,
        base_confidence: float,
        supporting_layer2: List[Detection],
        supporting_layer3: List[Detection]
    ) -> float:
        """
        Calculate confidence with supporting evidence adjustment
        
        Args:
            base_confidence: Original confidence from Layer 1
            supporting_layer2: Supporting denomination-specific features
            supporting_layer3: Supporting authenticity features
            
        Returns:
            Adjusted confidence score
        """
        adjusted = base_confidence
        
        # Layer 2 boost (denomination-specific features)
        if supporting_layer2:
            layer2_boost = max(det.confidence for det in supporting_layer2) * 0.15
            adjusted += layer2_boost
            
        # Layer 3 boost (authenticity features)
        if supporting_layer3:
            layer3_boost = max(det.confidence for det in supporting_layer3) * 0.2
            adjusted += layer3_boost
        else:
            # Penalty for missing authenticity (possible fake)
            if base_confidence > 0.5:
                adjusted *= 0.85  # 15% penalty
        
        return min(adjusted, 1.0)  # Cap at 1.0
    
    def _create_final_result(
        self, 
        denomination_results: Dict[int, Dict]
    ) -> Dict[str, Union[torch.Tensor, List[Dict]]]:
        """
        Create final result structure
        
        Args:
            denomination_results: Processed denomination results
            
        Returns:
            Final result dictionary
        """
        if not denomination_results:
            return self._empty_result()
        
        # Sort by confidence
        sorted_results = sorted(
            denomination_results.values(), 
            key=lambda x: x['confidence'], 
            reverse=True
        )
        
        # Extract tensors
        boxes = torch.stack([result['detection'].bbox for result in sorted_results])
        confidences = torch.tensor([result['confidence'] for result in sorted_results])
        labels = torch.tensor([result['denomination_id'] for result in sorted_results], dtype=torch.long)
        
        # Create denomination score vector
        denomination_scores = torch.zeros(7)
        for result in sorted_results:
            denom_id = result['denomination_id']
            denomination_scores[denom_id] = result['confidence']
        
        # Detailed results for analysis
        detailed_results = []
        for result in sorted_results:
            detailed_results.append({
                'denomination_id': result['denomination_id'],
                'denomination_name': result['denomination_name'],
                'confidence': result['confidence'],
                'bbox': result['detection'].bbox.tolist(),
                'supporting_layer2_count': len(result['supporting_layer2']),
                'supporting_layer3_count': len(result['supporting_layer3'])
            })
        
        return {
            'boxes': boxes,
            'scores': confidences,
            'labels': labels,
            'denomination_scores': denomination_scores,
            'detailed_results': detailed_results,
            'total_detections': len(sorted_results)
        }
    
    def _empty_result(self) -> Dict[str, Union[torch.Tensor, List]]:
        """Return empty result structure"""
        return {
            'boxes': torch.empty(0, 4),
            'scores': torch.empty(0),
            'labels': torch.empty(0, dtype=torch.long),
            'denomination_scores': torch.zeros(7),
            'detailed_results': [],
            'total_detections': 0
        }
    
    def get_mapping_info(self) -> Dict:
        """Get information about class mappings"""
        return {
            'class_mapping': self.class_mapping,
            'denomination_names': self.denomination_names,
            'confidence_threshold': self.confidence_threshold,
            'iou_threshold': self.iou_threshold,
            'total_training_classes': 17,
            'total_inference_classes': 7
        }


class BatchPostPredictionMapper:
    """
    Batch version of PostPredictionMapper for efficient processing
    """
    
    def __init__(self, confidence_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Initialize batch mapper
        
        Args:
            confidence_threshold: Minimum confidence threshold
            iou_threshold: IoU threshold for spatial matching
        """
        self.mapper = PostPredictionMapper(confidence_threshold, iou_threshold)
        self.logger = SmartCashLogger(__name__)
    
    def map_batch_predictions(
        self, 
        batch_predictions: List[torch.Tensor],
        image_shapes: Optional[List[Tuple[int, int]]] = None
    ) -> List[Dict]:
        """
        Map predictions for a batch of images
        
        Args:
            batch_predictions: List of prediction tensors
            image_shapes: Optional list of image shapes
            
        Returns:
            List of result dictionaries
        """
        results = []
        
        for i, predictions in enumerate(batch_predictions):
            image_shape = image_shapes[i] if image_shapes else None
            result = self.mapper._map_single_image(predictions, image_shape)
            results.append(result)
            
        return results


# Export key components
__all__ = [
    'PostPredictionMapper',
    'BatchPostPredictionMapper',
    'Detection'
]