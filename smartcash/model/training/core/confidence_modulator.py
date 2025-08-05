#!/usr/bin/env python3
"""
Confidence modulation module for hierarchical multi-layer predictions.

This module handles complex confidence modulation algorithms using spatial relationships
between different detection layers. Extracted from HierarchicalProcessor for better
separation of concerns and algorithm optimization.

Key Features:
- Spatial confidence computation using IoU overlap
- Denomination-specific confidence matching
- Memory-safe IoU matrix computation with chunking
- Class-based confidence modulation with memory optimization
- Hierarchical confidence boost/reduction based on money validation

Algorithmic Complexity:
- Spatial confidence: O(P₁ × P₃) for IoU matrix computation
- Denomination confidence: O(C × P₁ × P₂) where C is unique classes
- Memory optimization: O(chunk_size) for large prediction sets
"""

import torch
import numpy as np
from typing import Tuple, Dict, Set

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
from .yolo_utils_manager import get_box_iou

logger = get_logger(__name__, level="DEBUG")


class ConfidenceModulator:
    """
    Modulates confidence scores using spatial relationships between detection layers.
    
    This modulator implements complex confidence adjustment algorithms based on spatial
    overlap (IoU) between different detection layers, enabling hierarchical validation
    where Layer 1 (denomination) confidence is modulated by Layer 2 (features) and
    Layer 3 (money validation) predictions.
    
    Features:
    - Spatial confidence computation using IoU matrices
    - Denomination-specific confidence matching between layers
    - Memory-safe processing with chunking for large datasets
    - Class-aware confidence modulation with intelligent fallbacks
    - Hierarchical money validation with confidence boost/reduction
    
    Time Complexity: O(P₁ × max(P₂, P₃)) where P₁, P₂, P₃ are layer prediction counts
    Space Complexity: O(P₁ × max(P₂, P₃)) for IoU matrices, optimized with chunking
    """
    
    def __init__(self, device: torch.device = None, debug: bool = False):
        """
        Initialize confidence modulator.
        
        Args:
            device: Torch device for computations
            debug: Enable debug logging
        """
        self.device = device or torch.device('cpu')
        self.debug = debug
        self.memory_optimizer = get_memory_optimizer()
        
        # Memory safety thresholds
        self.max_matrix_combinations = 50_000_000  # 50M = ~400MB for IoU matrix
        self.max_class_matrix_size = 100_000       # 100K = ~800KB per class
        # TASK.md: More lenient thresholds for rotated images where Layer 2/3 may be partially outside Layer 1
        self.spatial_iou_threshold = 0.05          # Reduced IoU threshold for spatial overlap (was 0.1)
        self.intersection_threshold = 0.02         # Minimum intersection area for rotated images
        self.money_validation_threshold = 0.1      # Layer 3 confidence threshold
        
        # Track large matrix operations to avoid spam logging
        self._logged_large_matrix: Set[int] = set()
    
    def apply_confidence_modulation(
        self,
        all_predictions: torch.Tensor,
        layer_1_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply hierarchical confidence modulation using Layer 2 & 3 predictions.
        
        This is the main entry point for confidence modulation. It extracts Layer 2
        and Layer 3 predictions from all predictions and uses them to modulate
        Layer 1 confidence scores based on spatial relationships.
        
        Args:
            all_predictions: All predictions including all layers
            layer_1_predictions: Filtered Layer 1 predictions only
            
        Returns:
            Layer 1 predictions with modulated confidence scores
            
        Time Complexity: O(P₁ × max(P₂, P₃)) where P₁, P₂, P₃ are layer prediction counts
        Space Complexity: O(P₁ × max(P₂, P₃)) for IoU matrices
        """
        try:
            if len(layer_1_predictions) == 0:
                return layer_1_predictions
            
            # Ensure we're working with 2D tensors
            if all_predictions.dim() != 2 or layer_1_predictions.dim() != 2:
                logger.warning(f"Expected 2D tensors, got all_pred: {all_predictions.shape}, layer_1: {layer_1_predictions.shape}")
                return layer_1_predictions
            
            # Extract predictions by layer using optimized boolean masks
            layer_2_mask = (all_predictions[:, 5] >= 7) & (all_predictions[:, 5] < 14)
            layer_3_mask = all_predictions[:, 5] >= 14
            
            layer_2_preds = all_predictions[layer_2_mask]  # Classes 7-13
            layer_3_preds = all_predictions[layer_3_mask]  # Classes 14-16
            
            # Memory safety check for total combinations
            total_combinations = len(layer_1_predictions) * max(len(layer_2_preds), len(layer_3_preds))
            if total_combinations > self.max_matrix_combinations:
                if self.debug:
                    logger.debug(f"Memory-intensive operation detected ({total_combinations:,} combinations), using fallback")
                return layer_1_predictions  # Return unmodified to avoid OOM
            
            if self.debug:
                logger.debug(f"Confidence modulation: L1={len(layer_1_predictions)}, L2={len(layer_2_preds)}, L3={len(layer_3_preds)}")
                logger.debug(f"Memory estimate: {total_combinations:,} combinations")
            
            # Apply confidence modulation
            modified_predictions = layer_1_predictions.clone()
            original_conf = modified_predictions[:, 4]
            
            # Vectorized confidence computation for both layers
            layer_3_conf = self.compute_spatial_confidence(layer_1_predictions, layer_3_preds)
            layer_2_conf = self.compute_denomination_confidence(layer_1_predictions, layer_2_preds)
            
            # TASK.md compliant hierarchical confidence modulation
            money_mask = layer_3_conf > self.money_validation_threshold if len(layer_3_preds) > 0 else torch.ones_like(original_conf, dtype=torch.bool)
            
            # Check availability of each layer
            has_layer_2 = len(layer_2_preds) > 0
            has_layer_3 = len(layer_3_preds) > 0
            
            # TASK.md: Layer 1 and 2 are equal confidence score (OR logic)
            # If both exist, boost confidence by small number
            if has_layer_2:
                # Apply OR logic: max of layer_1 original confidence and layer_2 boost
                layer_12_confidence = torch.clamp(original_conf + layer_2_conf * 0.1, max=1.0)  # Small boost
            else:
                layer_12_confidence = original_conf
            
            # TASK.md: Layer 3 non-existence shouldn't penalize layer 1 and 2
            # It just decreases confidence score by small number
            if has_layer_3:
                # Layer 3 exists: apply money validation logic
                hierarchical_conf = torch.where(
                    money_mask,
                    torch.clamp(layer_12_confidence + layer_3_conf * 0.05, max=1.0),  # Small boost for money validation
                    layer_12_confidence * 0.95  # Small penalty (5% reduction) if Layer 3 disagrees - much gentler than 0.1
                )
            else:
                # TASK.md: Layer 3 non-existence - small decrease only
                hierarchical_conf = layer_12_confidence * 0.98  # Very small penalty (2% reduction) for Layer 3 absence
            
            modified_predictions[:, 4] = hierarchical_conf
            
            if self.debug and len(layer_1_predictions) > 0:
                has_layer_predictions = has_layer_2 or has_layer_3
                self._log_modulation_sample(layer_1_predictions, original_conf, hierarchical_conf, 
                                          layer_2_conf, layer_3_conf, has_layer_predictions)
            
            return modified_predictions
            
        except Exception as e:
            logger.warning(f"Error in confidence modulation: {e}")
            # Emergency cleanup and return unmodified predictions
            self.memory_optimizer.emergency_memory_cleanup()
            return layer_1_predictions
    
    def compute_spatial_confidence(
        self,
        layer_1_preds: torch.Tensor,
        layer_3_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Layer 3 confidence based on spatial overlap with Layer 1.
        
        Uses IoU computation between Layer 1 and Layer 3 predictions to determine
        spatial confidence. Layer 3 represents money validation features.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(P₁ × P₃) for IoU matrix computation
        Space Complexity: O(P₁ × P₃) for IoU matrix storage
        """
        if len(layer_3_preds) == 0:
            return torch.zeros(len(layer_1_preds), device=self.device)
        
        # Memory safety check for IoU matrix size
        matrix_size = len(layer_1_preds) * len(layer_3_preds)
        max_matrix_size = 1_000_000  # 1M elements = ~8MB for float32
        
        if matrix_size > max_matrix_size:
            return self._chunked_spatial_confidence(layer_1_preds, layer_3_preds)
        
        try:
            # TASK.md: Convert from xywh to xyxy format for YOLOv5 IoU calculation
            layer_1_xyxy = self._xywh_to_xyxy(layer_1_preds[:, :4])
            layer_3_xyxy = self._xywh_to_xyxy(layer_3_preds[:, :4])
            
            # Compute IoU matrix between all Layer 1 and Layer 3 predictions
            iou_matrix = get_box_iou()(layer_1_xyxy, layer_3_xyxy)
            
            # For each Layer 1 prediction, find best overlapping Layer 3 prediction
            max_ious, max_indices = torch.max(iou_matrix, dim=1)
            
            # TASK.md: Apply more lenient spatial validation for rotated images
            valid_mask = self._validate_spatial_overlap(
                layer_1_preds, layer_3_preds, max_ious, max_indices, iou_matrix
            )
            layer_3_conf = torch.zeros(len(layer_1_preds), device=self.device)
            layer_3_conf[valid_mask] = layer_3_preds[max_indices[valid_mask], 4]
            
            return layer_3_conf
            
        except Exception as e:
            logger.warning(f"Error in spatial confidence computation: {e}, using fallback")
            self.memory_optimizer.emergency_memory_cleanup()
            return torch.zeros(len(layer_1_preds), device=self.device)
    
    def compute_denomination_confidence(
        self,
        layer_1_preds: torch.Tensor,
        layer_2_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Layer 2 confidence for specific denominations.
        
        Matches Layer 1 denominations with corresponding Layer 2 confidence features
        using class correspondence (Layer 1 class N corresponds to Layer 2 class N+7).
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_2_preds: Layer 2 predictions
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(C × P₁ × P₂) where C is number of unique classes
        Space Complexity: O(P₁ × P₂) for IoU matrices per class
        """
        if len(layer_2_preds) == 0:
            return torch.zeros(len(layer_1_preds), device=self.device)
        
        # Map Layer 1 classes to corresponding Layer 2 classes
        layer_1_classes = layer_1_preds[:, 5].int()
        layer_2_classes = layer_1_classes + 7  # Classes 7-13 correspond to 0-6
        
        layer_2_conf = torch.zeros(len(layer_1_preds), device=self.device)
        
        try:
            unique_classes = torch.unique(layer_2_classes)
            
            if len(unique_classes) > 0:
                # Process each class with memory safety
                for layer_2_class in unique_classes:
                    layer_2_conf = self._process_class_confidence(
                        layer_1_preds, layer_2_preds, layer_2_classes, 
                        layer_2_class, layer_2_conf
                    )
            
            return layer_2_conf
            
        except Exception as e:
            logger.warning(f"Error in denomination confidence computation: {e}, using fallback")
            self.memory_optimizer.emergency_memory_cleanup()
            return torch.zeros(len(layer_1_preds), device=self.device)
    
    def _chunked_spatial_confidence(
        self,
        layer_1_preds: torch.Tensor,
        layer_3_preds: torch.Tensor,
        chunk_size: int = 500
    ) -> torch.Tensor:
        """
        Process spatial confidence in memory-safe chunks.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions
            chunk_size: Size of each processing chunk
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(P₁ × P₃) total, processed in O(chunk_size × P₃) iterations
        """
        layer_3_conf = torch.zeros(len(layer_1_preds), device=self.device)
        
        for i in range(0, len(layer_1_preds), chunk_size):
            end_idx = min(i + chunk_size, len(layer_1_preds))
            chunk = layer_1_preds[i:end_idx]
            
            try:
                # TASK.md: Convert from xywh to xyxy format for YOLOv5 IoU calculation
                chunk_xyxy = self._xywh_to_xyxy(chunk[:, :4])
                layer_3_xyxy = self._xywh_to_xyxy(layer_3_preds[:, :4])
                
                # Compute IoU for this chunk only
                iou_matrix = get_box_iou()(chunk_xyxy, layer_3_xyxy)
                max_ious, max_indices = torch.max(iou_matrix, dim=1)
                
                # TASK.md: Apply more lenient spatial validation for rotated images
                valid_mask = self._validate_spatial_overlap_simple(max_ious)
                layer_3_conf[i:end_idx][valid_mask] = layer_3_preds[max_indices[valid_mask], 4]
                
            except Exception as e:
                logger.warning(f"Error in chunk {i//chunk_size}: {e}")
                # Skip this chunk to avoid total failure
                continue
        
        return layer_3_conf
    
    def _validate_spatial_overlap(
        self,
        layer_1_preds: torch.Tensor,
        layer_3_preds: torch.Tensor,
        max_ious: torch.Tensor,
        max_indices: torch.Tensor,
        iou_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        TASK.md: Validate spatial overlap for rotated images with lenient criteria.
        
        In rotated images, Layer 2/3 might have small portions outside Layer 1 boxes.
        This method uses both IoU and intersection area to be more permissive.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions  
            max_ious: Maximum IoU values per Layer 1 prediction
            max_indices: Indices of best matching Layer 3 predictions
            iou_matrix: Full IoU matrix
            
        Returns:
            Boolean mask of valid spatial overlaps
        """
        # Basic IoU threshold (more lenient than before)  
        iou_valid = max_ious > self.spatial_iou_threshold
        
        # Additional intersection-based validation for rotated images
        intersection_valid = self._compute_intersection_validation(
            layer_1_preds, layer_3_preds, max_indices, max_ious
        )
        
        # Accept if either IoU or intersection criteria are met (OR logic)
        # This handles rotated images where IoU might be low but intersection is meaningful
        valid_mask = iou_valid | intersection_valid
        
        return valid_mask
    
    def _validate_spatial_overlap_simple(self, max_ious: torch.Tensor) -> torch.Tensor:
        """
        TASK.md: Simple spatial overlap validation with reduced threshold for rotated images.
        
        Args:
            max_ious: Maximum IoU values
            
        Returns:
            Boolean mask of valid overlaps
        """
        return max_ious > self.spatial_iou_threshold
    
    def _compute_intersection_validation(
        self,
        layer_1_preds: torch.Tensor,
        layer_3_preds: torch.Tensor,
        max_indices: torch.Tensor,
        max_ious: torch.Tensor
    ) -> torch.Tensor:
        """
        TASK.md: Compute intersection-based validation for rotated images.
        
        This validates based on absolute intersection area rather than IoU ratio,
        which is more permissive for cases where Layer 2/3 extends outside Layer 1.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions
            max_indices: Indices of best matching predictions
            max_ious: Maximum IoU values
            
        Returns:  
            Boolean mask of valid intersections
        """
        try:
            # Get matched pairs
            l1_boxes = layer_1_preds[:, :4]  # [x, y, w, h]
            l3_boxes = layer_3_preds[max_indices, :4]  # Best matching L3 boxes
            
            # Convert to [x1, y1, x2, y2] format for intersection calculation
            l1_x1 = l1_boxes[:, 0] - l1_boxes[:, 2] / 2
            l1_y1 = l1_boxes[:, 1] - l1_boxes[:, 3] / 2  
            l1_x2 = l1_boxes[:, 0] + l1_boxes[:, 2] / 2
            l1_y2 = l1_boxes[:, 1] + l1_boxes[:, 3] / 2
            
            l3_x1 = l3_boxes[:, 0] - l3_boxes[:, 2] / 2
            l3_y1 = l3_boxes[:, 1] - l3_boxes[:, 3] / 2
            l3_x2 = l3_boxes[:, 0] + l3_boxes[:, 2] / 2
            l3_y2 = l3_boxes[:, 1] + l3_boxes[:, 3] / 2
            
            # Compute intersection area
            inter_x1 = torch.max(l1_x1, l3_x1)
            inter_y1 = torch.max(l1_y1, l3_y1)
            inter_x2 = torch.min(l1_x2, l3_x2)
            inter_y2 = torch.min(l1_y2, l3_y2)
            
            # Calculate intersection area (0 if no overlap)
            inter_w = (inter_x2 - inter_x1).clamp(min=0)
            inter_h = (inter_y2 - inter_y1).clamp(min=0)
            intersection_area = inter_w * inter_h
            
            # Normalize by image area (assuming 640x640 for now)
            normalized_intersection = intersection_area / (640.0 * 640.0)
            
            # Accept if intersection area is above threshold
            intersection_valid = normalized_intersection > self.intersection_threshold
            
            return intersection_valid
            
        except Exception as e:
            logger.warning(f"Error in intersection validation: {e}")
            return torch.zeros(len(layer_1_preds), dtype=torch.bool, device=self.device)
    
    def _xywh_to_xyxy(self, boxes: torch.Tensor) -> torch.Tensor:
        """
        Convert boxes from [x_center, y_center, width, height] to [x1, y1, x2, y2] format.
        
        Args:
            boxes: Tensor of shape [N, 4] in xywh format
            
        Returns:
            Tensor of shape [N, 4] in xyxy format
        """
        xyxy = boxes.clone()
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1 = x_center - width/2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1 = y_center - height/2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2  # x2 = x_center + width/2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2  # y2 = y_center + height/2
        return xyxy
    
    def _process_class_confidence(
        self,
        layer_1_preds: torch.Tensor,
        layer_2_preds: torch.Tensor,
        layer_2_classes: torch.Tensor,
        layer_2_class: torch.Tensor,
        layer_2_conf: torch.Tensor
    ) -> torch.Tensor:
        """
        Process confidence for a specific class with memory optimization.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_2_preds: Layer 2 predictions
            layer_2_classes: Layer 2 class mappings
            layer_2_class: Current class being processed
            layer_2_conf: Confidence tensor to update
            
        Returns:
            Updated confidence tensor
            
        Time Complexity: O(P₁ × P₂) for IoU computation per class
        """
        l1_mask = layer_2_classes == layer_2_class
        l2_mask = layer_2_preds[:, 5].int() == layer_2_class
        
        if not l1_mask.any() or not l2_mask.any():
            return layer_2_conf
            
        matching_l1_preds = layer_1_preds[l1_mask]
        matching_l2_preds = layer_2_preds[l2_mask]
        
        # Memory safety check
        matrix_size = len(matching_l1_preds) * len(matching_l2_preds)
        if matrix_size > self.max_class_matrix_size:
            # Use simplified confidence assignment instead of IoU computation
            return self._handle_large_class_matrix(
                l1_mask, matching_l2_preds, layer_2_class, layer_2_conf
            )
        
        # Safe IoU computation for reasonable matrix sizes
        try:
            # TASK.md: Convert from xywh to xyxy format for YOLOv5 IoU calculation
            l1_xyxy = self._xywh_to_xyxy(matching_l1_preds[:, :4])
            l2_xyxy = self._xywh_to_xyxy(matching_l2_preds[:, :4])
            
            iou_matrix = get_box_iou()(l1_xyxy, l2_xyxy)
            max_ious, max_indices = torch.max(iou_matrix, dim=1)
            # TASK.md: Apply more lenient spatial validation for rotated images  
            valid_mask = self._validate_spatial_overlap_simple(max_ious)
            
            # Assign confidences
            l1_indices = torch.where(l1_mask)[0]
            layer_2_conf[l1_indices[valid_mask]] = matching_l2_preds[max_indices[valid_mask], 4]
            
        except Exception as e:
            logger.warning(f"Error processing class {layer_2_class}: {e}")
        
        return layer_2_conf
    
    def _handle_large_class_matrix(
        self,
        l1_mask: torch.Tensor,
        matching_l2_preds: torch.Tensor,
        layer_2_class: torch.Tensor,
        layer_2_conf: torch.Tensor
    ) -> torch.Tensor:
        """
        Handle large class matrices with memory optimization.
        
        Args:
            l1_mask: Layer 1 mask for current class
            matching_l2_preds: Matching Layer 2 predictions
            layer_2_class: Current class
            layer_2_conf: Confidence tensor to update
            
        Returns:
            Updated confidence tensor with average confidence assignment
            
        Time Complexity: O(P₂) for average computation
        """
        # Only log once per class per session to avoid spam
        class_id = layer_2_class.item()
        if class_id not in self._logged_large_matrix:
            matrix_size = l1_mask.sum().item() * len(matching_l2_preds)
            logger.debug(f"Memory optimization: Class {class_id} has {matrix_size:,} prediction pairs, using average confidence")
            self._logged_large_matrix.add(class_id)
        
        if len(matching_l2_preds) > 0:
            avg_conf = matching_l2_preds[:, 4].mean()
            l1_indices = torch.where(l1_mask)[0]
            layer_2_conf[l1_indices] = avg_conf
        
        return layer_2_conf
    
    def _log_modulation_sample(
        self,
        layer_1_predictions: torch.Tensor,
        original_conf: torch.Tensor,
        hierarchical_conf: torch.Tensor,
        layer_2_conf: torch.Tensor,
        layer_3_conf: torch.Tensor,
        has_layer_predictions: bool
    ) -> None:
        """
        Log sample predictions for debugging confidence modulation.
        
        Args:
            layer_1_predictions: Layer 1 predictions
            original_conf: Original confidence scores
            hierarchical_conf: Modulated confidence scores
            layer_2_conf: Layer 2 confidence contributions
            layer_3_conf: Layer 3 confidence contributions
            has_layer_predictions: Whether multi-layer predictions are available
            
        Time Complexity: O(1) - logs only first few predictions
        """
        if not has_layer_predictions:
            logger.debug("No layer 2/3 predictions - confidence unchanged")
            return
        
        # Log first few predictions for debugging
        for i in range(min(3, len(layer_1_predictions))):
            logger.debug(f"Pred {i}: class={layer_1_predictions[i, 5].int().item()}, "
                        f"conf={original_conf[i]:.3f}→{hierarchical_conf[i]:.3f}, "
                        f"L2={layer_2_conf[i]:.3f}, L3={layer_3_conf[i]:.3f}")
    
    def get_modulation_stats(self) -> Dict[str, any]:
        """
        Get statistics about confidence modulation operations.
        
        Returns:
            Dictionary with modulation statistics
            
        Time Complexity: O(1) - simple data collection
        """
        return {
            'memory_thresholds': {
                'max_matrix_combinations': self.max_matrix_combinations,
                'max_class_matrix_size': self.max_class_matrix_size
            },
            'confidence_thresholds': {
                'spatial_iou_threshold': self.spatial_iou_threshold,
                'money_validation_threshold': self.money_validation_threshold
            },
            'optimization_stats': {
                'large_matrix_classes_logged': len(self._logged_large_matrix),
                'device': str(self.device)
            }
        }


class ConfidenceAnalyzer:
    """
    Analyzer for confidence modulation results and performance.
    
    Provides analysis and insights into confidence modulation effectiveness
    and quality metrics.
    """
    
    @staticmethod
    def analyze_confidence_changes(
        original_conf: torch.Tensor,
        modulated_conf: torch.Tensor
    ) -> Dict[str, float]:
        """
        Analyze confidence changes from modulation.
        
        Args:
            original_conf: Original confidence scores
            modulated_conf: Modulated confidence scores
            
        Returns:
            Dictionary with confidence change analysis
            
        Time Complexity: O(N) where N is number of predictions
        """
        confidence_changes = modulated_conf - original_conf
        
        boosted_mask = confidence_changes > 1e-6
        reduced_mask = confidence_changes < -1e-6
        unchanged_mask = torch.abs(confidence_changes) <= 1e-6
        
        return {
            'total_predictions': len(original_conf),
            'boosted_count': boosted_mask.sum().item(),
            'reduced_count': reduced_mask.sum().item(),
            'unchanged_count': unchanged_mask.sum().item(),
            'avg_confidence_change': confidence_changes.mean().item(),
            'max_confidence_boost': confidence_changes.max().item(),
            'max_confidence_reduction': confidence_changes.min().item(),
            'boosted_percentage': (boosted_mask.sum().item() / len(original_conf) * 100),
            'reduced_percentage': (reduced_mask.sum().item() / len(original_conf) * 100)
        }
    
    @staticmethod
    def analyze_layer_contributions(
        layer_2_conf: torch.Tensor,
        layer_3_conf: torch.Tensor,
        money_validation_threshold: float = 0.1
    ) -> Dict[str, any]:
        """
        Analyze layer contributions to confidence modulation.
        
        Args:
            layer_2_conf: Layer 2 confidence contributions
            layer_3_conf: Layer 3 confidence contributions
            money_validation_threshold: Threshold for money validation
            
        Returns:
            Dictionary with layer contribution analysis
            
        Time Complexity: O(N) where N is number of predictions
        """
        layer_2_active = (layer_2_conf > 0.01).sum().item()
        layer_3_active = (layer_3_conf > money_validation_threshold).sum().item()
        
        return {
            'layer_2_active_contributions': layer_2_active,
            'layer_3_active_contributions': layer_3_active,
            'layer_2_active_percentage': (layer_2_active / len(layer_2_conf) * 100),
            'layer_3_active_percentage': (layer_3_active / len(layer_3_conf) * 100),
            'avg_layer_2_conf': layer_2_conf[layer_2_conf > 0.01].mean().item() if layer_2_active > 0 else 0.0,
            'avg_layer_3_conf': layer_3_conf[layer_3_conf > money_validation_threshold].mean().item() if layer_3_active > 0 else 0.0
        }


# Factory functions for backward compatibility
def create_confidence_modulator(device: torch.device = None, debug: bool = False) -> ConfidenceModulator:
    """
    Factory function to create confidence modulator.
    
    Args:
        device: Torch device for computations
        debug: Enable debug logging
        
    Returns:
        ConfidenceModulator instance
        
    Time Complexity: O(1) - simple object creation
    """
    return ConfidenceModulator(device, debug)


# Export public interface
__all__ = [
    'ConfidenceModulator',
    'ConfidenceAnalyzer',
    'create_confidence_modulator'
]