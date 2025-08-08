#!/usr/bin/env python3
"""
Batch processor for YOLOv5 mAP calculations in SmartCash.

Handles batch-level processing of predictions and targets, including:
- Tensor format validation and conversion
- Confidence filtering and coordinate transformation
- IoU computation and prediction-target matching
- Statistics generation for mAP calculation

Algorithmic Optimizations:
- Vectorized confidence filtering: O(1) operation across all batches
- Optimized coordinate conversion: O(N) batch operations
- Efficient IoU matrix computation: O(P*T) where P=predictions, T=targets
- Memory-conscious batch handling: O(batch_size) space complexity
"""

import torch
from typing import Optional, Tuple, Dict, Any

from smartcash.common.logger import get_logger
from .ultralytics_utils_manager import get_xywh2xyxy, get_box_iou
from .memory_optimized_processor import MemoryOptimizedProcessor

logger = get_logger(__name__, level="DEBUG")


class BatchProcessor:
    """
    Processes batches of predictions and targets for mAP calculation.
    
    This processor handles all the complex logic of tensor format validation,
    coordinate transformations, confidence filtering, and IoU-based matching
    between predictions and ground truth targets.
    
    Time Complexity: O(P*T) for IoU computation, O(P+T) for other operations
    Space Complexity: O(P*T) for IoU matrix, O(P+T) for tracking arrays
    """
    
    def __init__(
        self, 
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.5,
        device: Optional[torch.device] = None,
        debug: bool = False
    ):
        """
        Initialize batch processor.
        
        Args:
            conf_threshold: Confidence threshold for filtering predictions
            iou_threshold: IoU threshold for matching predictions to targets
            device: Torch device for computations
            debug: Enable debug logging
        """
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.device = device or torch.device('cpu')
        self.debug = debug
        
        # Initialize memory optimizer for efficient processing
        self.memory_processor = MemoryOptimizedProcessor(device=self.device, debug=debug)
        
    def process_batch(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Process a batch of predictions and targets for mAP calculation.
        
        Args:
            predictions: Processed predictions [N, 7] where each row is 
                        [batch_idx, x1, y1, x2, y2, conf, class]
            targets: Targets [M, 6] where each row is [batch_idx, class, x, y, w, h]
        
        Returns:
            Tuple of (tp, conf, pred_cls, target_cls) for mAP calculation,
            or None if processing fails
            
        Time Complexity: O(P*T) for IoU computation + O(P log P) for sorting
        Space Complexity: O(P*T) for IoU matrix + O(P+T) for tracking
        """
        try:
            # Handle empty prediction case
            if predictions.shape[0] == 0:
                return self._handle_empty_predictions(targets)
            
            # Handle empty targets case  
            if targets.shape[0] == 0:
                return self._handle_empty_targets(predictions)
            
            # Validate tensor formats
            if not self._validate_tensor_formats(predictions, targets):
                if self.debug:
                    logger.debug(f"❌ Tensor format validation failed - pred_shape: {predictions.shape}, target_shape: {targets.shape}")
                return None
            
            # Convert target format and validate
            target_boxes = self._convert_target_format(targets)
            if target_boxes is None:
                if self.debug:
                    logger.debug(f"❌ Target format conversion failed")
                return None
            
            # Compute IoU matrix with memory optimization
            iou_matrix = self._compute_iou_matrix(predictions, target_boxes)
            if iou_matrix is None:
                if self.debug:
                    logger.debug(f"❌ IoU matrix computation failed")
                return None
            
            # Perform optimized prediction-target matching
            tp = self._match_predictions_to_targets(
                predictions, target_boxes, iou_matrix
            )
            
            # Extract final statistics
            return self._extract_statistics(predictions, target_boxes, tp)
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}")
            if self.debug:
                import traceback
                logger.debug(f"Batch processing traceback: {traceback.format_exc()}")
            return None
        finally:
            # Clean up memory after batch processing
            self.memory_processor.cleanup_after_batch()
    
    def preprocess_predictions(
        self, 
        raw_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Preprocess raw predictions with confidence filtering and format conversion.
        
        Args:
            raw_predictions: Raw predictions tensor in various formats
            
        Returns:
            Processed predictions in standardized format [N, 7]
            
        Time Complexity: O(P) where P is number of predictions
        Space Complexity: O(P) for processed predictions
        """
        try:
            if raw_predictions.numel() == 0:
                return torch.empty((0, 7), device=self.device)
            
            # Handle different prediction tensor formats
            if raw_predictions.dim() == 3:
                return self._process_3d_predictions(raw_predictions)
            elif raw_predictions.dim() == 2:
                return self._process_2d_predictions(raw_predictions)
            else:
                logger.warning(f"Unexpected prediction tensor dimensions: {raw_predictions.shape}")
                return torch.empty((0, 7), device=raw_predictions.device)
                
        except Exception as e:
            logger.error(f"Error preprocessing predictions: {e}")
            return torch.empty((0, 7), device=self.device)
    
    def _process_3d_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Process 3D prediction tensor [batch, detections, 6].
        
        Args:
            predictions: 3D predictions tensor
            
        Returns:
            Standardized predictions [N, 7]
            
        Time Complexity: O(B*D) where B=batch_size, D=max_detections
        Space Complexity: O(B*D) for flattened predictions
        """
        batch_size, max_detections, _ = predictions.shape
        
        # Vectorized confidence filtering across all batches - O(1) operation
        conf_mask = predictions[:, :, 4] > self.conf_threshold
        
        # Flatten predictions and create batch indices - O(N) total
        valid_mask = conf_mask.flatten()
        
        if not valid_mask.any():
            return torch.empty((0, 7), device=predictions.device)
        
        # Flatten all predictions
        flat_predictions = predictions.view(-1, predictions.shape[-1])
        valid_predictions = flat_predictions[valid_mask]
        
        # Create batch indices efficiently - O(N)
        batch_indices = torch.arange(
            batch_size, 
            device=predictions.device, 
            dtype=torch.float32
        )
        batch_indices = batch_indices.repeat_interleave(max_detections)[valid_mask]
        
        # Convert coordinates vectorized - O(N)
        pred_xyxy = valid_predictions.clone()
        pred_xyxy[:, :4] = get_xywh2xyxy()(valid_predictions[:, :4])
        
        # Combine batch indices with predictions - O(N)
        # Final format: [batch_idx, x1, y1, x2, y2, conf, class]
        return torch.cat([
            batch_indices.unsqueeze(1), 
            pred_xyxy[:, :4], 
            valid_predictions[:, 4:]
        ], dim=1)
    
    def _process_2d_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Process 2D prediction tensor [detections, 6].
        
        Args:
            predictions: 2D predictions tensor
            
        Returns:
            Standardized predictions [N, 7]
            
        Time Complexity: O(D) where D=number of detections
        Space Complexity: O(D) for processed predictions
        """
        # Apply confidence threshold filtering
        conf_mask = predictions[:, 4] > self.conf_threshold
        valid_predictions = predictions[conf_mask]
        
        if len(valid_predictions) == 0:
            return torch.empty((0, 7), device=predictions.device)
        
        # Add batch indices (assume all predictions are from batch 0)
        batch_indices = torch.zeros(len(valid_predictions), 1, device=predictions.device)
        
        # Convert coordinates
        pred_xyxy = valid_predictions.clone()
        pred_xyxy[:, :4] = get_xywh2xyxy()(valid_predictions[:, :4])
        
        # Combine: [batch_idx, x1, y1, x2, y2, conf, class]
        return torch.cat([
            batch_indices, 
            pred_xyxy[:, :4], 
            valid_predictions[:, 4:]
        ], dim=1)
    
    def _handle_empty_predictions(
        self, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Handle case where there are no predictions but targets exist.
        
        Args:
            targets: Target tensor
            
        Returns:
            Statistics tuple for empty predictions case
            
        Time Complexity: O(T) where T is number of targets
        """
        if targets.shape[0] > 0:  # Have targets (all FN)
            return (
                torch.zeros((0, 1), dtype=torch.bool, device=self.device),  # tp
                torch.zeros(0, device=self.device),  # conf  
                torch.zeros(0, dtype=torch.int32, device=self.device),  # pred_cls
                targets[:, 1].int()  # target_cls
            )
        else:  # No predictions and no targets
            return (
                torch.zeros((0, 1), dtype=torch.bool, device=self.device),  # tp
                torch.zeros(0, device=self.device),  # conf
                torch.zeros(0, dtype=torch.int32, device=self.device),  # pred_cls
                torch.zeros(0, dtype=torch.int32, device=self.device)  # target_cls
            )
    
    def _handle_empty_targets(
        self, 
        predictions: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Handle case where there are no targets but predictions exist.
        
        Args:
            predictions: Predictions tensor
            
        Returns:
            Statistics tuple for empty targets case, or None if invalid format
            
        Time Complexity: O(1) - simple tensor access
        """
        # Validate prediction tensor format
        if predictions.shape[1] < 7:
            logger.error(f"🚨 Invalid prediction format: shape={predictions.shape}, expected [..., 7]")
            return None
            
        return (
            torch.zeros((predictions.shape[0], 1), dtype=torch.bool, device=self.device),  # tp
            predictions[:, 5],  # conf (column 5)
            predictions[:, 6].int(),  # pred_cls (column 6)
            torch.zeros(0, dtype=torch.int32, device=self.device)  # target_cls
        )
    
    def _validate_tensor_formats(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> bool:
        """
        Validate tensor formats before processing.
        
        Args:
            predictions: Predictions tensor
            targets: Targets tensor
            
        Returns:
            True if formats are valid, False otherwise
            
        Time Complexity: O(1) - simple shape checks
        """
        # Validate predictions format
        if predictions.shape[1] < 7:
            logger.error(f"🚨 Invalid prediction format for IoU: shape={predictions.shape}, expected [..., 7]")
            return False
            
        # Validate targets format
        if targets.shape[1] < 6:
            logger.error(f"🚨 Invalid target format for IoU: shape={targets.shape}, expected [..., 6]")
            return False
        
        return True
    
    def _convert_target_format(self, targets: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Convert target format from [batch_idx, class, x, y, w, h] to [batch_idx, x1, y1, x2, y2, class].
        
        UPDATED: Added filtering for empty/invalid targets based on mAP debug log analysis.
        Many targets were [0.0000, 0.0000, 0.0000, 0.0000] causing invalid IoU calculations.
        
        Args:
            targets: Targets tensor in original format
            
        Returns:
            Converted targets tensor with empty targets filtered out, or None if conversion fails
            
        Time Complexity: O(T) where T is number of targets
        Space Complexity: O(T) for converted targets
        """
        try:
            # CRITICAL FIX: Filter out empty/invalid targets first
            # Based on mAP debug logs, many targets are [0.0000, 0.0000, 0.0000, 0.0000]
            # These cause false IoU matches and corrupt mAP calculations
            
            # Check for valid targets (non-zero area)
            target_areas = targets[:, 4] * targets[:, 5]  # width * height
            valid_mask = target_areas > 1e-6  # Filter out near-zero area targets
            
            if self.debug and valid_mask.sum() < len(targets):
                removed_count = len(targets) - valid_mask.sum()
                logger.debug(f"🧹 Filtered out {removed_count} empty/invalid targets (area ≤ 1e-6)")
            
            # Keep only valid targets
            valid_targets = targets[valid_mask]
            
            if len(valid_targets) == 0:
                if self.debug:
                    logger.debug("⚠️ All targets filtered out - no valid targets remaining")
                return torch.empty((0, 6), device=targets.device)
            
            target_boxes = valid_targets.clone()
            
            # Convert xywh to xyxy coordinates
            target_boxes[:, 2:6] = get_xywh2xyxy()(valid_targets[:, 2:6])
            
            # Reorder to [batch_idx, x1, y1, x2, y2, class]
            target_boxes = target_boxes[:, [0, 2, 3, 4, 5, 1]]
            
            return target_boxes
            
        except Exception as e:
            logger.error(f"Error converting target format: {e}")
            return None
    
    def _compute_iou_matrix(
        self, 
        predictions: torch.Tensor, 
        target_boxes: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Compute IoU matrix between predictions and targets.
        
        Args:
            predictions: Predictions tensor [N, 7]
            target_boxes: Target boxes tensor [M, 6]
            
        Returns:
            IoU matrix [N, M], or None if computation fails
            
        Time Complexity: O(N*M) for IoU computation
        Space Complexity: O(N*M) for IoU matrix
        """
        try:
            # Extract bounding boxes
            pred_boxes = predictions[:, 1:5]  # columns 1-4: x1, y1, x2, y2
            target_boxes_xyxy = target_boxes[:, 1:5]  # columns 1-4: x1, y1, x2, y2
            
            # Check memory requirements
            memory_stats = self.memory_processor.estimate_memory_usage(
                len(pred_boxes), len(target_boxes_xyxy)
            )
            
            if memory_stats['recommend_chunking']:
                logger.debug(f"Large IoU matrix ({memory_stats['total_mb']:.1f}MB), using chunked computation")
                return self._compute_iou_matrix_chunked(pred_boxes, target_boxes_xyxy)
            
            # Direct computation for reasonable sizes
            iou_matrix = get_box_iou()(pred_boxes, target_boxes_xyxy)
            
            # IoU DISTRIBUTION DEBUG LOGGING - Simplified
            if self.debug and iou_matrix is not None and iou_matrix.numel() > 0:
                max_ious, _ = torch.max(iou_matrix, dim=1)
                valid_ious = max_ious[max_ious > 0]
                if len(valid_ious) > 0:
                    logger.debug(f"🔍 IoU DISTRIBUTION: Max={max_ious.max().item():.4f}, Mean={valid_ious.mean().item():.4f}")
                else:
                    logger.debug(f"❌ IoU ANALYSIS: ALL IoUs are ZERO! Matrix shape: {iou_matrix.shape}")
            
            return iou_matrix
            
        except Exception as e:
            logger.error(f"Error computing IoU matrix: {e}")
            return None
    
    def _compute_iou_matrix_chunked(
        self, 
        pred_boxes: torch.Tensor, 
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU matrix in memory-safe chunks.
        
        Args:
            pred_boxes: Prediction boxes tensor
            target_boxes: Target boxes tensor
            
        Returns:
            Complete IoU matrix
            
        Time Complexity: O(N*M) total, processed in chunks
        Space Complexity: O(chunk_size * M) per iteration
        """
        chunk_size = self.memory_processor.config.chunk_size
        num_chunks = (len(pred_boxes) + chunk_size - 1) // chunk_size
        
        iou_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = min((i + 1) * chunk_size, len(pred_boxes))
            
            pred_chunk = pred_boxes[start_idx:end_idx]
            iou_chunk = get_box_iou()(pred_chunk, target_boxes)
            iou_chunks.append(iou_chunk)
            
            # Periodic cleanup
            if i % 5 == 0:
                self.memory_processor.memory_optimizer.cleanup_memory()
        
        return torch.cat(iou_chunks, dim=0)
    
    def _match_predictions_to_targets(
        self,
        predictions: torch.Tensor,
        target_boxes: torch.Tensor,
        iou_matrix: torch.Tensor
    ) -> torch.Tensor:
        """
        Match predictions to targets using optimized greedy assignment.
        
        Args:
            predictions: Predictions tensor
            target_boxes: Target boxes tensor
            iou_matrix: IoU matrix between predictions and targets
            
        Returns:
            Boolean tensor indicating true positives
            
        Time Complexity: O(N log N) for sorting + O(N) for assignment
        Space Complexity: O(N+M) for tracking arrays
        """
        return self.memory_processor.optimize_greedy_assignment(
            predictions, target_boxes, iou_matrix, self.iou_threshold
        )
    
    def _extract_statistics(
        self,
        predictions: torch.Tensor,
        target_boxes: torch.Tensor,
        tp: torch.Tensor
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Extract final statistics for mAP calculation.
        
        Args:
            predictions: Predictions tensor
            target_boxes: Target boxes tensor
            tp: True positives tensor
            
        Returns:
            Statistics tuple (tp, conf, pred_cls, target_cls)
            
        Time Complexity: O(1) - simple tensor access
        """
        try:
            # Validate tensor formats before extraction
            if predictions.shape[1] < 7:
                logger.error(f"🚨 Invalid prediction format at return: shape={predictions.shape}, expected [..., 7]")
                return None
                
            if target_boxes.shape[1] < 6:
                logger.error(f"🚨 Invalid target format at return: shape={target_boxes.shape}, expected [..., 6]")
                return None
            
            # Extract statistics with bounds checking
            conf_scores = predictions[:, 5]  # Confidence scores (column 5)
            pred_classes = predictions[:, 6].int()  # Predicted classes (column 6)
            target_classes = target_boxes[:, 5].int()  # Target classes (column 5)
            
            return (tp, conf_scores, pred_classes, target_classes)
            
        except IndexError as e:
            logger.error(f"Index error during statistics extraction: {e}")
            logger.error(f"Prediction shape: {predictions.shape}, Target shape: {target_boxes.shape}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error during statistics extraction: {e}")
            return None
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Returns:
            Dictionary with processing statistics
            
        Time Complexity: O(1) - simple data collection
        """
        return {
            'conf_threshold': self.conf_threshold,
            'iou_threshold': self.iou_threshold,
            'device': str(self.device),
            'memory_processor_stats': self.memory_processor.get_processing_stats()
        }