#!/usr/bin/env python3
"""
Hierarchical processing module for multi-layer architecture in SmartCash.

Handles Phase 2 multi-layer hierarchical system:
- Layer 1: Denomination detection (classes 0-6)
- Layer 2: Confidence features (classes 7-13) 
- Layer 3: Money validation (classes 14-16)

Provides optimized confidence modulation using spatial relationships between layers.

Algorithmic Complexity:
- Filtering: O(P) where P is number of predictions
- Confidence modulation: O(P₁ * P₂) where P₁, P₂ are layer prediction counts
- Chunked processing: O(P₁) with controlled memory usage
"""

import torch
from typing import Tuple, Optional

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
from .yolo_utils_manager import get_box_iou

logger = get_logger(__name__, level="DEBUG")


class HierarchicalProcessor:
    """
    Processes multi-layer hierarchical predictions for Phase 2 architecture.
    
    This processor handles the complex logic of filtering and modulating confidence
    scores across multiple detection layers while maintaining memory efficiency
    and optimal performance.
    
    Time Complexity: O(P) for filtering, O(P₁ * P₂) for confidence modulation
    Space Complexity: O(P) where P is the number of predictions
    """
    
    def __init__(self, device: Optional[torch.device] = None, debug: bool = False, training_context: dict = None):
        """
        Initialize hierarchical processor.
        
        Args:
            device: Torch device for computations
            debug: Enable debug logging
            training_context: Training context information (backbone, phase, etc.)
        """
        self.device = device or torch.device('cpu')
        self.debug = debug
        self.training_context = training_context or {}
        self.memory_optimizer = get_memory_optimizer()
        
        # Memory safety thresholds
        self.max_predictions_per_chunk = 10000
        self.max_matrix_combinations = 50_000_000  # 50M = ~400MB for IoU matrix
        self.max_class_matrix_size = 100_000  # 100K = ~800KB per class
        
        # Layer configuration
        self.layer_1_classes = range(0, 7)   # Classes 0-6
        self.layer_2_classes = range(7, 14)  # Classes 7-13
        self.layer_3_classes = range(14, 17) # Classes 14-16
        
        # Initialize debug file logging if enabled (after layer configuration)
        self.debug_logger = None
        if debug:
            self._setup_hierarchical_debug_logging()
    
    def _setup_hierarchical_debug_logging(self):
        """
        Set up dedicated debug file logging for hierarchical processing.
        
        Creates debug log file in logs/validation_metrics/ with timestamp and training context.
        """
        from pathlib import Path
        import logging
        from datetime import datetime
        
        # Create debug log directory
        debug_log_dir = Path("logs/validation_metrics")
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract context information for filename
        backbone = self.training_context.get('backbone', 'unknown')
        phase = self.training_context.get('current_phase', 'unknown')
        training_mode = self.training_context.get('training_mode', 'unknown')
        
        # Create timestamped debug log file with context
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        debug_log_file = debug_log_dir / f"hierarchical_debug_{backbone}_{training_mode}_phase{phase}_{timestamp}.log"
        
        # Set up dedicated debug logger
        self.debug_logger = logging.getLogger(f"hierarchical_debug_{timestamp}")
        self.debug_logger.setLevel(logging.DEBUG)
        
        # Remove existing handlers to avoid duplicates
        self.debug_logger.handlers.clear()
        
        # Create file handler
        file_handler = logging.FileHandler(debug_log_file, mode='w')
        file_handler.setLevel(logging.DEBUG)
        
        # Create detailed formatter for debug file
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        
        # Add handler to debug logger
        self.debug_logger.addHandler(file_handler)
        
        # Log initialization with comprehensive context
        self.debug_logger.info("=" * 80)
        self.debug_logger.info("Hierarchical Processor Debug Session Started")
        self.debug_logger.info("=" * 80)
        
        # Log training context information
        self.debug_logger.info("Training Context:")
        self.debug_logger.info(f"  • Backbone: {backbone}")
        self.debug_logger.info(f"  • Training Mode: {training_mode}")
        self.debug_logger.info(f"  • Current Phase: {phase}")
        self.debug_logger.info(f"  • Session ID: {self.training_context.get('session_id', 'N/A')}")
        self.debug_logger.info(f"  • Model Name: {self.training_context.get('model_name', 'N/A')}")
        self.debug_logger.info(f"  • Layer Mode: {self.training_context.get('layer_mode', 'N/A')}")
        self.debug_logger.info(f"  • Detection Layers: {self.training_context.get('detection_layers', 'N/A')}")
        
        # Log processor configuration
        self.debug_logger.info("")
        self.debug_logger.info("Configuration:")
        self.debug_logger.info(f"  • device: {self.device}")
        self.debug_logger.info(f"  • layer_1_classes: {list(self.layer_1_classes)}")
        self.debug_logger.info(f"  • layer_2_classes: {list(self.layer_2_classes)}")
        self.debug_logger.info(f"  • layer_3_classes: {list(self.layer_3_classes)}")
        self.debug_logger.info(f"  • max_predictions_per_chunk: {self.max_predictions_per_chunk:,}")
        self.debug_logger.info(f"  • max_matrix_combinations: {self.max_matrix_combinations:,}")
        
        # Log file information
        self.debug_logger.info("")
        self.debug_logger.info(f"Debug log file: {debug_log_file}")
        self.debug_logger.info("")
        
        # Log to console about debug file creation
        logger.info(f"📄 Hierarchical debug log file created: {debug_log_file}")
    
    def _hierarchical_debug_log(self, message: str):
        """
        Write message to hierarchical debug log file if debug mode is enabled.
        
        Args:
            message: Message to log
        """
        if self.debug_logger:
            self.debug_logger.info(message)
        
    def process_hierarchical_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical filtering and confidence modulation.
        
        Args:
            predictions: Raw predictions tensor
            targets: Target tensor
            
        Returns:
            Tuple of (filtered_predictions, filtered_targets)
            
        Time Complexity: O(P) for filtering + O(P₁ * P₂) for modulation
        Space Complexity: O(P) for storing processed predictions
        """
        try:
            # Input validation
            if predictions is None or targets is None:
                return predictions, targets
                
            if predictions.numel() == 0:
                return predictions, targets
            
            # Detect processing phase based on prediction classes
            phase = self._detect_processing_phase(predictions)
            
            if self.debug:
                self._hierarchical_debug_log(f"\n{'='*60}")
                self._hierarchical_debug_log(f"🔍 HIERARCHICAL PROCESSING BATCH")
                self._hierarchical_debug_log(f"{'='*60}")
                self._hierarchical_debug_log(f"Input tensor shapes:")
                self._hierarchical_debug_log(f"  • predictions: {predictions.shape}")
                self._hierarchical_debug_log(f"  • targets: {targets.shape}")
            
            if phase == 1:
                # Phase 1: Standard single-layer processing
                if self.debug:
                    logger.debug("🔹 PHASE 1: Standard single-layer processing")
                    self._hierarchical_debug_log("🔹 PHASE 1 DETECTED: Standard single-layer processing")
                    self._hierarchical_debug_log("  • Hierarchical processing SKIPPED")
                    self._hierarchical_debug_log("  • Returning original predictions and targets unchanged")
                return predictions, targets
            
            # Phase 2: Multi-layer hierarchical processing
            if self.debug:
                logger.debug("🔹 PHASE 2: Hierarchical multi-layer processing")
                self._hierarchical_debug_log("🔹 PHASE 2 DETECTED: Hierarchical multi-layer processing")
                self._hierarchical_debug_log("  • Applying hierarchical filtering and confidence modulation")
                
            return self._process_phase2_predictions(predictions, targets)
            
        except Exception as e:
            logger.warning(f"Error in hierarchical processing: {e}")
            if self.debug:
                import traceback
                logger.debug(f"Hierarchical processing traceback: {traceback.format_exc()}")
            return predictions, targets
    
    def _detect_processing_phase(self, predictions: torch.Tensor) -> int:
        """
        Detect whether we're in Phase 1 or Phase 2 based on prediction classes.
        
        Args:
            predictions: Predictions tensor
            
        Returns:
            int: 1 for Phase 1, 2 for Phase 2
            
        Time Complexity: O(P) where P is number of predictions
        """
        try:
            # Extract unique classes from predictions
            if predictions.dim() == 3:
                # [batch, detections, features] -> flatten to check classes
                flat_predictions = predictions.view(-1, predictions.shape[-1])
                unique_classes = torch.unique(flat_predictions[:, 5].long()) if flat_predictions.numel() > 0 else torch.tensor([])
            elif predictions.dim() == 2:
                # [detections, features]
                unique_classes = torch.unique(predictions[:, 5].long()) if predictions.numel() > 0 else torch.tensor([])
            else:
                logger.warning(f"Unexpected prediction tensor shape: {predictions.shape}")
                return 1
                
            max_class = unique_classes.max().item() if len(unique_classes) > 0 else 0
            
            # Phase 1: classes 0-6 only, Phase 2: classes > 6
            return 1 if max_class < 7 else 2
            
        except Exception as e:
            logger.warning(f"Error detecting phase: {e}")
            return 1
    
    def _process_phase2_predictions(
        self, 
        predictions: torch.Tensor, 
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Process Phase 2 multi-layer predictions.
        
        Args:
            predictions: Raw predictions tensor
            targets: Targets tensor
            
        Returns:
            Tuple of (filtered_predictions, filtered_targets)
            
        Time Complexity: O(P) for filtering + O(P₁ * P₂) for confidence modulation
        """
        if self.debug:
            unique_classes = self._get_unique_classes(predictions)
            logger.debug(f"  • Original prediction classes: {unique_classes}")
            logger.debug(f"  • Prediction tensor shape: {predictions.shape}")
            
            self._hierarchical_debug_log("\n📊 PHASE 2 PREDICTION ANALYSIS:")
            self._hierarchical_debug_log(f"  • Original prediction classes detected: {unique_classes}")
            self._hierarchical_debug_log(f"  • Total predictions: {predictions.numel() // predictions.shape[-1] if predictions.numel() > 0 else 0}")
            self._hierarchical_debug_log(f"  • Layer breakdown by class ranges:")
            
            # Analyze predictions by layer
            if predictions.dim() >= 2:
                flat_preds = predictions.view(-1, predictions.shape[-1]) if predictions.dim() == 3 else predictions
                if flat_preds.numel() > 0:
                    pred_classes = flat_preds[:, 5].long()
                    layer_1_count = ((pred_classes >= 0) & (pred_classes < 7)).sum().item()
                    layer_2_count = ((pred_classes >= 7) & (pred_classes < 14)).sum().item()  
                    layer_3_count = (pred_classes >= 14).sum().item()
                    
                    self._hierarchical_debug_log(f"    - Layer 1 (classes 0-6): {layer_1_count} predictions")
                    self._hierarchical_debug_log(f"    - Layer 2 (classes 7-13): {layer_2_count} predictions")
                    self._hierarchical_debug_log(f"    - Layer 3 (classes 14-16): {layer_3_count} predictions")
        
        # Handle different tensor dimensions
        if predictions.dim() == 3:
            filtered_predictions = self._process_3d_predictions(predictions)
        elif predictions.dim() == 2:
            filtered_predictions = self._process_2d_predictions(predictions)
        else:
            logger.warning(f"Unsupported prediction tensor dimensions: {predictions.dim()}")
            return predictions, targets
        
        # Filter targets to Layer 1 only (classes 0-6)
        layer_1_targets = targets[targets[..., 1] < 7] if targets.numel() > 0 else targets
        
        if self.debug:
            logger.debug(f"  • Filtered Layer 1 predictions: {len(filtered_predictions)}")
            logger.debug(f"  • Filtered targets: {len(layer_1_targets)} (Layer 1 only)")
            if len(filtered_predictions) > 0:
                filtered_classes = torch.unique(filtered_predictions[:, 5].long())
                logger.debug(f"  • Final prediction classes: {filtered_classes.tolist()}")
            
            # Detailed hierarchical processing summary
            self._hierarchical_debug_log("\n🎯 HIERARCHICAL PROCESSING RESULTS:")
            self._hierarchical_debug_log(f"  • Layer 1 predictions after filtering: {len(filtered_predictions)}")
            self._hierarchical_debug_log(f"  • Layer 1 targets after filtering: {len(layer_1_targets)}")
            
            if len(filtered_predictions) > 0:
                # Confidence analysis
                confidences = filtered_predictions[:, 4]
                original_conf_sum = confidences.sum().item()
                avg_confidence = confidences.mean().item()
                min_confidence = confidences.min().item()
                max_confidence = confidences.max().item()
                
                self._hierarchical_debug_log(f"  • Confidence statistics:")
                self._hierarchical_debug_log(f"    - Average: {avg_confidence:.4f}")
                self._hierarchical_debug_log(f"    - Min: {min_confidence:.4f}")
                self._hierarchical_debug_log(f"    - Max: {max_confidence:.4f}")
                self._hierarchical_debug_log(f"    - Total confidence sum: {original_conf_sum:.4f}")
                
                # Final class distribution
                final_classes = torch.unique(filtered_predictions[:, 5].long())
                self._hierarchical_debug_log(f"  • Final prediction classes: {final_classes.tolist()}")
                
                # Class count breakdown
                for cls in final_classes:
                    cls_count = (filtered_predictions[:, 5] == cls).sum().item()
                    cls_avg_conf = filtered_predictions[filtered_predictions[:, 5] == cls, 4].mean().item()
                    self._hierarchical_debug_log(f"    - Class {cls}: {cls_count} predictions, avg_conf={cls_avg_conf:.4f}")
            else:
                self._hierarchical_debug_log("  ⚠️  NO LAYER 1 PREDICTIONS after filtering!")
                
            if len(layer_1_targets) == 0:
                self._hierarchical_debug_log("  ⚠️  NO LAYER 1 TARGETS after filtering!")
        
        return filtered_predictions, layer_1_targets
    
    def _process_3d_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Process 3D prediction tensor [batch, detections, features].
        
        Args:
            predictions: 3D predictions tensor
            
        Returns:
            Filtered and modulated predictions
            
        Time Complexity: O(P) where P is total number of predictions
        """
        _, _, num_features = predictions.shape
        flat_predictions = predictions.view(-1, num_features)
        
        # Filter to Layer 1 classes (0-6) only
        layer_1_mask = flat_predictions[:, 5] < 7
        layer_1_predictions = flat_predictions[layer_1_mask]
        
        if len(layer_1_predictions) == 0:
            return torch.empty((0, num_features), device=predictions.device)
        
        # Apply hierarchical confidence modulation
        return self._apply_confidence_modulation(flat_predictions, layer_1_predictions)
    
    def _process_2d_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        """
        Process 2D prediction tensor [detections, features].
        
        Args:
            predictions: 2D predictions tensor
            
        Returns:
            Filtered and modulated predictions
            
        Time Complexity: O(P) where P is number of predictions
        """
        # Filter to Layer 1 classes (0-6) only
        layer_1_mask = predictions[:, 5] < 7
        layer_1_predictions = predictions[layer_1_mask]
        
        if len(layer_1_predictions) == 0:
            return torch.empty((0, predictions.shape[1]), device=predictions.device)
        
        # Apply hierarchical confidence modulation
        return self._apply_confidence_modulation(predictions, layer_1_predictions)
    
    def _apply_confidence_modulation(
        self, 
        all_predictions: torch.Tensor, 
        layer_1_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply hierarchical confidence modulation using Layer 2 & 3 predictions.
        
        Args:
            all_predictions: All predictions including all layers
            layer_1_predictions: Filtered Layer 1 predictions only
            
        Returns:
            Layer 1 predictions with modulated confidence scores
            
        Time Complexity: O(P₁ * max(P₂, P₃)) where P₁, P₂, P₃ are layer prediction counts
        Space Complexity: O(P₁ * max(P₂, P₃)) for IoU matrices
        """
        try:
            if len(layer_1_predictions) == 0:
                return layer_1_predictions
            
            # Memory safety check - use chunked processing for large datasets
            if len(layer_1_predictions) > self.max_predictions_per_chunk:
                if self.debug:
                    logger.debug(f"Large prediction set ({len(layer_1_predictions)}), using chunked processing")
                return self._chunked_confidence_modulation(all_predictions, layer_1_predictions)
            
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
                logger.debug(f"  • Layer 1 predictions: {len(layer_1_predictions)}")
                logger.debug(f"  • Layer 2 predictions: {len(layer_2_preds)}")
                logger.debug(f"  • Layer 3 predictions: {len(layer_3_preds)}")
                logger.debug(f"  • Memory estimate: {total_combinations:,} combinations")
            
            # Apply confidence modulation
            modified_predictions = layer_1_predictions.clone()
            original_conf = modified_predictions[:, 4]
            
            # Vectorized confidence computation for both layers
            layer_3_conf = self._compute_spatial_confidence(layer_1_predictions, layer_3_preds)
            layer_2_conf = self._compute_denomination_confidence(layer_1_predictions, layer_2_preds)
            
            # Hierarchical confidence modulation with money validation
            money_mask = layer_3_conf > 0.1  # Money validation threshold
            
            # Check if we have any layer_2 or layer_3 predictions available
            has_layer_predictions = (len(layer_2_preds) > 0 or len(layer_3_preds) > 0)
            
            if has_layer_predictions:
                # Apply hierarchical modulation only when we have multi-layer predictions
                hierarchical_conf = torch.where(
                    money_mask,
                    torch.clamp(original_conf * (1.0 + layer_2_conf * layer_3_conf), max=1.0),
                    original_conf * 0.1  # Reduce confidence if Layer 3 disagrees
                )
            else:
                # No layer_2 or layer_3 predictions available - keep original confidence unchanged
                hierarchical_conf = original_conf
            
            modified_predictions[:, 4] = hierarchical_conf
            
            if self.debug and len(layer_1_predictions) > 0:
                # Log first few predictions for debugging
                for i in range(min(3, len(layer_1_predictions))):
                    logger.debug(f"  • Pred {i}: class={layer_1_predictions[i, 5].int().item()}, "
                               f"conf={original_conf[i]:.3f}→{hierarchical_conf[i]:.3f}, "
                               f"L2={layer_2_conf[i]:.3f}, L3={layer_3_conf[i]:.3f}")
                
                # Comprehensive confidence modulation analysis
                self._log_confidence_modulation_analysis(
                    layer_1_predictions, layer_2_preds, layer_3_preds,
                    original_conf, hierarchical_conf, layer_2_conf, layer_3_conf, 
                    has_layer_predictions
                )
            
            return modified_predictions
            
        except Exception as e:
            logger.warning(f"Error in confidence modulation: {e}")
            # Emergency cleanup and return unmodified predictions
            self.memory_optimizer.emergency_memory_cleanup()
            return layer_1_predictions
    
    def _chunked_confidence_modulation(
        self, 
        _: torch.Tensor, 
        layer_1_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Process confidence modulation in memory-safe chunks.
        
        Args:
            all_predictions: All predictions
            layer_1_predictions: Layer 1 predictions
            
        Returns:
            Modified predictions with updated confidence
            
        Time Complexity: O(P₁) with chunk-based processing
        Space Complexity: O(chunk_size) per iteration
        """
        try:
            chunk_size = self.max_predictions_per_chunk
            
            modified_predictions = layer_1_predictions.clone()
            
            # Process in chunks to avoid memory overflow
            for i in range(0, len(layer_1_predictions), chunk_size):
                end_idx = min(i + chunk_size, len(layer_1_predictions))
                chunk = layer_1_predictions[i:end_idx]
                
                # Apply simplified confidence modulation to chunk
                original_conf = chunk[:, 4] 
                # Apply conservative boost for hierarchical validation
                modified_conf = torch.clamp(original_conf * 1.1, max=1.0)
                modified_predictions[i:end_idx, 4] = modified_conf
                
                # Clean memory after each chunk
                if i % (chunk_size * 5) == 0:  # Every 5 chunks
                    self.memory_optimizer.cleanup_memory()
            
            return modified_predictions
            
        except Exception as e:
            logger.error(f"Error in chunked confidence modulation: {e}")
            self.memory_optimizer.emergency_memory_cleanup()
            return layer_1_predictions
    
    def _compute_spatial_confidence(
        self, 
        layer_1_preds: torch.Tensor, 
        layer_3_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Layer 3 confidence based on spatial overlap with Layer 1.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(P₁ * P₃) for IoU matrix computation
        Space Complexity: O(P₁ * P₃) for IoU matrix storage
        """
        if len(layer_3_preds) == 0:
            return torch.zeros(len(layer_1_preds), device=self.device)
        
        # Memory safety check for IoU matrix size
        matrix_size = len(layer_1_preds) * len(layer_3_preds)
        max_matrix_size = 1_000_000  # 1M elements = ~8MB for float32
        
        if matrix_size > max_matrix_size:
            return self._chunked_spatial_confidence(layer_1_preds, layer_3_preds)
        
        try:
            # Compute IoU matrix between all Layer 1 and Layer 3 predictions
            iou_matrix = get_box_iou()(layer_1_preds[:, :4], layer_3_preds[:, :4])
            
            # For each Layer 1 prediction, find best overlapping Layer 3 prediction
            max_ious, max_indices = torch.max(iou_matrix, dim=1)
            
            # Apply IoU threshold and get corresponding confidences
            valid_mask = max_ious > 0.1
            layer_3_conf = torch.zeros(len(layer_1_preds), device=self.device)
            layer_3_conf[valid_mask] = layer_3_preds[max_indices[valid_mask], 4]
            
            return layer_3_conf
            
        except Exception as e:
            logger.warning(f"Error in spatial confidence computation: {e}, using fallback")
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
            
        Time Complexity: O(P₁ * P₃) total, processed in O(chunk_size * P₃) iterations
        """
        layer_3_conf = torch.zeros(len(layer_1_preds), device=self.device)
        
        for i in range(0, len(layer_1_preds), chunk_size):
            end_idx = min(i + chunk_size, len(layer_1_preds))
            chunk = layer_1_preds[i:end_idx]
            
            try:
                # Compute IoU for this chunk only
                iou_matrix = get_box_iou()(chunk[:, :4], layer_3_preds[:, :4])
                max_ious, max_indices = torch.max(iou_matrix, dim=1)
                
                # Apply threshold and assign confidences
                valid_mask = max_ious > 0.1
                layer_3_conf[i:end_idx][valid_mask] = layer_3_preds[max_indices[valid_mask], 4]
                
            except Exception as e:
                logger.warning(f"Error in chunk {i//chunk_size}: {e}")
                # Skip this chunk to avoid total failure
                continue
        
        return layer_3_conf
    
    def _compute_denomination_confidence(
        self, 
        layer_1_preds: torch.Tensor, 
        layer_2_preds: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Layer 2 confidence for specific denominations.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_2_preds: Layer 2 predictions
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(C * P₁ * P₂) where C is number of unique classes
        Space Complexity: O(P₁ * P₂) for IoU matrices per class
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
                    l1_mask = layer_2_classes == layer_2_class
                    l2_mask = layer_2_preds[:, 5].int() == layer_2_class
                    
                    if not l1_mask.any() or not l2_mask.any():
                        continue
                        
                    matching_l1_preds = layer_1_preds[l1_mask]
                    matching_l2_preds = layer_2_preds[l2_mask]
                    
                    # Memory safety check
                    matrix_size = len(matching_l1_preds) * len(matching_l2_preds)
                    if matrix_size > self.max_class_matrix_size:
                        # Use simplified confidence assignment instead of IoU computation for memory efficiency
                        # This is normal behavior when there are many predictions for popular classes
                        if not hasattr(self, '_logged_large_matrix'):
                            self._logged_large_matrix = set()
                        
                        # Only log once per class per session to avoid spam
                        if layer_2_class.item() not in self._logged_large_matrix:
                            logger.debug(f"Memory optimization: Class {layer_2_class} has {matrix_size:,} prediction pairs, using average confidence instead of IoU computation")
                            self._logged_large_matrix.add(layer_2_class.item())
                        
                        if len(matching_l2_preds) > 0:
                            avg_conf = matching_l2_preds[:, 4].mean()
                            l1_indices = torch.where(l1_mask)[0]
                            layer_2_conf[l1_indices] = avg_conf
                        continue
                    
                    # Safe IoU computation for reasonable matrix sizes
                    iou_matrix = get_box_iou()(matching_l1_preds[:, :4], matching_l2_preds[:, :4])
                    max_ious, max_indices = torch.max(iou_matrix, dim=1)
                    valid_mask = max_ious > 0.1
                    
                    # Assign confidences
                    l1_indices = torch.where(l1_mask)[0]
                    layer_2_conf[l1_indices[valid_mask]] = matching_l2_preds[max_indices[valid_mask], 4]
            
            return layer_2_conf
            
        except Exception as e:
            logger.warning(f"Error in denomination confidence computation: {e}, using fallback")
            self.memory_optimizer.emergency_memory_cleanup()
            return torch.zeros(len(layer_1_preds), device=self.device)
    
    def _get_unique_classes(self, predictions: torch.Tensor) -> list:
        """
        Get unique classes from predictions tensor.
        
        Args:
            predictions: Predictions tensor
            
        Returns:
            List of unique class values
            
        Time Complexity: O(P) where P is number of predictions
        """
        try:
            if predictions.dim() == 3:
                flat_predictions = predictions.view(-1, predictions.shape[-1])
                unique_classes = torch.unique(flat_predictions[:, 5].long()) if flat_predictions.numel() > 0 else torch.tensor([])
            elif predictions.dim() == 2:
                unique_classes = torch.unique(predictions[:, 5].long()) if predictions.numel() > 0 else torch.tensor([])
            else:
                return []
                
            return unique_classes.tolist()
            
        except Exception:
            return []
    
    def _log_confidence_modulation_analysis(
        self, 
        layer_1_predictions: torch.Tensor,
        layer_2_preds: torch.Tensor, 
        layer_3_preds: torch.Tensor,
        original_conf: torch.Tensor,
        hierarchical_conf: torch.Tensor,
        layer_2_conf: torch.Tensor,
        layer_3_conf: torch.Tensor,
        has_layer_predictions: bool
    ):
        """
        Log comprehensive confidence modulation analysis to debug file.
        
        Args:
            layer_1_predictions: Layer 1 predictions
            layer_2_preds: Layer 2 predictions
            layer_3_preds: Layer 3 predictions  
            original_conf: Original confidence scores
            hierarchical_conf: Modulated confidence scores
            layer_2_conf: Layer 2 confidence contributions
            layer_3_conf: Layer 3 confidence contributions
            has_layer_predictions: Whether multi-layer predictions are available
        """
        if not self.debug_logger:
            return
            
        self._hierarchical_debug_log("\n🔧 CONFIDENCE MODULATION DETAILED ANALYSIS:")
        self._hierarchical_debug_log(f"{'='*60}")
        
        # Multi-layer availability analysis
        self._hierarchical_debug_log(f"📊 MULTI-LAYER PREDICTION AVAILABILITY:")
        self._hierarchical_debug_log(f"  • Layer 2 predictions available: {'Yes' if len(layer_2_preds) > 0 else 'No'} ({len(layer_2_preds)} predictions)")
        self._hierarchical_debug_log(f"  • Layer 3 predictions available: {'Yes' if len(layer_3_preds) > 0 else 'No'} ({len(layer_3_preds)} predictions)")
        self._hierarchical_debug_log(f"  • Has multi-layer predictions: {'Yes' if has_layer_predictions else 'No'}")
        
        if not has_layer_predictions:
            self._hierarchical_debug_log("  ✅ CONFIDENCE UNCHANGED: No layer 2/3 predictions - original confidence preserved")
            return
            
        # Confidence change analysis
        confidence_changes = hierarchical_conf - original_conf
        boosted_mask = confidence_changes > 1e-6  # Boosted if changed by more than a tiny epsilon
        reduced_mask = confidence_changes < -1e-6  # Reduced if changed by more than a tiny epsilon
        unchanged_mask = torch.abs(confidence_changes) <= 1e-6  # Unchanged if difference is negligible
        
        num_boosted = boosted_mask.sum().item()
        num_reduced = reduced_mask.sum().item()
        num_unchanged = unchanged_mask.sum().item()
        
        self._hierarchical_debug_log(f"\n🎚️  CONFIDENCE MODULATION SUMMARY:")
        self._hierarchical_debug_log(f"  • Total Layer 1 predictions: {len(layer_1_predictions)}")
        self._hierarchical_debug_log(f"  • Confidence BOOSTED: {num_boosted} predictions ({num_boosted/len(layer_1_predictions)*100:.1f}%)")
        self._hierarchical_debug_log(f"  • Confidence REDUCED: {num_reduced} predictions ({num_reduced/len(layer_1_predictions)*100:.1f}%)")
        self._hierarchical_debug_log(f"  • Confidence UNCHANGED: {num_unchanged} predictions ({num_unchanged/len(layer_1_predictions)*100:.1f}%)")
        
        # Statistical analysis
        avg_original = original_conf.mean().item()
        avg_hierarchical = hierarchical_conf.mean().item()
        avg_change = confidence_changes.mean().item()
        
        self._hierarchical_debug_log(f"\n📈 CONFIDENCE STATISTICS:")
        self._hierarchical_debug_log(f"  • Average original confidence: {avg_original:.4f}")
        self._hierarchical_debug_log(f"  • Average hierarchical confidence: {avg_hierarchical:.4f}")
        self._hierarchical_debug_log(f"  • Average confidence change: {avg_change:+.4f}")
        
        # Layer contribution analysis
        layer_2_active = (layer_2_conf > 0.01).sum().item()
        layer_3_active = (layer_3_conf > 0.1).sum().item()  # Money validation threshold
        
        self._hierarchical_debug_log(f"\n🎯 LAYER CONTRIBUTION ANALYSIS:")
        self._hierarchical_debug_log(f"  • Layer 2 active contributions: {layer_2_active}/{len(layer_1_predictions)} ({layer_2_active/len(layer_1_predictions)*100:.1f}%)")
        self._hierarchical_debug_log(f"  • Layer 3 active contributions: {layer_3_active}/{len(layer_1_predictions)} ({layer_3_active/len(layer_1_predictions)*100:.1f}%)")
        
        if layer_2_active > 0:
            avg_layer_2_conf = layer_2_conf[layer_2_conf > 0.01].mean().item()
            self._hierarchical_debug_log(f"  • Average Layer 2 confidence (when active): {avg_layer_2_conf:.4f}")
            
        if layer_3_active > 0:
            avg_layer_3_conf = layer_3_conf[layer_3_conf > 0.1].mean().item()
            self._hierarchical_debug_log(f"  • Average Layer 3 confidence (when active): {avg_layer_3_conf:.4f}")
        
        # Detailed per-class analysis
        self._hierarchical_debug_log(f"\n🔍 PER-CLASS CONFIDENCE MODULATION:")
        unique_classes = torch.unique(layer_1_predictions[:, 5].long())
        
        for cls in unique_classes:
            cls_mask = layer_1_predictions[:, 5] == cls
            cls_count = cls_mask.sum().item()
            
            if cls_count > 0:
                cls_original = original_conf[cls_mask].mean().item()
                cls_hierarchical = hierarchical_conf[cls_mask].mean().item()
                cls_change = cls_hierarchical - cls_original
                cls_l2_contrib = layer_2_conf[cls_mask].mean().item()
                cls_l3_contrib = layer_3_conf[cls_mask].mean().item()
                
                self._hierarchical_debug_log(f"  Class {cls}: {cls_count} predictions")
                self._hierarchical_debug_log(f"    • Confidence: {cls_original:.3f} → {cls_hierarchical:.3f} ({cls_change:+.3f})")
                self._hierarchical_debug_log(f"    • Layer 2 contrib: {cls_l2_contrib:.3f}, Layer 3 contrib: {cls_l3_contrib:.3f}")
        
        # Sample predictions for detailed inspection
        if len(layer_1_predictions) > 0:
            self._hierarchical_debug_log(f"\n🔬 SAMPLE PREDICTION ANALYSIS (first 5):")
            sample_size = min(5, len(layer_1_predictions))
            
            for i in range(sample_size):
                cls = layer_1_predictions[i, 5].int().item()
                orig_conf = original_conf[i].item()
                hier_conf = hierarchical_conf[i].item()
                l2_conf = layer_2_conf[i].item()
                l3_conf = layer_3_conf[i].item()
                change = hier_conf - orig_conf
                
                self._hierarchical_debug_log(f"  Prediction {i+1}: Class {cls}")
                self._hierarchical_debug_log(f"    • Original confidence: {orig_conf:.4f}")
                self._hierarchical_debug_log(f"    • Hierarchical confidence: {hier_conf:.4f} ({change:+.4f})")
                self._hierarchical_debug_log(f"    • Layer 2 contribution: {l2_conf:.4f}")
                self._hierarchical_debug_log(f"    • Layer 3 contribution: {l3_conf:.4f}")
                
                if l3_conf > 0.1:
                    boost_factor = 1.0 + l2_conf * l3_conf
                    self._hierarchical_debug_log(f"    • ✅ Money validated (L3>{0.1:.1f}) - boost factor: {boost_factor:.3f}")
                else:
                    self._hierarchical_debug_log(f"    • ❌ Money validation failed (L3≤{0.1:.1f}) - confidence reduced by 90%")
        
        self._hierarchical_debug_log(f"{'='*60}")