#!/usr/bin/env python3
"""
Hierarchical processing module for multi-layer architecture in SmartCash.

Handles Phase 2 multi-layer hierarchical system:
- Layer 1: Denomination detection (classes 0-6)
- Layer 2: Confidence features (classes 7-13) 
- Layer 3: Money validation (classes 14-16)

Provides optimized confidence modulation using spatial relationships between layers.
Refactored to use extracted modules for better separation of concerns.

Architecture:
- ConfidenceModulator: Handles spatial and denomination confidence algorithms
- ChunkedProcessor: Manages memory-safe processing for large datasets
- HierarchicalProcessor: Core coordination and phase detection (this file)

Algorithmic Complexity:
- Filtering: O(P) where P is number of predictions
- Phase detection: O(1) for context check, O(P) for fallback analysis
- Coordination: O(1) for module orchestration
"""

import torch
from typing import Tuple, Optional

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

# Import extracted modules
from .confidence_modulator import ConfidenceModulator
from .chunked_processor import ChunkedProcessor

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
    
    def __init__(self, device: Optional[torch.device] = None, debug: bool = False, training_context: dict = None, use_standard_map: bool = False):
        """
        Initialize hierarchical processor with extracted modules.
        
        Args:
            device: Torch device for computations
            debug: Enable debug logging
            training_context: Training context information (backbone, phase, etc.)
            use_standard_map: Enable standard (non-hierarchical) mAP calculation for all classes
        """
        self.device = device or torch.device('cpu')
        self.debug = debug
        self.training_context = training_context or {}
        self.use_standard_map = use_standard_map  # TASK.md: Enable standard mAP calculation for all classes
        
        # Check if hierarchical processing should be disabled (for SmartCash models)
        self.disable_hierarchical = self.training_context.get('disable_hierarchical', False)
        
        self.memory_optimizer = get_memory_optimizer()
        
        # Memory safety thresholds
        self.max_predictions_per_chunk = 10000
        self.max_matrix_combinations = 50_000_000  # 50M = ~400MB for IoU matrix
        self.max_class_matrix_size = 100_000  # 100K = ~800KB per class
        
        # Layer configuration
        self.layer_1_classes = range(0, 7)   # Classes 0-6
        self.layer_2_classes = range(7, 14)  # Classes 7-13
        self.layer_3_classes = range(14, 17) # Classes 14-16
        
        # Initialize debug file logging if enabled
        self.debug_logger = None
        self.current_epoch = 0  # Initialize current epoch
        self._current_debug_epoch = -1  # Track which epoch debug logger is configured for
        
        # Initialize extracted modules
        self.confidence_modulator = ConfidenceModulator(device=self.device, debug=self.debug)
        self.chunked_processor = ChunkedProcessor(device=self.device, debug=self.debug)
        
    
    def _setup_hierarchical_debug_logging(self):
        """
        Set up dedicated debug file logging for hierarchical processing.
        
        Creates debug log file in logs/validation_metrics/ with timestamp and training context.
        """
        from pathlib import Path
        import logging
         # Extract context information for filename
        backbone = self.training_context.get('backbone', 'unknown')
        # Try both 'current_phase' and 'phase' keys for phase detection
        phase = self.training_context.get('current_phase') or self.training_context.get('phase', 'unknown')
        training_mode = self.training_context.get('training_mode', 'unknown')
        
        # Create debug log directory
        debug_log_dir = Path(f"logs/validation_metrics/{backbone}/phase_{phase}")
        debug_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create timestamped debug log file with context
        debug_log_file = debug_log_dir / f"hierarchical_debug_epoch{self.current_epoch}.log"
        
        # Set up dedicated debug logger
        self.debug_logger = logging.getLogger(f"hierarchical_debug_phase{phase}_epoch{self.current_epoch}")
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
        targets: torch.Tensor,
        epoch: int = 0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply hierarchical filtering and confidence modulation.
        
        Args:
            predictions: Raw predictions tensor
            targets: Target tensor
            epoch: The current epoch number.
            
        Returns:
            Tuple of (filtered_predictions, filtered_targets)
            
        Time Complexity: O(P) for filtering + O(P₁ * P₂) for modulation
        Space Complexity: O(P) for storing processed predictions
        """
        # For SmartCash models, skip hierarchical processing entirely
        if self.disable_hierarchical:
            logger.debug("🆕 SmartCash model detected - bypassing hierarchical processing")
            return predictions, targets
        
        self.current_epoch = epoch
        
        # Setup or update debug logging for current epoch
        # SKIP hierarchical debug logging in Phase 1 to reduce log noise
        # Phase 1 only needs mAP debug logging (handled by YOLOv5MapCalculator)
        if self.debug and self._current_debug_epoch != epoch:
            # Detect current phase to determine if we should enable hierarchical debug logging
            current_phase = self._detect_processing_phase(predictions)
            
            if current_phase == 1:
                # Phase 1: Skip hierarchical debug logging, only mAP debug needed
                logger.debug("Phase 1: Skipping hierarchical debug logging setup (mAP debug only)")
                self._current_debug_epoch = epoch  # Mark epoch as processed but don't setup logging
            else:
                # Phase 2: Enable full hierarchical debug logging
                logger.debug("Phase 2: Setting up hierarchical debug logging")
                self._setup_hierarchical_debug_logging()
                self._current_debug_epoch = epoch
        
        try:
            # Input validation
            if predictions is None or targets is None:
                return predictions, targets
                
            if predictions.numel() == 0:
                return predictions, targets
            
            # Detect processing phase based on prediction classes
            phase = self._detect_processing_phase(predictions)
            
            # Only log hierarchical debug info in Phase 2
            if self.debug and phase == 2:
                self._hierarchical_debug_log(f"\n{'='*60}")
                self._hierarchical_debug_log(f"🔍 HIERARCHICAL PROCESSING BATCH")
                self._hierarchical_debug_log(f"{'='*60}")
                self._hierarchical_debug_log(f"Input tensor shapes:")
                self._hierarchical_debug_log(f"  • predictions: {predictions.shape}")
                self._hierarchical_debug_log(f"  • targets: {targets.shape}")
                self._hierarchical_debug_log("🔹 PHASE 2 DETECTED: Hierarchical multi-layer processing")
                self._hierarchical_debug_log("  • Applying hierarchical filtering and confidence modulation")
            elif self.debug and phase == 1:
                # Phase 1: Just log basic info to console, no file logging
                # logger.debug("Phase 1: Hierarchical processing skipped (single-layer mode)")
                return predictions, targets
            
            # Phase 2: Multi-layer hierarchical processing logging handled above
            
            if phase == 2:
                return self._process_phase2_predictions(predictions, targets)
            
            return predictions, targets
            
        except Exception as e:
            logger.warning(f"Error in hierarchical processing: {e}")
            if self.debug:
                import traceback
                logger.debug(f"Hierarchical processing traceback: {traceback.format_exc()}")
            return predictions, targets
    
    def _detect_processing_phase(self, predictions: torch.Tensor) -> int:
        """
        Detect whether we're in Phase 1 or Phase 2 based on training context.
        
        Args:
            predictions: Predictions tensor (used for fallback only)
            
        Returns:
            int: 1 for Phase 1, 2 for Phase 2
            
        Time Complexity: O(1) for context check, O(P) for fallback prediction analysis
        """
        try:
            # Primary: Use training context if available
            if self.training_context:
                current_phase = self.training_context.get('current_phase', '')
                
                # Handle both string and integer phase values
                phase_str = str(current_phase).lower() if current_phase else ''
                
                # Parse phase information
                if 'phase1' in phase_str or phase_str == '1' or current_phase == 1:
                    return 1
                elif 'phase2' in phase_str or phase_str == '2' or current_phase == 2:
                    return 2
                
                # Check training mode for phase indication
                training_mode = self.training_context.get('training_mode', '')
                training_mode_str = str(training_mode).lower() if training_mode else ''
                
                if 'single_phase' in training_mode_str or 'one_phase' in training_mode_str:
                    return 1
                elif 'two_phase' in training_mode_str and ('phase2' in phase_str or current_phase == 2):
                    return 2
            
            # Fallback: Analyze predictions (but with better logic)
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
                
            # Conservative fallback: Only assume Phase 2 if >80% of predictions are classes > 6
            if len(unique_classes) > 0:
                high_classes = unique_classes[unique_classes >= 7]
                if len(high_classes) > 0 and len(high_classes) > len(unique_classes) * 0.8:
                    return 2
            
            # Default to Phase 1 (safer assumption)
            return 1
            
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
            
            self._hierarchical_debug_log("\n📊 PHASE 2 PREDICTION ANALYSIS:")
            self._hierarchical_debug_log(f"  • Original prediction classes detected: {unique_classes}")
            self._hierarchical_debug_log(f"  • Prediction tensor shape: {predictions.shape}")
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
        
        # TASK.md: Apply target filtering based on mAP calculation mode
        if self.use_standard_map:
            # Standard mAP: Use all targets (all 17 classes: 0-16)
            filtered_targets = targets
            if self.debug:
                self._hierarchical_debug_log("🎯 TASK.md: Using standard mAP - keeping all targets (classes 0-16)")
        else:
            # Hierarchical mAP: Filter targets to Layer 1 only (classes 0-6)
            filtered_targets = targets[targets[..., 1] < 7] if targets.numel() > 0 else targets
            if self.debug:
                self._hierarchical_debug_log("🎯 Using hierarchical mAP - filtering to Layer 1 targets (classes 0-6)")
        
        if self.debug:
            # Detailed processing summary
            processing_mode = "STANDARD" if self.use_standard_map else "HIERARCHICAL"
            self._hierarchical_debug_log(f"\n🎯 {processing_mode} PROCESSING RESULTS:")
            self._hierarchical_debug_log(f"  • Predictions after filtering: {len(filtered_predictions)}")
            self._hierarchical_debug_log(f"  • Targets after filtering: {len(filtered_targets)}")
            
            if len(filtered_predictions) > 0:
                filtered_classes = torch.unique(filtered_predictions[:, 5].long())
                self._hierarchical_debug_log(f"  • Final prediction classes: {filtered_classes.tolist()}")
                
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
                
                # Class count breakdown
                for cls in filtered_classes:
                    cls_count = (filtered_predictions[:, 5] == cls).sum().item()
                    cls_avg_conf = filtered_predictions[filtered_predictions[:, 5] == cls, 4].mean().item()
                    self._hierarchical_debug_log(f"    - Class {cls}: {cls_count} predictions, avg_conf={cls_avg_conf:.4f}")
            else:
                self._hierarchical_debug_log("  ⚠️  NO PREDICTIONS after filtering!")
                
            if len(filtered_targets) == 0:
                self._hierarchical_debug_log("  ⚠️  NO TARGETS after filtering!")
        
        return filtered_predictions, filtered_targets
    
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
        
        # TASK.md: Apply prediction filtering based on mAP calculation mode
        if self.use_standard_map:
            # Standard mAP: Use all predictions (all 17 classes: 0-16)
            filtered_predictions = flat_predictions
            if self.debug:
                self._hierarchical_debug_log("📊 TASK.md: Using standard mAP - keeping all predictions (classes 0-16)")
        else:
            # Hierarchical mAP: Filter to Layer 1 classes (0-6) only
            layer_1_mask = flat_predictions[:, 5] < 7
            filtered_predictions = flat_predictions[layer_1_mask]
            if self.debug:
                self._hierarchical_debug_log("📊 Using hierarchical mAP - filtering to Layer 1 predictions (classes 0-6)")
        
        if len(filtered_predictions) == 0:
            return torch.empty((0, num_features), device=predictions.device)
        
        # Apply confidence modulation based on mode
        if self.use_standard_map:
            # Standard mAP: No hierarchical confidence modulation, return as-is
            return filtered_predictions
        else:
            # Hierarchical mAP: Apply hierarchical confidence modulation using extracted module
            layer_1_predictions = flat_predictions[flat_predictions[:, 5] < 7]
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
        # TASK.md: Apply prediction filtering based on mAP calculation mode
        if self.use_standard_map:
            # Standard mAP: Use all predictions (all 17 classes: 0-16)
            filtered_predictions = predictions
            if self.debug:
                self._hierarchical_debug_log("📊 TASK.md: Using standard mAP - keeping all predictions (classes 0-16)")
        else:
            # Hierarchical mAP: Filter to Layer 1 classes (0-6) only
            layer_1_mask = predictions[:, 5] < 7
            filtered_predictions = predictions[layer_1_mask]
            if self.debug:
                self._hierarchical_debug_log("📊 Using hierarchical mAP - filtering to Layer 1 predictions (classes 0-6)")
        
        if len(filtered_predictions) == 0:
            return torch.empty((0, predictions.shape[1]), device=predictions.device)
        
        # Apply confidence modulation based on mode
        if self.use_standard_map:
            # Standard mAP: No hierarchical confidence modulation, return as-is
            return filtered_predictions
        else:
            # Hierarchical mAP: Apply hierarchical confidence modulation using extracted module
            layer_1_predictions = predictions[predictions[:, 5] < 7]
            return self._apply_confidence_modulation(predictions, layer_1_predictions)
    
    def _apply_confidence_modulation(
        self, 
        all_predictions: torch.Tensor, 
        layer_1_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply hierarchical confidence modulation using extracted modules.
        
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
                    self._hierarchical_debug_log(f"Large prediction set ({len(layer_1_predictions)}), using chunked processing")
                return self.chunked_processor.process_chunked_confidence(all_predictions, layer_1_predictions)
            
            # Use confidence modulator for standard processing
            return self.confidence_modulator.apply_confidence_modulation(all_predictions, layer_1_predictions)
            
        except Exception as e:
            logger.warning(f"Error in confidence modulation: {e}")
            # Emergency cleanup and return unmodified predictions
            self.memory_optimizer.emergency_memory_cleanup()
            return layer_1_predictions
    
    
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
    
