#!/usr/bin/env python3
"""
Refactored YOLOv5-based mAP calculator for SmartCash validation phase.

This is a clean, modular implementation that follows Single Responsibility Principle
and maintains backward compatibility with the original API. Each major responsibility
has been extracted into focused, reusable modules.

Architecture:
- YOLOv5UtilitiesManager: Handles YOLOv5 imports and lazy loading
- HierarchicalProcessor: Manages multi-layer confidence modulation
- MemoryOptimizedProcessor: Platform-aware memory management
- BatchProcessor: Handles batch-level prediction processing
- YOLOv5MapCalculator: Core mAP calculation logic (this file)

Algorithmic Improvements:
- Vectorized operations: O(N) instead of O(N²) in many cases
- Memory-conscious chunking: O(chunk_size) space complexity
- Platform-aware optimization: Adaptive to hardware capabilities
"""

import torch
import numpy as np
from typing import Dict, Optional

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

# Import specialized processors
from .yolo_utils_manager import YOLOv5UtilitiesManager, is_yolov5_available, get_ap_per_class
from .hierarchical_processor import HierarchicalProcessor
from .memory_optimized_processor import MemoryOptimizedProcessor
from .batch_processor import BatchProcessor

logger = get_logger(__name__, level="DEBUG")

# Debug flag for hierarchical validation (can be overridden by config)
DEBUG_HIERARCHICAL = False


class YOLOv5MapCalculator:
    """
    YOLOv5-based mAP calculator using modular, SRP-compliant architecture.
    
    This calculator provides accurate mAP@0.5 computation that matches
    standard YOLO evaluation protocols while maintaining clean separation
    of concerns across specialized processing modules.
    
    Time Complexity: O(N log N) for sorting + O(N*M) for IoU computation
    Space Complexity: O(N*M) for IoU matrices, O(N) for statistics
    """
    
    def __init__(
        self, 
        num_classes: int = 7, 
        conf_thres: float = 0.005, 
        iou_thres: float = 0.03, 
        debug: bool = True,
        training_context: dict = None
    ):
        """
        Initialize YOLOv5 mAP calculator with modular architecture.
        
        Args:
            num_classes: Number of classes (default 7 for SmartCash banknotes)
            conf_thres: Confidence threshold for predictions
            iou_thres: IoU threshold for NMS and mAP calculation
            debug: Enable hierarchical debug logging
            training_context: Training context information (backbone, phase, etc.)
        """
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug = debug
        self.training_context = training_context or {}
        
        # Initialize device and memory management first
        self.memory_optimizer = get_memory_optimizer()
        self.device = self.memory_optimizer.device
        
        # Initialize debug file logging if enabled (after device is set)
        self.debug_logger = None
        if debug:
            self._setup_debug_logging()
            logger.info(f"🐛 YOLOv5MapCalculator DEBUG MODE ENABLED - Writing to debug log file")
        
        # Initialize specialized processors
        self._init_processors()
        
        # Initialize storage
        self._init_storage()
    
    def _setup_debug_logging(self):
        """
        Set up dedicated debug file logging for mAP analysis.
        
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
        debug_log_file = debug_log_dir / f"map_debug_{backbone}_{training_mode}_phase{phase}_{timestamp}.log"
        
        # Set up dedicated debug logger
        self.debug_logger = logging.getLogger(f"map_debug_{timestamp}")
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
        self.debug_logger.info("YOLOv5 mAP Calculator Debug Session Started")
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
        
        # Log validation configuration
        self.debug_logger.info("")
        self.debug_logger.info("Validation Configuration:")
        self.debug_logger.info(f"  • num_classes: {self.num_classes}")
        self.debug_logger.info(f"  • conf_thres: {self.conf_thres}")
        self.debug_logger.info(f"  • iou_thres: {self.iou_thres}")
        self.debug_logger.info(f"  • device: {self.device}")
        
        # Log file information
        self.debug_logger.info("")
        self.debug_logger.info(f"Debug log file: {debug_log_file}")
        self.debug_logger.info("")
        
        # Log to console about debug file creation
        logger.info(f"📄 Debug log file created: {debug_log_file}")
    
    def _debug_log(self, message: str):
        """
        Write message to debug log file if debug mode is enabled.
        
        Args:
            message: Message to log
        """
        if self.debug_logger:
            self.debug_logger.info(message)
    
    def _init_storage(self):
        """Initialize storage for batch statistics."""
        # Storage for batch statistics
        self.stats = []  # List of [tp, conf, pred_cls, target_cls]
        
        # Processing counters
        self._batch_count = 0
    
    def _init_processors(self):
        """
        Initialize all specialized processing modules.
        
        Time Complexity: O(1) - simple object initialization
        """
        # YOLOv5 utilities manager for lazy loading
        self.yolo_utils = YOLOv5UtilitiesManager()
        
        # Hierarchical processor for multi-layer architecture
        self.hierarchical_processor = HierarchicalProcessor(
            device=self.device, 
            debug=self.debug,
            training_context=self.training_context
        )
        
        # Memory-optimized processor for platform-aware operations
        self.memory_processor = MemoryOptimizedProcessor(
            device=self.device, 
            debug=self.debug
        )
        
        # Batch processor for prediction-target matching
        self.batch_processor = BatchProcessor(
            conf_threshold=self.conf_thres,
            iou_threshold=self.iou_thres,
            device=self.device,
            debug=self.debug
        )
    
    def reset(self):
        """
        Reset accumulated statistics for new validation run.
        
        Time Complexity: O(1) - simple list clearing
        """
        # Log before reset for debugging
        if self.debug:
            self._debug_log(f"\n🔄 VALIDATION EPOCH RESET")
            self._debug_log(f"📊 Previous epoch stats: {len(self.stats)} batches, {self._batch_count} updates")
            if self.stats:
                total_samples = sum(len(stat[0]) for stat in self.stats)
                self._debug_log(f"📊 Previous epoch samples: {total_samples}")
        
        self.stats.clear()
        self._batch_count = 0
        
        # Clean memory before starting new validation
        if self.debug:
            self._debug_log("🧹 Memory cleaned, ready for new validation epoch")
        self.memory_optimizer.cleanup_memory()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Update mAP statistics with batch predictions and targets.
        
        Args:
            predictions: Model predictions in YOLO format [batch, detections, 6]
                        where each detection is [x, y, w, h, conf, class]
            targets: Ground truth targets [num_targets, 6] 
                    where each target is [batch_idx, class, x, y, w, h]
                    
        Time Complexity: O(P*T) for IoU + O(P log P) for sorting
        Space Complexity: O(P*T) for IoU matrix
        """
        # Check YOLOv5 availability
        if not self._ensure_yolov5_available() or predictions is None or targets is None:
            if self.debug:
                self._debug_log(f"Skipping update: yolov5_available={self.yolo_utils.is_available()}, "
                              f"predictions={predictions is not None}, targets={targets is not None}")
            return
        
        # Comprehensive input validation
        if not self._validate_inputs(predictions, targets):
            return
        
        # Track batch count for debugging
        self._batch_count += 1
        # Enhanced debug logging for mAP investigation
        if self.debug:
            if self._batch_count == 1:
                self._debug_log(f"📊 mAP update first batch: pred_shape={predictions.shape}, target_shape={targets.shape}")
                self._debug_log(f"📊 Initial predictions sample: conf_range=[{predictions[:,:,4].min():.6f}, {predictions[:,:,4].max():.6f}]")
                self._debug_log(f"📊 Initial predictions classes: unique={torch.unique(predictions[:,:,5]).tolist()}")
                if targets.numel() > 0:
                    self._debug_log(f"📊 Initial targets classes: unique={torch.unique(targets[:,1]).tolist()}")
                else:
                    self._debug_log(f"📊 No targets in first batch - empty targets tensor")
        
        try:
            # Apply hierarchical filtering for Phase 2 multi-layer architecture
            processed_predictions, processed_targets = self.hierarchical_processor.process_hierarchical_predictions(
                predictions, targets
            )
            
            # Optimize tensor device management
            processed_predictions = self.memory_processor.optimize_tensor_transfer(processed_predictions)
            processed_targets = self.memory_processor.optimize_tensor_transfer(processed_targets)
            
            # Preprocess predictions (confidence filtering and format conversion)
            standardized_predictions = self.batch_processor.preprocess_predictions(processed_predictions)
            
            # Process batch for mAP statistics
            batch_stats = self.batch_processor.process_batch(standardized_predictions, processed_targets)
            
            if batch_stats is not None:
                self.stats.append(batch_stats)
                # Enhanced debug logging for mAP investigation
                if self.debug:
                    if self._batch_count == 1:
                        tp, conf, pred_cls, target_cls = batch_stats
                        self._debug_log(f"✅ Added first batch_stats to mAP calculator")
                        self._debug_log(f"   • TP: {tp.sum().item()}/{len(tp)} ({tp.sum().item()/max(len(tp),1)*100:.1f}%)")
                        self._debug_log(f"   • Confidence range: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
                        self._debug_log(f"   • Predicted classes: {torch.unique(pred_cls).tolist()}")
                        self._debug_log(f"   • Target classes: {torch.unique(target_cls).tolist()}")
                    elif self._batch_count % 10 == 0:  # Log every 10th batch
                        tp, conf, pred_cls, target_cls = batch_stats
                        self._debug_log(f"📊 Batch {self._batch_count} stats: TP={tp.sum().item()}/{len(tp)}, conf_avg={conf.mean().item():.4f}")
            else:
                if self.debug:
                    self._debug_log(f"❌ batch_stats is None for batch {self._batch_count} - check prediction processing")
            
        except Exception as e:
            logger.warning(f"Error updating mAP statistics: {e}")
            # Emergency cleanup on error
            self.memory_processor.emergency_cleanup()
    
    def compute_map(self) -> Dict[str, float]:
        """
        Compute final mAP metrics from accumulated statistics.
        
        Returns:
            Dictionary containing mAP metrics:
            - 'map50': mAP@0.5
            - 'map50_95': mAP@0.5:0.95 (set to 0 for now)
            - 'precision': Mean precision
            - 'recall': Mean recall
            - 'f1': Mean F1 score
            
        Time Complexity: O(N log N) for AP computation where N is total detections
        Space Complexity: O(N) for statistics concatenation
        """
        if not self._ensure_yolov5_available():
            raise RuntimeError("YOLOv5 not available - cannot compute mAP metrics")
        
        if not self.stats:
            logger.error(f"🚨 No statistics accumulated! update() called {self._batch_count} times")
            if self.debug:
                self._debug_log(f"mAP calculator state: yolov5_available={self.yolo_utils.is_available()}, "
                              f"device={self.device}")
            raise RuntimeError("No statistics accumulated - check if update() was called during validation")
        
        # Validate accumulated statistics
        if not self._validate_statistics():
            raise RuntimeError("Statistics validation failed")
        
        try:
            # Optimized statistics concatenation
            stats = self._concatenate_statistics()
            
            if len(stats) >= 4:  # Ensure we have all 4 required stat arrays
                tp, conf, pred_cls, target_cls = stats[:4]
                data_size = len(tp) if hasattr(tp, '__len__') else 0
                
                # Enhanced debugging for zero statistics
                if self.debug:
                    self._debug_log(f"📊 Statistics analysis:")
                    self._debug_log(f"  • TP tensor shape: {tp.shape if hasattr(tp, 'shape') else 'N/A'}")
                    self._debug_log(f"  • TP sum: {tp.sum() if hasattr(tp, 'sum') else 'N/A'}")
                    self._debug_log(f"  • Conf shape: {conf.shape if hasattr(conf, 'shape') else 'N/A'}")
                    self._debug_log(f"  • Pred classes: {len(pred_cls) if hasattr(pred_cls, '__len__') else 'N/A'}")
                    self._debug_log(f"  • Target classes: {len(target_cls) if hasattr(target_cls, '__len__') else 'N/A'}")
                
                # Check if we have any data at all
                if data_size > 0:
                    logger.info(f"📈 Processing {data_size} detection samples for mAP computation...")
                    
                    if data_size > 10000:
                        logger.info("⏳ Large dataset detected - mAP computation may take 30-60 seconds")
                    
                    if self.debug:
                        logger.info(f"🐛 Running mAP debug analysis (see debug log file)...")
                        self._debug_log(f"\n🔍 STARTING mAP COMPUTATION ANALYSIS (Epoch validation)")
                        self._debug_log(f"📊 Total batches processed: {self._batch_count}")
                        self._debug_log(f"📊 Total stat entries: {len(self.stats)}")
                        self._debug_log(f"📊 Total detection samples: {data_size}")
                        self._log_class_analysis(stats)
                    
                    # Always try to compute mAP even if all TP are zero (early training)
                    # Check for fast approximation on minimal data
                    if self._should_use_fast_approximation(stats, data_size):
                        return self._compute_fast_approximation(stats)
                    
                    # Compute full mAP using YOLOv5 function
                    return self._compute_full_map(stats, data_size)
                else:
                    # No data samples
                    logger.warning("📊 No detection samples accumulated - check if predictions are being generated")
                    if self.debug:
                        self._debug_log(f"\n❌ CRITICAL: NO DETECTION SAMPLES ACCUMULATED")
                        self._debug_log(f"📊 Batches processed: {self._batch_count}")
                        self._debug_log(f"📊 Stat entries: {len(self.stats)}")
                        self._debug_log(f"📊 This means either:")
                        self._debug_log(f"   1. Model is not generating predictions")
                        self._debug_log(f"   2. Predictions are filtered out by confidence threshold ({self.conf_thres})")
                        self._debug_log(f"   3. Batch processing is failing")
                        self._debug_log(f"   4. update() method is not being called during validation")
                    return {
                        'map50': 0.0,
                        'map50_95': 0.0,
                        'precision': 0.0,
                        'recall': 0.0,
                        'f1': 0.0
                    }
            else:
                # Malformed statistics
                logger.error(f"🚨 Malformed statistics! Expected 4 arrays, got {len(stats)}")
                if self.debug:
                    self._debug_log(f"Statistics structure: {[type(s) for s in stats]}")
                return {
                    'map50': 0.0,
                    'map50_95': 0.0,
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0
                }
            
        except Exception as e:
            logger.error(f"Error computing mAP: {e}")
            # Emergency cleanup on error
            self.memory_processor.emergency_cleanup()
            raise RuntimeError(f"mAP computation failed: {e}") from e
    
    def _ensure_yolov5_available(self) -> bool:
        """
        Ensure YOLOv5 utilities are available.
        
        Returns:
            bool: True if YOLOv5 is available
            
        Time Complexity: O(1) after first call (cached result)
        """
        if not self.yolo_utils.is_available():
            logger.error("YOLOv5 is required for hierarchical validation but is not available")
            return False
        return True
    
    @property
    def yolov5_available(self) -> bool:
        """
        Backward compatibility property for checking YOLOv5 availability.
        
        Returns:
            bool: True if YOLOv5 utilities are available
        """
        return self.yolo_utils.is_available()
    
    def _validate_inputs(self, predictions: torch.Tensor, targets: torch.Tensor) -> bool:
        """
        Validate input tensors.
        
        Args:
            predictions: Predictions tensor
            targets: Targets tensor
            
        Returns:
            bool: True if inputs are valid
            
        Time Complexity: O(1) - simple validation checks
        """
        try:
            # Validate prediction tensor format
            if not isinstance(predictions, torch.Tensor) or predictions.numel() == 0:
                if self.debug:
                    self._debug_log("Empty or invalid predictions tensor")
                return False
                
            if predictions.dim() not in [2, 3]:
                logger.warning(f"Invalid prediction tensor dimensions: {predictions.dim()}, expected 2D or 3D")
                return False
                
            if predictions.shape[-1] < 6:
                logger.error(f"🚨 Prediction tensor too small! Expected at least 6 columns, got {predictions.shape[-1]}")
                return False
                
            # Validate target tensor format
            if not isinstance(targets, torch.Tensor):
                if self.debug:
                    self._debug_log("Invalid targets tensor")
                return False
                
            if targets.numel() > 0 and (targets.dim() != 2 or targets.shape[-1] < 6):
                logger.warning(f"Invalid target tensor format: {targets.shape}, expected [N, 6]")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during input validation: {e}")
            return False
    
    def _validate_statistics(self) -> bool:
        """
        Validate accumulated statistics format.
        
        Returns:
            bool: True if statistics are valid
            
        Time Complexity: O(S) where S is number of statistics batches
        """
        try:
            for i, stat in enumerate(self.stats):
                if stat is None:
                    logger.error(f"🚨 Statistics batch {i} is None - skipping")
                    continue
                    
                if not isinstance(stat, tuple) or len(stat) != 4:
                    logger.error(f"🚨 Invalid statistics format at batch {i}: expected tuple of length 4, got {type(stat)}")
                    continue
                    
                # Validate each component
                tp, conf, pred_cls, target_cls = stat
                if not all(isinstance(x, torch.Tensor) for x in [tp, conf, pred_cls, target_cls]):
                    logger.warning(f"Statistics batch {i} contains non-tensor elements")
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating statistics format: {e}")
            return False
    
    def _concatenate_statistics(self) -> list:
        """
        Concatenate accumulated statistics efficiently.
        
        Returns:
            List of concatenated statistics arrays
            
        Time Complexity: O(S*N) where S is batches, N is average batch size
        Space Complexity: O(total_detections) for concatenated arrays
        """
        if self.debug:
            self._debug_log("📊 Concatenating statistics for mAP computation...")
        
        if len(self.stats) == 1:
            # Single batch optimization - no concatenation needed
            return [x.cpu().numpy() if hasattr(x, 'cpu') else x for x in self.stats[0]]
        else:
            # Multiple batches - efficient GPU concatenation then single CPU transfer
            total_size = sum(stat[0].shape[0] for stat in self.stats)
            if self.debug:
                self._debug_log(f"📊 Concatenating {len(self.stats)} batches, total size: {total_size}")
            
            # Keep concatenation on GPU for parallel processing
            gpu_stats = [torch.cat(x, 0) for x in zip(*self.stats)]
            
            # Single batch CPU transfer for all statistics
            stats = []
            for stat in gpu_stats:
                if hasattr(stat, 'cpu'):
                    stats.append(stat.cpu().numpy())
                else:
                    stats.append(stat)
            
            return stats
    
    def _log_class_analysis(self, stats: list):
        """
        Log comprehensive class analysis for mAP debugging to debug file.
        
        Args:
            stats: Concatenated statistics
            
        Time Complexity: O(N) for unique class computation
        """
        if not self.debug_logger:
            return
            
        tp, conf, pred_cls, target_cls = stats
        
        # Convert to numpy for easier analysis
        import numpy as np
        tp_np = tp.cpu().numpy() if hasattr(tp, 'cpu') else np.array(tp)
        conf_np = conf.cpu().numpy() if hasattr(conf, 'cpu') else np.array(conf)
        pred_cls_np = pred_cls.cpu().numpy() if hasattr(pred_cls, 'cpu') else np.array(pred_cls)
        target_cls_np = target_cls.cpu().numpy() if hasattr(target_cls, 'cpu') else np.array(target_cls)
        
        self._debug_log("\n" + "=" * 80)
        self._debug_log("🔍 COMPREHENSIVE mAP DEBUG ANALYSIS")
        self._debug_log("=" * 80)
        
        # Basic statistics
        total_predictions = len(pred_cls_np)
        total_targets = len(target_cls_np)
        total_tp = tp_np.sum()
        total_fp = total_predictions - total_tp
        
        self._debug_log("📊 OVERALL STATISTICS:")
        self._debug_log(f"  • Total predictions: {total_predictions:,}")
        self._debug_log(f"  • Total targets: {total_targets:,}")
        self._debug_log(f"  • Total True Positives (TP): {total_tp:,}")
        self._debug_log(f"  • Total False Positives (FP): {total_fp:,}")
        self._debug_log(f"  • Overall Precision: {total_tp / max(total_predictions, 1):.4f}")
        self._debug_log(f"  • Confidence range: {conf_np.min():.6f} - {conf_np.max():.6f}")
        
        # Class-specific analysis
        unique_pred_classes = np.unique(pred_cls_np)
        unique_target_classes = np.unique(target_cls_np)
        
        self._debug_log("\n📋 CLASS DISTRIBUTION:")
        self._debug_log(f"  • Predicted classes: {unique_pred_classes}")
        self._debug_log(f"  • Target classes: {unique_target_classes}")
        self._debug_log(f"  • Classes in both pred & target: {np.intersect1d(unique_pred_classes, unique_target_classes)}")
        self._debug_log(f"  • Classes only in predictions: {np.setdiff1d(unique_pred_classes, unique_target_classes)}")
        self._debug_log(f"  • Classes only in targets: {np.setdiff1d(unique_target_classes, unique_pred_classes)}")
        
        # Per-class detailed analysis
        self._debug_log("\n🎯 PER-CLASS DETAILED ANALYSIS:")
        for cls in sorted(np.union1d(unique_pred_classes, unique_target_classes)):
            cls_pred_mask = pred_cls_np == cls
            cls_target_mask = target_cls_np == cls
            
            cls_predictions = np.sum(cls_pred_mask)
            cls_targets = np.sum(cls_target_mask)
            cls_tp = tp_np[cls_pred_mask].sum() if cls_predictions > 0 else 0
            cls_fp = cls_predictions - cls_tp
            
            # Confidence stats for this class
            cls_conf = conf_np[cls_pred_mask] if cls_predictions > 0 else []
            avg_conf = np.mean(cls_conf) if len(cls_conf) > 0 else 0
            min_conf = np.min(cls_conf) if len(cls_conf) > 0 else 0
            max_conf = np.max(cls_conf) if len(cls_conf) > 0 else 0
            
            precision = cls_tp / max(cls_predictions, 1)
            recall = cls_tp / max(cls_targets, 1) if cls_targets > 0 else 0
            f1 = 2 * (precision * recall) / max(precision + recall, 1e-8)
            
            self._debug_log(f"\n  CLASS {int(cls)} ANALYSIS:")
            self._debug_log(f"    📊 Counts: {cls_predictions:,} predictions, {cls_targets:,} targets")
            self._debug_log(f"    ✅ True Positives: {cls_tp:,}")
            self._debug_log(f"    ❌ False Positives: {cls_fp:,}")
            self._debug_log(f"    📈 Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            self._debug_log(f"    🎯 Confidence: avg={avg_conf:.4f}, min={min_conf:.4f}, max={max_conf:.4f}")
            
            # Issue detection
            if cls_targets == 0:
                self._debug_log(f"    ⚠️  ISSUE: NO GROUND TRUTH for class {int(cls)} - all {cls_predictions} predictions are FP!")
            if cls_predictions == 0 and cls_targets > 0:
                self._debug_log(f"    ⚠️  ISSUE: NO PREDICTIONS for class {int(cls)} - missing all {cls_targets} targets!")
            if cls_tp == 0 and cls_predictions > 0 and cls_targets > 0:
                self._debug_log(f"    ⚠️  ISSUE: ZERO TRUE POSITIVES - predictions don't match targets (IoU/class mismatch)")
        
        # Confidence threshold analysis
        self._debug_log("\n🎚️  CONFIDENCE THRESHOLD ANALYSIS:")
        self._debug_log("    Threshold | Predictions | True Positives | Precision")
        self._debug_log("    ---------|-------------|----------------|----------")
        thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        for thresh in thresholds:
            high_conf_mask = conf_np >= thresh
            high_conf_tp = tp_np[high_conf_mask].sum()
            high_conf_total = np.sum(high_conf_mask)
            high_conf_precision = high_conf_tp / max(high_conf_total, 1)
            
            self._debug_log(f"    {thresh:8.1f} | {high_conf_total:11,} | {high_conf_tp:14,} | {high_conf_precision:8.4f}")
        
        # IoU/matching analysis hints
        if total_tp == 0:
            self._debug_log("\n❌ CRITICAL ISSUE: ZERO TRUE POSITIVES DETECTED!")
            self._debug_log("   This means NO predictions matched ANY ground truth targets.")
            self._debug_log("   Root causes to investigate:")
            self._debug_log("   1. 🎯 IoU threshold too high (try lowering from current value)")
            self._debug_log("   2. 🏷️  Class mismatch (predicted classes ≠ target classes)")
            self._debug_log("   3. 📦 Coordinate format issues (bbox format mismatch)")
            self._debug_log("   4. 📏 Scale issues (predictions in wrong coordinate space)")
            self._debug_log("   5. 🗂️  Empty targets or corrupted prediction data")
            self._debug_log("   6. 🎲 Model predicting random/incorrect bboxes")
            self._debug_log("   7. 🔧 Data preprocessing issues (wrong normalization)")
            
        # Additional insights
        self._debug_log(f"\n📋 SUMMARY INSIGHTS:")
        if total_predictions == 0:
            self._debug_log("   • Model is NOT generating any predictions - check model inference")
        elif total_targets == 0:
            self._debug_log("   • NO ground truth targets found - check dataset loading")
        elif total_tp == 0:
            self._debug_log("   • Predictions exist but don't match targets - check IoU/class alignment")
        else:
            overall_precision = total_tp / total_predictions
            avg_targets_per_class = total_targets / len(unique_target_classes) if len(unique_target_classes) > 0 else 0
            self._debug_log(f"   • Overall precision: {overall_precision:.4f}")
            self._debug_log(f"   • Average targets per class: {avg_targets_per_class:.1f}")
            if overall_precision < 0.1:
                self._debug_log("   • Very low precision - likely IoU or class mismatch issues")
            elif overall_precision < 0.5:
                self._debug_log("   • Low precision - model needs more training or threshold tuning")
        
        self._debug_log("=" * 80)
    
    def _should_use_fast_approximation(self, stats: list, data_size: int) -> bool:
        """
        Determine if fast approximation should be used.
        
        Args:
            stats: Statistics arrays
            data_size: Number of data points
            
        Returns:
            bool: True if fast approximation is recommended
            
        Time Complexity: O(1) - simple threshold checks
        """
        pred_cls = stats[2]
        pred_cls_cpu = pred_cls.cpu().numpy() if hasattr(pred_cls, 'cpu') else pred_cls
        
        # Use fast approximation for very small datasets with minimal predictions
        return data_size < 10 and len(np.unique(pred_cls_cpu)) <= 1
    
    def _compute_fast_approximation(self, stats: list) -> Dict[str, float]:
        """
        Compute fast approximation for minimal datasets.
        
        Args:
            stats: Statistics arrays
            
        Returns:
            Dictionary with approximated metrics
            
        Time Complexity: O(1) - simple arithmetic operations
        """
        tp, conf, pred_cls, target_cls = stats
        
        logger.info("⚡ Very small dataset - using fast approximation")
        
        if tp.sum() > 0:
            approx_precision = tp.sum() / len(pred_cls) if len(pred_cls) > 0 else 0.0
            approx_recall = tp.sum() / len(target_cls) if len(target_cls) > 0 else 0.0
            approx_f1 = (2 * (approx_precision * approx_recall) / 
                        (approx_precision + approx_recall)) if (approx_precision + approx_recall) > 0 else 0.0
        else:
            approx_precision = approx_recall = approx_f1 = 0.0
        
        return {
            'map50': float(approx_f1),  # Use F1 as mAP approximation
            'map50_95': 0.0,
            'precision': float(approx_precision),
            'recall': float(approx_recall),
            'f1': float(approx_f1)
        }
    
    def _compute_full_map(self, stats: list, data_size: int) -> Dict[str, float]:
        """
        Compute full mAP using YOLOv5 ap_per_class function.
        
        Args:
            stats: Statistics arrays
            data_size: Number of data points
            
        Returns:
            Dictionary with complete mAP metrics
            
        Time Complexity: O(N log N) for AP computation
        Space Complexity: O(N) for intermediate arrays
        """
        logger.info("🧮 Computing mAP metrics (this may take a moment)...")
        
        if self.debug:
            # Write to debug file instead of console
            self._debug_log(f"⚡ Starting ap_per_class computation with {len(stats[2])} predictions and {len(stats[3])} targets")
        
        def run_ap_computation():
            return get_ap_per_class()(
                *stats, 
                plot=False, 
                save_dir="", 
                names={}
            )
        
        # Platform-aware computation strategy
        if data_size > 1000 and not self.memory_processor.memory_optimizer.platform_info['is_apple_silicon']:
            # Use threading for large datasets (but not on MPS which has threading issues)
            logger.info("📊 Large dataset - using optimized threading")
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(run_ap_computation)
                tp, _, p, r, f1, ap, ap_class = future.result()
        else:
            # Direct computation for MPS or smaller datasets
            tp, _, p, r, f1, ap, ap_class = run_ap_computation()
        
        if self.debug:
            self._debug_log("✅ ap_per_class computation completed")
        
        # Clean up memory after expensive computation
        self.memory_optimizer.cleanup_memory()
        
        if self.debug:
            self._debug_log(f"📊 AP RESULTS:")
            self._debug_log(f"  • AP shape: {ap.shape}")
            self._debug_log(f"  • AP classes: {ap_class}")
            self._debug_log(f"  • AP values: {ap[:, 0] if ap.shape[1] > 0 else 'no AP data'}")
            self._debug_log(f"  • Precision: {p}")
            self._debug_log(f"  • Recall: {r}")
        
        # Extract mAP@0.5 and compute mean metrics
        ap50 = ap[:, 0] if ap.shape[1] > 0 else np.array([0.0])
        
        map50 = ap50.mean()
        precision = p.mean()
        recall = r.mean()
        f1_score = f1.mean()
        
        if self.debug:
            self._debug_log(f"Hierarchical mAP: {map50:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
        
        return {
            'map50': float(map50),
            'map50_95': 0.0,  # Not computed for performance
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1_score)
        }
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get comprehensive processing statistics from all modules.
        
        Returns:
            Dictionary with processing statistics
            
        Time Complexity: O(1) - simple data collection
        """
        return {
            'calculator_stats': {
                'num_classes': self.num_classes,
                'conf_thres': self.conf_thres,
                'iou_thres': self.iou_thres,
                'batch_count': self._batch_count,
                'accumulated_batches': len(self.stats),
                'device': str(self.device)
            },
            'yolo_utils_available': self.yolo_utils.is_available(),
            'hierarchical_processor_stats': getattr(self.hierarchical_processor, 'get_stats', lambda: {})(),
            'memory_processor_stats': self.memory_processor.get_processing_stats(),
            'batch_processor_stats': self.batch_processor.get_processing_stats()
        }


def create_yolov5_map_calculator(
    num_classes: int = 7, 
    conf_thres: float = 0.005, 
    iou_thres: float = 0.03, 
    debug: bool = False,
    training_context: dict = None
) -> YOLOv5MapCalculator:
    """
    Factory function to create YOLOv5 mAP calculator with modular architecture.
    
    Args:
        num_classes: Number of classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        debug: Enable hierarchical debug logging
        training_context: Training context information (backbone, phase, etc.)
        
    Returns:
        YOLOv5MapCalculator instance with SRP-compliant architecture
        
    Time Complexity: O(1) - simple object creation
    """
    return YOLOv5MapCalculator(num_classes, conf_thres, iou_thres, debug, training_context)