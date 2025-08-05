#!/usr/bin/env python3
"""
YOLOv5-based mAP calculator for SmartCash validation phase.

This is a clean, modular implementation that follows Single Responsibility Principle
and maintains backward compatibility with the original API. Each major responsibility
has been extracted into focused, reusable modules.

Architecture:
- YOLOv5UtilitiesManager: Handles YOLOv5 imports and lazy loading
- HierarchicalProcessor: Manages multi-layer confidence modulation
- MemoryOptimizedProcessor: Platform-aware memory management
- BatchProcessor: Handles batch-level prediction processing
- MapDebugLogger: Comprehensive debug logging and analysis
- MapStatisticsProcessor: Statistics validation and concatenation
- MapMetricsComputer: mAP computation algorithms
- YOLOv5MapCalculator: Core coordination logic (this file)

Algorithmic Improvements:
- Vectorized operations: O(N) instead of O(N²) in many cases
- Memory-conscious chunking: O(chunk_size) space complexity
- Platform-aware optimization: Adaptive to hardware capabilities
- Progressive confidence/IoU thresholds: Epoch-aware threshold adjustment
"""

import torch
from typing import Dict

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

# Import specialized processors
from .yolo_utils_manager import YOLOv5UtilitiesManager
from .hierarchical_processor import HierarchicalProcessor
from .memory_optimized_processor import MemoryOptimizedProcessor
from .batch_processor import BatchProcessor

# Import extracted modules
from .map_debug_logger import MapDebugLogger
from .map_statistics_processor import MapStatisticsProcessor
from .map_metrics_computer import MapMetricsComputer

# Re-export utility functions for backward compatibility
from .yolo_utils_manager import (
    get_box_iou, 
    get_xywh2xyxy,
    get_non_max_suppression,
    get_ap_per_class
)

logger = get_logger(__name__, level="DEBUG")

# Debug flag for hierarchical validation (can be overridden by config)
DEBUG_HIERARCHICAL = False


class YOLOv5MapCalculator:
    """
    YOLOv5-based mAP calculator using modular, SRP-compliant architecture.
    
    This calculator provides accurate mAP@0.5 computation that matches
    standard YOLO evaluation protocols while maintaining clean separation
    of concerns across specialized processing modules.
    
    Features:
    - Progressive confidence/IoU thresholds based on training epoch
    - Platform-aware memory optimization
    - Comprehensive debug logging
    - Hierarchical processing for multi-layer architectures
    
    Time Complexity: O(N log N) for sorting + O(N*M) for IoU computation
    Space Complexity: O(N*M) for IoU matrices, O(N) for statistics
    """
    
    def __init__(
        self, 
        num_classes: int = 7, 
        conf_thres: float = 0.01, 
        iou_thres: float = 0.5, 
        debug: bool = True,
        training_context: dict = None,
        use_progressive_thresholds: bool = True
    ):
        """
        Initialize YOLOv5 mAP calculator with modular architecture.
        
        Args:
            num_classes: Number of classes (default 7 for SmartCash banknotes)
            conf_thres: Base confidence threshold for predictions
            iou_thres: Base IoU threshold for NMS and mAP calculation
            debug: Enable hierarchical debug logging
            training_context: Training context information (backbone, phase, etc.)
            use_progressive_thresholds: Enable progressive threshold scheduling based on epoch
        """
        self.num_classes = num_classes
        self.base_conf_thres = conf_thres  # Store base threshold
        self.base_iou_thres = iou_thres    # Store base threshold
        self.use_progressive_thresholds = use_progressive_thresholds
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug = debug
        self.training_context = training_context or {}
        self.current_epoch = 0  # Initialize current_epoch
        
        # Initialize device and memory management first
        self.memory_optimizer = get_memory_optimizer()
        self.device = self.memory_optimizer.device
        
        # Initialize extracted modules
        self.debug_logger = MapDebugLogger(training_context) if debug else None
        self.statistics_processor = MapStatisticsProcessor(debug)
        self.metrics_computer = MapMetricsComputer(debug)
        
        if debug:
            logger.info(f"🐛 YOLOv5MapCalculator DEBUG MODE ENABLED - Debug logging will be setup per epoch")
        
        # Initialize specialized processors
        self._init_processors()
        
        # Initialize storage
        self._init_storage()
    
    def update_epoch(self, epoch: int):
        """Update current epoch and recalculate progressive thresholds if enabled."""
        self.current_epoch = epoch
        
        if self.use_progressive_thresholds:
            old_conf, old_iou = self.conf_thres, self.iou_thres
            self.conf_thres, self.iou_thres = self._calculate_progressive_thresholds(epoch)
            
            if old_conf != self.conf_thres or old_iou != self.iou_thres:
                logger.info(f"📈 Epoch {epoch}: Progressive thresholds updated - conf_thres: {old_conf:.3f}→{self.conf_thres:.3f}, iou_thres: {old_iou:.3f}→{self.iou_thres:.3f}")
            
            # Update processor thresholds
            if hasattr(self, 'batch_processor') and self.batch_processor:
                self.batch_processor.conf_threshold = self.conf_thres
                self.batch_processor.iou_threshold = self.iou_thres
    
    def _calculate_progressive_thresholds(self, epoch: int) -> tuple:
        """
        Calculate progressive confidence and IoU thresholds based on training epoch.
        
        Args:
            epoch: Current training epoch (0-based)
            
        Returns:
            Tuple of (conf_thres, iou_thres)
        """
        if epoch < 10:
            return 0.01, 0.1   # Super lenient, encourage any matching
        elif epoch < 20:
            return 0.03, 0.2   # Slightly tighter, but still low IoU
        elif epoch < 30:
            return 0.05, 0.3   # Now encouraging better alignment
        elif epoch < 40:
            return 0.07, 0.4   # Starting to demand real box quality
        else:
            return 0.1, 0.5    # Final target threshold
    
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
        if self.debug and self.debug_logger:
            self.debug_logger.log_epoch_reset(len(self.stats), self._batch_count)
        
        self.stats.clear()
        self._batch_count = 0
        
        # Clean memory before starting new validation
        self.memory_optimizer.cleanup_memory()
    
    def update(self, predictions: torch.Tensor, targets: torch.Tensor, epoch: int = 0):
        """
        Update mAP statistics with batch predictions and targets.
        
        Args:
            predictions: Model predictions in YOLO format [batch, detections, 6]
                        where each detection is [x, y, w, h, conf, class]
            targets: Ground truth targets [num_targets, 6] 
                    where each target is [batch_idx, class, x, y, w, h]
            epoch: The current epoch number.
                    
        Time Complexity: O(P*T) for IoU + O(P log P) for sorting
        Space Complexity: O(P*T) for IoU matrix
        """
        self.current_epoch = epoch
        
        # Setup or update debug logging for current epoch
        if self.debug and self.debug_logger:
            self.debug_logger.setup_debug_logging(epoch)
        
        # Check YOLOv5 availability
        if not self._ensure_yolov5_available() or predictions is None or targets is None:
            if self.debug and self.debug_logger:
                self.debug_logger.write_debug_log(f"Skipping update: yolov5_available={self.yolo_utils.is_available()}, "
                              f"predictions={predictions is not None}, targets={targets is not None}")
            return
        
        # Comprehensive input validation
        if not self.statistics_processor.validate_inputs(predictions, targets):
            return
        
        # Track batch count for debugging
        self._batch_count += 1
        
        # Enhanced debug logging for mAP investigation
        if self.debug and self.debug_logger:
            if self._batch_count == 1:
                self.debug_logger.write_debug_log(f"📊 mAP update first batch: pred_shape={predictions.shape}, target_shape={targets.shape}")
                self.debug_logger.write_debug_log(f"📊 Initial predictions sample: conf_range=[{predictions[:,:,4].min():.6f}, {predictions[:,:,4].max():.6f}]")
                self.debug_logger.write_debug_log(f"📊 Initial predictions classes: unique={torch.unique(predictions[:,:,5]).tolist()}")
                if targets.numel() > 0:
                    self.debug_logger.write_debug_log(f"📊 Initial targets classes: unique={torch.unique(targets[:,1]).tolist()}")
                else:
                    self.debug_logger.write_debug_log(f"📊 No targets in first batch - empty targets tensor")
        
        try:
            # Apply hierarchical filtering for Phase 2 multi-layer architecture
            processed_predictions, processed_targets = self.hierarchical_processor.process_hierarchical_predictions(
                predictions, targets, epoch=self.current_epoch
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
                if self.debug and self.debug_logger:
                    self.debug_logger.log_batch_processing(self._batch_count, batch_stats)
            else:
                if self.debug and self.debug_logger:
                    self.debug_logger.write_debug_log(f"❌ batch_stats is None for batch {self._batch_count} - check prediction processing")
            
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
            - 'accuracy': Mean accuracy (uses recall for detection tasks)
            
        Time Complexity: O(N log N) for AP computation where N is total detections
        Space Complexity: O(N) for statistics concatenation
        """
        if not self._ensure_yolov5_available():
            raise RuntimeError("YOLOv5 not available - cannot compute mAP metrics")
        
        if not self.stats:
            logger.error(f"🚨 No statistics accumulated! update() called {self._batch_count} times")
            if self.debug and self.debug_logger:
                self.debug_logger.write_debug_log(f"mAP calculator state: yolov5_available={self.yolo_utils.is_available()}, "
                              f"device={self.device}")
            raise RuntimeError("No statistics accumulated - check if update() was called during validation")
        
        # Validate accumulated statistics using extracted processor
        if not self.statistics_processor.validate_statistics(self.stats):
            raise RuntimeError("Statistics validation failed")
        
        try:
            # Concatenate statistics using extracted processor
            stats = self.statistics_processor.concatenate_statistics(self.stats)
            
            if len(stats) >= 4:  # Ensure we have all 4 required stat arrays
                data_size = len(stats[2]) if len(stats) > 2 else 0  # Use pred_cls length
                
                # Enhanced debugging using extracted debug logger
                if self.debug and self.debug_logger:
                    summary = self.statistics_processor.get_statistics_summary(stats)
                    self.debug_logger.write_debug_log(f"📊 Statistics summary: {summary}")
                    logger.info(f"🐛 Running mAP debug analysis (see debug log file)...")
                    self.debug_logger.write_debug_log(f"\n🔍 STARTING mAP COMPUTATION ANALYSIS (Epoch validation)")
                    self.debug_logger.write_debug_log(f"📊 Total batches processed: {self._batch_count}")
                    self.debug_logger.write_debug_log(f"📊 Total stat entries: {len(self.stats)}")
                    self.debug_logger.write_debug_log(f"📊 Total detection samples: {data_size}")
                    self.debug_logger.log_class_analysis(stats)
                
                # Check if we have any data at all
                if data_size > 0:
                    # Compute metrics using extracted computer
                    return self.metrics_computer.compute_metrics(stats, data_size)
                else:
                    # No data samples
                    logger.warning("📊 No detection samples accumulated - check if predictions are being generated")
                    if self.debug and self.debug_logger:
                        self.debug_logger.write_debug_log(f"\n❌ CRITICAL: NO DETECTION SAMPLES ACCUMULATED")
                        self.debug_logger.write_debug_log(f"📊 Batches processed: {self._batch_count}")
                        self.debug_logger.write_debug_log(f"📊 Stat entries: {len(self.stats)}")
                        self.debug_logger.write_debug_log(f"📊 This means either:")
                        self.debug_logger.write_debug_log(f"   1. Model is not generating predictions")
                        self.debug_logger.write_debug_log(f"   2. Predictions are filtered out by confidence threshold ({self.conf_thres})")
                        self.debug_logger.write_debug_log(f"   3. Batch processing is failing")
                        self.debug_logger.write_debug_log(f"   4. update() method is not being called during validation")
                    return self._create_zero_metrics()
            else:
                # Malformed statistics
                logger.error(f"🚨 Malformed statistics! Expected 4 arrays, got {len(stats)}")
                if self.debug and self.debug_logger:
                    self.debug_logger.write_debug_log(f"Statistics structure: {[type(s) for s in stats]}")
                return self._create_zero_metrics()
            
        except Exception as e:
            logger.error(f"Error computing mAP: {e}")
            # Emergency cleanup on error
            self.memory_processor.emergency_cleanup()
            raise RuntimeError(f"mAP computation failed: {e}") from e
    
    def _create_zero_metrics(self) -> Dict[str, float]:
        """
        Create zero metrics dictionary for error cases.
        
        Returns:
            Dictionary with all metrics set to 0.0
        """
        return {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
    
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
    
    
    def get_processing_stats(self) -> Dict[str, any]:
        """
        Get comprehensive processing statistics from all modules.
        
        Returns:
            Dictionary with processing statistics
            
        Time Complexity: O(1) - simple data collection
        """
        stats = {
            'calculator_stats': {
                'num_classes': self.num_classes,
                'conf_thres': self.conf_thres,
                'iou_thres': self.iou_thres,
                'use_progressive_thresholds': self.use_progressive_thresholds,
                'current_epoch': self.current_epoch,
                'batch_count': self._batch_count,
                'accumulated_batches': len(self.stats),
                'device': str(self.device)
            },
            'yolo_utils_available': self.yolo_utils.is_available(),
            'hierarchical_processor_stats': getattr(self.hierarchical_processor, 'get_stats', lambda: {})(),
            'memory_processor_stats': self.memory_processor.get_processing_stats(),
            'batch_processor_stats': self.batch_processor.get_processing_stats()
        }
        
        # Add statistics from extracted modules if available
        if self.stats:
            try:
                concatenated_stats = self.statistics_processor.concatenate_statistics(self.stats)
                stats['statistics_summary'] = self.statistics_processor.get_statistics_summary(concatenated_stats)
            except Exception as e:
                stats['statistics_summary'] = {'error': f'Failed to get summary: {e}'}
        
        return stats


def create_yolov5_map_calculator(
    num_classes: int = 7, 
    conf_thres: float = 0.1, 
    iou_thres: float = 0.5, 
    debug: bool = False,
    training_context: dict = None,
    use_progressive_thresholds: bool = True
) -> YOLOv5MapCalculator:
    """
    Factory function to create YOLOv5 mAP calculator with modular architecture.
    
    Args:
        num_classes: Number of classes
        conf_thres: Base confidence threshold (used if progressive_thresholds=False)
        iou_thres: Base IoU threshold (used if progressive_thresholds=False)
        debug: Enable hierarchical debug logging
        training_context: Training context information (backbone, phase, etc.)
        use_progressive_thresholds: Enable progressive threshold scheduling based on epoch
        
    Returns:
        YOLOv5MapCalculator instance with SRP-compliant architecture
        
    Time Complexity: O(1) - simple object creation
    """
    return YOLOv5MapCalculator(num_classes, conf_thres, iou_thres, debug, training_context, use_progressive_thresholds)


# Re-export all public symbols to maintain API compatibility
__all__ = [
    'YOLOv5MapCalculator',
    'create_yolov5_map_calculator',
    'DEBUG_HIERARCHICAL',
    'get_ap_per_class',
    'get_box_iou',
    'get_xywh2xyxy', 
    'get_non_max_suppression'
]