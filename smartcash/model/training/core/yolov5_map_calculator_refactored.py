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
        debug: bool = False
    ):
        """
        Initialize YOLOv5 mAP calculator with modular architecture.
        
        Args:
            num_classes: Number of classes (default 7 for SmartCash banknotes)
            conf_thres: Confidence threshold for predictions
            iou_thres: IoU threshold for NMS and mAP calculation
            debug: Enable hierarchical debug logging
        """
        self.num_classes = num_classes
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.debug = debug
        
        # Initialize device and memory management
        self.memory_optimizer = get_memory_optimizer()
        self.device = self.memory_optimizer.device
        
        # Initialize specialized processors
        self._init_processors()
        
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
            debug=self.debug
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
        self.stats.clear()
        self._batch_count = 0
        
        # Clean memory before starting new validation
        if self.debug:
            logger.debug("🧹 Cleaning memory before validation reset")
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
                logger.debug(f"Skipping update: yolov5_available={self.yolo_utils.is_available()}, "
                           f"predictions={predictions is not None}, targets={targets is not None}")
            return
        
        # Comprehensive input validation
        if not self._validate_inputs(predictions, targets):
            return
        
        # Track batch count for debugging
        self._batch_count += 1
        if self.debug and self._batch_count <= 3:
            logger.debug(f"📊 mAP update batch {self._batch_count}: "
                        f"pred_shape={predictions.shape}, target_shape={targets.shape}")
        
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
                if self.debug and self._batch_count <= 3:
                    logger.debug(f"✅ Added batch_stats to mAP calculator (total batches: {len(self.stats)})")
            else:
                if self.debug:
                    logger.warning(f"❌ batch_stats is None for batch {self._batch_count}")
            
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
                logger.debug(f"mAP calculator state: yolov5_available={self.yolo_utils.is_available()}, "
                           f"device={self.device}")
            raise RuntimeError("No statistics accumulated - check if update() was called during validation")
        
        # Validate accumulated statistics
        if not self._validate_statistics():
            raise RuntimeError("Statistics validation failed")
        
        try:
            # Optimized statistics concatenation
            stats = self._concatenate_statistics()
            
            if len(stats) and stats[0].any():
                # Log processing information
                data_size = len(stats[0]) if len(stats) > 0 else 0
                logger.info(f"📈 Processing {data_size} detection samples for mAP computation...")
                
                if data_size > 10000:
                    logger.info("⏳ Large dataset detected - mAP computation may take 30-60 seconds")
                
                if self.debug:
                    self._log_class_analysis(stats)
                
                # Check for fast approximation on minimal data
                if self._should_use_fast_approximation(stats, data_size):
                    return self._compute_fast_approximation(stats)
                
                # Compute full mAP using YOLOv5 function
                return self._compute_full_map(stats, data_size)
            
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
                    logger.debug("Empty or invalid predictions tensor")
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
                    logger.debug("Invalid targets tensor")
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
        logger.debug("📊 Concatenating statistics for mAP computation...")
        
        if len(self.stats) == 1:
            # Single batch optimization - no concatenation needed
            return [x.cpu().numpy() if hasattr(x, 'cpu') else x for x in self.stats[0]]
        else:
            # Multiple batches - efficient GPU concatenation then single CPU transfer
            total_size = sum(stat[0].shape[0] for stat in self.stats)
            logger.debug(f"📊 Concatenating {len(self.stats)} batches, total size: {total_size}")
            
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
        Log detailed class analysis for debugging.
        
        Args:
            stats: Concatenated statistics
            
        Time Complexity: O(N) for unique class computation
        """
        tp, conf, pred_cls, target_cls = stats
        logger.debug(f"📈 CLASS ANALYSIS:")
        logger.debug(f"  • Total predictions: {len(pred_cls)}")
        logger.debug(f"  • Total targets: {len(target_cls)}")
        logger.debug(f"  • Total TP: {tp.sum()}")
        logger.debug(f"  • Unique pred classes: {np.unique(pred_cls)}")
        logger.debug(f"  • Unique target classes: {np.unique(target_cls)}")
        logger.debug(f"  • Confidence range: {conf.min():.6f} - {conf.max():.6f}")
    
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
            logger.debug(f"⚡ Starting ap_per_class computation with {len(stats[2])} predictions and {len(stats[3])} targets")
        
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
        
        logger.debug("✅ ap_per_class computation completed")
        
        # Clean up memory after expensive computation
        self.memory_optimizer.cleanup_memory()
        
        if self.debug:
            logger.debug(f"📊 AP RESULTS:")
            logger.debug(f"  • AP shape: {ap.shape}")
            logger.debug(f"  • AP classes: {ap_class}")
            logger.debug(f"  • AP values: {ap[:, 0] if ap.shape[1] > 0 else 'no AP data'}")
            logger.debug(f"  • Precision: {p}")
            logger.debug(f"  • Recall: {r}")
        
        # Extract mAP@0.5 and compute mean metrics
        ap50 = ap[:, 0] if ap.shape[1] > 0 else np.array([0.0])
        
        map50 = ap50.mean()
        precision = p.mean()
        recall = r.mean()
        f1_score = f1.mean()
        
        if self.debug:
            logger.debug(f"Hierarchical mAP: {map50:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
        
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
    debug: bool = False
) -> YOLOv5MapCalculator:
    """
    Factory function to create YOLOv5 mAP calculator with modular architecture.
    
    Args:
        num_classes: Number of classes
        conf_thres: Confidence threshold
        iou_thres: IoU threshold
        debug: Enable hierarchical debug logging
        
    Returns:
        YOLOv5MapCalculator instance with SRP-compliant architecture
        
    Time Complexity: O(1) - simple object creation
    """
    return YOLOv5MapCalculator(num_classes, conf_thres, iou_thres, debug)