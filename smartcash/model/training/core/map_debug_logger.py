#!/usr/bin/env python3
"""
Debug logging module for mAP calculation analysis.

This module provides comprehensive debug logging functionality for mAP calculations,
including detailed class analysis, confidence threshold analysis, and comprehensive
debugging information. Extracted from YOLOv5MapCalculator for better separation of concerns.

Key Features:
- Epoch-specific debug file creation
- Comprehensive class-wise analysis
- Confidence threshold analysis 
- Memory-efficient statistics logging
- Training context integration

Time Complexity: O(N) for most logging operations where N is number of predictions
Space Complexity: O(1) for logging operations (file-based output)
"""

import logging
import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Optional, Union

from smartcash.common.logger import get_logger

logger = get_logger(__name__, level="DEBUG")


class MapDebugLogger:
    """
    Dedicated debug logger for mAP calculation analysis.
    
    Provides comprehensive debug logging with epoch-specific files, class analysis,
    confidence analysis, and detailed statistics logging for mAP debugging.
    
    Features:
    - Epoch-specific debug file management
    - Comprehensive class-wise analysis and statistics
    - Confidence threshold analysis across multiple thresholds
    - Training context integration
    - Memory-efficient logging operations
    
    Time Complexity: O(N) for analysis operations where N is number of predictions
    Space Complexity: O(1) - all output is file-based
    """
    
    def __init__(self, training_context: Optional[Dict] = None):
        """
        Initialize mAP debug logger.
        
        Args:
            training_context: Training context information (backbone, phase, etc.)
        """
        self.training_context = training_context or {}
        self.debug_logger = None
        self.current_epoch = 0
        self._current_debug_epoch = -1
        
        # Debug configuration
        self.confidence_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    def update_current_phase(self, phase_num: int):
        """
        Update the current phase in the training context.
        
        Args:
            phase_num: Current phase number (1 or 2)
        """
        self.training_context['current_phase'] = phase_num
        self.training_context['phase'] = phase_num  # Also set 'phase' key for compatibility
    
    def setup_debug_logging(self, epoch: int) -> None:
        """
        Set up dedicated debug file logging for mAP analysis.
        
        Creates debug log file in logs/validation_metrics/ with timestamp and training context.
        
        Args:
            epoch: Current training epoch
            
        Time Complexity: O(1) - simple file creation
        """
        self.current_epoch = epoch
        
        # Skip setup if already configured for this epoch
        if self._current_debug_epoch == epoch and self.debug_logger:
            return
            
        # Extract context information for filename
        backbone = self.training_context.get('backbone', 'unknown')
        # Try both 'current_phase' and 'phase' keys for phase detection
        phase = self.training_context.get('current_phase') or self.training_context.get('phase', 'unknown')
        training_mode = self.training_context.get('training_mode', 'unknown')
        
        # Create debug log directory
        debug_log_dir = Path(f"logs/validation_metrics/{backbone}/phase_{phase}")
        debug_log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped debug log file with context
        debug_log_file = debug_log_dir / f"map_debug_epoch{epoch}.log"
        
        # Set up dedicated debug logger
        logger_name = f"map_debug_epoch{epoch}"
        self.debug_logger = logging.getLogger(logger_name)
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
        self._write_session_header(debug_log_file)
        
        # Update current debug epoch
        self._current_debug_epoch = epoch
        
        # Log to console about debug file creation
        logger.info(f"📄 Debug log file created: {debug_log_file}")
    
    def write_debug_log(self, message: str) -> None:
        """
        Write message to debug log file if debug mode is enabled.
        
        Args:
            message: Message to log
            
        Time Complexity: O(1) - simple file write
        """
        if self.debug_logger:
            self.debug_logger.info(message)
    
    def log_class_analysis(self, stats: List[Union[torch.Tensor, np.ndarray]]) -> None:
        """
        Log comprehensive class analysis for mAP debugging.
        
        Provides detailed per-class statistics, confidence analysis, and issue detection
        to help debug mAP calculation problems.
        
        Args:
            stats: Concatenated statistics [tp, conf, pred_cls, target_cls]
            
        Time Complexity: O(N + C*N) where N is predictions, C is unique classes
        Space Complexity: O(C) for class-specific statistics
        """
        if not self.debug_logger or len(stats) < 4:
            return
            
        tp, conf, pred_cls, target_cls = stats[:4]
        
        # Convert to numpy for easier analysis
        tp_np = self._to_numpy(tp)
        conf_np = self._to_numpy(conf)
        pred_cls_np = self._to_numpy(pred_cls)
        target_cls_np = self._to_numpy(target_cls)
        
        self.write_debug_log("\n" + "=" * 80)
        self.write_debug_log("🔍 COMPREHENSIVE mAP DEBUG ANALYSIS")
        self.write_debug_log("=" * 80)
        
        # Overall statistics
        self._log_overall_statistics(tp_np, conf_np, pred_cls_np, target_cls_np)
        
        # Class distribution analysis
        self._log_class_distribution(pred_cls_np, target_cls_np)
        
        # Per-class detailed analysis
        self._log_per_class_analysis(tp_np, conf_np, pred_cls_np, target_cls_np)
        
        # Confidence threshold analysis
        self._log_confidence_threshold_analysis(tp_np, conf_np)
        
        # Issue detection and insights
        self._log_analysis_insights(tp_np, conf_np, pred_cls_np, target_cls_np)
        
        self.write_debug_log("=" * 80)
    
    def log_batch_processing(self, batch_count: int, batch_stats: tuple) -> None:
        """
        Log batch processing information for debugging.
        
        Args:
            batch_count: Current batch number
            batch_stats: Batch statistics tuple (tp, conf, pred_cls, target_cls)
            
        Time Complexity: O(1) - simple logging
        """
        if not self.debug_logger:
            return
            
        if batch_count == 1:
            tp, conf, pred_cls, target_cls = batch_stats
            self.write_debug_log(f"✅ Added first batch_stats to mAP calculator")
            self.write_debug_log(f"   • TP: {tp.sum().item()}/{len(tp)} ({tp.sum().item()/max(len(tp),1)*100:.1f}%)")
            self.write_debug_log(f"   • Confidence range: [{conf.min().item():.6f}, {conf.max().item():.6f}]")
            self.write_debug_log(f"   • Predicted classes: {torch.unique(pred_cls).tolist()}")
            self.write_debug_log(f"   • Target classes: {torch.unique(target_cls).tolist()}")
        elif batch_count % 10 == 0:  # Log every 10th batch
            tp, conf, pred_cls, target_cls = batch_stats
            self.write_debug_log(f"📊 Batch {batch_count} stats: TP={tp.sum().item()}/{len(tp)}, conf_avg={conf.mean().item():.4f}")
    
    def log_statistics_processing(self, stats_count: int, total_size: int) -> None:
        """
        Log statistics processing information.
        
        Args:
            stats_count: Number of statistics batches
            total_size: Total number of samples
            
        Time Complexity: O(1) - simple logging
        """
        if not self.debug_logger:
            return
            
        self.write_debug_log("📊 Concatenating statistics for mAP computation...")
        if stats_count > 1:
            self.write_debug_log(f"📊 Concatenating {stats_count} batches, total size: {total_size}")
    
    def log_epoch_reset(self, stats_count: int, batch_count: int) -> None:
        """
        Log epoch reset information.
        
        Args:
            stats_count: Number of accumulated statistics
            batch_count: Number of processed batches
            
        Time Complexity: O(1) - simple logging
        """
        if not self.debug_logger:
            return
            
        self.write_debug_log(f"\n🔄 VALIDATION EPOCH RESET")
        self.write_debug_log(f"📊 Previous epoch stats: {stats_count} batches, {batch_count} updates")
        
        if stats_count > 0:
            self.write_debug_log(f"📊 Previous epoch had accumulated statistics")
        
        self.write_debug_log("🧹 Memory cleaned, ready for new validation epoch")
    
    def _write_session_header(self, debug_log_file: Path) -> None:
        """
        Write comprehensive session header to debug log.
        
        Args:
            debug_log_file: Path to debug log file
            
        Time Complexity: O(1) - simple header writing
        """
        self.write_debug_log("=" * 80)
        self.write_debug_log("YOLOv5 mAP Calculator Debug Session Started")
        self.write_debug_log("=" * 80)
        
        # Log training context information
        self.write_debug_log("Training Context:")
        self.write_debug_log(f"  • Backbone: {self.training_context.get('backbone', 'unknown')}")
        self.write_debug_log(f"  • Training Mode: {self.training_context.get('training_mode', 'unknown')}")
        # Try both phase keys for better compatibility
        current_phase = self.training_context.get('current_phase') or self.training_context.get('phase', 'unknown')
        self.write_debug_log(f"  • Current Phase: {current_phase}")
        self.write_debug_log(f"  • Session ID: {self.training_context.get('session_id', 'N/A')}")
        self.write_debug_log(f"  • Model Name: {self.training_context.get('model_name', 'N/A')}")
        self.write_debug_log(f"  • Layer Mode: {self.training_context.get('layer_mode', 'N/A')}")
        self.write_debug_log(f"  • Detection Layers: {self.training_context.get('detection_layers', 'N/A')}")
        
        # Log file information
        self.write_debug_log("")
        self.write_debug_log(f"Debug log file: {debug_log_file}")
        self.write_debug_log("")
    
    def _to_numpy(self, tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """
        Convert tensor to numpy array safely.
        
        Args:
            tensor: Input tensor or array
            
        Returns:
            Numpy array
            
        Time Complexity: O(1) for CPU tensors, O(N) for GPU tensors
        """
        if hasattr(tensor, 'cpu'):
            return tensor.cpu().numpy()
        elif isinstance(tensor, np.ndarray):
            return tensor
        else:
            return np.array(tensor)
    
    def _log_overall_statistics(self, tp_np: np.ndarray, conf_np: np.ndarray, 
                              pred_cls_np: np.ndarray, target_cls_np: np.ndarray) -> None:
        """
        Log overall statistics summary.
        
        Args:
            tp_np: True positives array
            conf_np: Confidence scores array
            pred_cls_np: Predicted classes array
            target_cls_np: Target classes array
            
        Time Complexity: O(N) for aggregation operations
        """
        total_predictions = len(pred_cls_np)
        total_targets = len(target_cls_np)
        total_tp = tp_np.sum()
        total_fp = total_predictions - total_tp
        
        self.write_debug_log("📊 OVERALL STATISTICS:")
        self.write_debug_log(f"  • Total predictions: {total_predictions:,}")
        self.write_debug_log(f"  • Total targets: {total_targets:,}")
        self.write_debug_log(f"  • Total True Positives (TP): {total_tp:,}")
        self.write_debug_log(f"  • Total False Positives (FP): {total_fp:,}")
        self.write_debug_log(f"  • Overall Precision: {total_tp / max(total_predictions, 1):.4f}")
        self.write_debug_log(f"  • Confidence range: {conf_np.min():.6f} - {conf_np.max():.6f}")
    
    def _log_class_distribution(self, pred_cls_np: np.ndarray, target_cls_np: np.ndarray) -> None:
        """
        Log class distribution analysis.
        
        Args:
            pred_cls_np: Predicted classes array
            target_cls_np: Target classes array
            
        Time Complexity: O(N) for unique class computation
        """
        unique_pred_classes = np.unique(pred_cls_np)
        unique_target_classes = np.unique(target_cls_np)
        
        self.write_debug_log("\n📋 CLASS DISTRIBUTION:")
        self.write_debug_log(f"  • Predicted classes: {unique_pred_classes}")
        self.write_debug_log(f"  • Target classes: {unique_target_classes}")
        self.write_debug_log(f"  • Classes in both pred & target: {np.intersect1d(unique_pred_classes, unique_target_classes)}")
        self.write_debug_log(f"  • Classes only in predictions: {np.setdiff1d(unique_pred_classes, unique_target_classes)}")
        self.write_debug_log(f"  • Classes only in targets: {np.setdiff1d(unique_target_classes, unique_pred_classes)}")
    
    def _log_per_class_analysis(self, tp_np: np.ndarray, conf_np: np.ndarray,
                               pred_cls_np: np.ndarray, target_cls_np: np.ndarray) -> None:
        """
        Log detailed per-class analysis.
        
        Args:
            tp_np: True positives array
            conf_np: Confidence scores array  
            pred_cls_np: Predicted classes array
            target_cls_np: Target classes array
            
        Time Complexity: O(C*N) where C is unique classes, N is predictions
        """
        unique_pred_classes = np.unique(pred_cls_np)
        unique_target_classes = np.unique(target_cls_np)
        all_classes = np.union1d(unique_pred_classes, unique_target_classes)
        
        self.write_debug_log("\n🎯 PER-CLASS DETAILED ANALYSIS:")
        
        for cls in sorted(all_classes):
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
            
            self.write_debug_log(f"\n  CLASS {int(cls)} ANALYSIS:")
            self.write_debug_log(f"    📊 Counts: {cls_predictions:,} predictions, {cls_targets:,} targets")
            self.write_debug_log(f"    ✅ True Positives: {cls_tp:,}")
            self.write_debug_log(f"    ❌ False Positives: {cls_fp:,}")
            self.write_debug_log(f"    📈 Metrics: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
            self.write_debug_log(f"    🎯 Confidence: avg={avg_conf:.4f}, min={min_conf:.4f}, max={max_conf:.4f}")
            
            # Issue detection
            self._log_class_issues(cls, cls_predictions, cls_targets, cls_tp)
    
    def _log_class_issues(self, cls: int, cls_predictions: int, cls_targets: int, cls_tp: int) -> None:
        """
        Log class-specific issues for debugging.
        
        Args:
            cls: Class number
            cls_predictions: Number of predictions for class
            cls_targets: Number of targets for class
            cls_tp: Number of true positives for class
            
        Time Complexity: O(1) - simple issue detection
        """
        if cls_targets == 0:
            self.write_debug_log(f"    ⚠️  ISSUE: NO GROUND TRUTH for class {int(cls)} - all {cls_predictions} predictions are FP!")
        if cls_predictions == 0 and cls_targets > 0:
            self.write_debug_log(f"    ⚠️  ISSUE: NO PREDICTIONS for class {int(cls)} - missing all {cls_targets} targets!")
        if cls_tp == 0 and cls_predictions > 0 and cls_targets > 0:
            self.write_debug_log(f"    ⚠️  ISSUE: ZERO TRUE POSITIVES - predictions don't match targets (IoU/class mismatch)")
    
    def _log_confidence_threshold_analysis(self, tp_np: np.ndarray, conf_np: np.ndarray) -> None:
        """
        Log confidence threshold analysis across multiple thresholds.
        
        Args:
            tp_np: True positives array
            conf_np: Confidence scores array
            
        Time Complexity: O(T*N) where T is thresholds, N is predictions
        """
        self.write_debug_log("\n🎚️  CONFIDENCE THRESHOLD ANALYSIS:")
        self.write_debug_log("    Threshold | Predictions | True Positives | Precision")
        self.write_debug_log("    ---------|-------------|----------------|----------")
        
        for thresh in self.confidence_thresholds:
            high_conf_mask = conf_np >= thresh
            high_conf_tp = tp_np[high_conf_mask].sum()
            high_conf_total = np.sum(high_conf_mask)
            high_conf_precision = high_conf_tp / max(high_conf_total, 1)
            
            self.write_debug_log(f"    {thresh:8.1f} | {high_conf_total:11,} | {high_conf_tp:14,} | {high_conf_precision:8.4f}")
    
    def _log_analysis_insights(self, tp_np: np.ndarray, conf_np: np.ndarray,
                              pred_cls_np: np.ndarray, target_cls_np: np.ndarray) -> None:
        """
        Log analysis insights and recommendations.
        
        Args:
            tp_np: True positives array
            conf_np: Confidence scores array
            pred_cls_np: Predicted classes array
            target_cls_np: Target classes array
            
        Time Complexity: O(1) - simple analysis checks
        """
        total_predictions = len(pred_cls_np)
        total_targets = len(target_cls_np)
        total_tp = tp_np.sum()
        
        # Critical issue detection
        if total_tp == 0:
            self.write_debug_log("\n❌ CRITICAL ISSUE: ZERO TRUE POSITIVES DETECTED!")
            self.write_debug_log("   This means NO predictions matched ANY ground truth targets.")
            self.write_debug_log("   Root causes to investigate:")
            self.write_debug_log("   1. 🎯 IoU threshold too high (try lowering from current value)")
            self.write_debug_log("   2. 🏷️  Class mismatch (predicted classes ≠ target classes)")
            self.write_debug_log("   3. 📦 Coordinate format issues (bbox format mismatch)")
            self.write_debug_log("   4. 📏 Scale issues (predictions in wrong coordinate space)")
            self.write_debug_log("   5. 🗂️  Empty targets or corrupted prediction data")
            self.write_debug_log("   6. 🎲 Model predicting random/incorrect bboxes")
            self.write_debug_log("   7. 🔧 Data preprocessing issues (wrong normalization)")
        
        # General insights
        self.write_debug_log(f"\n📋 SUMMARY INSIGHTS:")
        if total_predictions == 0:
            self.write_debug_log("   • Model is NOT generating any predictions - check model inference")
        elif total_targets == 0:
            self.write_debug_log("   • NO ground truth targets found - check dataset loading")
        elif total_tp == 0:
            self.write_debug_log("   • Predictions exist but don't match targets - check IoU/class alignment")
        else:
            overall_precision = total_tp / total_predictions
            unique_target_classes = np.unique(target_cls_np)
            avg_targets_per_class = total_targets / len(unique_target_classes) if len(unique_target_classes) > 0 else 0
            
            self.write_debug_log(f"   • Overall precision: {overall_precision:.4f}")
            self.write_debug_log(f"   • Average targets per class: {avg_targets_per_class:.1f}")
            
            if overall_precision < 0.1:
                self.write_debug_log("   • Very low precision - likely IoU or class mismatch issues")
            elif overall_precision < 0.5:
                self.write_debug_log("   • Low precision - model needs more training or threshold tuning")
    
    def log_phase_configuration(self, model, phase_num: int, detailed: bool = True):
        """
        Log detailed Phase configuration analysis for debugging.
        
        Args:
            model: Training model instance
            phase_num: Current training phase (1 or 2)
            detailed: Whether to include detailed architecture analysis
        """
        try:
            self.write_debug_log(f"\n" + "="*80)
            self.write_debug_log(f"🔍 PHASE {phase_num} CONFIGURATION ANALYSIS")
            self.write_debug_log(f"="*80)
            
            # Basic phase information
            actual_phase = getattr(model, 'current_phase', None)
            self.write_debug_log(f"📊 PHASE STATUS:")
            self.write_debug_log(f"   • Expected phase: {phase_num}")
            self.write_debug_log(f"   • Model current_phase: {actual_phase}")
            self.write_debug_log(f"   • Phase match: {'✅' if actual_phase == phase_num else '❌'}")
            
            if phase_num == 1:
                self.write_debug_log(f"\n🎯 PHASE 1 SPECIFIC CHECKS:")
                self.write_debug_log(f"   Expected: Single-layer detection, frozen backbone, 7 classes")
                
                # Check backbone freezing
                frozen_params = 0
                trainable_params = 0
                backbone_layers = []
                
                for name, param in model.named_parameters():
                    if any(keyword in name.lower() for keyword in ['backbone', 'model.0', 'model.1', 'model.2', 'model.3', 'model.4']):
                        backbone_layers.append(name)
                        if param.requires_grad:
                            trainable_params += 1
                        else:
                            frozen_params += 1
                
                self.write_debug_log(f"\n❄️ BACKBONE FREEZING STATUS:")
                self.write_debug_log(f"   • Frozen parameters: {frozen_params}")
                self.write_debug_log(f"   • Trainable parameters: {trainable_params}")
                self.write_debug_log(f"   • Backbone properly frozen: {'✅' if trainable_params == 0 else '❌'}")
                
                if trainable_params > 0:
                    self.write_debug_log(f"   ⚠️ WARNING: Phase 1 should have ALL backbone parameters frozen!")
                    if detailed and len(backbone_layers) < 10:  # Only show if reasonable number
                        trainable_layers = [name for name, param in model.named_parameters() 
                                          if param.requires_grad and any(keyword in name.lower() 
                                          for keyword in ['backbone', 'model.0', 'model.1', 'model.2', 'model.3', 'model.4'])]
                        self.write_debug_log(f"   Trainable backbone layers: {trainable_layers[:5]}{'...' if len(trainable_layers) > 5 else ''}")
                
                # Check detection head configuration
                if hasattr(model, 'yolov5_model') and hasattr(model.yolov5_model, 'model'):
                    self.write_debug_log(f"\n🎯 DETECTION HEAD ANALYSIS:")
                    yolo_model = model.yolov5_model.model
                    
                    if hasattr(yolo_model, 'model') and len(yolo_model.model) > 0:
                        detection_head = yolo_model.model[-1]
                        self.write_debug_log(f"   • Detection head type: {type(detection_head).__name__}")
                        
                        # Check multi-layer head
                        if hasattr(detection_head, 'multi_layer_head'):
                            ml_head = detection_head.multi_layer_head
                            if ml_head:
                                ml_phase = getattr(ml_head, 'current_phase', None)
                                active_layers = getattr(ml_head, 'active_layers', None)
                                layer_mode = getattr(ml_head, 'layer_mode', None)
                                
                                self.write_debug_log(f"   • Multi-layer head present: ✅")
                                self.write_debug_log(f"   • ML head phase: {ml_phase}")
                                self.write_debug_log(f"   • Active layers: {active_layers}")
                                self.write_debug_log(f"   • Layer mode: {layer_mode}")
                                
                                # Phase 1 validation checks
                                if phase_num == 1:
                                    if active_layers != ['layer_1']:
                                        self.write_debug_log(f"   ❌ ISSUE: Phase 1 should have active_layers=['layer_1'], got {active_layers}")
                                    else:
                                        self.write_debug_log(f"   ✅ Active layers correct for Phase 1")
                                    
                                    if layer_mode != 'single':
                                        self.write_debug_log(f"   ❌ ISSUE: Phase 1 should have layer_mode='single', got {layer_mode}")
                                    else:
                                        self.write_debug_log(f"   ✅ Layer mode correct for Phase 1")
                            else:
                                self.write_debug_log(f"   ❌ Multi-layer head is None!")
                        
                        # Check detection head outputs
                        if hasattr(detection_head, 'm') and detection_head.m:
                            num_heads = len(detection_head.m)
                            self.write_debug_log(f"\n🔢 DETECTION HEAD OUTPUTS:")
                            self.write_debug_log(f"   • Number of detection heads: {num_heads}")
                            
                            for i, head in enumerate(detection_head.m):
                                if hasattr(head, 'weight'):
                                    output_size = head.weight.shape[0]
                                    input_size = head.weight.shape[1]
                                    self.write_debug_log(f"   • Head {i}: output_size={output_size}, input_size={input_size}")
                                    
                                    # Expected sizes for Phase 1 (7 classes)
                                    # Standard YOLO: 3 anchors * (5 + 7 classes) = 36 per scale
                                    expected_sizes = [36, 42, 84]  # Common configurations
                                    if output_size not in expected_sizes:
                                        self.write_debug_log(f"     ⚠️ Unusual output size for Phase 1 (expected one of {expected_sizes})")
                
                # Check loss configuration
                self.write_debug_log(f"\n💔 LOSS CONFIGURATION:")
                if hasattr(model, 'loss_coordinator'):
                    loss_coord = model.loss_coordinator
                    loss_phase = getattr(loss_coord, 'current_phase', None)
                    self.write_debug_log(f"   • Loss coordinator phase: {loss_phase}")
                    
                    if hasattr(loss_coord, 'loss_config'):
                        loss_type = loss_coord.loss_config.get('type', 'unknown')
                        self.write_debug_log(f"   • Loss type: {loss_type}")
                        
                        if phase_num == 1 and loss_type != 'standard':
                            self.write_debug_log(f"   ⚠️ WARNING: Phase 1 should use 'standard' YOLO loss, got '{loss_type}'")
                
            elif phase_num == 2:
                self.write_debug_log(f"\n🎯 PHASE 2 SPECIFIC CHECKS:")
                self.write_debug_log(f"   Expected: Multi-layer detection, unfrozen backbone, hierarchical classes")
                
                # Check backbone unfreezing
                frozen_params = 0
                trainable_params = 0
                
                for name, param in model.named_parameters():
                    if any(keyword in name.lower() for keyword in ['backbone', 'model.0', 'model.1', 'model.2', 'model.3', 'model.4']):
                        if param.requires_grad:
                            trainable_params += 1
                        else:
                            frozen_params += 1
                
                self.write_debug_log(f"\n🔥 BACKBONE UNFREEZING STATUS:")
                self.write_debug_log(f"   • Frozen parameters: {frozen_params}")
                self.write_debug_log(f"   • Trainable parameters: {trainable_params}")
                self.write_debug_log(f"   • Backbone properly unfrozen: {'✅' if trainable_params > 0 else '❌'}")
                
                if trainable_params == 0:
                    self.write_debug_log(f"   ❌ WARNING: Phase 2 should have backbone parameters trainable!")
            
            self.write_debug_log(f"="*80)
            
        except Exception as e:
            self.write_debug_log(f"❌ Error in phase configuration analysis: {e}")
            logger.error(f"Phase configuration logging failed: {e}")
    
    def log_coordinate_format_analysis(self, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Log detailed coordinate format analysis for debugging bbox alignment issues.
        
        Args:
            predictions: Prediction tensor
            targets: Target tensor
        """
        try:
            self.write_debug_log(f"\n🔍 COORDINATE FORMAT ANALYSIS")
            self.write_debug_log(f"="*50)
            
            if predictions.numel() > 0 and targets.numel() > 0:
                # Analyze prediction coordinate ranges
                pred_coords = predictions[:, :, :4]  # x, y, w, h
                pred_ranges = {
                    'x': (pred_coords[:, :, 0].min().item(), pred_coords[:, :, 0].max().item()),
                    'y': (pred_coords[:, :, 1].min().item(), pred_coords[:, :, 1].max().item()),
                    'w': (pred_coords[:, :, 2].min().item(), pred_coords[:, :, 2].max().item()),
                    'h': (pred_coords[:, :, 3].min().item(), pred_coords[:, :, 3].max().item())
                }
                
                # Analyze target coordinate ranges
                target_coords = targets[:, 2:6]  # x, y, w, h (skip batch_idx, class)
                target_ranges = {
                    'x': (target_coords[:, 0].min().item(), target_coords[:, 0].max().item()),
                    'y': (target_coords[:, 1].min().item(), target_coords[:, 1].max().item()),
                    'w': (target_coords[:, 2].min().item(), target_coords[:, 2].max().item()),
                    'h': (target_coords[:, 3].min().item(), target_coords[:, 3].max().item())
                }
                
                self.write_debug_log(f"📊 COORDINATE RANGES:")
                for coord in ['x', 'y', 'w', 'h']:
                    pred_min, pred_max = pred_ranges[coord]
                    tgt_min, tgt_max = target_ranges[coord]
                    self.write_debug_log(f"   {coord.upper()}: Pred[{pred_min:.4f}, {pred_max:.4f}] vs Target[{tgt_min:.4f}, {tgt_max:.4f}]")
                
                # Check normalization
                pred_normalized = all(v[1] <= 1.0 for v in pred_ranges.values())
                target_normalized = all(v[1] <= 1.0 for v in target_ranges.values())
                
                self.write_debug_log(f"\n🔍 NORMALIZATION STATUS:")
                self.write_debug_log(f"   • Predictions normalized (0-1): {pred_normalized}")
                self.write_debug_log(f"   • Targets normalized (0-1): {target_normalized}")
                
                if pred_normalized != target_normalized:
                    self.write_debug_log(f"   ❌ COORDINATE MISMATCH DETECTED!")
                    self.write_debug_log(f"      This is likely the cause of zero IoU/TP!")
                    self.write_debug_log(f"      Predictions: {'normalized' if pred_normalized else 'pixel coordinates'}")
                    self.write_debug_log(f"      Targets: {'normalized' if target_normalized else 'pixel coordinates'}")
                else:
                    self.write_debug_log(f"   ✅ Coordinate systems match")
                
                # Sample coordinate analysis
                self.write_debug_log(f"\n🔍 SAMPLE COORDINATES:")
                sample_size = min(3, predictions.shape[1], targets.shape[0])
                for i in range(sample_size):
                    if i < predictions.shape[1]:
                        pred_box = predictions[0, i, :4]
                        self.write_debug_log(f"   Pred {i}: [{pred_box[0]:.4f}, {pred_box[1]:.4f}, {pred_box[2]:.4f}, {pred_box[3]:.4f}]")
                    
                    if i < targets.shape[0]:
                        tgt_box = targets[i, 2:6]
                        self.write_debug_log(f"   Target {i}: [{tgt_box[0]:.4f}, {tgt_box[1]:.4f}, {tgt_box[2]:.4f}, {tgt_box[3]:.4f}]")
                
            else:
                self.write_debug_log(f"❌ Cannot analyze coordinates: empty tensors")
                
        except Exception as e:
            self.write_debug_log(f"❌ Error in coordinate analysis: {e}")
    
    def log_iou_distribution_analysis(self, iou_matrix: torch.Tensor, predictions: torch.Tensor, targets: torch.Tensor):
        """
        Log detailed IoU distribution analysis for debugging matching issues.
        
        Args:
            iou_matrix: IoU matrix between predictions and targets
            predictions: Prediction tensor  
            targets: Target tensor
        """
        try:
            self.write_debug_log(f"\n🔍 IoU DISTRIBUTION ANALYSIS")
            self.write_debug_log(f"="*50)
            
            if iou_matrix is not None and iou_matrix.numel() > 0:
                # Overall IoU statistics
                max_ious, best_target_idx = torch.max(iou_matrix, dim=1)
                valid_ious = max_ious[max_ious > 0]
                
                self.write_debug_log(f"📊 IoU STATISTICS:")
                self.write_debug_log(f"   • Matrix shape: {iou_matrix.shape}")
                self.write_debug_log(f"   • Max IoU achieved: {max_ious.max().item():.4f}")
                self.write_debug_log(f"   • Mean IoU (all): {max_ious.mean().item():.4f}")
                
                if len(valid_ious) > 0:
                    self.write_debug_log(f"   • Mean IoU (non-zero): {valid_ious.mean().item():.4f}")
                    self.write_debug_log(f"   • Non-zero IoUs: {len(valid_ious)}/{len(max_ious)}")
                else:
                    self.write_debug_log(f"   ❌ ALL IoUs are ZERO - complete bbox misalignment!")
                
                # IoU threshold analysis
                self.write_debug_log(f"\n🎚️ IoU THRESHOLD ANALYSIS:")
                thresholds = [0.1, 0.25, 0.5, 0.75]
                for thresh in thresholds:
                    count = (max_ious > thresh).sum().item()
                    percentage = (count / len(max_ious)) * 100
                    self.write_debug_log(f"   • IoU > {thresh}: {count}/{len(max_ious)} ({percentage:.1f}%)")
                
                # Show best matches
                if len(valid_ious) > 0:
                    top_k = min(5, len(max_ious))
                    top_ious, top_indices = torch.topk(max_ious, top_k)
                    self.write_debug_log(f"\n🏆 TOP {top_k} IoU MATCHES:")
                    for i, (iou_val, pred_idx) in enumerate(zip(top_ious, top_indices)):
                        target_idx = best_target_idx[pred_idx].item()
                        self.write_debug_log(f"   • Match {i+1}: IoU={iou_val:.4f}, pred_idx={pred_idx.item()}, target_idx={target_idx}")
                        
                        # Show coordinate details for top matches
                        if iou_val > 0 and pred_idx < predictions.shape[1] and target_idx < targets.shape[0]:
                            pred_box = predictions[0, pred_idx, :4]
                            target_box = targets[target_idx, 2:6]
                            self.write_debug_log(f"     Pred box:   [{pred_box[0]:.4f}, {pred_box[1]:.4f}, {pred_box[2]:.4f}, {pred_box[3]:.4f}]")
                            self.write_debug_log(f"     Target box: [{target_box[0]:.4f}, {target_box[1]:.4f}, {target_box[2]:.4f}, {target_box[3]:.4f}]")
                
            else:
                self.write_debug_log(f"❌ Cannot analyze IoU: invalid matrix")
                
        except Exception as e:
            self.write_debug_log(f"❌ Error in IoU analysis: {e}")
    


# Factory function for backward compatibility
def create_map_debug_logger(training_context: Optional[Dict] = None) -> MapDebugLogger:
    """
    Factory function to create mAP debug logger.
    
    Args:
        training_context: Training context information (backbone, phase, etc.)
        
    Returns:
        MapDebugLogger instance
        
    Time Complexity: O(1) - simple object creation
    """
    return MapDebugLogger(training_context)


# Export public interface
__all__ = [
    'MapDebugLogger',
    'create_map_debug_logger'
]