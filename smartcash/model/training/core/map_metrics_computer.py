#!/usr/bin/env python3
"""
Metrics computation module for mAP calculations.

This module handles all metrics computation algorithms including fast approximation,
full mAP computation, and performance optimizations. Extracted from YOLOv5MapCalculator
for better separation of concerns and algorithm-focused optimization.

Key Features:
- Fast approximation for minimal datasets
- Full mAP computation using YOLOv5 ap_per_class
- Platform-aware computation strategies
- Memory optimization and threading for large datasets
- Comprehensive metrics calculation (mAP, precision, recall, F1)

Algorithmic Complexity:
- Fast approximation: O(1) - simple arithmetic operations
- Full mAP computation: O(N log N) for AP computation where N is total detections
- Memory optimization: O(chunk_size) for chunked processing
"""

import torch
import numpy as np
import concurrent.futures
from typing import Dict, List, Union

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer
from .ultralytics_utils_manager import get_ap_per_class

logger = get_logger(__name__, level="DEBUG")


class MapMetricsComputer:
    """
    Computes mAP and related metrics using optimized algorithms.
    
    This computer handles all metrics computation including fast approximation for
    small datasets and full mAP computation for larger datasets. Includes platform-aware
    optimizations and memory management.
    
    Features:
    - Fast approximation using F1 score for minimal datasets
    - Full mAP computation using YOLOv5 ap_per_class function
    - Platform-aware threading optimization (avoids MPS issues)
    - Memory optimization for large datasets
    - Comprehensive metrics output (mAP@0.5, precision, recall, F1, accuracy)
    
    Time Complexity: O(N log N) for full mAP, O(1) for fast approximation
    Space Complexity: O(N) for intermediate arrays during computation
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize metrics computer.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        self.memory_optimizer = get_memory_optimizer()
        
        # Thresholds for computation strategy selection
        self.fast_approximation_threshold = 10  # Use fast approximation for datasets smaller than this
        self.max_unique_classes_for_fast = 1    # Max unique classes for fast approximation
        self.threading_threshold = 1000         # Use threading for datasets larger than this
    
    def compute_metrics(self, stats: List[Union[torch.Tensor, np.ndarray]], 
                       data_size: int) -> Dict[str, float]:
        """
        Compute comprehensive mAP metrics from statistics.
        
        Automatically selects between fast approximation and full computation
        based on dataset size and characteristics.
        
        Args:
            stats: Statistics arrays [tp, conf, pred_cls, target_cls]
            data_size: Number of data points
            
        Returns:
            Dictionary containing mAP metrics:
            - 'map50': mAP@0.5
            - 'map50_95': mAP@0.5:0.95 (set to 0 for performance)
            - 'precision': Mean precision
            - 'recall': Mean recall
            - 'f1': Mean F1 score
            - 'accuracy': Mean accuracy (uses recall for detection tasks)
            
        Time Complexity: O(N log N) for full computation, O(1) for fast approximation
        Space Complexity: O(N) for intermediate arrays
        """
        if len(stats) < 4:
            logger.error(f"🚨 Insufficient statistics! Expected 4 arrays, got {len(stats)}")
            return self._create_zero_metrics()
        
        if data_size == 0:
            logger.warning("📊 No detection samples accumulated - check if predictions are being generated")
            if self.debug:
                logger.debug("❌ CRITICAL: NO DETECTION SAMPLES ACCUMULATED")
            return self._create_zero_metrics()
        
        logger.info(f"📈 Processing {data_size} detection samples for mAP computation...")
        
        if data_size > 10000:
            logger.info("⏳ Large dataset detected - mAP computation may take 30-60 seconds")
        
        # Select computation strategy based on dataset characteristics
        if self._should_use_fast_approximation(stats, data_size):
            return self._compute_fast_approximation(stats)
        else:
            return self._compute_full_map(stats, data_size)
    
    def _should_use_fast_approximation(self, stats: List[Union[torch.Tensor, np.ndarray]], 
                                     data_size: int) -> bool:
        """
        Determine if fast approximation should be used.
        
        Uses fast approximation for very small datasets with minimal class diversity
        to avoid computational overhead.
        
        Args:
            stats: Statistics arrays
            data_size: Number of data points
            
        Returns:
            bool: True if fast approximation is recommended
            
        Time Complexity: O(1) - simple threshold checks
        """
        if data_size >= self.fast_approximation_threshold:
            return False
        
        pred_cls = stats[2]
        pred_cls_np = self._to_numpy(pred_cls)
        unique_classes = np.unique(pred_cls_np)
        
        # Use fast approximation for very small datasets with minimal predictions
        return len(unique_classes) <= self.max_unique_classes_for_fast
    
    def _compute_fast_approximation(self, stats: List[Union[torch.Tensor, np.ndarray]]) -> Dict[str, float]:
        """
        Compute fast approximation for minimal datasets.
        
        Uses simple precision, recall, and F1 calculations as approximations
        for mAP metrics when dataset is too small for meaningful AP computation.
        
        Args:
            stats: Statistics arrays [tp, conf, pred_cls, target_cls]
            
        Returns:
            Dictionary with approximated metrics
            
        Time Complexity: O(1) - simple arithmetic operations
        Space Complexity: O(1) - minimal memory usage
        """
        tp, _, pred_cls, target_cls = stats[:4]
        
        logger.info("⚡ Very small dataset - using fast approximation")
        
        # Convert to numpy for consistent operations
        tp_np = self._to_numpy(tp)
        pred_cls_np = self._to_numpy(pred_cls)
        target_cls_np = self._to_numpy(target_cls)
        
        # Calculate basic metrics
        total_tp = tp_np.sum()
        total_predictions = len(pred_cls_np)
        total_targets = len(target_cls_np)
        
        if total_tp > 0:
            approx_precision = total_tp / max(total_predictions, 1)
            approx_recall = total_tp / max(total_targets, 1)
            approx_f1 = (2 * (approx_precision * approx_recall) / 
                        max(approx_precision + approx_recall, 1e-8))
        else:
            approx_precision = approx_recall = approx_f1 = 0.0
        
        return {
            'map50': float(approx_f1),  # Use F1 as mAP approximation
            'map50_95': 0.0,
            'precision': float(approx_precision),
            'recall': float(approx_recall),
            'f1': float(approx_f1),
            'accuracy': float(approx_recall)  # Use recall as accuracy approximation
        }
    
    def _compute_full_map(self, stats: List[Union[torch.Tensor, np.ndarray]], 
                         data_size: int) -> Dict[str, float]:
        """
        Compute full mAP using YOLOv5 ap_per_class function.
        
        Performs complete Average Precision computation using the standard
        YOLOv5 evaluation methodology with platform-aware optimizations.
        
        Args:
            stats: Statistics arrays [tp, conf, pred_cls, target_cls]
            data_size: Number of data points
            
        Returns:
            Dictionary with complete mAP metrics
            
        Time Complexity: O(N log N) for AP computation
        Space Complexity: O(N) for intermediate arrays
        """
        if self.debug:
            logger.debug(f"⚡ Starting ap_per_class computation with {len(stats[2])} predictions and {len(stats[3])} targets")
        
        try:
            # Platform-aware computation strategy
            if self._should_use_threading(data_size):
                ap_results = self._compute_with_threading(stats)
            else:
                ap_results = self._compute_direct(stats)
            
            # Extract results
            _, _, p, r, f1, ap, ap_class = ap_results
            
            if self.debug:
                logger.debug("✅ ap_per_class computation completed")
                logger.debug(f"📊 AP RESULTS:")
                logger.debug(f"  • AP shape: {ap.shape}")
                logger.debug(f"  • AP classes: {ap_class}")
                logger.debug(f"  • AP values: {ap[:, 0] if ap.shape[1] > 0 else 'no AP data'}")
                logger.debug(f"  • Precision: {p}")
                logger.debug(f"  • Recall: {r}")
            
            # Clean up memory after expensive computation
            self.memory_optimizer.cleanup_memory()
            
            # Extract mAP@0.5 and compute mean metrics
            ap50 = ap[:, 0] if ap.shape[1] > 0 else np.array([0.0])
            
            map50 = ap50.mean()
            precision = p.mean()
            recall = r.mean()
            f1_score = f1.mean()
            
            if self.debug:
                logger.debug(f"Final mAP: {map50:.4f}, precision: {precision:.4f}, recall: {recall:.4f}")
            
            return {
                'map50': float(map50),
                'map50_95': 0.0,  # Not computed for performance
                'precision': float(precision),
                'recall': float(recall),
                'f1': float(f1_score),
                'accuracy': float(recall)  # Use recall as accuracy for detection tasks
            }
            
        except Exception as e:
            logger.error(f"Error in full mAP computation: {e}")
            # Emergency cleanup on error
            self.memory_optimizer.emergency_memory_cleanup()
            raise RuntimeError(f"Full mAP computation failed: {e}") from e
    
    def _should_use_threading(self, data_size: int) -> bool:
        """
        Determine if threading should be used for computation.
        
        Uses threading for large datasets but avoids it on Apple Silicon (MPS)
        due to threading issues.
        
        Args:
            data_size: Number of data points
            
        Returns:
            bool: True if threading should be used
            
        Time Complexity: O(1) - simple platform check
        """
        # Check if dataset is large enough to benefit from threading
        if data_size < self.threading_threshold:
            return False
        
        # Avoid threading on Apple Silicon (MPS) due to known issues
        platform_info = self.memory_optimizer.platform_info
        is_apple_silicon = platform_info.get('is_apple_silicon', False)
        
        return not is_apple_silicon
    
    def _compute_with_threading(self, stats: List[Union[torch.Tensor, np.ndarray]]) -> tuple:
        """
        Compute mAP with threading optimization.
        
        Args:
            stats: Statistics arrays
            
        Returns:
            Results from ap_per_class computation
            
        Time Complexity: O(N log N) with parallel optimization
        """
        logger.info("📊 Large dataset - using optimized threading")
        
        def run_ap_computation():
            return get_ap_per_class()(
                *stats, 
                plot=False, 
                save_dir="", 
                names={}
            )
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(run_ap_computation)
            return future.result()
    
    def _compute_direct(self, stats: List[Union[torch.Tensor, np.ndarray]]) -> tuple:
        """
        Compute mAP directly without threading.
        
        Args:
            stats: Statistics arrays
            
        Returns:
            Results from ap_per_class computation
            
        Time Complexity: O(N log N) for AP computation
        """
        return get_ap_per_class()(
            *stats, 
            plot=False, 
            save_dir="", 
            names={}
        )
    
    def _create_zero_metrics(self) -> Dict[str, float]:
        """
        Create zero metrics dictionary for error cases.
        
        Returns:
            Dictionary with all metrics set to 0.0
            
        Time Complexity: O(1) - simple dictionary creation
        """
        return {
            'map50': 0.0,
            'map50_95': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
    
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


class MetricsAnalyzer:
    """
    Analyzer for metrics computation results and performance.
    
    Provides analysis and insights into metrics computation performance
    and result quality.
    """
    
    @staticmethod
    def analyze_metrics_quality(metrics: Dict[str, float], data_size: int) -> Dict[str, str]:
        """
        Analyze metrics quality and provide insights.
        
        Args:
            metrics: Computed metrics dictionary
            data_size: Size of dataset used for computation
            
        Returns:
            Dictionary with quality analysis and recommendations
            
        Time Complexity: O(1) - simple analysis
        """
        analysis = {}
        
        # Overall quality assessment
        map50 = metrics.get('map50', 0.0)
        precision = metrics.get('precision', 0.0)
        recall = metrics.get('recall', 0.0)
        
        if map50 == 0.0:
            analysis['map_quality'] = 'CRITICAL: Zero mAP - no valid detections'
        elif map50 < 0.3:
            analysis['map_quality'] = 'LOW: mAP < 0.3 - model needs significant improvement'
        elif map50 < 0.5:
            analysis['map_quality'] = 'MODERATE: mAP < 0.5 - model shows promise but needs tuning'
        elif map50 < 0.7:
            analysis['map_quality'] = 'GOOD: mAP >= 0.5 - solid performance'
        else:
            analysis['map_quality'] = 'EXCELLENT: mAP >= 0.7 - high-quality model'
        
        # Precision vs recall balance
        if precision > 0 and recall > 0:
            pr_ratio = precision / recall
            if pr_ratio > 2.0:
                analysis['precision_recall_balance'] = 'HIGH PRECISION, LOW RECALL: Model is conservative'
            elif pr_ratio < 0.5:
                analysis['precision_recall_balance'] = 'LOW PRECISION, HIGH RECALL: Model is aggressive'
            else:
                analysis['precision_recall_balance'] = 'BALANCED: Good precision-recall trade-off'
        else:
            analysis['precision_recall_balance'] = 'INSUFFICIENT DATA: Cannot assess balance'
        
        # Dataset size assessment
        if data_size < 100:
            analysis['dataset_size'] = 'SMALL: Results may not be representative'
        elif data_size < 1000:
            analysis['dataset_size'] = 'MODERATE: Reasonable sample size'
        else:
            analysis['dataset_size'] = 'LARGE: Statistically significant results'
        
        return analysis
    
    @staticmethod
    def compare_metrics(metrics1: Dict[str, float], metrics2: Dict[str, float], 
                       labels: tuple = ('Model A', 'Model B')) -> Dict[str, str]:
        """
        Compare two sets of metrics and provide insights.
        
        Args:
            metrics1: First metrics dictionary
            metrics2: Second metrics dictionary
            labels: Labels for the two models
            
        Returns:
            Dictionary with comparison analysis
            
        Time Complexity: O(1) - simple comparison operations
        """
        comparison = {}
        
        label1, label2 = labels
        
        # mAP comparison
        map1 = metrics1.get('map50', 0.0)
        map2 = metrics2.get('map50', 0.0)
        map_diff = map2 - map1
        
        if abs(map_diff) < 0.01:
            comparison['map_comparison'] = f'SIMILAR: mAP difference < 0.01'
        elif map_diff > 0:
            comparison['map_comparison'] = f'{label2} BETTER: +{map_diff:.3f} mAP improvement'
        else:
            comparison['map_comparison'] = f'{label1} BETTER: {abs(map_diff):.3f} mAP advantage'
        
        # Precision comparison
        prec1 = metrics1.get('precision', 0.0)
        prec2 = metrics2.get('precision', 0.0)
        prec_diff = prec2 - prec1
        
        if abs(prec_diff) < 0.01:
            comparison['precision_comparison'] = f'SIMILAR: Precision difference < 0.01'
        elif prec_diff > 0:
            comparison['precision_comparison'] = f'{label2} MORE PRECISE: +{prec_diff:.3f} improvement'
        else:
            comparison['precision_comparison'] = f'{label1} MORE PRECISE: {abs(prec_diff):.3f} advantage'
        
        return comparison


# Factory functions for backward compatibility
def create_metrics_computer(debug: bool = False) -> MapMetricsComputer:
    """
    Factory function to create metrics computer.
    
    Args:
        debug: Enable debug logging
        
    Returns:
        MapMetricsComputer instance
        
    Time Complexity: O(1) - simple object creation
    """
    return MapMetricsComputer(debug)


def compute_quick_metrics(stats: List[Union[torch.Tensor, np.ndarray]]) -> Dict[str, float]:
    """
    Quick metrics computation function for simple use cases.
    
    Args:
        stats: Statistics arrays [tp, conf, pred_cls, target_cls]
        
    Returns:
        Dictionary with computed metrics
        
    Time Complexity: Depends on dataset size - O(1) for small, O(N log N) for large
    """
    computer = MapMetricsComputer(debug=False)
    data_size = len(stats[2]) if len(stats) > 2 else 0
    return computer.compute_metrics(stats, data_size)


# Export public interface
__all__ = [
    'MapMetricsComputer',
    'MetricsAnalyzer',
    'create_metrics_computer',
    'compute_quick_metrics'
]