#!/usr/bin/env python3
"""
Statistics processing module for mAP calculations.

This module handles all statistics-related operations including validation, concatenation,
and processing of accumulated statistics from mAP calculations. Extracted from 
YOLOv5MapCalculator for better separation of concerns and optimization opportunities.

Key Features:
- Input validation with comprehensive error checking
- Efficient statistics concatenation with GPU optimization
- Statistics format validation and error detection
- Memory-efficient processing for large datasets
- Performance optimizations for single vs. multiple batches

Algorithmic Complexity:
- Input validation: O(1) - simple validation checks
- Statistics concatenation: O(S*N) where S is batches, N is average batch size
- Memory optimization: O(total_detections) space complexity for concatenated arrays
"""

import torch
import numpy as np
from typing import List, Optional, Tuple, Union

from smartcash.common.logger import get_logger

logger = get_logger(__name__, level="DEBUG")


class MapStatisticsProcessor:
    """
    Processes and validates statistics for mAP calculations.
    
    This processor handles all statistics-related operations including input validation,
    statistics concatenation, and format validation. Designed for efficiency with both
    single-batch and multi-batch scenarios.
    
    Features:
    - Comprehensive input validation with detailed error reporting
    - GPU-optimized statistics concatenation with single CPU transfer
    - Memory-efficient processing for large datasets
    - Format validation with recovery strategies
    - Performance optimizations for different batch sizes
    
    Time Complexity: O(S*N) for concatenation where S is batches, N is batch size
    Space Complexity: O(total_detections) for concatenated statistics
    """
    
    def __init__(self, debug: bool = False):
        """
        Initialize statistics processor.
        
        Args:
            debug: Enable debug logging
        """
        self.debug = debug
        
        # Validation thresholds
        self.min_tensor_features = 6  # Minimum features for prediction tensors
        self.min_target_features = 6  # Minimum features for target tensors
    
    def validate_inputs(self, predictions: torch.Tensor, targets: torch.Tensor) -> bool:
        """
        Comprehensive validation of input tensors.
        
        Validates prediction and target tensors for correct format, dimensions,
        and feature counts required for mAP calculation.
        
        Args:
            predictions: Predictions tensor
            targets: Targets tensor
            
        Returns:
            bool: True if inputs are valid, False otherwise
            
        Time Complexity: O(1) - simple validation checks
        Space Complexity: O(1) - no additional memory allocation
        """
        try:
            # Validate prediction tensor format
            if not self._validate_prediction_tensor(predictions):
                return False
                
            # Validate target tensor format
            if not self._validate_target_tensor(targets):
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error during input validation: {e}")
            return False
    
    def validate_statistics(self, stats: List[Tuple]) -> bool:
        """
        Validate accumulated statistics format and content.
        
        Ensures all statistics batches have correct format and contain valid tensor data
        for mAP computation.
        
        Args:
            stats: List of statistics tuples [(tp, conf, pred_cls, target_cls), ...]
            
        Returns:
            bool: True if statistics are valid, False otherwise
            
        Time Complexity: O(S) where S is number of statistics batches
        Space Complexity: O(1) - validation only, no data copying
        """
        try:
            if not stats:
                logger.error("🚨 No statistics to validate - empty statistics list")
                return False
                
            valid_count = 0
            for i, stat in enumerate(stats):
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
                    continue
                
                # Validate tensor shapes are consistent
                if not (len(tp) == len(conf) == len(pred_cls)):
                    logger.error(f"🚨 Statistics batch {i} has inconsistent tensor lengths")
                    continue
                    
                valid_count += 1
            
            if valid_count == 0:
                logger.error("🚨 No valid statistics batches found")
                return False
                
            logger.debug(f"✅ Statistics validation passed: {valid_count}/{len(stats)} batches valid")
            return True
            
        except Exception as e:
            logger.error(f"Error validating statistics format: {e}")
            return False
    
    def concatenate_statistics(self, stats: List[Tuple]) -> List[Union[torch.Tensor, np.ndarray]]:
        """
        Efficiently concatenate accumulated statistics.
        
        Optimizes concatenation based on number of batches, using GPU operations
        when possible and minimizing CPU transfers.
        
        Args:
            stats: List of statistics tuples [(tp, conf, pred_cls, target_cls), ...]
            
        Returns:
            List of concatenated statistics arrays [tp, conf, pred_cls, target_cls]
            
        Time Complexity: O(S*N) where S is batches, N is average batch size
        Space Complexity: O(total_detections) for concatenated arrays
        """
        if not stats:
            raise ValueError("Cannot concatenate empty statistics list")
        
        if self.debug:
            logger.debug("📊 Concatenating statistics for mAP computation...")
        
        if len(stats) == 1:
            # Single batch optimization - no concatenation needed
            return self._process_single_batch(stats[0])
        else:
            # Multiple batches - efficient GPU concatenation
            return self._process_multiple_batches(stats)
    
    def get_statistics_summary(self, stats: List[Union[torch.Tensor, np.ndarray]]) -> dict:
        """
        Get comprehensive summary of statistics for analysis.
        
        Args:
            stats: Concatenated statistics [tp, conf, pred_cls, target_cls]
            
        Returns:
            Dictionary with statistics summary
            
        Time Complexity: O(N) where N is total number of predictions
        """
        if len(stats) < 4:
            return {'error': 'Insufficient statistics components'}
        
        tp, conf, pred_cls, target_cls = stats[:4]
        
        # Convert to numpy for analysis
        tp_np = self._to_numpy(tp)
        conf_np = self._to_numpy(conf)
        pred_cls_np = self._to_numpy(pred_cls)
        target_cls_np = self._to_numpy(target_cls)
        
        return {
            'total_predictions': len(pred_cls_np),
            'total_targets': len(target_cls_np),
            'total_true_positives': int(tp_np.sum()),
            'total_false_positives': len(pred_cls_np) - int(tp_np.sum()),
            'confidence_range': {
                'min': float(conf_np.min()),
                'max': float(conf_np.max()),
                'mean': float(conf_np.mean())
            },
            'unique_predicted_classes': np.unique(pred_cls_np).tolist(),
            'unique_target_classes': np.unique(target_cls_np).tolist(),
            'overall_precision': float(tp_np.sum()) / max(len(pred_cls_np), 1)
        }
    
    def _validate_prediction_tensor(self, predictions: torch.Tensor) -> bool:
        """
        Validate prediction tensor format and dimensions.
        
        Args:
            predictions: Predictions tensor to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Time Complexity: O(1) - simple validation checks
        """
        # Check if tensor exists and has data
        if not isinstance(predictions, torch.Tensor) or predictions.numel() == 0:
            if self.debug:
                logger.debug("Empty or invalid predictions tensor")
            return False
            
        # Check tensor dimensions
        if predictions.dim() not in [2, 3]:
            logger.warning(f"Invalid prediction tensor dimensions: {predictions.dim()}, expected 2D or 3D")
            return False
            
        # Check feature count
        if predictions.shape[-1] < self.min_tensor_features:
            logger.error(f"🚨 Prediction tensor too small! Expected at least {self.min_tensor_features} columns, got {predictions.shape[-1]}")
            return False
        
        return True
    
    def _validate_target_tensor(self, targets: torch.Tensor) -> bool:
        """
        Validate target tensor format and dimensions.
        
        Args:
            targets: Targets tensor to validate
            
        Returns:
            bool: True if valid, False otherwise
            
        Time Complexity: O(1) - simple validation checks
        """
        # Check if tensor exists
        if not isinstance(targets, torch.Tensor):
            if self.debug:
                logger.debug("Invalid targets tensor")
            return False
            
        # Check target tensor format (can be empty)
        if targets.numel() > 0 and (targets.dim() != 2 or targets.shape[-1] < self.min_target_features):
            logger.warning(f"Invalid target tensor format: {targets.shape}, expected [N, {self.min_target_features}]")
            return False
        
        return True
    
    def _process_single_batch(self, stat: Tuple) -> List[np.ndarray]:
        """
        Process single batch statistics with optimization.
        
        Args:
            stat: Single statistics tuple (tp, conf, pred_cls, target_cls)
            
        Returns:
            List of numpy arrays
            
        Time Complexity: O(N) where N is batch size
        """
        return [self._to_numpy(x) for x in stat]
    
    def _process_multiple_batches(self, stats: List[Tuple]) -> List[np.ndarray]:
        """
        Process multiple batch statistics with GPU optimization.
        
        Args:
            stats: List of statistics tuples
            
        Returns:
            List of concatenated numpy arrays
            
        Time Complexity: O(S*N) where S is batches, N is average batch size
        """
        total_size = sum(stat[0].shape[0] for stat in stats)
        
        if self.debug:
            logger.debug(f"📊 Concatenating {len(stats)} batches, total size: {total_size}")
        
        # Keep concatenation on GPU for parallel processing
        gpu_stats = []
        try:
            for i in range(4):  # Process each statistics component
                component_list = [stat[i] for stat in stats]
                concatenated = torch.cat(component_list, 0)
                gpu_stats.append(concatenated)
        except Exception as e:
            logger.error(f"Error during GPU concatenation: {e}")
            # Fallback to CPU concatenation
            return self._fallback_cpu_concatenation(stats)
        
        # Single batch CPU transfer for all statistics
        result_stats = []
        for stat in gpu_stats:
            result_stats.append(self._to_numpy(stat))
        
        return result_stats
    
    def _fallback_cpu_concatenation(self, stats: List[Tuple]) -> List[np.ndarray]:
        """
        Fallback CPU concatenation when GPU concatenation fails.
        
        Args:
            stats: List of statistics tuples
            
        Returns:
            List of concatenated numpy arrays
            
        Time Complexity: O(S*N) where S is batches, N is average batch size
        """
        logger.warning("Using fallback CPU concatenation")
        
        result_stats = []
        for i in range(4):  # Process each statistics component
            component_list = [stat[i] for stat in stats]
            
            # Convert to numpy and concatenate on CPU
            numpy_components = [self._to_numpy(comp) for comp in component_list]
            concatenated = np.concatenate(numpy_components, axis=0)
            result_stats.append(concatenated)
        
        return result_stats
    
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


class StatisticsValidator:
    """
    Standalone validator for statistics format and content.
    
    Provides static methods for validating different aspects of statistics
    without requiring processor instantiation.
    """
    
    @staticmethod
    def validate_statistics_format(stats: List[Tuple]) -> Tuple[bool, str]:
        """
        Validate statistics format with detailed error reporting.
        
        Args:
            stats: List of statistics tuples
            
        Returns:
            Tuple of (is_valid, error_message)
            
        Time Complexity: O(S) where S is number of statistics batches
        """
        if not stats:
            return False, "Empty statistics list"
        
        for i, stat in enumerate(stats):
            if stat is None:
                return False, f"Statistics batch {i} is None"
            
            if not isinstance(stat, tuple) or len(stat) != 4:
                return False, f"Statistics batch {i} has invalid format: expected tuple of length 4"
            
            tp, conf, pred_cls, target_cls = stat
            if not all(isinstance(x, torch.Tensor) for x in [tp, conf, pred_cls, target_cls]):
                return False, f"Statistics batch {i} contains non-tensor elements"
            
            # Check tensor shape consistency
            if not (len(tp) == len(conf) == len(pred_cls)):
                return False, f"Statistics batch {i} has inconsistent tensor shapes"
        
        return True, "Statistics format validation passed"
    
    @staticmethod
    def validate_tensor_compatibility(tensor1: torch.Tensor, tensor2: torch.Tensor) -> bool:
        """
        Validate that two tensors are compatible for operations.
        
        Args:
            tensor1: First tensor
            tensor2: Second tensor
            
        Returns:
            bool: True if compatible, False otherwise
            
        Time Complexity: O(1) - simple shape comparison
        """
        if not isinstance(tensor1, torch.Tensor) or not isinstance(tensor2, torch.Tensor):
            return False
        
        # Check device compatibility
        if tensor1.device != tensor2.device:
            return False
        
        # Check dimension compatibility for concatenation
        if tensor1.dim() != tensor2.dim():
            return False
        
        # Check all dimensions except the first (concatenation dimension)
        if tensor1.dim() > 1:
            return tensor1.shape[1:] == tensor2.shape[1:]
        
        return True


# Factory functions for backward compatibility
def create_statistics_processor(debug: bool = False) -> MapStatisticsProcessor:
    """
    Factory function to create statistics processor.
    
    Args:
        debug: Enable debug logging
        
    Returns:
        MapStatisticsProcessor instance
        
    Time Complexity: O(1) - simple object creation
    """
    return MapStatisticsProcessor(debug)


def validate_statistics_batch(stats: List[Tuple]) -> bool:
    """
    Quick validation function for statistics batch.
    
    Args:
        stats: List of statistics tuples
        
    Returns:
        bool: True if valid, False otherwise
        
    Time Complexity: O(S) where S is number of statistics
    """
    is_valid, _ = StatisticsValidator.validate_statistics_format(stats)
    return is_valid


# Export public interface
__all__ = [
    'MapStatisticsProcessor',
    'StatisticsValidator',
    'create_statistics_processor',
    'validate_statistics_batch'
]