#!/usr/bin/env python3
"""
Memory-optimized processor for SmartCash mAP calculations.

Provides platform-aware memory management and optimization strategies for 
different hardware configurations (Apple Silicon, CUDA, CPU). Handles
chunked processing and memory cleanup to prevent OOM issues during
large-scale validation runs.

Algorithmic Optimizations:
- Chunked processing: O(N/chunk_size) iterations instead of O(N²)
- Platform-aware chunk sizing: Optimal memory utilization per platform
- Parallel greedy assignment: O(N log N) instead of O(N²) matching
"""

import torch
from typing import Dict, Optional, Any
from dataclasses import dataclass

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

logger = get_logger(__name__, level="DEBUG")


@dataclass
class ProcessingConfig:
    """Configuration for memory-optimized processing."""
    chunk_size: int
    max_matrix_combinations: int
    cleanup_frequency: int
    use_parallel_assignment: bool
    

class MemoryOptimizedProcessor:
    """
    Memory-optimized processor for handling large-scale tensor operations.
    
    This processor adapts to different hardware platforms and provides
    optimized chunking strategies to prevent memory overflow while
    maintaining optimal performance.
    
    Time Complexity: O(N/chunk_size) iterations for chunked operations
    Space Complexity: O(chunk_size) per iteration instead of O(N)
    """
    
    def __init__(self, device: Optional[torch.device] = None, debug: bool = False):
        """
        Initialize memory-optimized processor.
        
        Args:
            device: Torch device for computations
            debug: Enable debug logging
        """
        self.device = device or torch.device('cpu')
        self.debug = debug
        self.memory_optimizer = get_memory_optimizer()
        
        # Get platform-specific configuration
        self.config = self._get_platform_config()
        
        # Processing counters
        self._batch_count = 0
        
    def _get_platform_config(self) -> ProcessingConfig:
        """
        Get platform-specific processing configuration.
        
        Returns:
            ProcessingConfig: Optimized configuration for current platform
            
        Time Complexity: O(1) - simple platform detection
        """
        platform_info = self.memory_optimizer.platform_info
        
        if platform_info['is_apple_silicon']:
            # Conservative settings for Apple Silicon MPS
            return ProcessingConfig(
                chunk_size=256,
                max_matrix_combinations=1_000_000,  # 1M = ~8MB
                cleanup_frequency=5,
                use_parallel_assignment=True
            )
        elif platform_info['is_cuda_workstation']:
            # Aggressive settings for CUDA GPUs with more memory
            return ProcessingConfig(
                chunk_size=2048,
                max_matrix_combinations=10_000_000,  # 10M = ~80MB
                cleanup_frequency=10,
                use_parallel_assignment=True
            )
        else:
            # Conservative CPU settings
            return ProcessingConfig(
                chunk_size=128,
                max_matrix_combinations=500_000,  # 500K = ~4MB
                cleanup_frequency=3,
                use_parallel_assignment=False
            )
    
    def optimize_tensor_transfer(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Optimize tensor device transfer with platform-aware strategies.
        
        Args:
            tensor: Input tensor to optimize
            
        Returns:
            Optimized tensor on target device
            
        Time Complexity: O(N) where N is tensor size
        Space Complexity: O(N) for tensor copy
        """
        if not isinstance(tensor, torch.Tensor):
            return tensor
            
        # Use non_blocking transfer for better performance
        return tensor.detach().to(self.device, non_blocking=True)
    
    def process_predictions_chunked(
        self, 
        predictions: torch.Tensor,
        processing_func: callable,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process predictions in memory-safe chunks.
        
        Args:
            predictions: Input predictions tensor
            processing_func: Function to apply to each chunk
            chunk_size: Override default chunk size
            
        Returns:
            Processed predictions tensor
            
        Time Complexity: O(N/chunk_size) iterations
        Space Complexity: O(chunk_size) per iteration
        """
        if predictions.numel() == 0:
            return predictions
            
        effective_chunk_size = chunk_size or self.config.chunk_size
        
        if len(predictions) <= effective_chunk_size:
            # No chunking needed
            return processing_func(predictions)
        
        results = []
        
        for i in range(0, len(predictions), effective_chunk_size):
            end_idx = min(i + effective_chunk_size, len(predictions))
            chunk = predictions[i:end_idx]
            
            try:
                processed_chunk = processing_func(chunk)
                results.append(processed_chunk)
                
                # Periodic memory cleanup
                if (i // effective_chunk_size) % self.config.cleanup_frequency == 0:
                    self.memory_optimizer.cleanup_memory()
                    
            except Exception as e:
                logger.warning(f"Error processing chunk {i//effective_chunk_size}: {e}")
                # Continue with next chunk to avoid total failure
                continue
        
        if not results:
            return torch.empty_like(predictions[:0])  # Empty tensor with same dtype/device
        
        return torch.cat(results, dim=0)
    
    def optimize_greedy_assignment(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        iou_matrix: torch.Tensor,
        iou_threshold: float = 0.5
    ) -> torch.Tensor:
        """
        Perform memory-optimized greedy assignment for prediction-target matching.
        
        Args:
            predictions: Predictions tensor
            targets: Targets tensor
            iou_matrix: IoU matrix between predictions and targets
            iou_threshold: IoU threshold for matching
            
        Returns:
            Boolean tensor indicating true positives
            
        Time Complexity: O(N log N) for sorting + O(N/chunk_size) for assignment
        Space Complexity: O(N) for tracking arrays
        """
        tp = torch.zeros((predictions.shape[0], 1), dtype=torch.bool, device=self.device)
        
        # Find potential matches above IoU threshold
        iou_mask = iou_matrix > iou_threshold
        matches = torch.where(iou_mask)
        
        if matches[0].shape[0] == 0:
            return tp
        
        # Get IoU values and class matching for valid matches
        match_ious = iou_matrix[matches]
        
        try:
            # Validate tensor access bounds
            if matches[0].max() >= predictions.shape[0] or matches[1].max() >= targets.shape[0]:
                logger.error("Index out of bounds in greedy assignment")
                return tp
                
            pred_classes = predictions[matches[0], 6].int()  # Prediction classes
            target_classes = targets[matches[1], 5].int()   # Target classes
            
            # Filter matches by class compatibility
            class_matches = pred_classes == target_classes
            valid_mask = class_matches
            
            if not valid_mask.any():
                return tp
            
            # Extract valid matches
            valid_pred_idx = matches[0][valid_mask]
            valid_target_idx = matches[1][valid_mask]
            valid_ious = match_ious[valid_mask]
            
            # Sort by IoU (highest first) for greedy matching - O(N log N)
            sort_indices = torch.argsort(valid_ious, descending=True)
            sorted_pred_idx = valid_pred_idx[sort_indices]
            sorted_target_idx = valid_target_idx[sort_indices]
            
            # Platform-aware parallel greedy assignment
            if self.config.use_parallel_assignment:
                tp = self._parallel_greedy_assignment(
                    tp, sorted_pred_idx, sorted_target_idx, 
                    predictions.shape[0], targets.shape[0]
                )
            else:
                tp = self._sequential_greedy_assignment(
                    tp, sorted_pred_idx, sorted_target_idx,
                    predictions.shape[0], targets.shape[0]
                )
            
            return tp
            
        except Exception as e:
            logger.error(f"Error in greedy assignment: {e}")
            return tp
    
    def _parallel_greedy_assignment(
        self,
        tp: torch.Tensor,
        sorted_pred_idx: torch.Tensor,
        sorted_target_idx: torch.Tensor,
        num_predictions: int,
        num_targets: int
    ) -> torch.Tensor:
        """
        Parallel greedy assignment with chunked processing.
        
        Args:
            tp: True positive tensor to update
            sorted_pred_idx: Sorted prediction indices
            sorted_target_idx: Sorted target indices
            num_predictions: Total number of predictions
            num_targets: Total number of targets
            
        Returns:
            Updated true positive tensor
            
        Time Complexity: O(N/chunk_size) iterations
        Space Complexity: O(N) for tracking arrays
        """
        # Initialize tracking arrays
        used_targets = torch.zeros(num_targets, dtype=torch.bool, device=self.device)
        used_predictions = torch.zeros(num_predictions, dtype=torch.bool, device=self.device)
        
        # Process in chunks for memory efficiency
        chunk_size = self.config.chunk_size
        
        for chunk_start in range(0, len(sorted_pred_idx), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(sorted_pred_idx))
            
            # Get chunk of indices
            chunk_pred_idx = sorted_pred_idx[chunk_start:chunk_end]
            chunk_target_idx = sorted_target_idx[chunk_start:chunk_end]
            
            # Parallel availability check for entire chunk
            pred_available = ~used_predictions[chunk_pred_idx]
            target_available = ~used_targets[chunk_target_idx]
            valid_matches = pred_available & target_available
            
            # Apply matches in parallel
            if valid_matches.any():
                valid_pred_idx = chunk_pred_idx[valid_matches]
                valid_target_idx = chunk_target_idx[valid_matches]
                
                # Parallel assignment - O(chunk_size)
                tp[valid_pred_idx, 0] = True
                used_targets[valid_target_idx] = True
                used_predictions[valid_pred_idx] = True
        
        return tp
    
    def _sequential_greedy_assignment(
        self,
        tp: torch.Tensor,
        sorted_pred_idx: torch.Tensor,
        sorted_target_idx: torch.Tensor,
        _: int,
        __: int
    ) -> torch.Tensor:
        """
        Sequential greedy assignment for CPU processing.
        
        Args:
            tp: True positive tensor to update
            sorted_pred_idx: Sorted prediction indices
            sorted_target_idx: Sorted target indices
            num_predictions: Total number of predictions
            num_targets: Total number of targets
            
        Returns:
            Updated true positive tensor
            
        Time Complexity: O(N) sequential processing
        Space Complexity: O(N) for tracking arrays
        """
        # Use Python sets for faster lookups on CPU
        used_targets = set()
        used_predictions = set()
        
        for pred_idx, target_idx in zip(sorted_pred_idx.cpu().numpy(), sorted_target_idx.cpu().numpy()):
            pred_idx = int(pred_idx)
            target_idx = int(target_idx)
            
            # Check availability
            if pred_idx not in used_predictions and target_idx not in used_targets:
                tp[pred_idx, 0] = True
                used_predictions.add(pred_idx)
                used_targets.add(target_idx)
        
        return tp
    
    def estimate_memory_usage(
        self, 
        num_predictions: int, 
        num_targets: int,
        dtype: torch.dtype = torch.float32
    ) -> Dict[str, Any]:
        """
        Estimate memory usage for tensor operations.
        
        Args:
            num_predictions: Number of predictions
            num_targets: Number of targets
            dtype: Tensor data type
            
        Returns:
            Dictionary with memory usage estimates
            
        Time Complexity: O(1) - simple arithmetic
        """
        # Size in bytes per element
        element_size = 4 if dtype == torch.float32 else 8  # float32=4, float64=8
        
        # IoU matrix size
        iou_matrix_elements = num_predictions * num_targets
        iou_matrix_bytes = iou_matrix_elements * element_size
        iou_matrix_mb = iou_matrix_bytes / (1024 * 1024)
        
        # Tracking arrays size
        tracking_bytes = (num_predictions + num_targets) * 1  # bool = 1 byte
        tracking_mb = tracking_bytes / (1024 * 1024)
        
        total_mb = iou_matrix_mb + tracking_mb
        
        # Recommend chunking if memory usage is high
        recommend_chunking = total_mb > 100  # 100MB threshold
        
        return {
            'iou_matrix_mb': iou_matrix_mb,
            'tracking_mb': tracking_mb,
            'total_mb': total_mb,
            'recommend_chunking': recommend_chunking,
            'suggested_chunk_size': max(64, int(self.config.chunk_size * (100 / max(total_mb, 1))))
        }
    
    def cleanup_after_batch(self):
        """
        Perform memory cleanup after batch processing.
        
        Time Complexity: O(1) - platform-specific cleanup operations
        """
        self._batch_count += 1
        
        # Periodic cleanup based on configuration
        if self._batch_count % self.config.cleanup_frequency == 0:
            if self.debug:
                logger.debug(f"🧹 Performing memory cleanup after batch {self._batch_count}")
            self.memory_optimizer.cleanup_memory()
    
    def emergency_cleanup(self):
        """
        Perform emergency memory cleanup on error.
        
        Time Complexity: O(1) - immediate cleanup operations
        """
        if self.debug:
            logger.debug("🚨 Emergency memory cleanup triggered")
        self.memory_optimizer.emergency_memory_cleanup()
    
    def get_processing_stats(self) -> Dict[str, Any]:
        """
        Get processing statistics and configuration.
        
        Returns:
            Dictionary with processing statistics
            
        Time Complexity: O(1) - simple data collection
        """
        return {
            'batch_count': self._batch_count,
            'device': str(self.device),
            'platform_info': self.memory_optimizer.platform_info,
            'config': {
                'chunk_size': self.config.chunk_size,
                'max_matrix_combinations': self.config.max_matrix_combinations,
                'cleanup_frequency': self.config.cleanup_frequency,
                'use_parallel_assignment': self.config.use_parallel_assignment
            }
        }