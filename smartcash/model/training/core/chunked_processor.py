#!/usr/bin/env python3
"""
Chunked processing module for memory-safe hierarchical operations.

This module provides memory-safe chunked processing capabilities for large datasets
in hierarchical prediction processing. Extracted from HierarchicalProcessor for
better separation of concerns and memory optimization.

Key Features:
- Memory-safe chunked confidence modulation
- Platform-aware chunk size optimization
- Fallback processing for memory-intensive operations
- Progressive memory cleanup during processing
- Conservative confidence adjustments for large datasets

Algorithmic Complexity:
- Chunked processing: O(P₁) with chunk-based processing
- Memory usage: O(chunk_size) per iteration
- Platform optimization: Adaptive chunk sizes based on available memory
"""

import torch
from typing import Optional

from smartcash.common.logger import get_logger
from smartcash.model.utils.memory_optimizer import get_memory_optimizer

logger = get_logger(__name__, level="DEBUG")


class ChunkedProcessor:
    """
    Processes large prediction sets in memory-safe chunks.
    
    This processor handles memory-intensive operations by breaking them into
    smaller, manageable chunks. Designed for situations where full tensor
    operations would exceed available memory.
    
    Features:
    - Adaptive chunk sizing based on available memory
    - Progressive memory cleanup during processing
    - Platform-aware optimization (different strategies for different hardware)
    - Conservative confidence modulation for large datasets
    - Fallback processing when chunk operations fail
    
    Time Complexity: O(P₁) with chunk-based processing where P₁ is predictions
    Space Complexity: O(chunk_size) per iteration, bounded memory usage
    """
    
    def __init__(self, device: torch.device = None, debug: bool = False):
        """
        Initialize chunked processor.
        
        Args:
            device: Torch device for computations
            debug: Enable debug logging
        """
        self.device = device or torch.device('cpu')
        self.debug = debug
        self.memory_optimizer = get_memory_optimizer()
        
        # Chunk size configuration based on platform
        self.base_chunk_size = 10000
        self.chunk_size = self._calculate_optimal_chunk_size()
        
        # Memory cleanup frequency
        self.cleanup_interval = 5  # Clean memory every 5 chunks
        
        # Conservative confidence adjustment for large datasets
        self.large_dataset_confidence_boost = 1.1  # 10% boost
    
    def process_chunked_confidence(
        self,
        all_predictions: torch.Tensor,
        layer_1_predictions: torch.Tensor
    ) -> torch.Tensor:
        """
        Process confidence modulation in memory-safe chunks.
        
        This method handles very large prediction sets that would cause memory
        issues if processed all at once. It applies conservative confidence
        modulation in chunks.
        
        Args:
            all_predictions: All predictions (not used in current implementation)
            layer_1_predictions: Layer 1 predictions to process
            
        Returns:
            Modified predictions with updated confidence
            
        Time Complexity: O(P₁) with chunk-based processing
        Space Complexity: O(chunk_size) per iteration
        """
        try:
            if self.debug:
                logger.debug(f"Large prediction set ({len(layer_1_predictions)}), using chunked processing")
            
            modified_predictions = layer_1_predictions.clone()
            
            # Process in chunks to avoid memory overflow
            for i in range(0, len(layer_1_predictions), self.chunk_size):
                end_idx = min(i + self.chunk_size, len(layer_1_predictions))
                chunk = layer_1_predictions[i:end_idx]
                
                # Apply simplified confidence modulation to chunk
                modified_chunk = self._process_chunk(chunk, i)
                modified_predictions[i:end_idx] = modified_chunk
                
                # Periodic memory cleanup
                if i % (self.chunk_size * self.cleanup_interval) == 0:
                    self.memory_optimizer.cleanup_memory()
            
            if self.debug:
                logger.debug(f"Chunked processing completed: {len(layer_1_predictions)} predictions in {(len(layer_1_predictions) + self.chunk_size - 1) // self.chunk_size} chunks")
            
            return modified_predictions
            
        except Exception as e:
            logger.error(f"Error in chunked confidence modulation: {e}")
            self.memory_optimizer.emergency_memory_cleanup()
            return layer_1_predictions
    
    def process_chunked_spatial(
        self,
        layer_1_preds: torch.Tensor,
        layer_3_preds: torch.Tensor,
        chunk_size: Optional[int] = None
    ) -> torch.Tensor:
        """
        Process spatial confidence in memory-safe chunks.
        
        This method handles spatial IoU computation for large prediction sets
        by processing in chunks to avoid memory overflow.
        
        Args:
            layer_1_preds: Layer 1 predictions
            layer_3_preds: Layer 3 predictions
            chunk_size: Optional custom chunk size
            
        Returns:
            Confidence scores for each Layer 1 prediction
            
        Time Complexity: O(P₁ × P₃) total, processed in O(chunk_size × P₃) iterations
        """
        if len(layer_3_preds) == 0:
            return torch.zeros(len(layer_1_preds), device=self.device)
        
        chunk_size = chunk_size or self.chunk_size
        layer_3_conf = torch.zeros(len(layer_1_preds), device=self.device)
        
        successful_chunks = 0
        failed_chunks = 0
        
        for i in range(0, len(layer_1_preds), chunk_size):
            end_idx = min(i + chunk_size, len(layer_1_preds))
            chunk = layer_1_preds[i:end_idx]
            
            try:
                # Process spatial confidence for this chunk
                chunk_confidence = self._process_spatial_chunk(chunk, layer_3_preds)
                layer_3_conf[i:end_idx] = chunk_confidence
                successful_chunks += 1
                
            except Exception as e:
                logger.warning(f"Error in spatial chunk {i//chunk_size}: {e}")
                failed_chunks += 1
                # Skip this chunk to avoid total failure - confidence remains 0
                continue
        
        if self.debug:
            logger.debug(f"Chunked spatial processing: {successful_chunks} successful, {failed_chunks} failed chunks")
        
        return layer_3_conf
    
    def _calculate_optimal_chunk_size(self) -> int:
        """
        Calculate optimal chunk size based on platform capabilities.
        
        Returns:
            Optimal chunk size for current platform
            
        Time Complexity: O(1) - simple platform check and calculation
        """
        platform_info = self.memory_optimizer.platform_info
        
        # Base chunk size
        chunk_size = self.base_chunk_size
        
        # Adjust based on available memory
        if platform_info.get('is_apple_silicon', False):
            # Apple Silicon - more conservative due to unified memory
            chunk_size = min(chunk_size, 5000)
        elif platform_info.get('has_cuda', False):
            # CUDA - can handle larger chunks with dedicated VRAM
            chunk_size = min(chunk_size, 15000)
        else:
            # CPU - moderate chunk size
            chunk_size = min(chunk_size, 8000)
        
        # Adjust based on total system memory if available
        total_memory = platform_info.get('total_memory_gb', 8)
        if total_memory < 8:
            chunk_size = min(chunk_size, 3000)  # Very conservative for low memory
        elif total_memory > 16:
            chunk_size = min(chunk_size, 20000)  # More aggressive for high memory
        
        if self.debug:
            logger.debug(f"Calculated optimal chunk size: {chunk_size} (platform: {platform_info})")
        
        return chunk_size
    
    def _process_chunk(self, chunk: torch.Tensor, chunk_index: int) -> torch.Tensor:
        """
        Process a single chunk with conservative confidence modulation.
        
        Args:
            chunk: Chunk of predictions to process
            chunk_index: Index of current chunk (for debugging)
            
        Returns:
            Modified chunk with adjusted confidence
            
        Time Complexity: O(chunk_size) - simple element-wise operations
        """
        try:
            # Clone chunk to avoid modifying original
            modified_chunk = chunk.clone()
            original_conf = chunk[:, 4]
            
            # Apply conservative boost for hierarchical validation
            # This is a simplified approach since we can't do full IoU computation
            modified_conf = torch.clamp(
                original_conf * self.large_dataset_confidence_boost,
                max=1.0
            )
            
            modified_chunk[:, 4] = modified_conf
            
            if self.debug and chunk_index % 10 == 0:  # Log every 10th chunk
                avg_boost = (modified_conf - original_conf).mean().item()
                logger.debug(f"Chunk {chunk_index//self.chunk_size}: size={len(chunk)}, avg_boost={avg_boost:.4f}")
            
            return modified_chunk
            
        except Exception as e:
            logger.warning(f"Error processing chunk {chunk_index//self.chunk_size}: {e}")
            return chunk  # Return original chunk if processing fails
    
    def _process_spatial_chunk(
        self,
        chunk: torch.Tensor,
        layer_3_preds: torch.Tensor,
        iou_threshold: float = 0.1
    ) -> torch.Tensor:
        """
        Process spatial confidence for a single chunk.
        
        Args:
            chunk: Chunk of Layer 1 predictions
            layer_3_preds: Layer 3 predictions for spatial comparison
            iou_threshold: IoU threshold for spatial overlap
            
        Returns:
            Confidence scores for chunk predictions
            
        Time Complexity: O(chunk_size × P₃) for IoU computation
        """
        from .yolo_utils_manager import get_box_iou
        
        try:
            # Compute IoU for this chunk only
            iou_matrix = get_box_iou()(chunk[:, :4], layer_3_preds[:, :4])
            max_ious, max_indices = torch.max(iou_matrix, dim=1)
            
            # Apply threshold and assign confidences
            valid_mask = max_ious > iou_threshold
            chunk_confidence = torch.zeros(len(chunk), device=self.device)
            chunk_confidence[valid_mask] = layer_3_preds[max_indices[valid_mask], 4]
            
            return chunk_confidence
            
        except Exception as e:
            logger.warning(f"Error in spatial chunk processing: {e}")
            return torch.zeros(len(chunk), device=self.device)
    
    def adjust_chunk_size(self, new_chunk_size: int) -> None:
        """
        Manually adjust chunk size for specific use cases.
        
        Args:
            new_chunk_size: New chunk size to use
            
        Time Complexity: O(1) - simple assignment
        """
        old_chunk_size = self.chunk_size
        self.chunk_size = max(100, min(new_chunk_size, 50000))  # Bounded between 100 and 50k
        
        if self.debug:
            logger.debug(f"Chunk size adjusted: {old_chunk_size} → {self.chunk_size}")
    
    def get_memory_usage_estimate(self, prediction_count: int) -> dict:
        """
        Estimate memory usage for given prediction count.
        
        Args:
            prediction_count: Number of predictions to process
            
        Returns:
            Dictionary with memory usage estimates
            
        Time Complexity: O(1) - simple calculation
        """
        chunks_needed = (prediction_count + self.chunk_size - 1) // self.chunk_size
        
        # Rough memory estimates (in MB)
        prediction_memory = prediction_count * 6 * 4 / (1024 * 1024)  # 6 features * 4 bytes per float
        chunk_memory = self.chunk_size * 6 * 4 / (1024 * 1024)
        
        return {
            'total_predictions': prediction_count,
            'chunk_size': self.chunk_size,
            'chunks_needed': chunks_needed,
            'estimated_total_memory_mb': prediction_memory,
            'estimated_chunk_memory_mb': chunk_memory,
            'memory_efficient': prediction_count > self.chunk_size
        }
    
    def get_processing_stats(self) -> dict:
        """
        Get statistics about chunked processing configuration.
        
        Returns:
            Dictionary with processing statistics
            
        Time Complexity: O(1) - simple data collection
        """
        return {
            'chunk_size': self.chunk_size,
            'base_chunk_size': self.base_chunk_size,
            'cleanup_interval': self.cleanup_interval,
            'confidence_boost': self.large_dataset_confidence_boost,
            'device': str(self.device),
            'platform_info': self.memory_optimizer.platform_info
        }


class ChunkedAnalyzer:
    """
    Analyzer for chunked processing performance and efficiency.
    
    Provides analysis of chunked processing effectiveness and recommendations
    for optimization.
    """
    
    @staticmethod
    def analyze_chunk_efficiency(
        prediction_count: int,
        chunk_size: int,
        processing_time: float
    ) -> dict:
        """
        Analyze efficiency of chunked processing.
        
        Args:
            prediction_count: Total number of predictions processed
            chunk_size: Chunk size used
            processing_time: Total processing time in seconds
            
        Returns:
            Dictionary with efficiency analysis
            
        Time Complexity: O(1) - simple calculations
        """
        chunks_processed = (prediction_count + chunk_size - 1) // chunk_size
        predictions_per_second = prediction_count / max(processing_time, 0.001)
        time_per_chunk = processing_time / max(chunks_processed, 1)
        
        return {
            'total_predictions': prediction_count,
            'chunk_size': chunk_size,
            'chunks_processed': chunks_processed,
            'processing_time_seconds': processing_time,
            'predictions_per_second': predictions_per_second,
            'time_per_chunk_seconds': time_per_chunk,
            'efficiency_rating': 'high' if predictions_per_second > 1000 else 'moderate' if predictions_per_second > 100 else 'low'
        }
    
    @staticmethod
    def recommend_chunk_size(
        prediction_count: int,
        available_memory_gb: float,
        target_memory_usage_percent: float = 50.0
    ) -> dict:
        """
        Recommend optimal chunk size based on constraints.
        
        Args:
            prediction_count: Number of predictions to process
            available_memory_gb: Available memory in GB
            target_memory_usage_percent: Target memory usage percentage
            
        Returns:
            Dictionary with chunk size recommendation
            
        Time Complexity: O(1) - simple calculations
        """
        # Rough calculation: each prediction uses ~24 bytes (6 floats)
        bytes_per_prediction = 24
        target_memory_bytes = available_memory_gb * 1024 * 1024 * 1024 * (target_memory_usage_percent / 100)
        
        recommended_chunk_size = int(target_memory_bytes / bytes_per_prediction)
        recommended_chunk_size = max(100, min(recommended_chunk_size, 50000))  # Bounded
        
        chunks_needed = (prediction_count + recommended_chunk_size - 1) // recommended_chunk_size
        
        return {
            'recommended_chunk_size': recommended_chunk_size,
            'chunks_needed': chunks_needed,
            'estimated_memory_usage_gb': (recommended_chunk_size * bytes_per_prediction) / (1024 ** 3),
            'memory_efficient': chunks_needed > 1,
            'reasoning': f'Optimized for {target_memory_usage_percent}% of {available_memory_gb}GB memory'
        }


# Factory functions for backward compatibility
def create_chunked_processor(device: torch.device = None, debug: bool = False) -> ChunkedProcessor:
    """
    Factory function to create chunked processor.
    
    Args:
        device: Torch device for computations
        debug: Enable debug logging
        
    Returns:
        ChunkedProcessor instance
        
    Time Complexity: O(1) - simple object creation
    """
    return ChunkedProcessor(device, debug)


# Export public interface
__all__ = [
    'ChunkedProcessor',
    'ChunkedAnalyzer',
    'create_chunked_processor'
]