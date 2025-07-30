#!/usr/bin/env python3
"""
mAP Calculator factory for choosing optimal implementation.

This module provides a factory to select the best mAP calculator
based on system capabilities and data size.
"""

import os
import psutil
import torch
from typing import Optional

from smartcash.common.logger import get_logger
from .map_calculator import MAPCalculator
from .parallel_map_calculator import ParallelMAPCalculator

logger = get_logger(__name__)


class MAPCalculatorFactory:
    """Factory for creating optimal mAP calculator implementations."""
    
    @staticmethod
    def create_calculator(
        force_parallel: Optional[bool] = None,
        max_workers: Optional[int] = None,
        batch_queue_size: int = 100
    ):
        """
        Create the optimal mAP calculator based on system capabilities.
        
        Args:
            force_parallel: Force parallel (True) or sequential (False) implementation.
                           If None, auto-detect based on system capabilities.
            max_workers: Maximum number of worker threads for parallel implementation.
                        If None, auto-detect based on CPU cores.
            batch_queue_size: Maximum size of batch processing queue
            
        Returns:
            MAPCalculator or ParallelMAPCalculator instance
        """
        # System capability detection
        cpu_count = os.cpu_count() or 1
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        has_cuda = torch.cuda.is_available()
        
        # Auto-detect optimal configuration
        if force_parallel is None:
            # Use parallel if we have multiple cores and sufficient memory
            should_use_parallel = (
                cpu_count >= 4 and  # At least 4 CPU cores
                available_memory >= 4.0  # At least 4GB available RAM
            )
        else:
            should_use_parallel = force_parallel
        
        if max_workers is None:
            # Optimal worker count: leave some cores for other processes
            if has_cuda:
                # If using GPU, use fewer CPU workers to avoid competition
                max_workers = max(2, min(4, cpu_count // 2))
            else:
                # If CPU-only, use more workers but leave some cores free
                max_workers = max(2, min(8, cpu_count - 1)) 
        
        # Create appropriate calculator
        if should_use_parallel:
            logger.info(f"🚀 Creating ParallelMAPCalculator: Workers={max_workers}, CPU cores={cpu_count}, Available memory={available_memory:.1f}GB, CUDA={has_cuda}")
            
            return ParallelMAPCalculator(
                max_workers=max_workers,
                batch_queue_size=batch_queue_size
            )
        else:
            reason = 'Force sequential' if force_parallel is False else 'Limited system resources'
            logger.info(f"📊 Creating Sequential MAPCalculator: {reason} (CPU cores={cpu_count}, Available memory={available_memory:.1f}GB)")
            
            return MAPCalculator()
    
    @staticmethod
    def get_system_info() -> dict:
        """
        Get system information for performance analysis.
        
        Returns:
            Dictionary containing system information
        """
        memory = psutil.virtual_memory()
        
        return {
            'cpu_count': os.cpu_count(),
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_percent': memory.percent,
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
        }
    
    @staticmethod
    def recommend_configuration(expected_batch_size: int, expected_image_count: int) -> dict:
        """
        Recommend optimal configuration based on expected workload.
        
        Args:
            expected_batch_size: Expected batch size for training
            expected_image_count: Expected number of images per epoch
            
        Returns:
            Dictionary with recommended configuration
        """
        system_info = MAPCalculatorFactory.get_system_info()
        
        # Calculate estimated workload
        estimated_workload = expected_batch_size * expected_image_count
        
        # Recommendations based on workload and system capabilities
        if estimated_workload > 50000 and system_info['cpu_count'] >= 4:
            # High workload + capable system = parallel with many workers
            recommended_workers = min(8, system_info['cpu_count'])
            use_parallel = True
            batch_queue_size = 200
        elif estimated_workload > 10000 and system_info['cpu_count'] >= 2:
            # Medium workload + decent system = parallel with fewer workers
            recommended_workers = min(4, system_info['cpu_count'])
            use_parallel = True
            batch_queue_size = 100
        else:
            # Low workload or limited system = sequential
            recommended_workers = 1
            use_parallel = False
            batch_queue_size = 50
        
        # Adjust for memory constraints
        if system_info['memory_available_gb'] < 2.0:
            # Limited memory = reduce parallelism
            use_parallel = False
            recommended_workers = 1
            batch_queue_size = 25
        elif system_info['memory_available_gb'] < 4.0:
            # Moderate memory = conservative parallelism
            recommended_workers = min(recommended_workers, 2)
            batch_queue_size = min(batch_queue_size, 50)
        
        return {
            'use_parallel': use_parallel,
            'max_workers': recommended_workers,
            'batch_queue_size': batch_queue_size,
            'estimated_workload': estimated_workload,
            'system_info': system_info,
            'reasoning': {
                'workload_category': (
                    'high' if estimated_workload > 50000 else
                    'medium' if estimated_workload > 10000 else 'low'
                ),
                'memory_category': (
                    'high' if system_info['memory_available_gb'] >= 4.0 else
                    'medium' if system_info['memory_available_gb'] >= 2.0 else 'low'
                ),
                'cpu_category': (
                    'high' if system_info['cpu_count'] >= 8 else
                    'medium' if system_info['cpu_count'] >= 4 else 'low'
                )
            }
        }


# Convenience function for simple usage
def create_optimal_map_calculator(**kwargs):
    """
    Create optimal mAP calculator with default settings.
    
    Args:
        **kwargs: Arguments to pass to MAPCalculatorFactory.create_calculator()
        
    Returns:
        Optimal mAP calculator instance
    """
    return MAPCalculatorFactory.create_calculator(**kwargs)