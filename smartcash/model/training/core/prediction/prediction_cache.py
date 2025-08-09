#!/usr/bin/env python3
"""
Caching utilities for prediction processing.

This module handles tensor caching, memory optimization,
and cached computations for prediction processing.
"""

import torch
from typing import Dict, Tuple
from functools import lru_cache

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class PredictionCache:
    """Handles caching for prediction processing operations."""
    
    def __init__(self):
        """Initialize prediction cache."""
        # Performance optimization: cached computations
        self._device_cache = {}
        self._tensor_cache = {}
        self._class_offsets = {
            'layer_1': 0,
            'layer_2': 7, 
            'layer_3': 14
        }
        
        # Debug counter for logging
        self._debug_counter = 0
    
    def get_cached_zero_tensor(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Get cached zero tensor to avoid repeated allocations.
        
        Time Complexity: O(1) for cache hit, O(n) for cache miss
        """
        cache_key = (batch_size, device, dtype)
        if cache_key not in self._tensor_cache:
            self._tensor_cache[cache_key] = torch.zeros(batch_size, dtype=dtype, device=device)
        return self._tensor_cache[cache_key].clone()
    
    def get_cached_random_tensor(self, batch_size: int, num_classes: int, device: torch.device) -> torch.Tensor:
        """Get cached random tensor for fallback predictions.
        
        Time Complexity: O(1) for cache hit, O(n*m) for cache miss
        """
        cache_key = (batch_size, num_classes, device)
        if cache_key not in self._tensor_cache:
            random_preds = torch.randn((batch_size, num_classes), device=device) * 0.1
            self._tensor_cache[cache_key] = torch.sigmoid(random_preds)
        return self._tensor_cache[cache_key].clone()
    
    def get_class_offset(self, layer_name: str) -> int:
        """Get cached class offset for layer."""
        return self._class_offsets.get(layer_name, 0)
    
    def increment_debug_counter(self) -> int:
        """Increment and return debug counter."""
        self._debug_counter += 1
        return self._debug_counter
    
    def reset_debug_counter(self) -> None:
        """Reset debug counter."""
        self._debug_counter = 0
    
    def get_debug_counter(self) -> int:
        """Get current debug counter value."""
        return self._debug_counter
    
    def clear_cache(self) -> None:
        """Clear tensor cache to free memory.
        
        Call this periodically during long training runs to prevent memory buildup.
        """
        self._tensor_cache.clear()
        self._device_cache.clear()
        self._debug_counter = 0
        
    def get_cache_info(self) -> Dict[str, int]:
        """Get cache statistics for monitoring.
        
        Returns:
            Dictionary with cache size information
        """
        return {
            'tensor_cache_size': len(self._tensor_cache),
            'device_cache_size': len(self._device_cache),
            'debug_counter': self._debug_counter
        }
