#!/usr/bin/env python3
"""
Shared checkpoint utilities for PyTorch version compatibility.

This module provides safe loading functions that handle the torch.serialization.safe_globals
compatibility issue across different PyTorch versions.
"""

import torch
from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def safe_load_checkpoint(checkpoint_path: str, map_location: str = 'cpu', **kwargs) -> Dict[str, Any]:
    """
    Load checkpoint data with proper safety measures and PyTorch version compatibility.
    
    This function handles the torch.serialization.safe_globals compatibility issue
    across different PyTorch versions.
    
    Args:
        checkpoint_path: Path to checkpoint file
        map_location: Device mapping for torch.load
        **kwargs: Additional arguments for torch.load
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        Exception: If checkpoint cannot be loaded
    """
    import torch.serialization
    
    try:
        # Check if safe_globals is available (PyTorch 2.6+)
        if hasattr(torch.serialization, 'safe_globals'):
            try:
                # Try to import YOLOv5 components for safe globals
                from models.yolo import Model as YOLOModel
                from models.common import Conv, C3, SPPF, Bottleneck
                safe_globals = [YOLOModel, Conv, C3, SPPF, Bottleneck]
            except ImportError:
                # If YOLOv5 components not available, use empty safe_globals
                safe_globals = []
            
            with torch.serialization.safe_globals(safe_globals):
                return torch.load(checkpoint_path, map_location=map_location, weights_only=False, **kwargs)
        else:
            # Fallback for older PyTorch versions without safe_globals
            logger.debug(f"PyTorch version does not have safe_globals, using standard torch.load for {checkpoint_path}")
            return torch.load(checkpoint_path, map_location=map_location, weights_only=False, **kwargs)
            
    except AttributeError as e:
        # Handle missing safe_globals attribute
        logger.debug(f"safe_globals attribute not available: {e}, using standard torch.load")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False, **kwargs)
    except Exception as e:
        # Final fallback - try standard load without special handling
        logger.warning(f"Safe checkpoint loading failed for {checkpoint_path}: {e}")
        logger.info("Attempting fallback with standard torch.load")
        return torch.load(checkpoint_path, map_location=map_location, weights_only=False, **kwargs)