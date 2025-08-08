"""
Validation utilities for visualization components.

This module provides functions to validate inputs and data structures
used in visualization, ensuring data integrity and proper error handling.
"""

from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np


def validate_metrics_dict(metrics: Dict[str, Any], 
                        required_metrics: List[str] = None) -> Tuple[bool, str]:
    """
    Validate a metrics dictionary.
    
    Args:
        metrics: Dictionary of metrics to validate
        required_metrics: List of required metric names
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(metrics, dict):
        return False, f"Metrics must be a dictionary, got {type(metrics).__name__}"
    
    if required_metrics:
        for metric in required_metrics:
            if metric not in metrics:
                return False, f"Missing required metric: {metric}"
    
    return True, ""


def validate_confusion_matrix(
    cm: Any, 
    num_classes: Optional[int] = None,
    require_square: bool = True
) -> Tuple[bool, str]:
    """
    Validate a confusion matrix.
    
    Args:
        cm: Confusion matrix to validate
        num_classes: Expected number of classes (optional)
        require_square: Whether the matrix must be square
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(cm, np.ndarray):
        return False, f"Confusion matrix must be a numpy array, got {type(cm).__name__}"
    
    if cm.ndim != 2:
        return False, f"Confusion matrix must be 2D, got {cm.ndim}D"
    
    if require_square and cm.shape[0] != cm.shape[1]:
        return False, f"Confusion matrix must be square, got shape {cm.shape}"
    
    if num_classes is not None and cm.shape[0] != num_classes:
        return (
            False,
            f"Confusion matrix has {cm.shape[0]} classes, expected {num_classes}"
        )
    
    return True, ""


def validate_layer_config(
    layer_config: Dict[str, Any], 
    required_keys: List[str] = None
) -> Tuple[bool, str]:
    """
    Validate a layer configuration dictionary.
    
    Args:
        layer_config: Layer configuration to validate
        required_keys: List of required keys
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(layer_config, dict):
        return False, f"Layer config must be a dictionary, got {type(layer_config).__name__}"
    
    if required_keys:
        for key in required_keys:
            if key not in layer_config:
                return False, f"Missing required key in layer config: {key}"
    
    return True, ""


def validate_learning_rates(
    learning_rates: List[float], 
    min_lr: float = 1e-10, 
    max_lr: float = 1.0
) -> Tuple[bool, str]:
    """
    Validate learning rates.
    
    Args:
        learning_rates: List of learning rates to validate
        min_lr: Minimum allowed learning rate
        max_lr: Maximum allowed learning rate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(learning_rates, (list, np.ndarray)):
        return False, f"Learning rates must be a list or numpy array, got {type(learning_rates).__name__}"
    
    if not learning_rates:
        return False, "Learning rates list is empty"
    
    for i, lr in enumerate(learning_rates):
        if not isinstance(lr, (int, float, np.number)):
            return False, f"Learning rate at index {i} is not a number: {lr}"
        
        if not (min_lr <= lr <= max_lr):
            return (
                False,
                f"Learning rate {lr} at index {i} is outside valid range "
                f"[{min_lr}, {max_lr}]"
            )
    
    return True, ""


def validate_epoch_metrics(
    metrics: Dict[str, Any], 
    epoch: int
) -> Tuple[bool, str]:
    """
    Validate metrics for a specific epoch.
    
    Args:
        metrics: Dictionary of metrics to validate
        epoch: Epoch number for error messages
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not isinstance(metrics, dict):
        return False, f"Metrics for epoch {epoch} must be a dictionary, got {type(metrics).__name__}"
    
    if not metrics:
        return False, f"No metrics provided for epoch {epoch}"
    
    for key, value in metrics.items():
        if not isinstance(key, str):
            return False, f"Metric name must be a string, got {type(key).__name__} for epoch {epoch}"
        
        if not isinstance(value, (int, float, np.number)):
            return False, f"Metric value for '{key}' must be a number, got {type(value).__name__} for epoch {epoch}"
    
    return True, ""


def validate_visualization_available() -> Tuple[bool, str]:
    """
    Check if visualization libraries are available.
    
    Returns:
        Tuple of (is_available, error_message)
    """
    try:
        import matplotlib  # noqa: F401
        import seaborn  # noqa: F401
        return True, ""
    except ImportError as e:
        return False, f"Visualization libraries not available: {str(e)}"
