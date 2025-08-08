"""
Helper functions for visualization components.

This module provides utility functions that support the visualization
functionality, including data processing and formatting.
"""

from typing import Dict, List, Any, Optional, Union
import numpy as np
from pathlib import Path


def get_default_layer_metrics() -> Dict[str, List[float]]:
    """
    Get the default structure for layer metrics.
    
    Returns:
        Dictionary with default metrics initialized to empty lists
    """
    return {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'loss': []
    }


def initialize_layer_metrics(layer_names: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Initialize metrics storage for multiple layers.
    
    Args:
        layer_names: List of layer names
        
    Returns:
        Nested dictionary with metrics for each layer
    """
    return {layer: get_default_layer_metrics() for layer in layer_names}


def normalize_confusion_matrix(confusion_matrix: np.ndarray) -> np.ndarray:
    """
    Normalize a confusion matrix.
    
    Args:
        confusion_matrix: Input confusion matrix
        
    Returns:
        Normalized confusion matrix
    """
    if not isinstance(confusion_matrix, np.ndarray):
        raise ValueError("Input must be a numpy array")
        
    if confusion_matrix.size == 0:
        return confusion_matrix
        
    # Normalize by row (true labels)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums[row_sums == 0] = 1
    return confusion_matrix / row_sums


def ensure_directory_exists(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: Directory path
        
    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def format_metric_name(metric_name: str) -> str:
    """
    Format a metric name for display.
    
    Args:
        metric_name: Raw metric name (e.g., 'val_accuracy')
        
    Returns:
        Formatted display name (e.g., 'Validation Accuracy')
    """
    if not metric_name:
        return ""
        
    # Common replacements
    replacements = {
        'val_': 'Validation ',
        'train_': 'Training ',
        '_': ' ',
        'f1': 'F1',
        'iou': 'IoU',
        'map': 'mAP',
        'ap': 'AP'
    }
    
    # Apply replacements
    display_name = metric_name.lower()
    for old, new in replacements.items():
        display_name = display_name.replace(old.lower(), new)
    
    # Capitalize first letter of each word
    display_name = ' '.join(word.capitalize() for word in display_name.split())
    
    return display_name


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers, returning a default value if division by zero.
    
    Args:
        numerator: Numerator
        denominator: Denominator
        default: Value to return if denominator is zero
        
    Returns:
        Result of division or default value
    """
    try:
        return numerator / denominator if denominator != 0 else default
    except (TypeError, ZeroDivisionError):
        return default
