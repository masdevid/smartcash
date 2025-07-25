#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/metrics_utils.py

Metrics calculation utilities for unified training pipeline.
"""

import torch
import numpy as np
from typing import Dict
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def calculate_multilayer_metrics(predictions: Dict[str, torch.Tensor], 
                                targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Optimized metrics calculation for multi-layer detection model.
    
    Args:
        predictions: Dictionary of predictions per layer
        targets: Dictionary of ground truth targets per layer
        
    Returns:
        Dictionary containing accuracy, precision, recall, F1 for each layer
    """
    metrics = {}
    
    if not predictions or not targets:
        return metrics
    
    for layer_name in predictions.keys():
        if layer_name not in targets:
            continue
            
        pred_tensor = predictions[layer_name]
        target_tensor = targets[layer_name]
        
        # Skip if either is None or empty
        if pred_tensor is None or target_tensor is None:
            continue
            
        try:
            # Convert to numpy efficiently
            if isinstance(pred_tensor, torch.Tensor):
                pred_np = pred_tensor.detach().cpu().numpy()
            elif isinstance(pred_tensor, (list, tuple)):
                pred_np = np.concatenate([t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in pred_tensor])
            else:
                pred_np = np.asarray(pred_tensor)
                
            if isinstance(target_tensor, torch.Tensor):
                target_np = target_tensor.detach().cpu().numpy()
            elif isinstance(target_tensor, (list, tuple)):
                target_np = np.concatenate([t.detach().cpu().numpy() if isinstance(t, torch.Tensor) else np.array(t) for t in target_tensor])
            else:
                target_np = np.asarray(target_tensor)
            
            # Skip if empty arrays
            if pred_np.size == 0 or target_np.size == 0:
                continue
            
            # Determine classification type and process accordingly
            if pred_np.ndim > 1 and pred_np.shape[-1] > 1:
                # Multi-class: use argmax
                pred_classes = np.argmax(pred_np, axis=-1).flatten()
            else:
                # Binary: use threshold
                pred_classes = (pred_np.flatten() > 0.5).astype(int)
            
            target_classes = target_np.flatten().astype(int)
            
            # Ensure same shape
            min_len = min(len(pred_classes), len(target_classes))
            if min_len == 0:
                continue
                
            pred_classes = pred_classes[:min_len]
            target_classes = target_classes[:min_len]
            
            # Calculate metrics using sklearn (handles edge cases well)
            metrics[f'{layer_name}_accuracy'] = float(accuracy_score(target_classes, pred_classes))
            metrics[f'{layer_name}_precision'] = float(precision_score(target_classes, pred_classes, average='weighted', zero_division=0))
            metrics[f'{layer_name}_recall'] = float(recall_score(target_classes, pred_classes, average='weighted', zero_division=0))
            metrics[f'{layer_name}_f1'] = float(f1_score(target_classes, pred_classes, average='weighted', zero_division=0))
            
        except Exception as e:
            logger.debug(f"Error calculating metrics for {layer_name}: {e}")
            # Set default values for failed calculation
            metrics[f'{layer_name}_accuracy'] = 0.0
            metrics[f'{layer_name}_precision'] = 0.0
            metrics[f'{layer_name}_recall'] = 0.0
            metrics[f'{layer_name}_f1'] = 0.0
    
    return metrics


def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for console display.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted string for display
    """
    if not metrics:
        return "No metrics available"
    
    formatted_lines = []
    
    # Group metrics by layer
    layers = set()
    for key in metrics.keys():
        if '_' in key:
            layer = key.split('_')[0] + '_' + key.split('_')[1]  # e.g., 'layer_1'
            if layer.startswith('layer_'):
                layers.add(layer)
    
    # Display layer metrics
    for layer in sorted(layers):
        layer_metrics = {k: v for k, v in metrics.items() if k.startswith(layer)}
        if layer_metrics:
            formatted_lines.append(f"{layer.upper()}:")
            for metric_name, value in layer_metrics.items():
                metric_type = metric_name.replace(f"{layer}_", "").title()
                formatted_lines.append(f"  {metric_type}: {value:.4f}")
    
    # Display other metrics
    other_metrics = {k: v for k, v in metrics.items() 
                    if not any(k.startswith(layer) for layer in layers)}
    if other_metrics:
        formatted_lines.append("Other Metrics:")
        for metric_name, value in other_metrics.items():
            formatted_lines.append(f"  {metric_name.title()}: {value:.4f}")
    
    return "\n".join(formatted_lines)