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
    
    logger.debug(f"Calculating metrics for {len(predictions) if predictions else 0} prediction layers, {len(targets) if targets else 0} target layers")
    
    if not predictions or not targets:
        logger.debug("No predictions or targets provided")
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
                logger.debug(f"Layer {layer_name}: No valid predictions or targets after processing")
                continue
                
            pred_classes = pred_classes[:min_len]
            target_classes = target_classes[:min_len]
            
            # Debug info
            unique_preds = np.unique(pred_classes)
            unique_targets = np.unique(target_classes)
            logger.debug(f"Layer {layer_name}: {min_len} samples, pred_classes={unique_preds}, target_classes={unique_targets}")
            
            # Calculate metrics using sklearn (handles edge cases well)
            # Add small epsilon to avoid exactly zero values that might be interpreted as static
            epsilon = 1e-6
            try:
                accuracy = float(accuracy_score(target_classes, pred_classes))
                metrics[f'{layer_name}_accuracy'] = max(epsilon, accuracy)
                
                precision = float(precision_score(target_classes, pred_classes, average='weighted', zero_division=0))
                metrics[f'{layer_name}_precision'] = max(epsilon, precision)
                
                recall = float(recall_score(target_classes, pred_classes, average='weighted', zero_division=0))
                metrics[f'{layer_name}_recall'] = max(epsilon, recall)
                
                f1 = float(f1_score(target_classes, pred_classes, average='weighted', zero_division=0))
                metrics[f'{layer_name}_f1'] = max(epsilon, f1)
            except Exception as e:
                logger.warning(f"Error calculating sklearn metrics for {layer_name}: {e}")
                # Fallback to simple calculation with epsilon
                metrics[f'{layer_name}_accuracy'] = epsilon
                metrics[f'{layer_name}_precision'] = epsilon
                metrics[f'{layer_name}_recall'] = epsilon
                metrics[f'{layer_name}_f1'] = epsilon
            
        except Exception as e:
            logger.warning(f"Error calculating metrics for {layer_name}: {e}")
            import traceback
            logger.debug(f"Traceback: {traceback.format_exc()}")
            # Set default values for failed calculation
            metrics[f'{layer_name}_accuracy'] = 0.0
            metrics[f'{layer_name}_precision'] = 0.0
            metrics[f'{layer_name}_recall'] = 0.0
            metrics[f'{layer_name}_f1'] = 0.0
    
    return metrics


def filter_phase_relevant_metrics(metrics: Dict[str, float], phase_num: int) -> Dict[str, float]:
    """
    Filter metrics to show only relevant ones for the current training phase.
    
    Phase 1: Focus on train_loss, val_loss, and layer_1_* metrics (only Layer 1 is training)
    Phase 2: Show all metrics (full model fine-tuning)
    
    Args:
        metrics: Complete metrics dictionary
        phase_num: Current training phase (1 or 2)
        
    Returns:
        Filtered metrics dictionary containing only relevant metrics
    """
    if not metrics:
        return metrics
    
    if phase_num == 1:
        # Phase 1: Only show metrics relevant to Layer 1 training
        relevant_metrics = {}
        
        # Always include core training metrics
        core_metrics = ['train_loss', 'val_loss', 'val_accuracy','val_precision', 'val_recall', 'val_f1', 'learning_rate', 'epoch']
        for metric in core_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # Include all layer_1_* metrics (these should be improving)
        for key, value in metrics.items():
            if key.startswith('layer_1_'):
                relevant_metrics[key] = value
        
        # Optionally include uncertainty/loss components for debugging
        debug_metrics = ['total_loss', 'layer_1_total_loss', 'layer_1_weighted_loss', 
                        'layer_1_regularization', 'layer_1_uncertainty']
        for metric in debug_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        logger.debug(f"Phase {phase_num}: Filtered {len(metrics)} metrics to {len(relevant_metrics)} relevant ones")
        return relevant_metrics
    
    else:
        # Phase 2: Show all metrics (full model is being fine-tuned)
        logger.debug(f"Phase {phase_num}: Showing all {len(metrics)} metrics")
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