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
            
            # Handle target format - check if one-hot encoded or class indices
            if target_np.ndim > 1 and target_np.shape[-1] > 1:
                # One-hot encoded targets: convert to class indices
                target_classes = np.argmax(target_np, axis=-1).flatten()
                logger.debug(f"Layer {layer_name}: Converting one-hot targets to class indices")
            else:
                # Already class indices
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
                
                # Calculate confusion matrix for visualization
                try:
                    from sklearn.metrics import confusion_matrix
                    # Determine number of classes for this layer
                    layer_num_classes = {
                        'layer_1': 7,  # Banknote denominations
                        'layer_2': 7,  # Denomination features  
                        'layer_3': 3   # Common features
                    }
                    num_classes = layer_num_classes.get(layer_name, 7)
                    
                    cm = confusion_matrix(target_classes, pred_classes, 
                                        labels=list(range(num_classes)))
                    # Store as list of lists for JSON serialization
                    metrics[f'{layer_name}_confusion_matrix'] = cm.tolist()
                    
                    logger.debug(f"Layer {layer_name}: Confusion matrix calculated ({cm.shape})")
                except Exception as cm_e:
                    logger.debug(f"Failed to calculate confusion matrix for {layer_name}: {cm_e}")
                    # Don't fail the whole process if CM calculation fails
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
        # Phase 1: Simple YOLO loss - focus on basic YOLO training metrics
        relevant_metrics = {}
        
        # Core YOLO training metrics
        core_metrics = ['train_loss', 'val_loss', 'learning_rate', 'epoch']
        for metric in core_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # YOLO detection metrics (if available)
        yolo_metrics = ['val_map50', 'val_map50_95', 'val_precision', 'val_recall', 'val_f1', 'val_accuracy']
        for metric in yolo_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # Include layer_1 metrics (since Phase 1 focuses on single layer)
        for key, value in metrics.items():
            if key.startswith('layer_1_') and isinstance(value, (int, float)) and value > 0.0001:
                relevant_metrics[key] = value
        
        # Include loss breakdown components for visualization charts
        loss_breakdown_metrics = [
            'train_box_loss', 'train_obj_loss', 'train_cls_loss',
            'val_box_loss', 'val_obj_loss', 'val_cls_loss'
        ]
        for metric in loss_breakdown_metrics:
            if metric in metrics and isinstance(metrics[metric], (int, float)):
                relevant_metrics[metric] = metrics[metric]
        
        # Skip multi-task loss components in Phase 1 (they shouldn't exist with simple YOLO loss)
        # Only include if they appear (for debugging purposes)
        if 'total_loss' in metrics and metrics['total_loss'] != metrics.get('train_loss', 0):
            relevant_metrics['total_loss'] = metrics['total_loss']
            logger.warning(f"Phase 1 showing total_loss ({metrics['total_loss']:.4f}) different from train_loss - this may indicate multi-task loss is being used incorrectly")
        
        logger.debug(f"Phase {phase_num}: Filtered {len(metrics)} metrics to {len(relevant_metrics)} core YOLO metrics")
        return relevant_metrics
    
    else:
        # Phase 2: Organize metrics into logical groups for better readability
        relevant_metrics = {}
        
        # Core multi-task training metrics (always shown first)
        core_metrics = ['train_loss', 'val_loss', 'val_map50', 'learning_rate', 'epoch']
        for metric in core_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # Multi-task loss components (specific to Phase 2)
        if 'total_loss' in metrics and metrics['total_loss'] != metrics.get('train_loss', 0):
            relevant_metrics['total_loss'] = metrics['total_loss']
            # Include uncertainty metrics if available
            uncertainty_metrics = ['layer_1_uncertainty', 'layer_2_uncertainty', 'layer_3_uncertainty']
            for metric in uncertainty_metrics:
                if metric in metrics:
                    relevant_metrics[metric] = metrics[metric]
        
        # Standard validation metrics (same as Phase 1 + mAP50)
        standard_val_metrics = [
            'val_precision', 'val_recall', 'val_f1', 'val_accuracy'
        ]
        for metric in standard_val_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # Training layer performance (show actual layer training progress)
        training_layer_metrics = [
            'layer_1_accuracy', 'layer_1_precision', 'layer_1_recall', 'layer_1_f1',
            'layer_2_accuracy', 'layer_2_precision', 'layer_2_recall', 'layer_2_f1',
            'layer_3_accuracy', 'layer_3_precision', 'layer_3_recall', 'layer_3_f1'
        ]
        for metric in training_layer_metrics:
            if metric in metrics:
                relevant_metrics[metric] = metrics[metric]
        
        # Skip confusing or redundant metrics:
        # - Detailed layer metrics (val_layer_X_accuracy/precision/recall/f1 when confusing)
        # - Redundant detection metrics (val_detection_map50 when val_map50 exists)
        # - High-precision metrics that show as 0.0000 and confuse users
        
        logger.debug(f"Phase {phase_num}: Organized {len(metrics)} metrics into {len(relevant_metrics)} key metrics")
        return relevant_metrics


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