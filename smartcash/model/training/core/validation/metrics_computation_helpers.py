#!/usr/bin/env python3
"""
Helper functions for validation metrics computation.

This module contains helper functions extracted from ValidationMetricsProcessor
to keep the main processor under 400 lines while maintaining functionality.
"""

import torch
from typing import Dict, List

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def compute_phase_aware_primary_metrics(base_metrics: Dict, computed_metrics: Dict, phase_num: int = None):
    """Update base metrics with computed classification metrics based on training phase."""
    if not computed_metrics:
        base_metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
        logger.warning("No classification metrics computed. Primary metrics set to 0.")
        return

    epsilon = 1e-6

    # Phase 1: Use Layer 1 metrics (denomination detection focus)
    if phase_num == 1:
        base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', epsilon)
        base_metrics['precision'] = computed_metrics.get('layer_1_precision', epsilon)
        base_metrics['recall'] = computed_metrics.get('layer_1_recall', epsilon)
        base_metrics['f1'] = computed_metrics.get('layer_1_f1', epsilon)
        logger.debug("Phase 1: Primary validation metrics updated from Layer 1.")

    # Phase 2: Use Layer 1 metrics as basis with hierarchical refinement
    elif phase_num == 2:
        layer_1_acc = computed_metrics.get('layer_1_accuracy', 0.0)
        layer_1_prec = computed_metrics.get('layer_1_precision', 0.0)
        layer_1_rec = computed_metrics.get('layer_1_recall', 0.0)
        layer_1_f1 = computed_metrics.get('layer_1_f1', 0.0)

        base_metrics['accuracy'] = max(epsilon, layer_1_acc)
        base_metrics['precision'] = max(epsilon, layer_1_prec)
        base_metrics['recall'] = max(epsilon, layer_1_rec)
        base_metrics['f1'] = max(epsilon, layer_1_f1)
        logger.info(f"Phase 2: Hierarchical metrics based on Layer 1: acc={layer_1_acc:.4f}")

    # Fallback for unexpected phases
    else:
        logger.error(f"Unsupported training phase: {phase_num}")
        base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', 0.0)


def standardize_validation_metrics_with_prefix(raw_metrics: Dict, computed_metrics: Dict, phase_num: int) -> Dict[str, float]:
    """Standardize validation metrics with val_ prefix and layer metrics."""
    standardized = {}
    
    # Add validation loss
    if 'loss' in raw_metrics:
        standardized['val_loss'] = raw_metrics['loss']
    
    # Add primary validation metrics
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        if metric in raw_metrics:
            standardized[f'val_{metric}'] = raw_metrics[metric]
    
    # Add mAP metrics with val_ prefix
    for metric in ['map50', 'map50_95', 'map50_precision', 'map50_recall', 'map50_f1', 'map50_accuracy']:
        if metric in raw_metrics:
            standardized[f'val_{metric}'] = raw_metrics[metric]
    
    # Add all layer-specific metrics with val_ prefix
    for key, value in computed_metrics.items():
        if key.startswith('layer_'):
            standardized[f'val_{key}'] = value
    
    return standardized


def log_validation_summary(phase_num: int, metrics: Dict[str, float]):
    """Log essential validation metrics summary."""
    val_loss = metrics.get('val_loss', 0.0)
    val_map50 = metrics.get('val_map50', 0.0)
    val_accuracy = metrics.get('val_accuracy', 0.0)
    
    if val_map50 > 0:
        logger.info(f"📊 Phase {phase_num}: Val Loss={val_loss:.4f}, mAP@0.5={val_map50:.4f}, Accuracy={val_accuracy:.4f}")
    else:
        logger.info(f"📊 Phase {phase_num}: Val Loss={val_loss:.4f}, Accuracy={val_accuracy:.4f}")


def compute_classification_metrics_from_tensors(all_predictions: Dict, all_targets: Dict) -> Dict[str, float]:
    """Compute classification metrics from collected predictions and targets."""
    computed_metrics = {}
    logger.debug(f"Classification metrics input: predictions={len(all_predictions) if all_predictions else 0} layers, targets={len(all_targets) if all_targets else 0} layers")
    
    if all_predictions and all_targets:
        # Concatenate predictions and targets by layer
        final_predictions = {}
        final_targets = {}
        
        for layer_name in all_predictions.keys():
            if all_predictions[layer_name] and all_targets[layer_name]:
                try:
                    pred_tensors = [t for t in all_predictions[layer_name] if t.numel() > 0]
                    target_tensors = [t for t in all_targets[layer_name] if t.numel() > 0]
                    
                    if pred_tensors and target_tensors:
                        final_predictions[layer_name] = torch.cat(pred_tensors, dim=0)
                        final_targets[layer_name] = torch.cat(target_tensors, dim=0)
                except Exception as e:
                    logger.warning(f"Error processing {layer_name}: {e}")
        
        # Calculate metrics if data available
        if final_predictions and final_targets:
            from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics
            computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
            logger.info(f"Computed validation metrics for {len(final_predictions)} layers")
    
    return computed_metrics