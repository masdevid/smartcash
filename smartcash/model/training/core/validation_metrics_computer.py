#!/usr/bin/env python3
"""
Validation metrics computation for the unified training pipeline.

This module handles computation of validation metrics including
classification metrics, mAP metrics, and research-focused standardization.
"""

import torch

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics

logger = get_logger(__name__)


class ValidationMetricsComputer:
    """Handles computation of validation metrics."""
    
    def __init__(self, model, config, map_calculator):
        """
        Initialize validation metrics computer.
        
        Args:
            model: PyTorch model
            config: Training configuration
            map_calculator: YOLOv5 mAP calculator instance
        """
        self.model = model
        self.config = config
        self.map_calculator = map_calculator
        
        # Cache for mAP results to avoid duplicate expensive computation
        self._cached_map_results = None
        
        # Both YOLOv5 hierarchical metrics and per-layer metrics are always used
    
    def compute_final_metrics(self, running_val_loss, num_batches, all_predictions, all_targets, phase_num: int = None):
        """Compute final validation metrics with research-focused naming."""
        from smartcash.model.training.utils.research_metrics import get_research_metrics_manager
        
        # Start with loss
        raw_metrics = {
            'loss': running_val_loss / num_batches if num_batches > 0 else 0.0
        }

        # Phase-aware metrics computation
        logger.info(f"PHASE {phase_num}: Computing validation metrics.")

        # Always compute per-layer classification metrics
        computed_metrics = self._compute_classification_metrics(all_predictions, all_targets)
        if computed_metrics:
            raw_metrics.update(computed_metrics)
            logger.debug(f"Computed per-layer metrics: {list(computed_metrics.keys())}")

        # Compute and add YOLOv5 mAP metrics FIRST (so they don't override layer metrics)
        try:
            logger.debug("Computing YOLOv5 mAP metrics...")
            if self._cached_map_results is None:
                self._cached_map_results = self.map_calculator.compute_map()
            
            map_metrics = {
                'map50': self._cached_map_results.get('map50', 0.0),
                'map50_95': self._cached_map_results.get('map50_95', 0.0),
                'map50_precision': self._cached_map_results.get('precision', 0.0),
                'map50_recall': self._cached_map_results.get('recall', 0.0),
                'map50_f1': self._cached_map_results.get('f1', 0.0),
                'map50_accuracy': self._cached_map_results.get('accuracy', 0.0)
            }
            raw_metrics.update(map_metrics)
            logger.debug("YOLOv5 mAP metrics computed and cached.")

        except Exception as e:
            logger.warning(f"Error computing YOLOv5 mAP metrics: {e}")
            # Set fallback mAP values
            raw_metrics.update({
                'map50': 0.0, 'map50_95': 0.0, 'map50_precision': 0.0, 
                'map50_recall': 0.0, 'map50_f1': 0.0, 'map50_accuracy': 0.0
            })

        # Update primary metrics (accuracy, precision, etc.) based on the current phase
        # This MUST come after YOLOv5 metrics to ensure layer metrics take precedence
        logger.debug(f"Before layer metrics update - raw_metrics keys: {list(raw_metrics.keys())}")
        if phase_num == 1:
            logger.debug(f"Phase 1 - Before update: val_accuracy={raw_metrics.get('accuracy', 'missing')}, layer_1_accuracy={computed_metrics.get('layer_1_accuracy', 'missing')}")
        
        self._update_with_classification_metrics(raw_metrics, computed_metrics, phase_num)
        
        if phase_num == 1:
            logger.debug(f"Phase 1 - After update: val_accuracy={raw_metrics.get('accuracy', 'missing')}, layer_1_accuracy={computed_metrics.get('layer_1_accuracy', 'missing')}")
            # Verify they match
            if abs(raw_metrics.get('accuracy', 0) - computed_metrics.get('layer_1_accuracy', 0)) > 0.0001:
                logger.warning(f"⚠️ Phase 1 metrics mismatch: val_accuracy={raw_metrics.get('accuracy')} vs layer_1_accuracy={computed_metrics.get('layer_1_accuracy')}")

        # Standardize metric names for research and logging
        research_metrics_manager = get_research_metrics_manager()
        standardized_metrics = research_metrics_manager.standardize_metric_names(
            raw_metrics, phase_num, is_validation=True
        )

        # Log a summary of the final, standardized metrics
        research_metrics_manager.log_phase_appropriate_metrics(phase_num, standardized_metrics)

        return standardized_metrics
    
    def _compute_classification_metrics(self, all_predictions, all_targets):
        """Compute classification metrics from collected predictions and targets."""
        computed_metrics = {}
        logger.debug(f"Classification metrics input: predictions={len(all_predictions) if all_predictions else 0} layers, targets={len(all_targets) if all_targets else 0} layers")
        
        # Debug: Check if this is the static validation issue
        if all_predictions:
            for layer_name, layer_preds in all_predictions.items():
                batch_count = len(layer_preds) if layer_preds else 0
                logger.debug(f"Layer {layer_name}: {batch_count} prediction batches")
                if batch_count == 0:
                    logger.warning(f"Layer {layer_name} has zero prediction batches - no metrics available")
        
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
                computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
                logger.info(f"Computed validation metrics for {len(final_predictions)} layers")
                logger.debug(f"Layer metrics computed: {list(computed_metrics.keys())}")
                if 'layer_1_accuracy' in computed_metrics:
                    logger.debug(f"layer_1_accuracy = {computed_metrics['layer_1_accuracy']:.6f}")
        
        return computed_metrics
    
    def _update_with_classification_metrics(self, base_metrics, computed_metrics, phase_num: int = None):
        """Update base metrics with computed classification metrics."""
        if not computed_metrics:
            # Fallback if no classification metrics are available
            base_metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            logger.warning("No classification metrics computed. Primary metrics set to 0.")
            return

        epsilon = 1e-6

        # Phase 1: Use Layer 1 metrics directly for primary validation.
        # This is because Phase 1 is focused on training the denomination detection layer.
        if phase_num == 1:
            base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', epsilon)
            base_metrics['precision'] = computed_metrics.get('layer_1_precision', epsilon)
            base_metrics['recall'] = computed_metrics.get('layer_1_recall', epsilon)
            base_metrics['f1'] = computed_metrics.get('layer_1_f1', epsilon)
            logger.debug("Phase 1: Primary validation metrics updated from Layer 1.")

        # Phase 2: Use Layer 1 metrics as the basis, as they are enhanced by hierarchical filtering.
        # The hierarchical logic in the mAP calculator already refines these predictions.
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

        # This case should not be reached in standard operation.
        else:
            error_msg = f"Unsupported training phase: {phase_num}. Metrics update skipped."
            logger.error(f"CRITICAL ERROR: {error_msg}")
            # As a fallback, use generic accuracy if available
            if 'accuracy' not in base_metrics:
                base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', 0.0)
    
    