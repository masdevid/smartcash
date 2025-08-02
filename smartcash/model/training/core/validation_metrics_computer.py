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
        
        # Initialize base metrics
        raw_metrics = {
            'loss': running_val_loss / num_batches if num_batches > 0 else 0.0
        }
        
        # Compute classification metrics
        computed_metrics = self._compute_classification_metrics(all_predictions, all_targets)
        logger.debug(f"Computed classification metrics: {list(computed_metrics.keys()) if computed_metrics else 'None'}")
        
        # Compute validation metrics using both YOLOv5 hierarchical and per-layer metrics
        logger.info(f"📊 Phase {phase_num}: Computing hierarchical validation metrics")
        
        # Get YOLOv5 hierarchical metrics (mAP-based accuracy, precision, recall, F1)
        yolo_metrics = self._compute_yolov5_builtin_metrics()
        raw_metrics.update(yolo_metrics)
        
        # Add per-layer metrics for research analysis
        if computed_metrics:
            raw_metrics.update(computed_metrics)
            # Update primary metrics with phase-appropriate hierarchical approach
            self._update_with_classification_metrics(raw_metrics, computed_metrics, phase_num)
        else:
            logger.warning("No per-layer metrics computed")
            raw_metrics.update({'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'accuracy': 0.0})
        
        # Compute mAP metrics using YOLOv5 calculator (cached to avoid duplicate computation)
        try:
            logger.debug("🧮 Computing YOLOv5 mAP metrics...")
            import time
            start_time = time.time()
            
            self._cached_map_results = self.map_calculator.compute_map()
            
            computation_time = time.time() - start_time
            if computation_time > 2.0:  # Only log if it takes significant time
                logger.info(f"✅ YOLOv5 mAP metrics computed in {computation_time:.2f}s")
            else:
                logger.debug(f"✅ YOLOv5 mAP metrics computed in {computation_time:.2f}s")
            
            # Add mAP metrics to raw_metrics
            raw_metrics.update({
                'map50': self._cached_map_results['map50'],
                'map50_95': self._cached_map_results['map50_95'],  # Will be 0 for now
                'map_precision': self._cached_map_results['precision'],
                'map_recall': self._cached_map_results['recall'],
                'map_f1': self._cached_map_results['f1']
            })
            logger.debug("📊 YOLOv5 mAP metrics cached for reuse")
            
        except Exception as e:
            logger.warning(f"Error computing YOLOv5 mAP metrics: {e}")
            # Set fallback mAP values and cache them
            self._cached_map_results = {
                'map50': 0.0,
                'map50_95': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }
            raw_metrics.update({
                'map50': 0.0,
                'map50_95': 0.0,
                'map_precision': 0.0,
                'map_recall': 0.0,
                'map_f1': 0.0
            })
        
        # Convert to research-focused metrics with clear naming
        research_metrics_manager = get_research_metrics_manager()
        
        standardized_metrics = research_metrics_manager.standardize_metric_names(
            raw_metrics, phase_num, is_validation=True
        )
        
        # Log phase-appropriate metrics only
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
        
        return computed_metrics
    
    def _update_with_classification_metrics(self, base_metrics, computed_metrics, phase_num: int = None):
        """Update base metrics with computed classification metrics."""
        if not computed_metrics:
            base_metrics.update({'accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0})
            return
        
        # Hierarchical metric computation
        epsilon = 1e-6
        
        if phase_num == 1:
            # Phase 1: Standard processing - use Layer 1 metrics directly
            # No hierarchical filtering in Phase 1 (frozen backbone, single layer focus)
            base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', epsilon)
            base_metrics['precision'] = computed_metrics.get('layer_1_precision', epsilon)
            base_metrics['recall'] = computed_metrics.get('layer_1_recall', epsilon)
            base_metrics['f1'] = computed_metrics.get('layer_1_f1', epsilon)
            logger.debug(f"Phase 1: Standard processing - using layer_1 metrics directly")
            
        elif phase_num == 2:
            # Phase 2: Apply hierarchical validation approach
            # In Phase 2, hierarchical filtering is applied during prediction processing
            # The validation metrics (accuracy, precision, recall, F1) should focus on 
            # Layer 1 performance enhanced by hierarchical confidence from Layer 2 & 3
            
            # Get Layer 1 metrics (primary focus in hierarchical system)
            layer_1_acc = computed_metrics.get('layer_1_accuracy', 0.0)
            layer_1_prec = computed_metrics.get('layer_1_precision', 0.0)
            layer_1_rec = computed_metrics.get('layer_1_recall', 0.0)
            layer_1_f1 = computed_metrics.get('layer_1_f1', 0.0)
            
            # Phase 2 Hierarchical Approach:
            # Use Layer 1 metrics as primary since hierarchical confidence modulation
            # already enhances Layer 1 predictions using Layer 2 & 3 validation
            # This aligns with hierarchical mAP calculation where we evaluate Layer 1 performance
            base_metrics['accuracy'] = max(epsilon, layer_1_acc)
            base_metrics['precision'] = max(epsilon, layer_1_prec) 
            base_metrics['recall'] = max(epsilon, layer_1_rec)
            base_metrics['f1'] = max(epsilon, layer_1_f1)
            
            logger.info(f"Phase 2 Hierarchical: Layer 1 metrics (acc={layer_1_acc:.4f}, prec={layer_1_prec:.4f}, rec={layer_1_rec:.4f}, f1={layer_1_f1:.4f})")
            
        else:
            # Unknown phase - this should not happen in production
            error_msg = f"Unknown training phase: {phase_num}. Only Phase 1 and Phase 2 are supported."
            logger.error(f"🚨 CRITICAL ERROR: {error_msg}")
            raise ValueError(error_msg)
    
    def _compute_yolov5_builtin_metrics(self):
        """
        Compute validation metrics using YOLOv5's built-in evaluation with hierarchical filtering.
        
        This uses cached mAP results to avoid duplicate expensive computation.
        The hierarchical filtering was already applied during the mAP calculation.
        
        Returns:
            Dictionary with YOLOv5 computed metrics (hierarchically filtered)
        """
        try:
            # Use cached mAP results to avoid duplicate expensive computation
            if self._cached_map_results is None:
                logger.debug("No cached mAP results - computing fresh metrics")
                map_results = self.map_calculator.compute_map()
            else:
                logger.debug("🔄 Using cached YOLOv5 mAP results")
                map_results = self._cached_map_results
            
            # For object detection, accuracy is often approximated as mAP or computed differently
            # We'll use mAP@0.5 as a proxy for overall detection accuracy
            detection_accuracy = map_results['map50']
            
            # Log hierarchical metrics application
            logger.debug(f"YOLOv5 metrics with hierarchical filtering (cached):")
            logger.debug(f"  • Detection accuracy (mAP@0.5): {detection_accuracy:.6f}")
            logger.debug(f"  • Precision (hierarchical): {map_results['precision']:.6f}")
            logger.debug(f"  • Recall (hierarchical): {map_results['recall']:.6f}")
            logger.debug(f"  • F1 (hierarchical): {map_results['f1']:.6f}")
            
            return {
                'accuracy': detection_accuracy,  # Use mAP@0.5 as detection accuracy
                'precision': map_results['precision'],  # Already hierarchically filtered
                'recall': map_results['recall'],        # Already hierarchically filtered
                'f1': map_results['f1']                 # Already hierarchically filtered
            }
            
        except Exception as e:
            logger.warning(f"Error computing YOLOv5 built-in metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }