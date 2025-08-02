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
        
        # Configuration for metrics calculation method
        validation_config = config.get('training', {}).get('validation', {})
        nested_yolov5 = validation_config.get('use_yolov5_builtin_metrics', False)
        toplevel_yolov5 = config.get('use_yolov5_builtin_metrics', False)
        
        self.use_yolov5_metrics = nested_yolov5 or toplevel_yolov5
        self.use_hierarchical_metrics = (
            validation_config.get('use_hierarchical_metrics', True) and 
            config.get('use_hierarchical_metrics', True)
        )
    
    def compute_final_metrics(self, running_val_loss, num_batches, all_predictions, all_targets, phase_num: int = None):
        """Compute final validation metrics with research-focused naming."""
        from smartcash.model.training.utils.research_metrics import get_research_metrics_manager
        
        # Base metrics - only set loss, let other metrics be computed properly
        raw_metrics = {
            'loss': running_val_loss / num_batches if num_batches > 0 else 0.0
        }
        
        # Debug: Ensure no static values are pre-loaded in base_metrics
        suspicious_values = {
            'accuracy': 0.0321,
            'precision': 0.0010,
            'recall': 0.0321,
            'f1': 0.0020
        }
        
        for metric_name, suspicious_value in suspicious_values.items():
            if metric_name in raw_metrics:
                if abs(raw_metrics[metric_name] - suspicious_value) < 0.0001:
                    logger.error(f"🚨 DETECTED SUSPICIOUS PRE-LOADED VALUE: {metric_name}={raw_metrics[metric_name]:.6f}")
                    logger.error(f"   • This matches the static pattern from debug.md!")
                    logger.error(f"   • Removing this value to force proper computation")
                    del raw_metrics[metric_name]
        
        # Compute classification metrics (core research data)
        computed_metrics = self._compute_classification_metrics(all_predictions, all_targets)
        logger.debug(f"Computed classification metrics: {computed_metrics}")
        
        # Debug: Check why validation metrics might be static
        if not computed_metrics:
            logger.warning(f"⚠️ No computed_metrics returned - validation metrics will remain static!")
            logger.warning(f"   • all_predictions keys: {list(all_predictions.keys()) if all_predictions else 'None'}")
            logger.warning(f"   • all_targets keys: {list(all_targets.keys()) if all_targets else 'None'}")
            if all_predictions:
                for layer_name, layer_preds in all_predictions.items():
                    pred_count = len(layer_preds) if layer_preds else 0
                    logger.warning(f"   • {layer_name}: {pred_count} prediction batches")
                    if layer_preds and len(layer_preds) > 0:
                        first_batch_shape = getattr(layer_preds[0], 'shape', 'no shape') if hasattr(layer_preds[0], 'shape') else 'not tensor'
                        logger.warning(f"     - First batch shape: {first_batch_shape}")
        elif len(computed_metrics) == 0:
            logger.warning(f"⚠️ Empty computed_metrics - validation metrics will remain static!")
        else:
            logger.info(f"✅ Computed {len(computed_metrics)} validation metrics: {list(computed_metrics.keys())}")
            # Check if computed metrics have the same values repeatedly (indicates static issue)
            first_accuracy = None
            for key, value in computed_metrics.items():
                if 'accuracy' in key:
                    if first_accuracy is None:
                        first_accuracy = value
                    elif abs(first_accuracy - value) < 0.0001:
                        logger.warning(f"⚠️ Potential static validation metrics - {key}={value:.6f} matches first accuracy")
                    break
        
        # Choose metrics computation method based on configuration
        if self.use_yolov5_metrics:
            logger.info(f"📊 Phase {phase_num}: Using YOLOv5 built-in metrics for validation")
            # Use YOLOv5 built-in metrics (includes accuracy, precision, recall, F1)
            yolo_metrics = self._compute_yolov5_builtin_metrics()
            raw_metrics.update(yolo_metrics)
            
            # Still preserve individual layer metrics if hierarchical is also enabled
            if self.use_hierarchical_metrics and computed_metrics:
                # Add layer-specific metrics alongside YOLOv5 metrics
                for key, value in computed_metrics.items():
                    if key.startswith('layer_'):
                        raw_metrics[key] = value
                logger.debug(f"Added hierarchical layer metrics alongside YOLOv5 metrics")
        
        elif self.use_hierarchical_metrics:
            logger.info(f"📊 Phase {phase_num}: Using hierarchical multi-layer metrics")
            if computed_metrics:
                # Include individual layer metrics first (preserve exact values)
                raw_metrics.update(computed_metrics)
                
                # Layer metrics preserved for research consistency
                    
                # Average metrics across active layers and add to base metrics (for legacy compatibility)
                # IMPORTANT: Don't let this overwrite the individual layer metrics
                base_metrics_backup = raw_metrics.copy()  # Backup the layer metrics
                self._update_with_classification_metrics(raw_metrics, computed_metrics, phase_num)
                
                # Restore individual layer metrics (ensure they're not overwritten by averaging)
                for key, value in base_metrics_backup.items():
                    if key.startswith('layer_'):
                        raw_metrics[key] = value
                        
                # Layer metrics restored successfully
            else:
                logger.warning("No classification metrics computed - check prediction/target data")
                # Set fallback values only if no computed metrics at all
                raw_metrics.update({
                    'precision': 0.0,
                    'recall': 0.0,
                    'f1': 0.0,
                    'accuracy': 0.0
                })
        
        else:
            logger.warning("No metrics computation method enabled - using fallback values")
            raw_metrics.update({
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'accuracy': 0.0
            })
        
        # Compute mAP metrics using YOLOv5 calculator
        try:
            map_metrics = self.map_calculator.compute_map()
            
            # Add mAP metrics to raw_metrics
            raw_metrics.update({
                'map50': map_metrics['map50'],
                'map50_95': map_metrics['map50_95'],  # Will be 0 for now
                'map_precision': map_metrics['precision'],
                'map_recall': map_metrics['recall'],
                'map_f1': map_metrics['f1']
            })
            
        except Exception as e:
            logger.warning(f"Error computing YOLOv5 mAP metrics: {e}")
            # Set fallback mAP values
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
            # Concatenate all predictions and targets
            final_predictions = {}
            final_targets = {}
            
            for layer_name in all_predictions.keys():
                if all_predictions[layer_name] and all_targets[layer_name]:
                    try:
                        pred_batches = len(all_predictions[layer_name])
                        target_batches = len(all_targets[layer_name])
                        logger.debug(f"Layer {layer_name}: {pred_batches} prediction batches, {target_batches} target batches")
                        
                        # Handle case where we might have empty tensors or single-element tensors
                        pred_tensors = [t for t in all_predictions[layer_name] if t.numel() > 0]
                        target_tensors = [t for t in all_targets[layer_name] if t.numel() > 0]
                        
                        if pred_tensors and target_tensors:
                            final_predictions[layer_name] = torch.cat(pred_tensors, dim=0)
                            final_targets[layer_name] = torch.cat(target_tensors, dim=0)
                            
                            logger.debug(f"Layer {layer_name}: concatenated pred shape {final_predictions[layer_name].shape}, target shape {final_targets[layer_name].shape}")
                        else:
                            # Create dummy tensors to avoid completely missing metrics
                            device = next(self.model.parameters()).device
                            final_predictions[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                            final_targets[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                            logger.warning(f"Layer {layer_name}: using dummy tensors for metrics computation")
                        
                    except Exception as e:
                        logger.error(f"❌ Error concatenating {layer_name} data: {e}")
                        logger.error(f"   • This will cause missing {layer_name}_* metrics!")
                        import traceback
                        logger.error(f"   • Traceback: {traceback.format_exc()}")
                        # Create dummy tensors as fallback
                        device = next(self.model.parameters()).device
                        final_predictions[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                        final_targets[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                        logger.warning(f"Layer {layer_name}: using dummy tensors as fallback")
                else:
                    logger.error(f"❌ Layer {layer_name}: empty predictions or targets - missing layer_1_* metrics incoming!")
                    if not all_predictions[layer_name]:
                        logger.error(f"   • No predictions collected for {layer_name}")
                    if not all_targets[layer_name]:
                        logger.error(f"   • No targets collected for {layer_name}")
                    # Create dummy tensors as fallback
                    device = next(self.model.parameters()).device
                    final_predictions[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                    final_targets[layer_name] = torch.zeros(1, dtype=torch.long, device=device)
                    logger.warning(f"Layer {layer_name}: using dummy tensors as fallback")
            
            # Calculate metrics using the metrics utils
            if final_predictions and final_targets:
                logger.debug(f"Computing metrics for {len(final_predictions)} layers")
                computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
                logger.info(f"✅ Computed validation metrics: {list(computed_metrics.keys())}")
                
                # Verify layer_1_* metrics are present for Phase 1
                has_layer_1_metrics = any(key.startswith('layer_1_') for key in computed_metrics.keys())
                if not has_layer_1_metrics:
                    logger.error(f"🚨 CRITICAL: No layer_1_* metrics in computed results!")
                    logger.error(f"   • Available metrics: {list(computed_metrics.keys())}")
                    logger.error(f"   • This will cause static validation metrics (val_accuracy=0.0321)")
                    # Add fallback metrics to avoid static values
                    computed_metrics['layer_1_accuracy'] = 0.0
                    computed_metrics['layer_1_precision'] = 0.0
                    computed_metrics['layer_1_recall'] = 0.0
                    computed_metrics['layer_1_f1'] = 0.0
                else:
                    layer_1_acc = computed_metrics.get('layer_1_accuracy', 'MISSING')
                    logger.info(f"✅ Found layer_1_accuracy = {layer_1_acc}")
                    
            else:
                logger.error("❌ No final predictions or targets available for metric computation - static metrics incoming!")
                logger.error(f"   • final_predictions keys: {list(final_predictions.keys()) if final_predictions else 'None'}")
                logger.error(f"   • final_targets keys: {list(final_targets.keys()) if final_targets else 'None'}")
                # Add fallback metrics to avoid static values
                device = next(self.model.parameters()).device
                computed_metrics['layer_1_accuracy'] = 0.0
                computed_metrics['layer_1_precision'] = 0.0
                computed_metrics['layer_1_recall'] = 0.0
                computed_metrics['layer_1_f1'] = 0.0
        else:
            logger.error("❌ No predictions or targets collected during validation - static metrics incoming!")
            # Add fallback metrics to avoid static values
            device = next(self.model.parameters()).device
            computed_metrics['layer_1_accuracy'] = 0.0
            computed_metrics['layer_1_precision'] = 0.0
            computed_metrics['layer_1_recall'] = 0.0
            computed_metrics['layer_1_f1'] = 0.0
        
        return computed_metrics
    
    def _update_with_classification_metrics(self, base_metrics, computed_metrics, phase_num: int = None):
        """Update base metrics with computed classification metrics from active layers."""
        # Debug: Check if we received computed metrics
        if not computed_metrics:
            logger.error(f"🚨 No computed_metrics provided to _update_with_classification_metrics!")
            logger.error(f"   • This will result in base_metrics having only loss and zeros for other metrics")
            logger.error(f"   • Research metrics will fall back to generic accuracy/precision (likely static)")
            # Set fallback zeros to avoid issues downstream
            base_metrics.update({
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            })
            return
        
        # Collect metrics by type from all returned layers (already filtered at model level)
        accuracy_metrics = []
        precision_metrics = []
        recall_metrics = []
        f1_metrics = []
        
        for key, value in computed_metrics.items():
            if 'accuracy' in key:
                accuracy_metrics.append(value)
            elif 'precision' in key:
                precision_metrics.append(value)
            elif 'recall' in key:
                recall_metrics.append(value)
            elif 'f1' in key:
                f1_metrics.append(value)
        
        # Hierarchical metric computation - Layer 1 is primary, others are auxiliary
        epsilon = 1e-6
        
        if phase_num == 1:
            # Phase 1: Use Layer 1 metrics directly (only layer_1 is training)
            base_metrics['accuracy'] = computed_metrics.get('layer_1_accuracy', epsilon)
            base_metrics['precision'] = computed_metrics.get('layer_1_precision', epsilon)
            base_metrics['recall'] = computed_metrics.get('layer_1_recall', epsilon)
            base_metrics['f1'] = computed_metrics.get('layer_1_f1', epsilon)
            logger.debug(f"Phase 1: Using layer_1 metrics directly (single-layer training)")
            
        elif phase_num == 2:
            # Phase 2: Hierarchical weighted combination
            # Layer 1 (70%): Primary denomination detection - main research goal
            # Layer 2 (20%): Denomination-specific features - supporting Layer 1
            # Layer 3 (10%): Common security features - general banknote validation
            
            layer_1_acc = computed_metrics.get('layer_1_accuracy', 0.0)
            layer_2_acc = computed_metrics.get('layer_2_accuracy', 0.0)
            layer_3_acc = computed_metrics.get('layer_3_accuracy', 0.0)
            
            layer_1_prec = computed_metrics.get('layer_1_precision', 0.0)
            layer_2_prec = computed_metrics.get('layer_2_precision', 0.0)
            layer_3_prec = computed_metrics.get('layer_3_precision', 0.0)
            
            layer_1_rec = computed_metrics.get('layer_1_recall', 0.0)
            layer_2_rec = computed_metrics.get('layer_2_recall', 0.0)
            layer_3_rec = computed_metrics.get('layer_3_recall', 0.0)
            
            layer_1_f1 = computed_metrics.get('layer_1_f1', 0.0)
            layer_2_f1 = computed_metrics.get('layer_2_f1', 0.0)
            layer_3_f1 = computed_metrics.get('layer_3_f1', 0.0)
            
            # Weighted hierarchical combination
            base_metrics['accuracy'] = max(epsilon, 0.7 * layer_1_acc + 0.2 * layer_2_acc + 0.1 * layer_3_acc)
            base_metrics['precision'] = max(epsilon, 0.7 * layer_1_prec + 0.2 * layer_2_prec + 0.1 * layer_3_prec)
            base_metrics['recall'] = max(epsilon, 0.7 * layer_1_rec + 0.2 * layer_2_rec + 0.1 * layer_3_rec)
            base_metrics['f1'] = max(epsilon, 0.7 * layer_1_f1 + 0.2 * layer_2_f1 + 0.1 * layer_3_f1)
            
            logger.info(f"Phase 2 Hierarchical Metrics: layer_1={layer_1_acc:.4f} (70%), layer_2={layer_2_acc:.4f} (20%), layer_3={layer_3_acc:.4f} (10%)")
            logger.info(f"Weighted val_accuracy: {base_metrics['accuracy']:.4f} (should reflect Layer 1 dominance)")
            
        else:
            # Fallback: Simple averaging for unknown phases
            logger.warning(f"Unknown phase {phase_num}, falling back to simple averaging")
            base_metrics['accuracy'] = max(epsilon, sum(accuracy_metrics) / len(accuracy_metrics)) if accuracy_metrics else epsilon
            base_metrics['precision'] = max(epsilon, sum(precision_metrics) / len(precision_metrics)) if precision_metrics else epsilon
            base_metrics['recall'] = max(epsilon, sum(recall_metrics) / len(recall_metrics)) if recall_metrics else epsilon
            base_metrics['f1'] = max(epsilon, sum(f1_metrics) / len(f1_metrics)) if f1_metrics else epsilon
        
        # Debug: Check if we're creating static metrics from the generic averaging
        if (abs(base_metrics['accuracy'] - 0.0321) < 0.0001 and 
            abs(base_metrics['precision'] - 0.0010) < 0.0001):
            logger.error(f"🚨 STATIC METRICS DETECTED in _update_with_classification_metrics!")
            logger.error(f"   • accuracy: {base_metrics['accuracy']:.6f} (suspicious)")
            logger.error(f"   • precision: {base_metrics['precision']:.6f} (suspicious)")
            logger.error(f"   • These values match the static pattern seen in debug.md")
            logger.error(f"   • Check if these are coming from old cached values")
        
        logger.debug(f"Phase {phase_num} computed metrics: {list(computed_metrics.keys())}")
        logger.debug(f"Collected accuracy metrics: {accuracy_metrics}")
        logger.debug(f"Aggregated validation metrics: accuracy={base_metrics['accuracy']:.4f}, precision={base_metrics['precision']:.4f}, recall={base_metrics['recall']:.4f}, f1={base_metrics['f1']:.4f}")
        logger.debug(f"Metrics count: accuracy={len(accuracy_metrics)}, precision={len(precision_metrics)}, recall={len(recall_metrics)}, f1={len(f1_metrics)}")
        
        # Log individual layer accuracies for comparison
        for key, value in computed_metrics.items():
            if 'accuracy' in key:
                logger.debug(f"Individual metric: {key} = {value:.4f}")
    
    def _compute_yolov5_builtin_metrics(self):
        """
        Compute validation metrics using YOLOv5's built-in evaluation.
        
        This uses the same statistics accumulated by the mAP calculator
        to provide YOLOv5's standard accuracy, precision, recall, and F1 scores.
        
        Returns:
            Dictionary with YOLOv5 computed metrics
        """
        try:
            # Get the computed mAP metrics which include precision, recall, F1
            map_results = self.map_calculator.compute_map()
            
            # For object detection, accuracy is often approximated as mAP or computed differently
            # We'll use mAP@0.5 as a proxy for overall detection accuracy
            detection_accuracy = map_results['map50']
            
            return {
                'accuracy': detection_accuracy,  # Use mAP@0.5 as detection accuracy
                'precision': map_results['precision'],
                'recall': map_results['recall'],
                'f1': map_results['f1']
            }
            
        except Exception as e:
            logger.warning(f"Error computing YOLOv5 built-in metrics: {e}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0
            }