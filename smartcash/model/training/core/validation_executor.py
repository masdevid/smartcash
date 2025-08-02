#!/usr/bin/env python3
"""
Validation execution for the unified training pipeline.

This module handles validation epoch execution, metrics computation,
and mAP calculation.
"""

import torch
from typing import Dict

from smartcash.common.logger import get_logger
from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics
from .prediction_processor import PredictionProcessor
from .yolov5_map_calculator import create_yolov5_map_calculator

logger = get_logger(__name__)


class ValidationExecutor:
    """Handles validation epoch execution and metrics computation."""
    
    def __init__(self, model, config, progress_tracker):
        """
        Initialize validation executor.
        
        Args:
            model: PyTorch model
            config: Training configuration
            progress_tracker: Progress tracking instance
        """
        self.model = model
        self.config = config
        self.progress_tracker = progress_tracker
        self.prediction_processor = PredictionProcessor(config, model)
        
        # Initialize YOLOv5-based mAP calculator and metrics configuration
        num_classes = config.get('model', {}).get('num_classes', 7)
        self.map_calculator = create_yolov5_map_calculator(
            num_classes=num_classes,
            conf_thres=0.005,  # Very low threshold for early training with new anchors
            iou_thres=0.03   # AGGRESSIVE: Very low threshold for scale learning phase
        )
        
        # Configuration for metrics calculation method - check both locations
        validation_config = config.get('training', {}).get('validation', {})
        # Also check top-level config (set by ConfigurationBuilder)
        nested_yolov5 = validation_config.get('use_yolov5_builtin_metrics', False)
        toplevel_yolov5 = config.get('use_yolov5_builtin_metrics', False)
        
        # Debug logging to track configuration
        logger.debug(f"YOLOv5 config debug:")
        logger.debug(f"  • Nested path (training.validation): {nested_yolov5}")
        logger.debug(f"  • Top-level path: {toplevel_yolov5}")
        logger.debug(f"  • Full config keys: {list(config.keys())}")
        if 'training' in config:
            logger.debug(f"  • Training config keys: {list(config['training'].keys())}")
            if 'validation' in config['training']:
                logger.debug(f"  • Validation config keys: {list(config['training']['validation'].keys())}")
        
        self.use_yolov5_metrics = nested_yolov5 or toplevel_yolov5
        self.use_hierarchical_metrics = (
            validation_config.get('use_hierarchical_metrics', True) and 
            config.get('use_hierarchical_metrics', True)
        )
        
        logger.info(f"Validation metrics configuration:")
        logger.info(f"  • YOLOv5 mAP calculator: {num_classes} classes")
        logger.info(f"  • Use YOLOv5 built-in metrics: {self.use_yolov5_metrics}")
        logger.info(f"  • Use hierarchical metrics: {self.use_hierarchical_metrics}")
        
        if self.use_yolov5_metrics and not self.map_calculator.yolov5_available:
            logger.warning("YOLOv5 metrics requested but YOLOv5 not available - falling back to hierarchical metrics")
            self.use_yolov5_metrics = False
    
    def validate_epoch(self, val_loader, loss_manager, 
                      epoch: int, total_epochs: int, phase_num: int, display_epoch: int = None) -> Dict[str, float]:
        """
        Run validation for one epoch.
        
        Args:
            val_loader: Validation data loader
            loss_manager: Loss manager instance
            epoch: Current epoch number (0-based)
            total_epochs: Total number of epochs
            phase_num: Current phase number
            display_epoch: Display epoch number (1-based, for progress/logging)
            
        Returns:
            Dictionary containing validation metrics
        """
        # Optimized model switching to eval mode with reduced overhead
        self._switch_to_eval_mode()
        
        running_val_loss = 0.0
        num_batches = len(val_loader)
        
        # Calculate display epoch if not provided
        if display_epoch is None:
            display_epoch = epoch + 1
        
        # Reset mAP calculator for new validation epoch
        self.map_calculator.reset()
        
        # Debug logging with optimization info
        logger.debug(f"Starting validation epoch {display_epoch} with {num_batches} batches")
        logger.debug(f"⚡ YOLOv5 mAP calculation enabled for Phase {phase_num}")
        if num_batches == 0:
            logger.warning("Validation loader is empty!")
            return self._get_empty_validation_metrics()
        
        # Log validation batch size information for first epoch
        if epoch == 0:
            first_batch = next(iter(val_loader))
            actual_batch_size = first_batch[0].shape[0] if len(first_batch) > 0 else "unknown"
            logger.info(f"📊 Validation Batch Configuration:")
            logger.info(f"   • Configured Batch Size: {val_loader.batch_size}")
            logger.info(f"   • Actual Batch Size: {actual_batch_size}")
            logger.info(f"   • Total Batches: {num_batches}")
            logger.info(f"   • Samples per Epoch: ~{num_batches * val_loader.batch_size}")
        
        # Initialize collectors
        all_predictions = {}
        all_targets = {}
        all_loss_breakdowns = []  # Collect loss breakdowns for aggregation
        
        # Start batch tracking for validation
        self.progress_tracker.start_batch_tracking(num_batches)
        
        # Process all validation batches with optimizations
        with torch.no_grad():
            
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Check for shutdown signal every few batches for responsive interruption
                if batch_idx % 10 == 0:  # Check every 10 batches
                    from smartcash.model.training.utils.signal_handler import is_shutdown_requested
                    if is_shutdown_requested():
                        logger.info("🛑 Shutdown requested during validation batch processing")
                        break
                
                batch_metrics = self._process_validation_batch(
                    images, targets, loss_manager, batch_idx, num_batches, 
                    phase_num, all_predictions, all_targets
                )
                
                running_val_loss += batch_metrics['loss']
                
                # Collect loss breakdown for aggregation
                if 'loss_breakdown' in batch_metrics and batch_metrics['loss_breakdown']:
                    all_loss_breakdowns.append(batch_metrics['loss_breakdown'])
                
                # Optimized progress updates - reduce frequency for better performance
                update_freq = max(1, num_batches // 10)  # Update 10 times max per validation
                if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                    avg_loss = running_val_loss / (batch_idx + 1)
                    self.progress_tracker.update_batch_progress(
                        batch_idx + 1, num_batches,
                        f"Validation batch {batch_idx + 1}/{num_batches}",
                        loss=avg_loss,
                        epoch=display_epoch
                    )
                
                # Optional: Skip detailed metrics computation for some batches to speed up
                # Can be enabled with a config flag for faster validation
                if hasattr(self.config, 'fast_validation') and self.config.get('fast_validation', False):
                    if batch_idx > 0 and batch_idx % 3 != 0:  # Skip 2/3 of batches for mAP
                        continue
        
        # Complete batch tracking
        self.progress_tracker.complete_batch_tracking()
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics(
            running_val_loss, num_batches, all_predictions, all_targets, phase_num
        )
        
        # Aggregate loss breakdowns and add to final metrics
        if all_loss_breakdowns:
            aggregated_loss_breakdown = self._aggregate_loss_breakdowns(all_loss_breakdowns)
            final_metrics['loss_breakdown'] = aggregated_loss_breakdown
            logger.debug(f"Added aggregated loss breakdown with {len(aggregated_loss_breakdown)} components")
        
        return final_metrics
    
    def _process_validation_batch(self, images, targets, loss_manager, batch_idx, num_batches,
                                phase_num, all_predictions, all_targets):
        """Process a single validation batch."""
        if batch_idx == 0 or batch_idx % 10 == 0:  # Log first and every 10th batch
            logger.debug(f"Processing validation batch {batch_idx+1}/{num_batches}, images: {images.shape}, targets: {targets.shape}")
        
        device = next(self.model.parameters()).device
        
        # Optimized non-blocking transfers
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # For first batch, ensure transfer is complete to avoid later sync overhead
        if batch_idx == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Get model predictions
        predictions = self.model(images)
        
        if batch_idx == 0:  # Only log on first batch to reduce noise
            logger.debug(f"Model predictions type: {type(predictions)}, structure: {type(predictions).__name__}")
            if isinstance(predictions, (list, tuple)):
                logger.debug(f"  Predictions list length: {len(predictions)}")
                if len(predictions) > 0:
                    logger.debug(f"  First prediction type: {type(predictions[0])}, shape: {getattr(predictions[0], 'shape', 'N/A')}")
        
        # Normalize predictions format
        predictions = self.prediction_processor.normalize_validation_predictions(
            predictions, phase_num, batch_idx
        )
        
        
        # Compute loss
        try:
            loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
            
            # Debug: Check if loss is zero
            if loss.item() == 0.0:
                logger.warning(f"Validation batch {batch_idx}: Loss is zero. Targets shape: {targets.shape}, Predictions keys: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
            
            if batch_idx == 0:
                logger.debug(f"First batch loss: {loss.item():.4f}, loss_breakdown keys: {list(loss_breakdown.keys())}")
                
        except Exception as e:
            logger.error(f"Error computing loss in validation batch {batch_idx}: {e}")
            return {'loss': 0.0}
        
        # Collect predictions and targets for metrics computation
        self._collect_predictions_and_targets(
            predictions, targets, images, device, batch_idx, all_predictions, all_targets, phase_num
        )
        
        # Update mAP calculator with batch data for object detection metrics
        try:
            # Convert predictions to format expected by YOLOv5 mAP calculator
            # YOLOv5 expects predictions in [batch, detections, 6] format: [x, y, w, h, conf, class]
            # and targets in [num_targets, 6] format: [batch_idx, class, x, y, w, h]
            
            # Debug: Log raw prediction data (only when debug logging is enabled)
            if batch_idx == 0:
                logger.debug(f"🔍 RAW PREDICTIONS DEBUG - Batch {batch_idx}:")
                if isinstance(predictions, dict):
                    for key, value in predictions.items():
                        if hasattr(value, 'shape'):
                            logger.debug(f"  • {key}: {value.shape}")
                else:
                    logger.debug(f"  • predictions type: {type(predictions)}")
                    if hasattr(predictions, 'shape'):
                        logger.debug(f"  • predictions shape: {predictions.shape}")
                
                logger.debug(f"  • targets shape: {targets.shape}")
                logger.debug(f"  • targets sample: {targets[:3] if targets.shape[0] > 0 else 'empty'}")
            
            map_predictions = self._prepare_map_predictions(predictions, images.shape[0])
            map_targets = self._prepare_map_targets(targets)
            
            # Debug: Log conversion results (only when debug logging is enabled)
            if batch_idx == 0:
                logger.debug(f"🔄 CONVERSION RESULTS:")
                logger.debug(f"  • map_predictions: {map_predictions.shape if map_predictions is not None else 'None'}")
                logger.debug(f"  • map_targets: {map_targets.shape if map_targets is not None else 'None'}")
                if map_predictions is not None and map_predictions.numel() > 0:
                    logger.debug(f"  • prediction sample: {map_predictions[0, 0] if map_predictions.shape[1] > 0 else 'no detections'}")
                if map_targets is not None and map_targets.numel() > 0:
                    logger.debug(f"  • target sample: {map_targets[0] if map_targets.shape[0] > 0 else 'no targets'}")
            
            if map_predictions is not None and map_targets is not None:
                # Debug: Log detailed mAP data for first batch (only when debug logging is enabled)
                if batch_idx == 0:
                    logger.debug(f"🔍 mAP DEBUG - Batch {batch_idx}:")
                    logger.debug(f"  • Predictions shape: {map_predictions.shape}")
                    logger.debug(f"  • Targets shape: {map_targets.shape}")
                    if map_predictions.numel() > 0:
                        conf_scores = map_predictions[:, :, 4] if map_predictions.dim() == 3 else map_predictions[:, 4]
                        logger.debug(f"  • Confidence range: {conf_scores.min():.6f} - {conf_scores.max():.6f}")
                        logger.debug(f"  • Non-zero confidences: {(conf_scores > 0.01).sum().item()}/{conf_scores.numel()}")
                        logger.debug(f"  • Classes present: {map_predictions[:, :, 5].unique() if map_predictions.dim() == 3 else map_predictions[:, 5].unique()}")
                
                self.map_calculator.update(map_predictions, map_targets)
                
                if batch_idx == 0:
                    logger.debug(f"mAP data - Predictions: {map_predictions.shape}, Targets: {map_targets.shape}")
            
        except Exception as e:
            logger.warning(f"Error updating mAP calculator for batch {batch_idx}: {e}")
            if batch_idx == 0:  # Only log details for first batch to avoid spam
                logger.debug(f"mAP update error details: predictions type={type(predictions)}, targets shape={targets.shape}")
        
        # Debug: Track collection progress for missing layer metrics issue
        if batch_idx % 5 == 0:  # Every 5th batch
            total_collected = sum(len(preds) for preds in all_predictions.values()) if all_predictions else 0
            logger.debug(f"Batch {batch_idx}: Collected {total_collected} prediction batches across {len(all_predictions)} layers")
        
        return {'loss': loss.item(), 'loss_breakdown': loss_breakdown}
    
    def _collect_predictions_and_targets(self, predictions, targets, images, device, batch_idx,
                                       all_predictions, all_targets, phase_num):
        """Collect predictions and targets for metrics computation."""
        try:
            # Optimization: Only process active layers for current phase
            active_layers = self._get_active_layers_for_phase(phase_num)
            
            # Debug: Log what we're receiving vs what we expect
            if batch_idx == 0:
                logger.info(f"🔍 Phase {phase_num} - Received prediction layers: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
                logger.info(f"🔍 Phase {phase_num} - Expected active layers: {active_layers}")
            
            # Track successful layer processing for debugging static metrics
            processed_layers = 0
            failed_layers = []
            
            # Handle case where predictions might not be a dictionary
            if not isinstance(predictions, dict):
                logger.warning(f"Predictions is not a dict: {type(predictions)}, converting to dict with 'layer_1' key")
                predictions = {'layer_1': predictions}
            
            # Ensure we have at least layer_1 for Phase 1
            if phase_num == 1 and 'layer_1' not in predictions:
                logger.error(f"🚨 CRITICAL: Phase 1 missing layer_1 predictions! Available: {list(predictions.keys())}")
                # This will cause the static validation metrics issue
            
            for layer_name, layer_preds in predictions.items():
                # Skip inactive layers to reduce complexity O(L) -> O(L_active)
                if layer_name not in active_layers:
                    if batch_idx == 0:
                        logger.debug(f"Skipping inactive layer {layer_name} in Phase {phase_num}")
                    continue
                    
                if layer_name not in all_predictions:
                    all_predictions[layer_name] = []
                    all_targets[layer_name] = []
                
                try:
                    # Focus only on classification metrics (mAP processing removed)
                    
                    # Process for classification metrics
                    layer_output = self.prediction_processor.extract_classification_predictions(
                        layer_preds, images.shape[0], device
                    )
                    
                    # Extract target classes with phase-aware filtering
                    layer_targets = self.prediction_processor.extract_target_classes(
                        targets, images.shape[0], device, layer_name
                    )
                    
                    # Validate outputs before adding to collections
                    if layer_output is not None and layer_targets is not None:
                        if layer_output.numel() > 0 and layer_targets.numel() > 0:
                            all_predictions[layer_name].append(layer_output)
                            all_targets[layer_name].append(layer_targets)
                            processed_layers += 1
                            
                            # Log successful processing
                            if batch_idx <= 1:  # Log first two batches
                                logger.debug(f"✅ {layer_name} batch {batch_idx}: pred shape {layer_output.shape}, target shape {layer_targets.shape}")
                        else:
                            logger.warning(f"⚠️ {layer_name} batch {batch_idx}: Empty tensors - pred numel={layer_output.numel()}, target numel={layer_targets.numel()}")
                            # Try to create dummy tensors to avoid completely missing metrics
                            if layer_output.numel() == 0:
                                layer_output = torch.zeros(1, dtype=torch.long, device=device)
                            if layer_targets.numel() == 0:
                                layer_targets = torch.zeros(1, dtype=torch.long, device=device)
                            all_predictions[layer_name].append(layer_output)
                            all_targets[layer_name].append(layer_targets)
                            processed_layers += 1
                    else:
                        logger.warning(f"⚠️ {layer_name} batch {batch_idx}: None outputs - pred={layer_output is not None}, target={layer_targets is not None}")
                        # Create dummy tensors to avoid completely missing metrics
                        if layer_output is None:
                            layer_output = torch.zeros(1, dtype=torch.long, device=device)
                        if layer_targets is None:
                            layer_targets = torch.zeros(1, dtype=torch.long, device=device)
                        all_predictions[layer_name].append(layer_output)
                        all_targets[layer_name].append(layer_targets)
                        processed_layers += 1
                    
                    # Optimization: Memory-efficient accumulation (configurable)
                    COMPACT_FREQUENCY = self.config.get('training', {}).get('validation', {}).get('memory_compact_freq', 20)
                    if batch_idx > 0 and batch_idx % COMPACT_FREQUENCY == 0:
                        if len(all_predictions[layer_name]) > 1:
                            # Compact tensors to improve memory locality and reduce fragmentation
                            try:
                                all_predictions[layer_name] = [torch.cat(all_predictions[layer_name], dim=0)]
                                all_targets[layer_name] = [torch.cat(all_targets[layer_name], dim=0)]
                            except Exception as compact_e:
                                logger.warning(f"Warning compacting tensors for {layer_name}: {compact_e}")
                    
                    if batch_idx == 0:
                        logger.debug(f"Layer {layer_name}: prediction shape {layer_output.shape}, target shape {layer_targets.shape}")
                        logger.debug(f"  Prediction sample: {layer_output[:2] if layer_output.numel() > 0 else 'empty'}")
                        logger.debug(f"  Target sample: {layer_targets[:2] if layer_targets.numel() > 0 else 'empty'}")
                        
                except Exception as layer_e:
                    logger.error(f"❌ Error processing {layer_name} in batch {batch_idx}: {layer_e}")
                    failed_layers.append(f"{layer_name}(exception)")
                    import traceback
                    logger.error(f"   • Traceback: {traceback.format_exc()}")
            
            # Summary logging for debugging static metrics issue
            if batch_idx % 5 == 0 or failed_layers:  # Every 5th batch or if failures
                logger.info(f"📊 Batch {batch_idx} collection summary: {processed_layers} successful, {len(failed_layers)} failed")
                if failed_layers:
                    logger.error(f"   • Failed layers: {failed_layers}")
                    logger.error(f"   • This will contribute to missing layer_1_* metrics and static validation values!")
                    
        except Exception as e:
            logger.error(f"❌ Critical error processing predictions in batch {batch_idx}: {e}")
            logger.error(f"   • This will cause missing layer_1_* metrics and static validation fallback!")
            import traceback
            logger.error(f"   • Traceback: {traceback.format_exc()}")
    
    
    def _compute_final_metrics(self, running_val_loss, num_batches, all_predictions, all_targets, phase_num: int = None):
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
        logger.debug(f"📊 Phase {phase_num}: Computing YOLOv5 mAP metrics")
        try:
            map_metrics = self.map_calculator.compute_map()
            logger.info(f"YOLOv5 mAP Results - mAP@0.5: {map_metrics['map50']:.4f}, Precision: {map_metrics['precision']:.4f}, Recall: {map_metrics['recall']:.4f}")
            
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
    
    # _should_compute_map_metrics method removed - mAP computation completely disabled
    
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
        
        # Note: Old constraint checking removed - now handled by research metrics system
        # Phase constraint validation is now handled in the research metrics standardization
    
    def _get_active_layers_for_phase(self, phase_num: int) -> list:
        """Get list of active layers for the given phase."""
        if phase_num == 1:
            return ['layer_1']  # Phase 1: only layer_1 is active
        elif phase_num == 2:
            return ['layer_1', 'layer_2', 'layer_3']  # Phase 2: all layers are active
        else:
            # Default to all layers for unknown phases or single-phase mode
            return ['layer_1', 'layer_2', 'layer_3']
    
    def _extract_layer_name_from_metric(self, metric_key: str) -> str:
        """Extract layer name from metric key (e.g., 'layer_1_accuracy' -> 'layer_1')."""
        if 'layer_1' in metric_key:
            return 'layer_1'
        elif 'layer_2' in metric_key:
            return 'layer_2'
        elif 'layer_3' in metric_key:
            return 'layer_3'
        return ''
    
    def _get_empty_validation_metrics(self):
        """Get empty validation metrics when no data is available."""
        return {
            'val_loss': 0.0,
            'val_map50': 0.0,
            'val_map50_95': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0
        }
    
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
            
            logger.info(f"YOLOv5 built-in metrics:")
            logger.info(f"  • Precision: {map_results['precision']:.4f}")
            logger.info(f"  • Recall: {map_results['recall']:.4f}")
            logger.info(f"  • F1: {map_results['f1']:.4f}")
            logger.info(f"  • mAP@0.5: {map_results['map50']:.4f}")
            
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

    def _prepare_map_predictions(self, predictions, batch_size=None):
        """
        Convert SmartCash model predictions to YOLOv5 mAP format.
        
        Args:
            predictions: Model predictions (dict or tensor)
            batch_size: Batch size
            
        Returns:
            Tensor in YOLOv5 format [batch, detections, 6] where each detection is [x, y, w, h, conf, class]
        """
        try:
            # For object detection, we primarily use layer_1 predictions (main banknote detection)
            if isinstance(predictions, dict):
                if 'layer_1' in predictions:
                    layer_1_preds = predictions['layer_1']
                else:
                    # Fallback to first available layer
                    first_key = next(iter(predictions.keys()))
                    layer_1_preds = predictions[first_key]
                    logger.debug(f"Using {first_key} predictions for mAP (layer_1 not found)")
            else:
                layer_1_preds = predictions
            
            # Handle different prediction formats
            if isinstance(layer_1_preds, (list, tuple)):
                layer_1_preds = layer_1_preds[0] if len(layer_1_preds) > 0 else None
            
            if layer_1_preds is None:
                return None
            
            # Convert to tensor if not already
            if not isinstance(layer_1_preds, torch.Tensor):
                return None
            
            # Handle YOLOv5 feature map format: [batch, anchors, height, width, features]
            if layer_1_preds.dim() == 5:
                batch_size, num_anchors, height, width, num_features = layer_1_preds.shape
                logger.debug(f"Converting YOLOv5 feature map: {layer_1_preds.shape} -> [batch, detections, features]")
                
                # Reshape to [batch, detections, features] where detections = anchors * height * width
                layer_1_preds = layer_1_preds.permute(0, 1, 2, 3, 4).contiguous()
                layer_1_preds = layer_1_preds.view(batch_size, num_anchors * height * width, num_features)
                
                # Filter out predictions with low confidence (optional optimization)
                if num_features >= 5:  # Has confidence score
                    conf_scores = layer_1_preds[:, :, 4]  # Objectness/confidence at index 4
                    conf_threshold = 0.005  # Extra low threshold for early training with new anchors
                    
                    # Keep detections above threshold per batch
                    valid_detections = []
                    for batch_idx in range(batch_size):
                        batch_preds = layer_1_preds[batch_idx]
                        valid_mask = conf_scores[batch_idx] > conf_threshold
                        valid_preds = batch_preds[valid_mask]
                        
                        # Ensure we have some detections even if all are low confidence
                        if valid_preds.shape[0] == 0:
                            # Take top 100 predictions by confidence
                            top_k = min(100, batch_preds.shape[0])
                            _, top_indices = conf_scores[batch_idx].topk(top_k)
                            valid_preds = batch_preds[top_indices]
                        
                        valid_detections.append(valid_preds)
                    
                    # Pad to same length for batch processing
                    max_detections = max(pred.shape[0] for pred in valid_detections)
                    max_detections = min(max_detections, 300)  # Cap at 300 detections per image
                    
                    padded_preds = []
                    for batch_preds in valid_detections:
                        if batch_preds.shape[0] > max_detections:
                            # Take top predictions by confidence
                            conf = batch_preds[:, 4]
                            _, top_indices = conf.topk(max_detections)
                            batch_preds = batch_preds[top_indices]
                        elif batch_preds.shape[0] < max_detections:
                            # Pad with zeros
                            padding = torch.zeros(max_detections - batch_preds.shape[0], batch_preds.shape[1], 
                                                device=batch_preds.device, dtype=batch_preds.dtype)
                            batch_preds = torch.cat([batch_preds, padding], dim=0)
                        padded_preds.append(batch_preds)
                    
                    layer_1_preds = torch.stack(padded_preds, dim=0)
            
            # Ensure we have the right format [batch, detections, features] for other cases
            elif layer_1_preds.dim() == 2:
                # Add batch dimension if missing
                layer_1_preds = layer_1_preds.unsqueeze(0)
            
            elif layer_1_preds.dim() != 3:
                logger.warning(f"Cannot handle prediction tensor dimensions: {layer_1_preds.shape}")
                return None
            
            # Expected format: [batch, detections, features]
            # For YOLO: features should include [x, y, w, h, objectness, class_probs...]
            num_features = layer_1_preds.shape[-1]
            if num_features < 6:  # Need at least x,y,w,h,conf,class
                logger.debug(f"Prediction features too few for mAP: {num_features} < 6")
                return None
            
            # Check if we have class probabilities (features > 6 means we have class probs after objectness)
            if num_features > 6:
                # Format: [x, y, w, h, objectness, class0, class1, ..., classN]
                # Convert to: [x, y, w, h, conf, class_index]
                logger.debug(f"Converting {num_features} features to YOLO format with class probabilities")
                
                coords = layer_1_preds[:, :, :4]  # x, y, w, h
                objectness = layer_1_preds[:, :, 4]  # objectness score
                class_probs = layer_1_preds[:, :, 5:]  # class probabilities
                
                # Get maximum class and its probability
                max_class_prob, max_class = torch.max(class_probs, dim=-1)
                
                # Combine objectness with class probability for final confidence  
                # Clamp to valid range [0, 1] to prevent issues with mAP calculation
                confidence = torch.clamp(objectness * max_class_prob, 0.0, 1.0)
                
                # Create final prediction tensor [x, y, w, h, conf, class]
                map_preds = torch.cat([
                    coords,                      # x, y, w, h
                    confidence.unsqueeze(-1),    # confidence
                    max_class.float().unsqueeze(-1)  # class index
                ], dim=-1)
                
            else:
                # Already in the right format [x, y, w, h, conf, class] 
                map_preds = layer_1_preds.clone()
            
            # Legacy handling (should not be needed with above logic)
            if map_preds.shape[-1] > 6:
                # Take objectness as confidence and get max class
                objectness = map_preds[:, :, 4]  # Objectness score
                class_probs = map_preds[:, :, 5:]  # Class probabilities
                max_class = torch.argmax(class_probs, dim=-1).float()
                
                # Combine objectness with max class probability for confidence
                max_class_prob = torch.max(class_probs, dim=-1)[0]
                confidence = objectness * max_class_prob
                
                # Reconstruct as [x, y, w, h, conf, class]
                map_preds = torch.cat([
                    map_preds[:, :, :4],  # x, y, w, h
                    confidence.unsqueeze(-1),  # conf
                    max_class.unsqueeze(-1)   # class
                ], dim=-1)
            
            # CRITICAL FIX: Normalize coordinates to match target format
            # Predictions are in pixel coordinates, targets are normalized (0-1)
            # We need to normalize predictions to match
            
            # Get input image dimensions from batch_size parameter context
            # Assume standard input size for now - this should be passed as parameter
            input_height, input_width = 640, 640  # Standard YOLO input size
            
            # Normalize coordinates from pixel to 0-1 range
            map_preds[:, :, 0] = map_preds[:, :, 0] / input_width   # x center
            map_preds[:, :, 1] = map_preds[:, :, 1] / input_height  # y center  
            map_preds[:, :, 2] = map_preds[:, :, 2] / input_width   # width
            map_preds[:, :, 3] = map_preds[:, :, 3] / input_height  # height
            
            # Clamp to valid range [0, 1] to prevent out-of-bounds coordinates
            map_preds[:, :, :4] = torch.clamp(map_preds[:, :, :4], 0.0, 1.0)
            
            logger.debug(f"✅ Normalized predictions: coord range {map_preds[:, :, :4].min():.3f}-{map_preds[:, :, :4].max():.3f}")
            
            return map_preds.detach()
            
        except Exception as e:
            logger.warning(f"Error preparing mAP predictions: {e}")
            return None
    
    def _prepare_map_targets(self, targets):
        """
        Convert SmartCash targets to YOLOv5 mAP format.
        
        Args:
            targets: Ground truth targets [N, 6] format: [batch_idx, class, x, y, w, h]
            
        Returns:
            Tensor in YOLOv5 format [num_targets, 6]: [batch_idx, class, x, y, w, h]
        """
        try:
            if targets is None:
                return None
            
            # Convert to tensor if not already
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            
            # Ensure targets are on CPU for mAP calculation
            targets = targets.detach().cpu()
            
            if targets.numel() == 0:
                return torch.empty((0, 6))
            
            # Expected format: [batch_idx, class, x, y, w, h]
            if targets.dim() == 2 and targets.shape[-1] >= 6:
                map_targets = targets[:, :6].clone()  # Take first 6 columns
                
                # CRITICAL FIX: Filter out invalid class indices
                # SmartCash model has 7 classes (0-6), but targets may contain invalid classes (7-13)
                num_classes = 7  # SmartCash banknote classes: 0-6
                valid_mask = map_targets[:, 1] < num_classes  # Class index is at position 1
                
                if valid_mask.sum() == 0:
                    logger.debug("⚠️ No valid target classes found after filtering")
                    return torch.empty((0, 6))
                
                # Filter to only valid classes
                map_targets = map_targets[valid_mask]
                
                logger.debug(f"✅ Filtered targets: {targets.shape[0]} -> {map_targets.shape[0]} (kept classes: {map_targets[:, 1].unique()})")
                
                return map_targets
            
            logger.debug(f"Unexpected target format for mAP: {targets.shape}")
            return None
            
        except Exception as e:
            logger.warning(f"Error preparing mAP targets: {e}")
            return None

    def _switch_to_eval_mode(self):
        """
        Optimized model switching to evaluation mode.
        
        This method implements optimizations to reduce the time required
        to switch the model from training to evaluation mode.
        """
        import torch
        
        try:
            # Check if model is already in eval mode to avoid unnecessary work
            if not self.model.training:
                logger.debug("Model already in eval mode, skipping switch")
                return
            
            # Switch to eval mode
            self.model.eval()
            
            # For CUDA models, ensure synchronization happens now to avoid later sync overhead
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                torch.cuda.synchronize()
            
            # Enable fast validation optimizations if configured
            self._enable_fast_validation_mode()
            
            logger.debug("🔄 Model switched to eval mode (optimized)")
            
        except Exception as e:
            # Fallback to simple eval() if optimization fails
            logger.debug(f"⚠️ Optimized eval switch failed, using fallback: {e}")
            self.model.eval()
    
    def _enable_fast_validation_mode(self):
        """Enable fast validation mode with reduced metrics computation."""
        # This can be called when validation speed is more important than complete accuracy
        self.fast_validation_enabled = getattr(self, 'fast_validation_enabled', False)
        if hasattr(self.config, 'fast_validation') and self.config.get('fast_validation', False):
            self.fast_validation_enabled = True
            logger.info("⚡ Fast validation mode enabled - using sampling optimizations")
    
    def _aggregate_loss_breakdowns(self, all_loss_breakdowns: list) -> dict:
        """
        Aggregate loss breakdowns from all validation batches.
        
        Args:
            all_loss_breakdowns: List of loss breakdown dictionaries from each batch
            
        Returns:
            Dictionary containing averaged loss breakdown components
        """
        if not all_loss_breakdowns:
            return {}
        
        logger.debug(f"Aggregating {len(all_loss_breakdowns)} loss breakdowns")
        
        # Collect all unique keys across all breakdowns
        all_keys = set()
        for breakdown in all_loss_breakdowns:
            all_keys.update(breakdown.keys())
        
        aggregated = {}
        
        for key in all_keys:
            values = []
            for breakdown in all_loss_breakdowns:
                if key in breakdown:
                    value = breakdown[key]
                    # Convert tensor values to float
                    if hasattr(value, 'item'):
                        value = value.item()
                    elif hasattr(value, 'cpu'):
                        value = value.cpu().numpy()
                    
                    if isinstance(value, (int, float)):
                        values.append(value)
            
            # Average the values for this component
            if values:
                aggregated[key] = sum(values) / len(values)
                logger.debug(f"  {key}: averaged {len(values)} values = {aggregated[key]:.6f}")
        
        return aggregated