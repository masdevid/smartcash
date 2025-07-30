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
from .map_calculator_factory import create_optimal_map_calculator

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
        
        # Create optimal mAP calculator based on system capabilities
        self.map_calculator = create_optimal_map_calculator()
    
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
        
        # Debug logging with optimization info
        map_sample_rate = self.config.get('training', {}).get('validation', {}).get('map_sample_rate', 5)
        logger.debug(f"Starting validation epoch {display_epoch} with {num_batches} batches")
        logger.debug(f"⚡ Optimizations: mAP sampling every {map_sample_rate} batches, Phase {phase_num} layer filtering")
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
        return self._compute_final_metrics(
            running_val_loss, num_batches, all_predictions, all_targets, phase_num
        )
    
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
        
        # Debug: Log which layers are actually returned after normalization
        if batch_idx == 0:
            logger.info(f"🔍 Phase {phase_num} predictions after normalization: {list(predictions.keys()) if isinstance(predictions, dict) else 'not dict'}")
            if isinstance(predictions, dict):
                for layer_name in predictions.keys():
                    logger.info(f"    • {layer_name}: {type(predictions[layer_name])}")
            expected_layers = self._get_active_layers_for_phase(phase_num)
            logger.info(f"🎯 Expected for Phase {phase_num}: {expected_layers}")
        
        # Compute loss
        try:
            loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
            
            if batch_idx == 0:
                logger.debug(f"First batch loss: {loss.item():.4f}, loss_breakdown keys: {list(loss_breakdown.keys())}")
                
        except Exception as e:
            logger.error(f"Error computing loss in validation batch {batch_idx}: {e}")
            return {'loss': 0.0}
        
        # Collect predictions and targets for metrics computation
        self._collect_predictions_and_targets(
            predictions, targets, images, device, batch_idx, all_predictions, all_targets, phase_num
        )
        
        return {'loss': loss.item(), 'loss_breakdown': loss_breakdown}
    
    def _collect_predictions_and_targets(self, predictions, targets, images, device, batch_idx,
                                       all_predictions, all_targets, phase_num):
        """Collect predictions and targets for metrics computation."""
        try:
            # Optimization: Only process active layers for current phase
            active_layers = self._get_active_layers_for_phase(phase_num)
            
            # Debug: Log what we're receiving vs what we expect
            if batch_idx == 0:
                logger.info(f"🔍 Phase {phase_num} - Received prediction layers: {list(predictions.keys())}")
                logger.info(f"🔍 Phase {phase_num} - Expected active layers: {active_layers}")
            
            for layer_name, layer_preds in predictions.items():
                # Skip inactive layers to reduce complexity O(L) -> O(L_active)
                if layer_name not in active_layers:
                    if batch_idx == 0:
                        logger.debug(f"Skipping inactive layer {layer_name} in Phase {phase_num}")
                    continue
                    
                if layer_name not in all_predictions:
                    all_predictions[layer_name] = []
                    all_targets[layer_name] = []
                
                # Process predictions for mAP and classification metrics
                # Optimization: Sample mAP computation for speed (configurable)
                MAP_SAMPLE_RATE = self.config.get('training', {}).get('validation', {}).get('map_sample_rate', 5)
                if batch_idx % MAP_SAMPLE_RATE == 0 or batch_idx < 3:  # Always include first 3 batches
                    self.map_calculator.process_batch_for_map(
                        layer_preds, targets, images.shape, device, batch_idx
                    )
                
                # Process for classification metrics
                layer_output = self.prediction_processor.extract_classification_predictions(
                    layer_preds, images.shape[0], device
                )
                all_predictions[layer_name].append(layer_output)
                
                # Extract target classes
                layer_targets = self.prediction_processor.extract_target_classes(
                    targets, images.shape[0], device
                )
                all_targets[layer_name].append(layer_targets)
                
                # Optimization: Memory-efficient accumulation (configurable)
                COMPACT_FREQUENCY = self.config.get('training', {}).get('validation', {}).get('memory_compact_freq', 20)
                if batch_idx > 0 and batch_idx % COMPACT_FREQUENCY == 0:
                    if len(all_predictions[layer_name]) > 1:
                        # Compact tensors to improve memory locality and reduce fragmentation
                        all_predictions[layer_name] = [torch.cat(all_predictions[layer_name], dim=0)]
                        all_targets[layer_name] = [torch.cat(all_targets[layer_name], dim=0)]
                
                if batch_idx == 0:
                    logger.debug(f"Layer {layer_name}: prediction shape {layer_output.shape}, target shape {layer_targets.shape}")
                    logger.debug(f"  Prediction sample: {layer_output[:2] if layer_output.numel() > 0 else 'empty'}")
                    logger.debug(f"  Target sample: {layer_targets[:2] if layer_targets.numel() > 0 else 'empty'}")
                    
        except Exception as e:
            logger.error(f"Error processing predictions in batch {batch_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    
    def _compute_final_metrics(self, running_val_loss, num_batches, all_predictions, all_targets, phase_num: int = None):
        """Compute final validation metrics with research-focused naming."""
        from smartcash.model.training.utils.research_metrics import get_research_metrics_manager
        
        # Base metrics
        raw_metrics = {
            'loss': running_val_loss / num_batches if num_batches > 0 else 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'accuracy': 0.0
        }
        
        # Compute classification metrics (core research data)
        computed_metrics = self._compute_classification_metrics(all_predictions, all_targets)
        logger.debug(f"Computed classification metrics: {computed_metrics}")
        
        if computed_metrics:
            # Average metrics across active layers and update base metrics
            self._update_with_classification_metrics(raw_metrics, computed_metrics, phase_num)
            # Include individual layer metrics
            raw_metrics.update(computed_metrics)
        else:
            logger.warning("No classification metrics computed - check prediction/target data")
        
        # Only compute mAP if needed for research (avoid confusing metrics)
        if self._should_compute_map_metrics(phase_num):
            logger.debug(f"Computing mAP for Phase 2 additional detection information")
            map_metrics = self.map_calculator.compute_final_map()
            logger.debug(f"mAP metrics result: {map_metrics}")
            # Include map50 as additional detection information in Phase 2
            if 'val_map50' in map_metrics:
                raw_metrics['map50'] = map_metrics['val_map50']
                raw_metrics['detection_map50'] = map_metrics['val_map50']  # Also include with research name
                logger.info(f"📊 Phase 2 additional detection info: mAP@0.5 = {map_metrics['val_map50']:.4f}")
        
        # Convert to research-focused metrics with clear naming
        research_metrics_manager = get_research_metrics_manager()
        standardized_metrics = research_metrics_manager.standardize_metric_names(
            raw_metrics, phase_num, is_validation=True
        )
        
        # Log phase-appropriate metrics only
        research_metrics_manager.log_phase_appropriate_metrics(phase_num, standardized_metrics)
        
        return standardized_metrics
    
    def _should_compute_map_metrics(self, phase_num: int) -> bool:
        """Determine if mAP metrics should be computed for this phase."""
        # Only compute mAP in Phase 2 where detection performance is relevant
        # Phase 1 focuses on classification accuracy, Phase 2 on hierarchical detection
        return phase_num == 2
    
    def _compute_classification_metrics(self, all_predictions, all_targets):
        """Compute classification metrics from collected predictions and targets."""
        computed_metrics = {}
        logger.debug(f"Classification metrics input: predictions={len(all_predictions) if all_predictions else 0} layers, targets={len(all_targets) if all_targets else 0} layers")
        
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
                        
                        final_predictions[layer_name] = torch.cat(all_predictions[layer_name], dim=0)
                        final_targets[layer_name] = torch.cat(all_targets[layer_name], dim=0)
                        
                        logger.debug(f"Layer {layer_name}: concatenated pred shape {final_predictions[layer_name].shape}, target shape {final_targets[layer_name].shape}")
                        
                    except Exception as e:
                        logger.debug(f"Error concatenating {layer_name} data: {e}")
                        continue
                else:
                    logger.debug(f"Layer {layer_name}: empty predictions or targets")
            
            # Calculate metrics using the metrics utils
            if final_predictions and final_targets:
                logger.debug(f"Computing metrics for {len(final_predictions)} layers")
                computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
                logger.debug(f"Computed validation metrics: {computed_metrics}")
            else:
                logger.warning("No final predictions or targets available for metric computation")
        else:
            logger.warning("No predictions or targets collected during validation")
        
        return computed_metrics
    
    def _update_with_classification_metrics(self, base_metrics, computed_metrics, phase_num: int = None):
        """Update base metrics with computed classification metrics from active layers."""
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
        
        # Update base metrics - these will be converted by research metrics system later
        base_metrics['accuracy'] = sum(accuracy_metrics) / len(accuracy_metrics) if accuracy_metrics else 0.0
        base_metrics['precision'] = sum(precision_metrics) / len(precision_metrics) if precision_metrics else 0.0
        base_metrics['recall'] = sum(recall_metrics) / len(recall_metrics) if recall_metrics else 0.0
        base_metrics['f1'] = sum(f1_metrics) / len(f1_metrics) if f1_metrics else 0.0
        
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
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0
        }
    
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