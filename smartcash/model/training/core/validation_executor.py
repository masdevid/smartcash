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
        self.prediction_processor = PredictionProcessor(config)
        
        # Create optimal mAP calculator based on system capabilities
        self.map_calculator = create_optimal_map_calculator()
    
    def validate_epoch(self, val_loader, loss_manager, 
                      epoch: int, total_epochs: int, phase_num: int) -> Dict[str, float]:
        """
        Run validation for one epoch.
        
        Args:
            val_loader: Validation data loader
            loss_manager: Loss manager instance
            epoch: Current epoch number
            total_epochs: Total number of epochs
            phase_num: Current phase number
            
        Returns:
            Dictionary containing validation metrics
        """
        self.model.eval()
        running_val_loss = 0.0
        num_batches = len(val_loader)
        
        # Debug logging
        logger.info(f"Starting validation epoch {epoch+1} with {num_batches} batches")
        if num_batches == 0:
            logger.warning("Validation loader is empty!")
            return self._get_empty_validation_metrics()
        
        # Initialize collectors
        metrics = {}
        all_predictions = {}
        all_targets = {}
        
        # Start batch tracking for validation
        self.progress_tracker.start_batch_tracking(num_batches)
        
        # Process all validation batches
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                batch_metrics = self._process_validation_batch(
                    images, targets, loss_manager, batch_idx, num_batches, 
                    phase_num, all_predictions, all_targets
                )
                
                running_val_loss += batch_metrics['loss']
                
                # Update batch progress
                if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                    avg_loss = running_val_loss / (batch_idx + 1)
                    self.progress_tracker.update_batch_progress(
                        batch_idx + 1, num_batches,
                        f"Validation batch {batch_idx + 1}/{num_batches}",
                        loss=avg_loss,
                        epoch=epoch + 1
                    )
        
        # Complete batch tracking
        self.progress_tracker.complete_batch_tracking()
        
        # Compute final metrics
        return self._compute_final_metrics(
            running_val_loss, num_batches, all_predictions, all_targets
        )
    
    def _process_validation_batch(self, images, targets, loss_manager, batch_idx, num_batches,
                                phase_num, all_predictions, all_targets):
        """Process a single validation batch."""
        if batch_idx == 0 or batch_idx % 10 == 0:  # Log first and every 10th batch
            logger.info(f"Processing validation batch {batch_idx+1}/{num_batches}, images: {images.shape}, targets: {targets.shape}")
        
        device = next(self.model.parameters()).device
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        
        # Get model predictions
        predictions = self.model(images)
        if batch_idx == 0:  # Only log on first batch to reduce noise
            logger.info(f"Model predictions type: {type(predictions)}, structure: {type(predictions).__name__}")
            if isinstance(predictions, (list, tuple)):
                logger.info(f"  Predictions list length: {len(predictions)}")
                if len(predictions) > 0:
                    logger.info(f"  First prediction type: {type(predictions[0])}, shape: {getattr(predictions[0], 'shape', 'N/A')}")
        
        # Normalize predictions format
        predictions = self.prediction_processor.normalize_validation_predictions(
            predictions, phase_num, batch_idx
        )
        
        # Compute loss
        try:
            loss, loss_breakdown = loss_manager.compute_loss(predictions, targets, images.shape[-1])
            
            if batch_idx == 0:
                logger.info(f"First batch loss: {loss.item():.4f}, loss_breakdown keys: {list(loss_breakdown.keys())}")
                
        except Exception as e:
            logger.error(f"Error computing loss in validation batch {batch_idx}: {e}")
            return {'loss': 0.0}
        
        # Collect predictions and targets for metrics computation
        self._collect_predictions_and_targets(
            predictions, targets, images, device, batch_idx, all_predictions, all_targets
        )
        
        return {'loss': loss.item(), 'loss_breakdown': loss_breakdown}
    
    def _collect_predictions_and_targets(self, predictions, targets, images, device, batch_idx,
                                       all_predictions, all_targets):
        """Collect predictions and targets for metrics computation."""
        try:
            for layer_name, layer_preds in predictions.items():
                if layer_name not in all_predictions:
                    all_predictions[layer_name] = []
                    all_targets[layer_name] = []
                
                # Process predictions for mAP and classification metrics
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
                
                if batch_idx == 0:
                    logger.info(f"Layer {layer_name}: prediction shape {layer_output.shape}, target shape {layer_targets.shape}")
                    
        except Exception as e:
            logger.error(f"Error processing predictions in batch {batch_idx}: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    
    def _compute_final_metrics(self, running_val_loss, num_batches, all_predictions, all_targets):
        """Compute final validation metrics."""
        # Base metrics
        base_metrics = {
            'val_loss': running_val_loss / num_batches if num_batches > 0 else 0.0,
            'val_map50': 0.0,
            'val_map50_95': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0
        }
        
        # Compute mAP metrics
        map_metrics = self.map_calculator.compute_final_map()
        base_metrics.update(map_metrics)
        
        # Compute classification metrics
        computed_metrics = self._compute_classification_metrics(all_predictions, all_targets)
        if computed_metrics:
            # Average metrics across layers and update base metrics
            self._update_with_classification_metrics(base_metrics, computed_metrics)
            # Include individual layer metrics
            base_metrics.update(computed_metrics)
        
        logger.info(f"Validation completed: {num_batches} batches processed")
        logger.info(f"Final metrics: mAP@0.5={base_metrics['val_map50']:.4f}, accuracy={base_metrics['val_accuracy']:.4f}")
        
        return base_metrics
    
    def _compute_classification_metrics(self, all_predictions, all_targets):
        """Compute classification metrics from collected predictions and targets."""
        computed_metrics = {}
        if all_predictions and all_targets:
            # Concatenate all predictions and targets
            final_predictions = {}
            final_targets = {}
            
            for layer_name in all_predictions.keys():
                if all_predictions[layer_name] and all_targets[layer_name]:
                    try:
                        final_predictions[layer_name] = torch.cat(all_predictions[layer_name], dim=0)
                        final_targets[layer_name] = torch.cat(all_targets[layer_name], dim=0)
                    except Exception as e:
                        logger.debug(f"Error concatenating {layer_name} data: {e}")
                        continue
            
            # Calculate metrics using the metrics utils
            if final_predictions and final_targets:
                computed_metrics = calculate_multilayer_metrics(final_predictions, final_targets)
                logger.info(f"Computed validation metrics: {computed_metrics}")
        
        return computed_metrics
    
    def _update_with_classification_metrics(self, base_metrics, computed_metrics):
        """Update base metrics with computed classification metrics."""
        # For multi-layer models, average the metrics across layers
        layer_count = 0
        total_accuracy = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        
        for key, value in computed_metrics.items():
            if 'accuracy' in key:
                total_accuracy += value
                layer_count += 1
            elif 'precision' in key:
                total_precision += value
            elif 'recall' in key:
                total_recall += value
            elif 'f1' in key:
                total_f1 += value
        
        # Update validation metrics with averages
        if layer_count > 0:
            base_metrics['val_accuracy'] = total_accuracy / layer_count
            base_metrics['val_precision'] = total_precision / layer_count
            base_metrics['val_recall'] = total_recall / layer_count
            base_metrics['val_f1'] = total_f1 / layer_count
    
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