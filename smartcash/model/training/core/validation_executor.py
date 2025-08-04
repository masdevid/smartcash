#!/usr/bin/env python3
"""
Refactored validation execution for the unified training pipeline.

This module orchestrates validation epoch execution using SRP-compliant components
for better maintainability and separation of concerns.
"""

import torch
from typing import Dict

from smartcash.common.logger import get_logger
from .prediction_processor import PredictionProcessor
from .yolov5_map_calculator import create_yolov5_map_calculator
from .validation_batch_processor import ValidationBatchProcessor
from .validation_metrics_computer import ValidationMetricsComputer
from .validation_model_manager import ValidationModelManager
from .validation_map_processor import ValidationMapProcessor

# Pre-import signal handler to avoid blocking import during validation loop
try:
    from smartcash.model.training.utils.signal_handler import is_shutdown_requested
except ImportError:
    # Fallback if signal handler not available
    def is_shutdown_requested():
        return False

logger = get_logger(__name__)


class ValidationExecutor:
    """Orchestrates validation epoch execution using SRP-compliant components."""
    
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
        
        # Initialize core components
        self.prediction_processor = PredictionProcessor(config, model)
        
        # Initialize YOLOv5-based mAP calculator with enhanced debug verification
        num_classes = config.get('model', {}).get('num_classes', 7)
        debug_map = config.get('debug_map', False)
        
        # Extract training context from config for debug logging
        training_context = {
            'backbone': config.get('backbone', 'unknown'),
            'training_mode': config.get('training_mode', 'unknown'),
            'current_phase': getattr(model, 'current_phase', None) or 'unknown',
            'session_id': config.get('session_id', 'N/A'),
            'model_name': config.get('model', {}).get('model_name', 'N/A'),
            'layer_mode': config.get('model', {}).get('layer_mode', 'N/A'),
            'detection_layers': config.get('model', {}).get('detection_layers', 'N/A')
        }
        
        self.map_calculator = create_yolov5_map_calculator(
            num_classes=num_classes,
            conf_thres=0.005,  # Very low threshold for early training with new anchors
            iou_thres=0.03,   # AGGRESSIVE: Very low threshold for scale learning phase
            debug=debug_map,
            training_context=training_context
        )
        
        
        # Initialize SRP components
        self.batch_processor = ValidationBatchProcessor(model, config, self.prediction_processor)
        self.metrics_computer = ValidationMetricsComputer(model, config, self.map_calculator)
        self.model_manager = ValidationModelManager(model)
        self.map_processor = ValidationMapProcessor(self.map_calculator)
        
        logger.info(f"Validation metrics configuration:")
        logger.info(f"  • YOLOv5 mAP calculator: {num_classes} classes")
        logger.info(f"  • Using hierarchical validation (YOLOv5 + per-layer metrics)")
        
        if not self.map_calculator.yolov5_available:
            logger.warning("YOLOv5 not available - using fallback metrics")
        
    
    def validate_epoch(self, val_loader, loss_manager, 
                      epoch: int, phase_num: int, display_epoch: int = None) -> Dict[str, float]:
        """
        Run validation for one epoch.
        
        Args:
            val_loader: Validation data loader
            loss_manager: Loss manager instance
            epoch: Current epoch number (0-based)
            phase_num: Current phase number
            display_epoch: Display epoch number (1-based, for progress/logging)
            
        Returns:
            Dictionary containing validation metrics
        """
        # Switch model to evaluation mode
        self.model_manager.switch_to_eval_mode()
        
        running_val_loss = 0.0
        num_batches = len(val_loader)
        
        # Calculate display epoch if not provided
        if display_epoch is None:
            display_epoch = epoch + 1
        
        # Reset mAP calculator for new validation epoch
        self.map_calculator.reset()
        
        # Clear cached mAP results in metrics computer to ensure fresh computation
        self.metrics_computer._cached_map_results = None
        
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
        
        # Process all validation batches
        with torch.no_grad():
            for batch_idx, (images, targets) in enumerate(val_loader):
                # Check for shutdown signal every few batches for responsive interruption
                if batch_idx % 10 == 0:  # Check every 10 batches
                    if is_shutdown_requested():
                        logger.info("🛑 Shutdown requested during validation batch processing")
                        break
                
                # Process batch using batch processor
                batch_metrics = self.batch_processor.process_batch(
                    images, targets, loss_manager, batch_idx, num_batches, 
                    phase_num, all_predictions, all_targets
                )
                
                running_val_loss += batch_metrics['loss']
                
                # Collect loss breakdown for aggregation
                if 'loss_breakdown' in batch_metrics and batch_metrics['loss_breakdown']:
                    all_loss_breakdowns.append(batch_metrics['loss_breakdown'])
                
                # Update mAP calculator with batch data
                # Note: Using the latest predictions from batch processing
                if hasattr(self.prediction_processor, 'last_normalized_predictions') and self.prediction_processor.last_normalized_predictions is not None:
                    self.map_processor.update_map_calculator(
                        self.prediction_processor.last_normalized_predictions,
                        targets, images, batch_idx
                    )
                else:
                    # Fallback: Try to get predictions from batch processor if available
                    if hasattr(self.batch_processor, 'prediction_processor') and hasattr(self.batch_processor.prediction_processor, 'last_normalized_predictions'):
                        if self.batch_processor.prediction_processor.last_normalized_predictions is not None:
                            self.map_processor.update_map_calculator(
                                self.batch_processor.prediction_processor.last_normalized_predictions,
                                targets, images, batch_idx
                            )
                    else:
                        # Log the issue for debugging
                        if batch_idx == 0:  # Only log once per epoch
                            logger.warning(f"⚠️ No normalized predictions available for mAP calculation at batch {batch_idx}")
                            logger.debug(f"prediction_processor has last_normalized_predictions: {hasattr(self.prediction_processor, 'last_normalized_predictions')}")
                            if hasattr(self.prediction_processor, 'last_normalized_predictions'):
                                logger.debug(f"last_normalized_predictions is None: {self.prediction_processor.last_normalized_predictions is None}")
                
                # Progress updates - more responsive frequency
                # For responsive UX, update more frequently for larger datasets
                if num_batches <= 10:
                    update_freq = 1  # Update every batch for small sets
                elif num_batches <= 50:
                    update_freq = max(1, num_batches // 5)  # Update ~5 times 
                else:
                    update_freq = max(1, num_batches // 20)  # Update ~20 times for large sets
                
                if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                    avg_loss = running_val_loss / (batch_idx + 1)
                    self.progress_tracker.update_batch_progress(
                        batch_idx + 1, num_batches,
                        f"Validation batch {batch_idx + 1}/{num_batches}",
                        loss=avg_loss,
                        epoch=display_epoch
                    )
        
        # Complete batch tracking
        self.progress_tracker.complete_batch_tracking()
        
        # Add progress feedback for final metrics computation
        logger.info(f"📊 Computing final validation metrics for epoch {display_epoch}...")
        if self.progress_tracker.progress_callback:
            self.progress_tracker.progress_callback('batch', 100, 100, f"Computing validation metrics for epoch {display_epoch}...")
        
        # Compute final metrics using metrics computer
        final_metrics = self.metrics_computer.compute_final_metrics(
            running_val_loss, num_batches, all_predictions, all_targets, phase_num
        )
        
        # Confirm metrics computation completed
        logger.debug(f"✅ Final validation metrics computed for epoch {display_epoch}")
        
        # Aggregate loss breakdowns and add to final metrics
        if all_loss_breakdowns:
            aggregated_loss_breakdown = self._aggregate_loss_breakdowns(all_loss_breakdowns)
            final_metrics['loss_breakdown'] = aggregated_loss_breakdown
            logger.debug(f"Added aggregated loss breakdown with {len(aggregated_loss_breakdown)} components")
        
        return final_metrics
    
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
