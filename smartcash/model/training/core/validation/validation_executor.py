#!/usr/bin/env python3
"""
Validation execution for the SmartCash training pipeline.

This module orchestrates validation epoch execution using SRP-compliant components
for better maintainability and separation of concerns.
"""

import torch
from typing import Dict, Optional, List, Any

from smartcash.common.logger import get_logger
from smartcash.model.architectures.model import SmartCashYOLOv5Model
from ..prediction.prediction_processor import PredictionProcessor
from ..yolov5_map_calculator import YOLOv5MapCalculator
from .batch_processor import ValidationBatchProcessor
from .validation_metrics_processor import ValidationMetricsProcessor
from .validation_model_manager import ValidationModelManager
from .validation_map_processor import ValidationMapProcessor
from smartcash.model.training.utils.signal_handler import is_shutdown_requested

logger = get_logger(__name__)

class ValidationExecutor:
    """Orchestrates validation epoch execution using SRP-compliant components."""
    
    def __init__(self, model: SmartCashYOLOv5Model, config: Dict, progress_tracker, phase_num: int = 1):
        """
        Initialize validation executor.
        
        Args:
            model: SmartCashYOLOv5Model instance
            config: Training configuration
            progress_tracker: Progress tracking instance
            phase_num: Current phase number (default: 1)
        """
        self.model = model
        self.config = config
        self.progress_tracker = progress_tracker
        
        # Initialize core components
        self.prediction_processor = PredictionProcessor(config, model)
        
        # Get model configuration
        model_info = model.get_model_config()
        num_classes = 17  # Fixed for SmartCashYOLO
        
        # Training context for logging
        training_context = {
            'backbone': config.get('backbone', 'unknown'),
            'training_mode': config.get('training_mode', 'unknown'),
            'current_phase': getattr(model, 'current_phase', phase_num),
            'session_id': config.get('session_id', 'N/A'),
            'model_name': 'SmartCashYOLOv5',
            'smartcash_model': True
        }
        
        # Initialize mAP calculator
        self.map_calculator = YOLOv5MapCalculator(
            num_classes=num_classes,
            conf_thres=0.001,  # Low threshold for early training
            iou_thres=0.5,
            debug=config.get('debug_map', False),
            training_context=training_context,
            use_standard_map=True
        )
        
        # Initialize SRP components
        self.batch_processor = ValidationBatchProcessor(model, config, self.prediction_processor)
        self.metrics_processor = ValidationMetricsProcessor(model, config, map_calculator=self.map_calculator)
        self.model_manager = ValidationModelManager(model)
        self.map_processor = ValidationMapProcessor(self.map_calculator, phase_num)
        
        logger.debug(f"Initialized validation executor with {num_classes} classes")
        
    
    def validate_epoch(
        self, 
        val_loader: torch.utils.data.DataLoader, 
        loss_manager: Any,
        epoch: int, 
        phase_num: int = 1, 
        display_epoch: Optional[int] = None
    ) -> Dict[str, float]:
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
        display_epoch = display_epoch if display_epoch is not None else epoch + 1
        
        # Reset mAP calculator for new validation epoch
        self.map_calculator.reset()
        
        # Update mAP calculator with current epoch for progressive thresholds
        if hasattr(self.map_calculator, 'update_epoch'):
            self.map_calculator.update_epoch(epoch)
        
        # Clear cached mAP results in metrics computer
        self.metrics_processor._cached_map_results = None
        
        if num_batches == 0:
            logger.warning("Validation loader is empty!")
            return self._get_empty_validation_metrics()
        
        # Log validation batch size information for first epoch
        if epoch == 0:
            first_batch = next(iter(val_loader))
            actual_batch_size = first_batch[0].shape[0] if first_batch[0].dim() > 0 else "unknown"
            logger.debug(f"Validation batch size: {actual_batch_size}, batches: {num_batches}")
        
        # Initialize collectors
        all_predictions: Dict[str, List[torch.Tensor]] = {}
        all_targets: Dict[str, List[torch.Tensor]] = {}
        all_loss_breakdowns: List[Dict[str, float]] = []
        
        # Start batch tracking
        self.progress_tracker.start_batch_tracking(num_batches)
        
        try:
            # Optimized validation batch processing
            update_freq = self._get_progress_update_frequency(num_batches)
            
            with torch.no_grad():
                for batch_idx, (images, targets) in enumerate(val_loader):
                    # Reduced shutdown check frequency for better performance
                    if batch_idx % 50 == 0 and is_shutdown_requested():
                        logger.info("🛑 Shutdown requested during validation")
                        break
                    
                    # Optimized batch processing
                    batch_metrics = self.batch_processor.process_batch(
                        batch=(images, targets),
                        batch_idx=batch_idx,
                        num_batches=num_batches,
                        phase_num=phase_num,
                        progress_callback=self.progress_tracker.update_batch_progress if batch_idx % 10 == 0 else None
                    )
                    
                    # Fast loss accumulation
                    running_val_loss += batch_metrics.get('loss', 0.0)
                    
                    # Collect loss breakdown less frequently
                    if batch_idx % 5 == 0 and 'loss_breakdown' in batch_metrics:
                        loss_breakdown = batch_metrics['loss_breakdown']
                        if loss_breakdown:
                            all_loss_breakdowns.append(loss_breakdown)
                    
                    # Optimized mAP processing (less frequent)
                    if batch_idx % 10 == 0 or batch_idx == num_batches - 1:
                        try:
                            if hasattr(self.prediction_processor, 'last_normalized_predictions'):
                                preds = self.prediction_processor.last_normalized_predictions
                                if preds is not None:
                                    self.map_processor.update_map_calculator(
                                        preds, targets, images, batch_idx, epoch
                                    )
                        except Exception as e:
                            logger.debug(f"mAP update error batch {batch_idx}: {e}")
                    
                    # Less frequent progress updates
                    if batch_idx % update_freq == 0 or batch_idx == num_batches - 1:
                        self._update_progress(
                            batch_idx, num_batches, running_val_loss, display_epoch
                        )
        finally:
            # Ensure batch tracking is always completed
            self.progress_tracker.complete_batch_tracking()
        
        # Compute and return final metrics
        return self._compute_final_metrics(
            running_val_loss, num_batches, all_predictions, 
            all_targets, all_loss_breakdowns, phase_num, display_epoch
        )
    
    def _get_progress_update_frequency(self, num_batches: int) -> int:
        """Optimized progress update frequency for better performance."""
        if num_batches <= 5:
            return 1  # Update every batch for very small sets
        elif num_batches <= 20:
            return max(2, num_batches // 3)  # Update ~3 times
        elif num_batches <= 100:
            return max(5, num_batches // 10)  # Update ~10 times
        return max(10, num_batches // 20)  # Update ~20 times for large sets
    
    def _update_progress(
        self, 
        batch_idx: int, 
        num_batches: int, 
        running_loss: float, 
        display_epoch: int
    ) -> None:
        """Update progress tracking for the current batch."""
        avg_loss = running_loss / (batch_idx + 1)
        self.progress_tracker.update_batch_progress(
            batch_idx + 1, 
            num_batches,
            f"Validation batch {batch_idx + 1}/{num_batches}",
            loss=avg_loss,
            epoch=display_epoch
        )
    
    def _compute_final_metrics(
        self,
        running_loss: float,
        num_batches: int,
        all_predictions: Dict[str, List[torch.Tensor]],
        all_targets: Dict[str, List[torch.Tensor]],
        all_loss_breakdowns: List[Dict[str, float]],
        phase_num: int,
        display_epoch: int
    ) -> Dict[str, float]:
        """Compute and return final validation metrics."""
        logger.info(f"Computing final validation metrics for epoch {display_epoch}...")
        
        if self.progress_tracker.progress_callback:
            self.progress_tracker.progress_callback(
                'batch', 100, 100, 
                f"Computing validation metrics for epoch {display_epoch}..."
            )
        
        # Get base metrics
        final_metrics = self.metrics_processor.compute_final_metrics(
            running_loss, num_batches, all_predictions, all_targets, phase_num
        )
        
        # Add loss breakdown if available
        if all_loss_breakdowns:
            final_metrics['loss_breakdown'] = self._aggregate_loss_breakdowns(all_loss_breakdowns)
        
        return final_metrics
    
    def _get_empty_validation_metrics(self) -> Dict[str, float]:
        """
        Get empty validation metrics when no data is available.
        
        Returns:
            Dictionary with default zero values for all metrics
        """
        return {
            'val_loss': 0.0,
            'val_map50': 0.0,
            'val_map50_95': 0.0,
            'val_precision': 0.0,
            'val_recall': 0.0,
            'val_f1': 0.0,
            'val_accuracy': 0.0,
            'loss_breakdown': {}
        }
    
    def _aggregate_loss_breakdowns(self, all_loss_breakdowns: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Aggregate loss breakdowns from all validation batches.
        
        Args:
            all_loss_breakdowns: List of loss breakdown dictionaries from each batch
            
        Returns:
            Dictionary containing averaged loss breakdown components
        """
        if not all_loss_breakdowns:
            return {}
        
        # Collect all unique keys and their values
        key_values: Dict[str, List[float]] = {}
        
        for breakdown in all_loss_breakdowns:
            for key, value in breakdown.items():
                # Convert tensor to float if needed
                if hasattr(value, 'item'):
                    value = value.item()
                elif hasattr(value, 'cpu'):
                    value = value.cpu().numpy()
                
                if isinstance(value, (int, float)):
                    if key not in key_values:
                        key_values[key] = []
                    key_values[key].append(float(value))
        
        # Calculate average for each key
        return {
            key: sum(values) / len(values)
            for key, values in key_values.items()
            if values  # Only include keys with valid values
        }
