"""
Metrics processing mixin for phase management.

Provides functionality for processing and formatting metrics during training.
"""

from typing import Dict, Any
from smartcash.common.logger import get_logger

logger = get_logger(__name__)

class MetricsProcessingMixin:
    """Mixin for metrics processing and formatting capabilities."""
    
    def process_epoch_metrics(self, train_metrics: Dict, val_metrics: Dict, 
                             phase_num: int, epoch: int) -> Dict[str, Any]:
        """
        Process and combine training and validation metrics.
        
        Args:
            train_metrics: Training metrics
            val_metrics: Validation metrics  
            phase_num: Current phase number
            epoch: Current epoch
            
        Returns:
            Combined and processed metrics
        """
        # Combine metrics
        final_metrics = {**train_metrics, **val_metrics}
        
        # Add phase and epoch info
        final_metrics['phase'] = phase_num
        final_metrics['epoch'] = epoch + 1  # Convert 0-based to 1-based for display
        
        # Compute layer-specific metrics if needed
        layer_metrics = self._compute_layer_metrics(train_metrics, phase_num)
        final_metrics.update(layer_metrics)
        
        # Apply formatting
        final_metrics = self._apply_research_metrics_format(final_metrics, phase_num)
        
        # Filter unnecessary metrics
        final_metrics = self._filter_unnecessary_metrics(final_metrics, phase_num)
        
        # Ensure required metrics are present
        self._ensure_required_metrics(final_metrics)
        
        return final_metrics
    
    def _compute_layer_metrics(self, train_metrics: Dict, phase_num: int = None) -> Dict[str, float]:
        """Compute layer-specific metrics from accumulated training data."""
        
        # Extract accumulated predictions and targets from training metrics
        accumulated_predictions = train_metrics.get('_accumulated_predictions', {})
        accumulated_targets = train_metrics.get('_accumulated_targets', {})
        
        if not accumulated_predictions or not accumulated_targets:
            logger = get_logger(self.__class__.__name__)
            logger.debug("No accumulated predictions/targets available for layer metrics")
            return {}
        
        try:
            from smartcash.model.training.utils.metrics_utils import calculate_multilayer_metrics
            
            # Concatenate accumulated tensors for each layer
            layer_predictions = {}
            layer_targets = {}
            
            for layer_name in accumulated_predictions.keys():
                if (layer_name in accumulated_targets and 
                    accumulated_predictions[layer_name] and 
                    accumulated_targets[layer_name]):
                    
                    # Concatenate all batches for this layer
                    import torch
                    layer_predictions[layer_name] = torch.cat(accumulated_predictions[layer_name], dim=0)
                    layer_targets[layer_name] = torch.cat(accumulated_targets[layer_name], dim=0)
            
            if layer_predictions and layer_targets:
                return calculate_multilayer_metrics(layer_predictions, layer_targets)
            else:
                return {}
                
        except Exception as e:
            logger = get_logger(self.__class__.__name__)
            logger.debug(f"Could not compute layer metrics: {e}")
            return {}
    
    def _apply_research_metrics_format(self, final_metrics: Dict[str, float], 
                                     phase_num: int) -> Dict[str, float]:
        """Apply research paper formatting to metrics."""
        # Round metrics to appropriate precision
        formatted_metrics = {}
        for key, value in final_metrics.items():
            if isinstance(value, float):
                if 'loss' in key.lower():
                    formatted_metrics[key] = round(value, 6)
                elif any(metric in key.lower() for metric in ['accuracy', 'f1', 'precision', 'recall', 'map']):
                    formatted_metrics[key] = round(value, 4)
                else:
                    formatted_metrics[key] = round(value, 4)
            else:
                formatted_metrics[key] = value
        
        return formatted_metrics
    
    def _filter_unnecessary_metrics(self, metrics: Dict[str, Any], phase_num: int) -> Dict[str, Any]:
        """Filter out unnecessary metrics based on phase."""
        from smartcash.model.training.utils.metrics_utils import filter_phase_relevant_metrics
        return filter_phase_relevant_metrics(metrics, phase_num)
    
    def _ensure_required_metrics(self, final_metrics: Dict[str, Any]):
        """Ensure all required metrics are present with default values."""
        required_metrics = {
            'train_loss': 0.0,
            'val_loss': 0.0,
            'val_accuracy': 0.0,
            'learning_rate': 0.0
        }
        
        for metric, default_value in required_metrics.items():
            if metric not in final_metrics:
                final_metrics[metric] = default_value
