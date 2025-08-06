"""
Multi-metric early stopping implementation.

Monitors multiple metrics simultaneously with configurable combination logic.
"""

from typing import Dict, Any, List, Optional
import torch
from .base import BaseEarlyStopping
from .standard import StandardEarlyStopping


class MultiMetricEarlyStopping(BaseEarlyStopping):
    """Early stopping with multiple metrics monitoring."""
    
    def __init__(self, metrics_config: List[Dict[str, Any]], 
                 combination_mode: str = 'any', verbose: bool = True):
        """
        Initialize multi-metric early stopping.
        
        Args:
            metrics_config: List of metric configs, each containing:
                - metric: metric name
                - patience: patience for this metric
                - min_delta: minimum delta
                - mode: 'max' or 'min'
                - weight: weight for combination (optional)
            combination_mode: 'any' (stop if any metric triggers), 'all' (stop if all trigger)
            verbose: Print messages
        """
        # Use average patience for base class
        avg_patience = sum(config.get('patience', 15) for config in metrics_config) // len(metrics_config)
        super().__init__(patience=avg_patience, verbose=verbose)
        
        self.combination_mode = combination_mode.lower()
        
        # Create individual early stopping for each metric
        self.stoppers = {}
        for config in metrics_config:
            metric_name = config['metric']
            self.stoppers[metric_name] = StandardEarlyStopping(
                patience=config.get('patience', 15),
                min_delta=config.get('min_delta', 0.001),
                metric=metric_name,
                mode=config.get('mode', 'max'),
                restore_best_weights=False,  # Handle at this level
                verbose=False  # Handle messaging at this level
            )
        
        # State tracking
        self.triggered_metrics = []
        
        if self.verbose:
            metric_names = list(self.stoppers.keys())
            print(f"   Monitoring metrics: {metric_names}")
            print(f"   Combination mode: {combination_mode}")
    
    def __call__(self, metrics: Dict[str, float], model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """
        Check multiple metrics for early stopping.
        
        Args:
            metrics: Dictionary of metric values
            model: Model for weight management
            epoch: Current epoch
            
        Returns:
            True if should stop
        """
        epoch = epoch or 0
        triggered_stoppers = []
        
        # Check each metric
        for metric_name, stopper in self.stoppers.items():
            if metric_name in metrics:
                should_stop = stopper(metrics, model, epoch)
                if should_stop:
                    triggered_stoppers.append(metric_name)
        
        # Determine overall stopping decision
        if self.combination_mode == 'any':
            self.should_stop = len(triggered_stoppers) > 0
        else:  # combination_mode == 'all'
            self.should_stop = len(triggered_stoppers) == len(self.stoppers)
        
        if self.should_stop and not self.triggered_metrics:
            self.stopped_epoch = epoch
            self.triggered_metrics = triggered_stoppers.copy()
            
            if self.verbose:
                triggered_str = ", ".join(triggered_stoppers)
                print(f"🛑 Multi-metric early stopping triggered! Metrics: {triggered_str}")
        
        return self.should_stop
    
    def get_status_summary(self) -> Dict[str, Any]:
        """Get status summary for all metrics."""
        summary = {
            'should_stop': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'triggered_metrics': self.triggered_metrics,
            'combination_mode': self.combination_mode,
            'individual_status': {}
        }
        
        for metric_name, stopper in self.stoppers.items():
            summary['individual_status'][metric_name] = stopper.get_best_info()
        
        return summary
    
    def reset(self) -> None:
        """Reset all stoppers."""
        for stopper in self.stoppers.values():
            stopper.reset()
        
        self.should_stop = False
        self.stopped_epoch = 0
        self.triggered_metrics = []
        self.best_weights = None
        
        # Clear history
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        if self.verbose:
            print("🔄 Multi-metric early stopping state reset")
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best models across all metrics."""
        best_info = {
            'stopped': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'triggered_metrics': self.triggered_metrics,
            'combination_mode': self.combination_mode,
            'individual_best': {}
        }
        
        for metric_name, stopper in self.stoppers.items():
            best_info['individual_best'][metric_name] = stopper.get_best_info()
        
        return best_info