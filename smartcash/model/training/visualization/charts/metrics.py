"""
Metrics visualization utilities.

This module provides functionality for visualizing various metrics across
different layers and training phases.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Import base chart class
from .base import BaseChart, Figure, Axes, VISUALIZATION_AVAILABLE
from smartcash.common.logger import get_logger

if VISUALIZATION_AVAILABLE:
    import matplotlib.pyplot as plt
    import seaborn as sns

class MetricsCharts(BaseChart):
    """Charts for visualizing model metrics across layers and phases."""
    
    def __init__(self, save_dir: Union[str, Path] = "data/visualization", 
                 verbose: bool = False):
        """
        Initialize the metrics chart generator.
        
        Args:
            save_dir: Directory to save generated charts
            verbose: Enable verbose logging
        """
        super().__init__(save_dir=save_dir, verbose=verbose)
        self.logger = get_logger(self.__class__.__name__)
    
    def plot_metrics_comparison(
            self,
            metrics: Dict[str, Dict[str, List[float]]],
            metric_name: str,
            title: str = None) -> Optional[Path]:
        """
        Plot comparison of a specific metric across different layers.
        
        Args:
            metrics: Nested dictionary of metrics by layer and metric type
            metric_name: Name of the metric to plot (e.g., 'accuracy', 'f1_score')
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not metrics:
            return None
            
        # Filter out layers that don't have the requested metric
        layers_data = {
            layer: layer_metrics[metric_name] 
            for layer, layer_metrics in metrics.items()
            if metric_name in layer_metrics and layer_metrics[metric_name]
        }
        
        if not layers_data:
            return None
            
        # Create title if not provided
        if title is None:
            title = f"{metric_name.replace('_', ' ').title()} Comparison by Layer"
        
        # Create figure
        fig, ax = self.create_figure(title, figsize=(12, 6))
        
        # Plot each layer's metric
        for layer, values in layers_data.items():
            epochs = list(range(1, len(values) + 1))
            ax.plot(epochs, values, 'o-', label=layer, alpha=0.7)
        
        self.apply_common_styling(
            ax,
            xlabel='Epoch',
            ylabel=metric_name.replace('_', ' ').title(),
            legend=True,
            grid=True
        )
        
        # Adjust layout to prevent title overlap
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, f"{metric_name}_comparison")
    
    def plot_metrics_heatmap(
            self,
            metrics: Dict[str, Dict[str, float]],
            title: str = "Metrics Comparison") -> Optional[Path]:
        """
        Plot a heatmap comparing multiple metrics across layers.
        
        Args:
            metrics: Dictionary mapping layer names to metric dictionaries
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not metrics:
            return None
            
        # Convert metrics to a 2D array for the heatmap
        layers = list(metrics.keys())
        if not layers:
            return None
            
        # Get all unique metric names
        all_metrics = set()
        for layer_metrics in metrics.values():
            all_metrics.update(layer_metrics.keys())
        all_metrics = sorted(all_metrics)
        
        if not all_metrics:
            return None
            
        # Create data matrix
        data = []
        for metric in all_metrics:
            row = [metrics[layer].get(metric, np.nan) for layer in layers]
            data.append(row)
        
        data = np.array(data)
        
        # Create figure
        fig, ax = self.create_figure(title, figsize=(max(8, len(layers) * 1.5), 
                                                    max(6, len(all_metrics) * 0.5)))
        
        # Plot heatmap
        sns.heatmap(
            data,
            annot=True,
            fmt=".3f",
            cmap="YlGnBu",
            xticklabels=layers,
            yticklabels=all_metrics,
            ax=ax
        )
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Adjust layout to prevent label cutoff
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, "metrics_heatmap")
    
    def plot_phase_transition_analysis(
            self,
            phase_metrics: Dict[str, Dict[str, List[float]]],
            metric_name: str,
            title: str = None) -> Optional[Path]:
        """
        Plot metrics across different training phases.
        
        Args:
            phase_metrics: Dictionary mapping phase names to metrics
            metric_name: Name of the metric to plot
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not phase_metrics:
            return None
            
        # Create figure
        if title is None:
            title = f"{metric_name.replace('_', ' ').title()} Across Training Phases"
            
        fig, ax = self.create_figure(title, figsize=(12, 6))
        
        # Plot each phase's metrics
        for phase, metrics in phase_metrics.items():
            if metric_name in metrics and metrics[metric_name]:
                epochs = list(range(1, len(metrics[metric_name]) + 1))
                ax.plot(epochs, metrics[metric_name], 'o-', label=phase, alpha=0.7)
        
        self.apply_common_styling(
            ax,
            xlabel='Epoch',
            ylabel=metric_name.replace('_', ' ').title(),
            legend=True,
            grid=True
        )
        
        # Add vertical lines for phase transitions
        if len(phase_metrics) > 1:
            max_epoch = max(
                len(metrics[metric_name])
                for metrics in phase_metrics.values()
                if metric_name in metrics
            )
            
            if max_epoch > 0:
                for i in range(1, len(phase_metrics)):
                    ax.axvline(x=i * max_epoch, color='r', linestyle='--', alpha=0.5)
        
        # Adjust layout to prevent title overlap
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, f"phase_transition_{metric_name}")
