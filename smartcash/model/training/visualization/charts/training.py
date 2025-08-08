"""
Training-specific chart generation utilities.

This module provides visualization components for training metrics such as
loss curves, learning rate schedules, and training progress.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np

# Flag to indicate if visualization dependencies are available
VISUALIZATION_AVAILABLE = True

# Import base chart class
from .base import BaseChart, Figure, Axes
from smartcash.common.logger import get_logger

class TrainingCharts(BaseChart):
    """Charts for visualizing training progress and metrics."""
    
    def __init__(self, save_dir: Union[str, Path] = "data/visualization", 
                 verbose: bool = False):
        """
        Initialize the training charts generator.
        
        Args:
            save_dir: Directory to save generated charts
            verbose: Enable verbose logging
        """
        super().__init__(save_dir=save_dir, verbose=verbose)
        self.logger = get_logger(self.__class__.__name__)
    
    def plot_loss_curves(self, train_losses: List[float], val_losses: List[float],
                        title: str = "Training and Validation Loss") -> Optional[Path]:
        """
        Plot training and validation loss curves.
        
        Args:
            train_losses: List of training losses per epoch
            val_losses: List of validation losses per epoch
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not train_losses:
            return None
            
        epochs = list(range(1, len(train_losses) + 1))
        
        fig, ax = self.create_figure(title)
        
        # Plot training loss
        ax.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
        
        # Plot validation loss if available
        if val_losses and len(val_losses) == len(train_losses):
            ax.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
        
        self.apply_common_styling(
            ax,
            xlabel='Epoch',
            ylabel='Loss',
            legend=True,
            grid=True
        )
        
        # Adjust layout to prevent title overlap
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, 'loss_curves')
    
    def plot_learning_rate_schedule(self, learning_rates: List[float],
                                  title: str = "Learning Rate Schedule") -> Optional[Path]:
        """
        Plot the learning rate schedule over training.
        
        Args:
            learning_rates: List of learning rates per epoch
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not learning_rates:
            return None
            
        epochs = list(range(1, len(learning_rates) + 1))
        
        fig, ax = self.create_figure(title)
        
        # Plot learning rate
        ax.semilogy(epochs, learning_rates, 'g-', label='Learning Rate', alpha=0.7)
        
        self.apply_common_styling(
            ax,
            xlabel='Epoch',
            ylabel='Learning Rate (log scale)',
            legend=True,
            grid=True
        )
        
        # Adjust layout to prevent title overlap
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, 'learning_rate_schedule')
    
    def plot_metric_trends(self, metrics: Dict[str, List[float]],
                          title: str = "Training Metrics") -> Optional[Path]:
        """
        Plot trends of multiple metrics over training.
        
        Args:
            metrics: Dictionary of metric names to lists of values
            title: Chart title
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or not metrics:
            return None
            
        # Determine number of epochs from the first metric
        first_metric = next(iter(metrics.values()))
        epochs = list(range(1, len(first_metric) + 1))
        
        fig, ax = self.create_figure(title, figsize=(12, 6))
        
        # Plot each metric
        for name, values in metrics.items():
            if len(values) == len(epochs):
                ax.plot(epochs, values, '.-', label=name, alpha=0.7)
        
        self.apply_common_styling(
            ax,
            xlabel='Epoch',
            ylabel='Metric Value',
            legend=True,
            grid=True
        )
        
        # Adjust layout to prevent title overlap
        fig.tight_layout()
        
        # Save and return path
        return self.save_figure(fig, 'metric_trends')
