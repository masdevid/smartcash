"""
Research-focused visualization utilities.

This module provides specialized visualizations for research and analysis,
including comprehensive training dashboards and detailed metric breakdowns.
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
    from matplotlib.gridspec import GridSpec

class ResearchCharts(BaseChart):
    """Research-focused visualization components."""
    
    def __init__(self, save_dir: Union[str, Path] = "data/visualization", 
                 verbose: bool = False):
        """
        Initialize the research chart generator.
        
        Args:
            save_dir: Directory to save generated charts
            verbose: Enable verbose logging
        """
        super().__init__(save_dir=save_dir, verbose=verbose)
        self.logger = get_logger(self.__class__.__name__)
    
    def create_research_dashboard(
            self,
            metrics: Dict[str, Any],
            title: str = "Training Dashboard") -> Optional[Path]:
        """
        Create a comprehensive research dashboard with multiple visualizations.
        
        Args:
            metrics: Dictionary containing various metrics and training data
            title: Dashboard title
            
        Returns:
            Path to saved dashboard or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE:
            return None
            
        try:
            # Create figure with multiple subplots
            fig = plt.figure(figsize=(16, 12), constrained_layout=True)
            gs = GridSpec(3, 2, figure=fig)
            
            # Set main title
            fig.suptitle(title, fontsize=16, fontweight='bold')
            
            # Plot 1: Training and validation loss
            ax1 = fig.add_subplot(gs[0, 0])
            self._plot_loss_curves(ax1, metrics.get('losses', {}))
            
            # Plot 2: Learning rate schedule
            ax2 = fig.add_subplot(gs[0, 1])
            self._plot_learning_rate(ax2, metrics.get('learning_rates', []))
            
            # Plot 3: Accuracy metrics
            ax3 = fig.add_subplot(gs[1, 0])
            self._plot_accuracy_metrics(ax3, metrics.get('metrics', {}))
            
            # Plot 4: Confusion matrix (if available)
            ax4 = fig.add_subplot(gs[1, 1])
            self._plot_confusion_matrix_preview(ax4, metrics.get('confusion_matrices', {}))
            
            # Plot 5: Phase analysis
            ax5 = fig.add_subplot(gs[2, :])
            self._plot_phase_analysis(ax5, metrics.get('phase_metrics', {}))
            
            # Adjust layout
            fig.tight_layout()
            
            # Save and return path
            return self.save_figure(fig, 'research_dashboard')
            
        except Exception as e:
            if self.verbose:
                self.logger.error(f"Error creating research dashboard: {str(e)}")
            return None
    
    def _plot_loss_curves(self, ax: Axes, losses: Dict[str, List[float]]) -> None:
        """Plot training and validation loss curves."""
        if not losses:
            return
            
        train_losses = losses.get('train', [])
        val_losses = losses.get('val', [])
        
        epochs = list(range(1, max(len(train_losses), len(val_losses)) + 1))
        
        if train_losses:
            ax.plot(epochs[:len(train_losses)], train_losses, 'b-', label='Training Loss', alpha=0.7)
        if val_losses:
            ax.plot(epochs[:len(val_losses)], val_losses, 'r-', label='Validation Loss', alpha=0.7)
        
        ax.set_title('Training & Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_learning_rate(self, ax: Axes, learning_rates: List[float]) -> None:
        """Plot learning rate schedule."""
        if not learning_rates:
            return
            
        epochs = list(range(1, len(learning_rates) + 1))
        ax.semilogy(epochs, learning_rates, 'g-', alpha=0.7)
        
        ax.set_title('Learning Rate Schedule')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Learning Rate (log scale)')
        ax.grid(True, alpha=0.3, which='both')
    
    def _plot_accuracy_metrics(self, ax: Axes, metrics: Dict[str, List[float]]) -> None:
        """Plot accuracy metrics over time."""
        if not metrics:
            return
            
        # Find the maximum length of any metric list
        max_len = max((len(v) for v in metrics.values()), default=0)
        if max_len == 0:
            return
            
        epochs = list(range(1, max_len + 1))
        
        for name, values in metrics.items():
            if values and 'loss' not in name.lower():
                ax.plot(epochs[:len(values)], values, 'o-', label=name.replace('_', ' ').title(), 
                       alpha=0.7, markersize=4)
        
        ax.set_title('Accuracy Metrics')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_confusion_matrix_preview(self, ax: Axes, confusion_matrices: Dict[str, np.ndarray]) -> None:
        """Plot a preview of the confusion matrix."""
        if not confusion_matrices:
            return
            
        # Just show the first confusion matrix as a preview
        layer_name, cm = next(iter(confusion_matrices.items()))
        
        if cm is None or cm.size == 0:
            return
            
        # Normalize
        cm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt=".2f",
            cmap="Blues",
            cbar=True,
            square=True,
            ax=ax
        )
        
        ax.set_title(f'Confusion Matrix Preview\n({layer_name})')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
    
    def _plot_phase_analysis(self, ax: Axes, phase_metrics: Dict[str, Dict[str, List[float]]]) -> None:
        """Plot metrics across different training phases."""
        if not phase_metrics:
            return
            
        # Choose a metric to plot (e.g., 'val_accuracy')
        target_metric = None
        for metric in ['val_accuracy', 'accuracy', 'f1_score', 'val_loss']:
            if any(metric in metrics for metrics in phase_metrics.values()):
                target_metric = metric
                break
                
        if not target_metric:
            return
            
        # Plot each phase
        for phase, metrics in phase_metrics.items():
            if target_metric in metrics and metrics[target_metric]:
                values = metrics[target_metric]
                epochs = list(range(1, len(values) + 1))
                ax.plot(epochs, values, 'o-', label=f"{phase} ({target_metric})", alpha=0.7)
        
        ax.set_title(f'Phase Analysis - {target_metric.replace("_", " ").title()}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(target_metric.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)
