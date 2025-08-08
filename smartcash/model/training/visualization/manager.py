"""
Visualization manager for training metrics and charts.

This module provides a high-level interface for managing and generating
training visualizations, including loss curves, confusion matrices,
and research dashboards.
"""

from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import numpy as np
import json
from datetime import datetime

from .charts import (
    TrainingCharts,
    ConfusionMatrixCharts,
    MetricsCharts,
    ResearchCharts
)
from ..utils.resume_utils import get_logger

# Flag to indicate if visualization dependencies are available
VISUALIZATION_AVAILABLE = True

class VisualizationManager:
    """
    Centralized manager for all training visualizations.
    
    This class provides a unified interface for generating and managing
    all training-related visualizations, including metrics tracking,
    confusion matrices, and research dashboards.
    """
    
    def __init__(
        self,
        num_classes_per_layer: Dict[str, int],
        class_names: Optional[Dict[str, List[str]]] = None,
        save_dir: Union[str, Path] = "data/visualization",
        verbose: bool = False
    ):
        """
        Initialize the visualization manager.
        
        Args:
            num_classes_per_layer: Dictionary mapping layer names to number of classes
            class_names: Optional mapping of layer names to class name lists
            save_dir: Directory to save visualization outputs
            verbose: Enable verbose logging
        """
        self.num_classes_per_layer = num_classes_per_layer
        self.class_names = class_names or {}
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        
        # Initialize logger
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize chart generators
        self.training_charts = TrainingCharts(save_dir=save_dir, verbose=verbose)
        self.confusion_charts = ConfusionMatrixCharts(save_dir=save_dir, verbose=verbose)
        self.metrics_charts = MetricsCharts(save_dir=save_dir, verbose=verbose)
        self.research_charts = ResearchCharts(save_dir=save_dir, verbose=verbose)
        
        # Initialize metrics storage
        self.metrics_history = []
        self.confusion_matrices = {}
        self.learning_rates = []
        self.phase_metrics = {}
        
        if verbose:
            self.logger.info("✅ Visualization manager initialized")
            self.logger.info(f"   • Layers: {list(num_classes_per_layer.keys())}")
            self.logger.info(f"   • Save directory: {self.save_dir}")
    
    def update_metrics(
        self,
        epoch: int,
        metrics: Dict[str, Any],
        phase: str = "train",
        learning_rate: Optional[float] = None,
        confusion_matrices: Optional[Dict[str, np.ndarray]] = None
    ) -> None:
        """
        Update metrics and store them for visualization.
        
        Args:
            epoch: Current epoch number
            metrics: Dictionary of metrics for this epoch
            phase: Training phase (e.g., 'train', 'val', 'test')
            learning_rate: Current learning rate
            confusion_matrices: Dictionary of confusion matrices by layer
        """
        # Store metrics
        metrics_entry = {
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat(),
            **metrics
        }
        self.metrics_history.append(metrics_entry)
        
        # Store learning rate if provided
        if learning_rate is not None:
            self.learning_rates.append(learning_rate)
        
        # Store confusion matrices if provided
        if confusion_matrices:
            for layer_name, cm in confusion_matrices.items():
                if layer_name not in self.confusion_matrices:
                    self.confusion_matrices[layer_name] = []
                self.confusion_matrices[layer_name].append(cm)
    
    def generate_training_curves(self) -> Optional[Path]:
        """
        Generate training and validation loss curves.
        
        Returns:
            Path to the saved chart or None if generation failed
        """
        if not self.metrics_history:
            return None
            
        # Extract losses
        train_losses = [m['loss'] for m in self.metrics_history if m.get('phase') == 'train']
        val_losses = [m.get('val_loss') for m in self.metrics_history if m.get('phase') == 'val']
        
        # Filter out None values
        val_losses = [v for v in val_losses if v is not None]
        
        return self.training_charts.plot_loss_curves(train_losses, val_losses)
    
    def generate_learning_rate_chart(self) -> Optional[Path]:
        """
        Generate learning rate schedule chart.
        
        Returns:
            Path to the saved chart or None if generation failed
        """
        if not self.learning_rates:
            return None
            
        return self.training_charts.plot_learning_rate_schedule(self.learning_rates)
    
    def generate_confusion_matrices(self) -> Dict[str, Path]:
        """
        Generate confusion matrices for all layers.
        
        Returns:
            Dictionary mapping layer names to saved chart paths
        """
        if not self.confusion_matrices:
            return {}
            
        # Get the most recent confusion matrix for each layer
        latest_matrices = {
            layer: matrices[-1]  # Get the most recent matrix
            for layer, matrices in self.confusion_matrices.items()
            if matrices
        }
        
        return self.confusion_charts.plot_multi_layer_confusion_matrices(
            confusion_matrices=latest_matrices,
            class_names=self.class_names
        )
    
    def generate_metrics_comparison(self, metric_name: str) -> Optional[Path]:
        """
        Generate comparison chart for a specific metric across layers.
        
        Args:
            metric_name: Name of the metric to compare
            
        Returns:
            Path to the saved chart or None if generation failed
        """
        if not self.metrics_history:
            return None
            
        # Extract metrics by layer
        metrics_by_layer = {}
        for layer in self.num_classes_per_layer.keys():
            layer_metric = f"{layer}_{metric_name}"
            metrics = [m.get(layer_metric) for m in self.metrics_history]
            metrics = [m for m in metrics if m is not None]
            if metrics:
                metrics_by_layer[layer] = metrics
        
        if not metrics_by_layer:
            return None
            
        return self.metrics_charts.plot_metrics_comparison(
            metrics=metrics_by_layer,
            metric_name=metric_name
        )
    
    def generate_research_dashboard(self) -> Optional[Path]:
        """
        Generate a comprehensive research dashboard.
        
        Returns:
            Path to the saved dashboard or None if generation failed
        """
        # Prepare metrics for the dashboard
        dashboard_metrics = {
            'losses': {
                'train': [m['loss'] for m in self.metrics_history if m.get('phase') == 'train'],
                'val': [m.get('val_loss') for m in self.metrics_history if m.get('phase') == 'val']
            },
            'learning_rates': self.learning_rates,
            'metrics': self._extract_metrics_for_dashboard(),
            'confusion_matrices': {
                layer: matrices[-1] if matrices else None
                for layer, matrices in self.confusion_matrices.items()
            },
            'phase_metrics': self.phase_metrics
        }
        
        return self.research_charts.create_research_dashboard(
            metrics=dashboard_metrics,
            title="Training Analysis Dashboard"
        )
    
    def _extract_metrics_for_dashboard(self) -> Dict[str, List[float]]:
        """
        Extract metrics in a format suitable for the dashboard.
        
        Returns:
            Dictionary of metric names to lists of values
        """
        metrics = {}
        
        # Get all unique metric names
        all_metrics = set()
        for entry in self.metrics_history:
            all_metrics.update(entry.keys())
        
        # Filter out non-numeric metrics and special fields
        numeric_metrics = {
            m for m in all_metrics 
            if m not in ['epoch', 'phase', 'timestamp'] 
            and any(isinstance(entry.get(m), (int, float)) 
                   for entry in self.metrics_history)
        }
        
        # Extract values for each metric
        for metric in numeric_metrics:
            values = [entry.get(metric) for entry in self.metrics_history]
            metrics[metric] = values
        
        return metrics
    
    def save_metrics_summary(self, filename: str = "metrics_summary.json") -> Path:
        """
        Save a summary of all metrics to a JSON file.
        
        Args:
            filename: Output filename
            
        Returns:
            Path to the saved file
        """
        summary = {
            'num_classes_per_layer': self.num_classes_per_layer,
            'class_names': self.class_names,
            'metrics_history': self.metrics_history,
            'learning_rates': self.learning_rates,
            'phase_metrics': self.phase_metrics,
            'generated_at': datetime.now().isoformat()
        }
        
        save_path = self.save_dir / filename
        with open(save_path, 'w') as f:
            json.dump(summary, f, indent=2)
            
        if self.verbose:
            self.logger.info(f"💾 Saved metrics summary to {save_path}")
            
        return save_path
    
    def generate_all_charts(self) -> Dict[str, Union[str, Dict[str, str]]]:
        """
        Generate all available charts and visualizations.
        
        Returns:
            Dictionary mapping chart types to file paths
        """
        results = {}
        
        # Generate individual charts
        results['training_curves'] = str(self.generate_training_curves() or "")
        results['learning_rate'] = str(self.generate_learning_rate_chart() or "")
        
        # Generate confusion matrices
        confusion_matrices = self.generate_confusion_matrices()
        results['confusion_matrices'] = {
            layer: str(path) for layer, path in confusion_matrices.items()
        }
        
        # Generate metrics comparisons
        for metric in ['accuracy', 'precision', 'recall', 'f1_score']:
            path = self.generate_metrics_comparison(metric)
            if path:
                results[f"{metric}_comparison"] = str(path)
        
        # Generate research dashboard
        results['research_dashboard'] = str(self.generate_research_dashboard() or "")
        
        # Save metrics summary
        results['metrics_summary'] = str(self.save_metrics_summary())
        
        return results
    
    def cleanup(self):
        """Clean up resources used by the visualization manager.
        
        This method releases any resources (e.g., figure handles, caches)
        used by the visualization manager. It should be called when the
        manager is no longer needed to free up system resources.
        """
        if self.verbose:
            self.logger.info("🧹 Cleaning up visualization resources")
        
        # Clear any cached figures or data
        if hasattr(self, 'metrics_history'):
            self.metrics_history.clear()
        
        # Clear any chart caches
        if hasattr(self, 'training_charts') and hasattr(self.training_charts, 'clear_cache'):
            self.training_charts.clear_cache()
        if hasattr(self, 'confusion_charts') and hasattr(self.confusion_charts, 'clear_cache'):
            self.confusion_charts.clear_cache()
        if hasattr(self, 'metrics_charts') and hasattr(self.metrics_charts, 'clear_cache'):
            self.metrics_charts.clear_cache()
        if hasattr(self, 'research_charts') and hasattr(self.research_charts, 'clear_cache'):
            self.research_charts.clear_cache()
        
        # Clear any other resources
        if hasattr(self, 'confusion_matrices'):
            self.confusion_matrices.clear()
        if hasattr(self, 'learning_rates'):
            self.learning_rates.clear()
        if hasattr(self, 'phase_metrics'):
            self.phase_metrics.clear()
    
    def __del__(self):
        """Clean up resources when the manager is garbage collected."""
        if hasattr(self, 'logger') and self.verbose:
            self.logger.info("Cleaning up visualization manager")
        self.cleanup()
