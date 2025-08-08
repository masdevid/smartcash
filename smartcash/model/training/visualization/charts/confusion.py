"""
Confusion matrix visualization utilities.

This module provides functionality for generating and visualizing confusion matrices
for model evaluation across different detection layers.
"""

from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes

# Import base chart class
from .base import BaseChart, VISUALIZATION_AVAILABLE
from smartcash.common.logger import get_logger

if VISUALIZATION_AVAILABLE:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LogNorm

class ConfusionMatrixCharts(BaseChart):
    """Charts for visualizing confusion matrices."""
    
    def __init__(self, save_dir: Union[str, Path] = "data/visualization", 
                 verbose: bool = False):
        """
        Initialize the confusion matrix chart generator.
        
        Args:
            save_dir: Directory to save generated charts
            verbose: Enable verbose logging
        """
        super().__init__(save_dir=save_dir, verbose=verbose)
        self.logger = get_logger(self.__class__.__name__)
    
    def plot_confusion_matrix(self, confusion_matrix: np.ndarray,
                            class_names: List[str] = None,
                            layer_name: str = None,
                            normalize: bool = True,
                            title: str = None) -> Optional[Path]:
        """
        Plot a confusion matrix with optional normalization.
        
        Args:
            confusion_matrix: 2D numpy array of shape [n_classes, n_classes]
            class_names: List of class names for axis labels
            layer_name: Name of the layer for the title
            normalize: Whether to normalize the confusion matrix
            title: Custom title (overrides auto-generated title)
            
        Returns:
            Path to saved chart or None if generation failed
        """
        if not VISUALIZATION_AVAILABLE or confusion_matrix.size == 0:
            return None
            
        # Create title if not provided
        if title is None:
            title = f"Confusion Matrix{f' - {layer_name}' if layer_name else ''}"
            if normalize:
                title += " (Normalized)"
        
        # Normalize if requested
        if normalize:
            confusion_matrix = confusion_matrix.astype('float') / (
                confusion_matrix.sum(axis=1)[:, np.newaxis] + 1e-8)
        
        # Create figure
        fig, ax = self.create_figure(title, figsize=(10, 8))
        
        # Plot confusion matrix
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt=".2f" if normalize else "d",
            cmap="Blues",
            cbar=True,
            square=True,
            xticklabels=class_names if class_names else "auto",
            yticklabels=class_names if class_names else "auto",
            ax=ax
        )
        
        # Rotate x-axis labels for better readability
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        
        # Set labels
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        
        # Adjust layout to prevent label cutoff
        fig.tight_layout()
        
        # Generate filename
        filename = f"confusion_matrix"
        if layer_name:
            filename += f"_{layer_name.lower().replace(' ', '_')}"
        if normalize:
            filename += "_normalized"
            
        # Save and return path
        return self.save_figure(fig, filename)
    
    def plot_multi_layer_confusion_matrices(
            self,
            confusion_matrices: Dict[str, np.ndarray],
            class_names: Dict[str, List[str]] = None,
            normalize: bool = True) -> Dict[str, Path]:
        """
        Plot confusion matrices for multiple layers.
        
        Args:
            confusion_matrices: Dictionary mapping layer names to confusion matrices
            class_names: Dictionary mapping layer names to class name lists
            normalize: Whether to normalize the confusion matrices
            
        Returns:
            Dictionary mapping layer names to saved chart paths
        """
        if not VISUALIZATION_AVAILABLE or not confusion_matrices:
            return {}
            
        saved_paths = {}
        
        for layer_name, cm in confusion_matrices.items():
            if cm is None or cm.size == 0:
                continue
                
            # Get class names for this layer if available
            layer_class_names = None
            if class_names and layer_name in class_names:
                layer_class_names = class_names[layer_name]
                
            # Generate the confusion matrix plot
            path = self.plot_confusion_matrix(
                confusion_matrix=cm,
                class_names=layer_class_names,
                layer_name=layer_name,
                normalize=normalize
            )
            
            if path:
                saved_paths[layer_name] = path
        
        return saved_paths
