"""
Base chart classes and utilities for visualization.

This module provides the foundation for all chart generation in the visualization system.
"""

from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

# Check if visualization libraries are available
try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.figure import Figure
    from matplotlib.axes import Axes
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    Figure = Any
    Axes = Any

class BaseChart:
    """Base class for all chart generators."""
    
    def __init__(self, save_dir: Union[str, Path] = "data/visualization", 
                 verbose: bool = False):
        """
        Initialize the base chart generator.
        
        Args:
            save_dir: Directory to save generated charts
            verbose: Enable verbose logging
        """
        self.save_dir = Path(save_dir)
        self.verbose = verbose
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def save_figure(self, fig: Figure, filename: str, dpi: int = 300, 
                   bbox_inches: str = 'tight', **kwargs) -> Optional[Path]:
        """
        Save a matplotlib figure to disk.
        
        Args:
            fig: Matplotlib figure to save
            filename: Output filename (without extension)
            dpi: DPI for the output image
            bbox_inches: Bounding box in inches
            **kwargs: Additional arguments to savefig()
            
        Returns:
            Path to saved file or None if saving failed
        """
        if not VISUALIZATION_AVAILABLE:
            return None
            
        try:
            save_path = self.save_dir / f"{filename}.png"
            fig.savefig(save_path, dpi=dpi, bbox_inches=bbox_inches, **kwargs)
            plt.close(fig)
            if self.verbose:
                print(f"📊 Saved chart to {save_path}")
            return save_path
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to save {filename}: {str(e)}")
            return None
    
    @staticmethod
    def create_figure(title: str, figsize: Tuple[int, int] = (10, 6), 
                     style: str = 'whitegrid', **kwargs) -> Tuple[Figure, Axes]:
        """
        Create a new matplotlib figure with consistent styling.
        
        Args:
            title: Chart title
            figsize: Figure size (width, height)
            style: Seaborn style to use
            **kwargs: Additional arguments to subplots()
            
        Returns:
            Tuple of (figure, axes) objects
        """
        if not VISUALIZATION_AVAILABLE:
            return None, None
            
        sns.set_style(style)
        fig, ax = plt.subplots(figsize=figsize, **kwargs)
        fig.suptitle(title, fontsize=12, fontweight='bold')
        return fig, ax
    
    @staticmethod
    def apply_common_styling(ax: Axes, xlabel: str = None, ylabel: str = None,
                            grid: bool = True, legend: bool = True, **kwargs) -> None:
        """
        Apply common styling to a matplotlib axes.
        
        Args:
            ax: Matplotlib axes to style
            xlabel: X-axis label
            ylabel: Y-axis label
            grid: Whether to show grid
            legend: Whether to show legend
            **kwargs: Additional styling parameters
        """
        if not VISUALIZATION_AVAILABLE or ax is None:
            return
            
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
            
        if grid:
            ax.grid(True, alpha=0.3)
            
        if legend and ax.get_legend_handles_labels()[0]:
            ax.legend()
            
        sns.despine(ax=ax, **kwargs)
