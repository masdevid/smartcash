"""
Type definitions and data classes for visualization components.

This module provides type hints and data structures used throughout
the visualization package to ensure type safety and code clarity.
"""

from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np

# Type aliases
PathLike = Union[str, Path]
ColorType = Union[str, Tuple[float, float, float], Tuple[float, float, float, float]]
MetricValue = Union[int, float, List[float], Dict[str, float]]


class ChartType(Enum):
    """Supported chart types."""
    LINE = auto()
    BAR = auto()
    SCATTER = auto()
    HEATMAP = auto()
    CONFUSION_MATRIX = auto()
    PIE = auto()
    HISTOGRAM = auto()
    BOXPLOT = auto()
    VIOLIN = auto()
    AREA = auto()


class MetricType(Enum):
    """Supported metric types."""
    LOSS = auto()
    ACCURACY = auto()
    PRECISION = auto()
    RECALL = auto()
    F1_SCORE = auto()
    IOU = auto()
    MAP = auto()
    CUSTOM = auto()


@dataclass
class ChartConfig:
    """Configuration for a chart."""
    chart_type: ChartType
    title: str
    x_label: str = ""
    y_label: str = ""
    legend: bool = True
    grid: bool = True
    figsize: Tuple[int, int] = (10, 6)
    style: str = "seaborn"
    colors: List[ColorType] = field(default_factory=list)
    alpha: float = 0.7
    dpi: int = 100
    save_format: str = "png"
    tight_layout: bool = True


@dataclass
class MetricConfig:
    """Configuration for a metric to track."""
    name: str
    metric_type: MetricType
    label: str = ""
    higher_is_better: bool = True
    format_str: str = ".4f"
    visible: bool = True
    chart_config: Optional[ChartConfig] = None


@dataclass
class LayerConfig:
    """Configuration for a model layer."""
    name: str
    num_classes: int
    class_names: List[str] = field(default_factory=list)
    metrics: List[MetricConfig] = field(default_factory=list)
    visible: bool = True
    color: Optional[ColorType] = None


@dataclass
class TrainingPhase:
    """Configuration for a training phase."""
    name: str
    epochs: int
    learning_rate: float
    metrics: List[str] = field(default_factory=list)
    layers: List[str] = field(default_factory=list)
    description: str = ""


@dataclass
class VisualizationConfig:
    """Main configuration for visualization."""
    output_dir: PathLike = "visualizations"
    save_figures: bool = True
    show_figures: bool = False
    dpi: int = 100
    style: str = "seaborn"
    colors: List[ColorType] = field(default_factory=list)
    default_figsize: Tuple[int, int] = (10, 6)
    tight_layout: bool = True
    verbose: bool = False


@dataclass
class PlotData:
    """Data for plotting."""
    x: List[Any]
    y: List[Any]
    label: str = ""
    color: Optional[ColorType] = None
    alpha: float = 0.7
    linewidth: float = 2.0
    linestyle: str = "-"
    marker: Optional[str] = None
    markersize: int = 6


@dataclass
class FigureExportConfig:
    """Configuration for exporting figures."""
    filename: str
    format: str = "png"
    dpi: int = 300
    bbox_inches: str = "tight"
    pad_inches: float = 0.1
    transparent: bool = False
    metadata: Optional[Dict[str, Any]] = None


# Type aliases for common patterns
LayerMetrics = Dict[str, List[float]]
PhaseMetrics = Dict[str, Dict[str, List[float]]]
ConfusionMatrix = Dict[str, np.ndarray]
EpochMetrics = Dict[str, float]
TrainingHistory = List[Dict[str, float]]

# Callback types
FigureCallback = Callable[[Any], Any]  # Callback that takes a matplotlib figure
DataProcessor = Callable[[Dict[str, Any]], Dict[str, Any]]  # Data processing function

# Type for chart generation functions
ChartGenerator = Callable[..., Optional[Path]]
