#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/metric_color_utils.py

Metric color utilities for visual indication of training performance levels.
Provides color coding and status indicators for different types of metrics.
"""

from typing import Dict, Tuple, Optional, Union
from enum import Enum
import math


class MetricStatus(Enum):
    """Metric performance status levels."""
    EXCELLENT = "excellent"
    GOOD = "good" 
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"


class ColorScheme(Enum):
    """Color schemes for different display contexts."""
    TERMINAL = "terminal"
    HTML = "html"
    RGB = "rgb"
    EMOJI = "emoji"


# Color mappings for different schemes
COLOR_MAPPINGS = {
    ColorScheme.TERMINAL: {
        MetricStatus.EXCELLENT: "\033[92m",  # Bright green
        MetricStatus.GOOD: "\033[32m",       # Green
        MetricStatus.FAIR: "\033[93m",       # Yellow
        MetricStatus.POOR: "\033[91m",       # Red
        MetricStatus.CRITICAL: "\033[95m",   # Magenta
    },
    ColorScheme.HTML: {
        MetricStatus.EXCELLENT: "#00FF00",  # Bright green
        MetricStatus.GOOD: "#008000",       # Green
        MetricStatus.FAIR: "#FFA500",       # Orange
        MetricStatus.POOR: "#FF4500",       # Red-orange
        MetricStatus.CRITICAL: "#FF0000",   # Red
    },
    ColorScheme.RGB: {
        MetricStatus.EXCELLENT: (0, 255, 0),    # Bright green
        MetricStatus.GOOD: (0, 128, 0),         # Green
        MetricStatus.FAIR: (255, 165, 0),       # Orange
        MetricStatus.POOR: (255, 69, 0),        # Red-orange
        MetricStatus.CRITICAL: (255, 0, 0),     # Red
    },
    ColorScheme.EMOJI: {
        MetricStatus.EXCELLENT: "🟢",  # Green circle
        MetricStatus.GOOD: "🔵",       # Blue circle
        MetricStatus.FAIR: "🟡",       # Yellow circle
        MetricStatus.POOR: "🟠",       # Orange circle
        MetricStatus.CRITICAL: "🔴",   # Red circle
    }
}

# Reset code for terminal colors
TERMINAL_RESET = "\033[0m"


class MetricColorizer:
    """
    Utility class for colorizing and categorizing training metrics.
    
    Provides context-aware color coding for different types of metrics
    commonly used in machine learning training.
    """
    
    def __init__(self, color_scheme: ColorScheme = ColorScheme.TERMINAL):
        """
        Initialize the metric colorizer.
        
        Args:
            color_scheme: Color scheme to use for output
        """
        self.color_scheme = color_scheme
        self.colors = COLOR_MAPPINGS[color_scheme]
    
    def _should_skip_coloring(self, metric_name: str) -> bool:
        """Check if a metric should skip color coding (loss metrics and non-performance fields)."""
        skip_patterns = ['loss', 'learning_rate', 'epoch', 'phase_completed', 'phase_status', 'best_val_loss']
        return any(pattern in metric_name.lower() for pattern in skip_patterns)
    
    def get_accuracy_status(self, accuracy: float, phase: int = 1, epoch: int = 1, max_epochs: int = 100) -> MetricStatus:
        """
        Determine status for accuracy metrics with phase awareness.
        
        Args:
            accuracy: Accuracy value (0.0 to 1.0)
            phase: Training phase (1=single layer, 2=multi-layer)
            epoch: Current training epoch
            max_epochs: Total training epochs
            
        Returns:
            MetricStatus indicating performance level
        """
        progress_ratio = epoch / max_epochs
        early_training = progress_ratio < 0.3
        
        if phase == 1:
            # Phase 1: Single-layer training - higher accuracy expected
            if accuracy >= 0.95: return MetricStatus.EXCELLENT
            elif accuracy >= 0.85: return MetricStatus.GOOD
            elif accuracy >= 0.70: return MetricStatus.FAIR
            elif accuracy >= 0.50: return MetricStatus.POOR
            else: return MetricStatus.CRITICAL
        else:
            # Phase 2: Multi-layer training - lower initial accuracy is normal
            if early_training:
                # Early Phase 2: Very low accuracy is expected as layers sync up
                if accuracy >= 0.60: return MetricStatus.EXCELLENT
                elif accuracy >= 0.40: return MetricStatus.GOOD
                elif accuracy >= 0.20: return MetricStatus.FAIR
                elif accuracy >= 0.05: return MetricStatus.POOR
                else: return MetricStatus.CRITICAL
            else:
                # Later Phase 2: Should improve towards Phase 1 levels
                if accuracy >= 0.90: return MetricStatus.EXCELLENT
                elif accuracy >= 0.75: return MetricStatus.GOOD
                elif accuracy >= 0.60: return MetricStatus.FAIR
                elif accuracy >= 0.40: return MetricStatus.POOR
                else: return MetricStatus.CRITICAL
    
    def get_map_status(self, map_value: float) -> MetricStatus:
        """
        Determine status for mAP (mean Average Precision) metrics.
        
        Args:
            map_value: mAP value (0.0 to 1.0)
            
        Returns:
            MetricStatus indicating performance level
        """
        if map_value >= 0.8: return MetricStatus.EXCELLENT
        elif map_value >= 0.6: return MetricStatus.GOOD
        elif map_value >= 0.4: return MetricStatus.FAIR
        elif map_value >= 0.2: return MetricStatus.POOR
        else: return MetricStatus.CRITICAL
    
    def get_precision_recall_f1_status(self, value: float) -> MetricStatus:
        """
        Determine status for precision, recall, or F1 metrics.
        
        Args:
            value: Metric value (0.0 to 1.0)
            
        Returns:
            MetricStatus indicating performance level
        """
        if value >= 0.9: return MetricStatus.EXCELLENT
        elif value >= 0.75: return MetricStatus.GOOD
        elif value >= 0.6: return MetricStatus.FAIR
        elif value >= 0.4: return MetricStatus.POOR
        else: return MetricStatus.CRITICAL
    
    def colorize_metric(self, metric_name: str, metric_value: float, 
                        epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> str:
        """
        Apply color coding to performance metrics (excludes loss metrics).
        
        Args:
            metric_name: Name of the metric
            metric_value: The metric value
            epoch: Current training epoch
            max_epochs: Total training epochs
            phase: Training phase
            
        Returns:
            Colorized string or plain string for loss metrics
        """
        # Skip coloring for loss metrics and non-performance fields
        if self._should_skip_coloring(metric_name):
            return f"{metric_value:.4f}"
        
        # Determine status for performance metrics only
        if "accuracy" in metric_name.lower():
            status = self.get_accuracy_status(metric_value, phase, epoch, max_epochs)
        elif "map" in metric_name.lower():
            status = self.get_map_status(metric_value)
        elif any(x in metric_name.lower() for x in ["precision", "recall", "f1"]):
            status = self.get_precision_recall_f1_status(metric_value)
        else:
            # Default to accuracy-like thresholds for unknown performance metrics
            status = self.get_accuracy_status(metric_value, phase, epoch, max_epochs)
        
        return self.apply_color(f"{metric_value:.4f}", status)
    
    def apply_color(self, text: str, status: MetricStatus) -> str:
        """
        Apply color formatting to text based on status.
        
        Args:
            text: Text to colorize
            status: Performance status
            
        Returns:
            Colorized text
        """
        if self.color_scheme == ColorScheme.TERMINAL:
            color_code = self.colors[status]
            return f"{color_code}{text}{TERMINAL_RESET}"
        elif self.color_scheme == ColorScheme.EMOJI:
            emoji = self.colors[status]
            return f"{emoji} {text}"
        else:
            # For HTML/RGB, return the color value alongside text
            color = self.colors[status]
            return f"[{color}] {text}"
    
    def get_status_indicator(self, status: MetricStatus) -> str:
        """
        Get a visual indicator for a status level.
        
        Args:
            status: Performance status
            
        Returns:
            Visual indicator (emoji, color code, etc.)
        """
        return self.colors[status]
    
    def format_metric_summary(self, metrics: Dict[str, float], 
                            epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> str:
        """
        Format metrics summary with color coding for performance metrics only.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current training epoch
            max_epochs: Total training epochs  
            phase: Training phase
            
        Returns:
            Formatted metrics summary (loss metrics without colors)
        """
        lines = []
        
        # Group metrics
        loss_metrics = {k: v for k, v in metrics.items() if "loss" in k.lower()}
        performance_metrics = {k: v for k, v in metrics.items() 
                             if not self._should_skip_coloring(k)}
        
        # Loss metrics (no colors)
        if loss_metrics:
            lines.append("📉 Loss Metrics:")
            for name, value in loss_metrics.items():
                lines.append(f"   {name}: {value:.4f}")
        
        # Performance metrics (with colors)
        if performance_metrics:
            lines.append("🎯 Performance Metrics:")
            for name, value in performance_metrics.items():
                colorized = self.colorize_metric(name, value, epoch, max_epochs, phase)
                lines.append(f"   {name}: {colorized}")
        
        return "\n".join(lines)


# Convenience functions
def get_metric_color(metric_name: str, metric_value: float, 
                    color_scheme: ColorScheme = ColorScheme.TERMINAL,
                    epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> str:
    """
    Quick function to get colorized metric value.
    
    Args:
        metric_name: Name of the metric
        metric_value: The metric value
        color_scheme: Color scheme to use
        epoch: Current training epoch
        max_epochs: Total training epochs
        phase: Training phase (1=single layer, 2=multi-layer)
        
    Returns:
        Colorized metric string
    """
    colorizer = MetricColorizer(color_scheme)
    return colorizer.colorize_metric(metric_name, metric_value, epoch, max_epochs, phase)


def get_metric_status(metric_name: str, metric_value: float,
                     epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> MetricStatus:
    """
    Get metric performance status for performance metrics only (excludes loss metrics).
    
    Args:
        metric_name: Name of the metric
        metric_value: The metric value
        epoch: Current training epoch
        max_epochs: Total training epochs
        phase: Training phase (1=single layer, 2=multi-layer)
        
    Returns:
        MetricStatus indicating performance level
    """
    colorizer = MetricColorizer()
    
    # Skip status calculation for loss metrics
    if colorizer._should_skip_coloring(metric_name):
        return MetricStatus.GOOD  # Default neutral status for non-performance metrics
    
    # Performance metrics only
    if "accuracy" in metric_name.lower():
        return colorizer.get_accuracy_status(metric_value, phase, epoch, max_epochs)
    elif "map" in metric_name.lower():
        return colorizer.get_map_status(metric_value)
    elif any(x in metric_name.lower() for x in ["precision", "recall", "f1"]):
        return colorizer.get_precision_recall_f1_status(metric_value)
    else:
        return colorizer.get_accuracy_status(metric_value, phase, epoch, max_epochs)


def get_metrics_with_colors(metrics: Dict[str, float], 
                          epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> Dict[str, Dict]:
    """
    Get metrics with color information for performance metrics only (excludes loss metrics).
    
    Args:
        metrics: Dictionary of metric names and values
        epoch: Current training epoch
        max_epochs: Total training epochs
        phase: Training phase (1=single layer, 2=multi-layer)
        
    Returns:
        Dictionary with metric data including colors for performance metrics
    """
    result = {}
    colorizer = MetricColorizer()
    
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            # Check if this metric should be colored
            if colorizer._should_skip_coloring(metric_name):
                # Loss and non-performance metrics: no colors
                result[metric_name] = {
                    'value': value,
                    'status': 'neutral',
                    'colors': {},
                    'formatted': {scheme.value: f"{value:.4f}" for scheme in ColorScheme}
                }
            else:
                # Performance metrics: with colors
                status = get_metric_status(metric_name, value, epoch, max_epochs, phase)
                
                colors = {}
                formatted = {}
                
                for scheme in ColorScheme:
                    scheme_colorizer = MetricColorizer(scheme)
                    colors[scheme.value] = scheme_colorizer.get_status_indicator(status)
                    formatted[scheme.value] = scheme_colorizer.colorize_metric(metric_name, value, epoch, max_epochs, phase)
                
                result[metric_name] = {
                    'value': value,
                    'status': status.value,
                    'colors': colors,
                    'formatted': formatted
                }
        else:
            # Non-numeric metrics
            result[metric_name] = {
                'value': value,
                'status': 'unknown',
                'colors': {},
                'formatted': {}
            }
    
    return result


def format_colorized_metrics(metrics: Dict[str, float], 
                           color_scheme: ColorScheme = ColorScheme.TERMINAL,
                           epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> str:
    """
    Quick function to format a full metrics dictionary with colors.
    
    Args:
        metrics: Dictionary of metric names and values
        color_scheme: Color scheme to use
        epoch: Current training epoch
        max_epochs: Total training epochs
        phase: Training phase (1=single layer, 2=multi-layer)
        
    Returns:
        Formatted, colorized metrics summary
    """
    colorizer = MetricColorizer(color_scheme)
    return colorizer.format_metric_summary(metrics, epoch, max_epochs, phase)
