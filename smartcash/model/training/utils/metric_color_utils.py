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
    
    def get_loss_status(self, loss_value: float, loss_type: str = "yolo", 
                       epoch: int = 1, max_epochs: int = 100, phase: int = 1) -> MetricStatus:
        """
        Determine status for loss metrics with phase awareness.
        
        Args:
            loss_value: The loss value to evaluate
            loss_type: Type of loss ("yolo", "classification", "mse", "bce")
            epoch: Current training epoch
            max_epochs: Total training epochs
            phase: Training phase (1=single layer, 2=multi-layer)
            
        Returns:
            MetricStatus indicating performance level
        """
        # Adjust thresholds based on training progress
        progress_ratio = epoch / max_epochs
        early_training = progress_ratio < 0.3
        mid_training = 0.3 <= progress_ratio < 0.7
        late_training = progress_ratio >= 0.7
        
        if loss_type.lower() == "yolo":
            # Phase-aware YOLO loss thresholds
            if phase == 1:
                # Phase 1: Single-layer training (7 classes, simpler loss)
                if early_training:
                    if loss_value <= 1.5: return MetricStatus.EXCELLENT
                    elif loss_value <= 2.5: return MetricStatus.GOOD
                    elif loss_value <= 4.0: return MetricStatus.FAIR
                    elif loss_value <= 6.0: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
                elif mid_training:
                    if loss_value <= 1.0: return MetricStatus.EXCELLENT
                    elif loss_value <= 2.0: return MetricStatus.GOOD
                    elif loss_value <= 3.0: return MetricStatus.FAIR
                    elif loss_value <= 4.5: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
                else:  # late_training
                    if loss_value <= 0.5: return MetricStatus.EXCELLENT
                    elif loss_value <= 1.0: return MetricStatus.GOOD
                    elif loss_value <= 2.0: return MetricStatus.FAIR
                    elif loss_value <= 3.0: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
            else:
                # Phase 2: Multi-layer training (17 classes, more complex loss)
                # Higher thresholds are normal due to multi-layer complexity
                if early_training:
                    if loss_value <= 3.0: return MetricStatus.EXCELLENT
                    elif loss_value <= 6.0: return MetricStatus.GOOD
                    elif loss_value <= 10.0: return MetricStatus.FAIR
                    elif loss_value <= 15.0: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
                elif mid_training:
                    if loss_value <= 2.0: return MetricStatus.EXCELLENT
                    elif loss_value <= 4.0: return MetricStatus.GOOD
                    elif loss_value <= 7.0: return MetricStatus.FAIR
                    elif loss_value <= 12.0: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
                else:  # late_training
                    if loss_value <= 1.0: return MetricStatus.EXCELLENT
                    elif loss_value <= 2.5: return MetricStatus.GOOD
                    elif loss_value <= 5.0: return MetricStatus.FAIR
                    elif loss_value <= 8.0: return MetricStatus.POOR
                    else: return MetricStatus.CRITICAL
                
        elif loss_type.lower() == "classification":
            # Classification loss thresholds (cross-entropy)
            if early_training:
                if loss_value <= 0.5: return MetricStatus.EXCELLENT
                elif loss_value <= 1.0: return MetricStatus.GOOD
                elif loss_value <= 2.0: return MetricStatus.FAIR
                elif loss_value <= 3.0: return MetricStatus.POOR
                else: return MetricStatus.CRITICAL
            elif mid_training:
                if loss_value <= 0.3: return MetricStatus.EXCELLENT
                elif loss_value <= 0.7: return MetricStatus.GOOD
                elif loss_value <= 1.5: return MetricStatus.FAIR
                elif loss_value <= 2.5: return MetricStatus.POOR
                else: return MetricStatus.CRITICAL
            else:  # late_training
                if loss_value <= 0.1: return MetricStatus.EXCELLENT
                elif loss_value <= 0.3: return MetricStatus.GOOD
                elif loss_value <= 0.8: return MetricStatus.FAIR
                elif loss_value <= 1.5: return MetricStatus.POOR
                else: return MetricStatus.CRITICAL
                
        elif loss_type.lower() == "mse":
            # Mean Squared Error thresholds
            if loss_value <= 0.01: return MetricStatus.EXCELLENT
            elif loss_value <= 0.05: return MetricStatus.GOOD
            elif loss_value <= 0.1: return MetricStatus.FAIR
            elif loss_value <= 0.5: return MetricStatus.POOR
            else: return MetricStatus.CRITICAL
            
        elif loss_type.lower() == "bce":
            # Binary Cross-Entropy thresholds
            if loss_value <= 0.1: return MetricStatus.EXCELLENT
            elif loss_value <= 0.3: return MetricStatus.GOOD
            elif loss_value <= 0.7: return MetricStatus.FAIR
            elif loss_value <= 1.2: return MetricStatus.POOR
            else: return MetricStatus.CRITICAL
        
        # Default fallback
        if loss_value <= 0.5: return MetricStatus.EXCELLENT
        elif loss_value <= 1.0: return MetricStatus.GOOD
        elif loss_value <= 2.0: return MetricStatus.FAIR
        elif loss_value <= 4.0: return MetricStatus.POOR
        else: return MetricStatus.CRITICAL
    
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
        Apply color coding to a metric value based on its type and performance.
        
        Args:
            metric_name: Name of the metric (e.g., "train_loss", "val_accuracy")
            metric_value: The metric value
            epoch: Current training epoch
            max_epochs: Total training epochs
            phase: Training phase (1=single layer, 2=multi-layer)
            
        Returns:
            Colorized string representation of the metric
        """
        # Determine metric type and get status
        if "loss" in metric_name.lower():
            if "yolo" in metric_name.lower() or any(x in metric_name.lower() for x in ["train_loss", "val_loss", "total_loss"]):
                status = self.get_loss_status(metric_value, "yolo", epoch, max_epochs, phase)
            elif "mse" in metric_name.lower():
                status = self.get_loss_status(metric_value, "mse", epoch, max_epochs, phase)
            elif "bce" in metric_name.lower():
                status = self.get_loss_status(metric_value, "bce", epoch, max_epochs, phase)
            else:
                status = self.get_loss_status(metric_value, "classification", epoch, max_epochs, phase)
        elif "accuracy" in metric_name.lower():
            status = self.get_accuracy_status(metric_value, phase, epoch, max_epochs)
        elif "map" in metric_name.lower():
            status = self.get_map_status(metric_value)
        elif any(x in metric_name.lower() for x in ["precision", "recall", "f1"]):
            status = self.get_precision_recall_f1_status(metric_value)
        else:
            # Default to accuracy-like thresholds
            status = self.get_accuracy_status(metric_value, phase, epoch, max_epochs)
        
        # Apply colorization
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
        Format a complete metrics summary with color coding.
        
        Args:
            metrics: Dictionary of metric names and values
            epoch: Current training epoch
            max_epochs: Total training epochs
            phase: Training phase (1=single layer, 2=multi-layer)
            
        Returns:
            Formatted, colorized metrics summary
        """
        lines = []
        
        # Group metrics by type
        loss_metrics = {k: v for k, v in metrics.items() if "loss" in k.lower()}
        accuracy_metrics = {k: v for k, v in metrics.items() if "accuracy" in k.lower()}
        map_metrics = {k: v for k, v in metrics.items() if "map" in k.lower()}
        other_metrics = {k: v for k, v in metrics.items() 
                        if k not in loss_metrics and k not in accuracy_metrics and k not in map_metrics}
        
        # Format each group
        if loss_metrics:
            lines.append("📉 Loss Metrics:")
            for name, value in loss_metrics.items():
                colorized = self.colorize_metric(name, value, epoch, max_epochs, phase)
                lines.append(f"   {name}: {colorized}")
        
        if accuracy_metrics:
            lines.append("🎯 Accuracy Metrics:")
            for name, value in accuracy_metrics.items():
                colorized = self.colorize_metric(name, value, epoch, max_epochs, phase)
                lines.append(f"   {name}: {colorized}")
        
        if map_metrics:
            lines.append("📊 mAP Metrics:")
            for name, value in map_metrics.items():
                colorized = self.colorize_metric(name, value, epoch, max_epochs, phase)
                lines.append(f"   {name}: {colorized}")
        
        if other_metrics:
            lines.append("📋 Other Metrics:")
            for name, value in other_metrics.items():
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
    Quick function to get metric performance status.
    
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
    
    if "loss" in metric_name.lower():
        if "yolo" in metric_name.lower() or any(x in metric_name.lower() for x in ["train_loss", "val_loss", "total_loss"]):
            return colorizer.get_loss_status(metric_value, "yolo", epoch, max_epochs, phase)
        else:
            return colorizer.get_loss_status(metric_value, "classification", epoch, max_epochs, phase)
    elif "accuracy" in metric_name.lower():
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
    Get metrics with comprehensive color information for UI usage.
    
    Args:
        metrics: Dictionary of metric names and values
        epoch: Current training epoch
        max_epochs: Total training epochs
        phase: Training phase (1=single layer, 2=multi-layer)
        
    Returns:
        Dictionary with metric data including colors for all schemes
        Format: {
            metric_name: {
                'value': float,
                'status': MetricStatus,
                'colors': {
                    'terminal': str,
                    'html': str, 
                    'rgb': tuple,
                    'emoji': str
                },
                'formatted': {
                    'terminal': str,
                    'html': str,
                    'emoji': str
                }
            }
        }
    """
    result = {}
    
    for metric_name, value in metrics.items():
        if isinstance(value, (int, float)):
            status = get_metric_status(metric_name, value, epoch, max_epochs, phase)
            
            # Get colors for all schemes
            colors = {}
            formatted = {}
            
            for scheme in ColorScheme:
                colorizer = MetricColorizer(scheme)
                colors[scheme.value] = colorizer.get_status_indicator(status)
                formatted[scheme.value] = colorizer.colorize_metric(metric_name, value, epoch, max_epochs, phase)
            
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


# Example usage and testing
if __name__ == "__main__":
    # Demo different color schemes and metrics
    sample_metrics = {
        "train_loss": 2.3381,
        "val_loss": 2.1234,
        "val_accuracy": 0.25,
        "val_map50": 0.0,
        "layer_1_accuracy": 0.83,
        "layer_1_precision": 0.75,
        "layer_1_recall": 0.92,
        "layer_1_f1": 0.83
    }
    
    print("=== YOLO Metric Color Coding Demo ===\n")
    
    # Terminal colors
    print("🖥️  Terminal Colors:")
    colorizer_terminal = MetricColorizer(ColorScheme.TERMINAL)
    print(colorizer_terminal.format_metric_summary(sample_metrics, epoch=1, max_epochs=100))
    print()
    
    # Emoji indicators
    print("😊 Emoji Indicators:")
    colorizer_emoji = MetricColorizer(ColorScheme.EMOJI)
    print(colorizer_emoji.format_metric_summary(sample_metrics, epoch=1, max_epochs=100))
    print()
    
    # Show how thresholds change with training progress
    print("📈 Loss Status by Training Progress:")
    loss_value = 2.3
    for epoch_pct, label in [(10, "Early"), (50, "Mid"), (90, "Late")]:
        epoch = int(epoch_pct)
        status = get_metric_status("val_loss", loss_value, epoch, 100)
        color = get_metric_color("val_loss", loss_value, ColorScheme.EMOJI, epoch, 100)
        print(f"   {label} training (epoch {epoch}): {color} ({status.value})")