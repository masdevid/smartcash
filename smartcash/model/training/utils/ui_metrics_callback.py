#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/ui_metrics_callback.py

Enhanced metrics callback that provides color information for both console and UI usage.
"""

from typing import Dict, Any, Callable, Optional, Tuple
from smartcash.model.training.utils.metric_color_utils import (
    MetricColorizer, ColorScheme, get_metrics_with_colors, get_metric_status
)


class UIMetricsCallback:
    """
    Enhanced metrics callback that provides comprehensive color and formatting information
    for both console display and UI integration.
    """
    
    def __init__(self, verbose: bool = True, console_scheme: ColorScheme = ColorScheme.EMOJI):
        """
        Initialize the UI metrics callback.
        
        Args:
            verbose: Whether to print verbose output
            console_scheme: Color scheme for console output
        """
        self.verbose = verbose
        self.console_scheme = console_scheme
        self.colorizer = MetricColorizer(console_scheme)
        
        # Storage for UI access
        self.latest_metrics = {}
        self.latest_colored_metrics = {}
        self.current_epoch = 0
        self.current_phase = ""
        
        # Optional UI callback function
        self.ui_callback: Optional[Callable] = None
    
    def set_ui_callback(self, callback: Callable[[str, int, Dict, Dict], None]):
        """
        Set a UI callback function that will receive metrics data.
        
        Args:
            callback: Function that receives (phase, epoch, metrics, colored_metrics)
        """
        self.ui_callback = callback
    
    def __call__(self, phase: str, epoch: int, metrics: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Process metrics and provide both console output and UI-ready data.
        
        Args:
            phase: Training phase name
            epoch: Current epoch number
            metrics: Dictionary of metric values
            **kwargs: Additional parameters (max_epochs, etc.)
            
        Returns:
            Dictionary containing original metrics plus color information
        """
        # Store current state
        self.current_epoch = epoch
        self.current_phase = phase
        self.latest_metrics = metrics.copy()
        
        # Get max epochs for context-aware coloring
        max_epochs = kwargs.get('max_epochs', 100)
        
        # Generate comprehensive color information
        colored_metrics = get_metrics_with_colors(metrics, epoch, max_epochs)
        self.latest_colored_metrics = colored_metrics
        
        # Console output if verbose
        if self.verbose:
            self._print_console_metrics(phase, epoch, metrics, colored_metrics, **kwargs)
        
        # Call UI callback if registered
        if self.ui_callback:
            try:
                self.ui_callback(phase, epoch, metrics, colored_metrics)
            except Exception as e:
                print(f"Warning: UI callback failed: {e}")
        
        # Return enhanced metrics data
        return {
            'original_metrics': metrics,
            'colored_metrics': colored_metrics,
            'phase': phase,
            'epoch': epoch,
            'max_epochs': max_epochs
        }
    
    def _print_console_metrics(self, phase: str, epoch: int, metrics: Dict[str, Any], 
                             colored_metrics: Dict[str, Dict], **kwargs):
        """Print formatted metrics to console."""
        print(f"📊 METRICS [{phase.upper()}] Epoch {epoch}:")
        
        if not metrics:
            print("    No metrics available")
            return
        
        # Determine layer filtering based on phase
        show_layers, filter_zeros = self._determine_layer_display(phase, metrics)
        
        # Print core metrics
        core_metrics = ['train_loss', 'val_loss', 'val_map50', 'val_map50_95', 
                       'val_precision', 'val_recall', 'val_f1', 'val_accuracy']
        
        for metric_name in core_metrics:
            if metric_name in metrics:
                self._print_metric(metric_name, metrics[metric_name], colored_metrics)
        
        # Print layer-specific metrics
        layer_metrics = self._filter_layer_metrics(metrics, show_layers, filter_zeros)
        for metric_name, value in layer_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
        
        # Print AP metrics
        ap_metrics = {k: v for k, v in metrics.items() 
                     if k.startswith('val_ap_') and isinstance(v, (int, float)) and v > 0.0001}
        for metric_name, value in ap_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
        
        # Print remaining metrics
        remaining_metrics = self._get_remaining_metrics(metrics, core_metrics, layer_metrics, ap_metrics)
        for metric_name, value in remaining_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
    
    def _print_metric(self, metric_name: str, value: Any, colored_metrics: Dict[str, Dict]):
        """Print a single metric with color coding."""
        if isinstance(value, (int, float)) and metric_name in colored_metrics:
            color_info = colored_metrics[metric_name]
            status_indicator = color_info['colors'][self.console_scheme.value]
            status_text = color_info['status']
            print(f"    {metric_name}: {status_indicator} {value:.4f} ({status_text})")
        else:
            print(f"    {metric_name}: {value}")
    
    def _determine_layer_display(self, phase: str, metrics: Dict[str, Any]) -> Tuple[list, bool]:
        """
        Determine which layers to show and whether to filter zeros based on training phase.
        
        This implements intelligent phase-aware metrics logic:
        - Phase 1: Only show layer_1, filter zeros for clean output
        - Phase 2: Show all layers (layer_1, layer_2, layer_3), don't filter
        - Single Phase: Auto-detect active layers and adapt display
        """
        if phase.lower() == 'training_phase_1':
            # Two-phase mode, phase 1: Only show layer_1 
            return ['layer_1'], True  # Filter zeros to reduce clutter
        elif phase.lower() == 'training_phase_2':
            # Two-phase mode, phase 2: Show all layers
            return ['layer_1', 'layer_2', 'layer_3'], False
        elif phase.lower() == 'training_phase_single':
            # Single-phase mode: Determine active layers from actual metrics
            # Check which layers have meaningful (non-zero) metrics to determine if it's single or multi layer mode
            layer_activity = {}
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                # Check if this layer has any meaningful metrics
                has_activity = any(
                    metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                    metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                    for metric in ['accuracy', 'precision', 'recall', 'f1']
                )
                layer_activity[layer] = has_activity
            
            # Determine active layers
            active_layers = [layer for layer, active in layer_activity.items() if active]
            if len(active_layers) == 1:
                # Single-phase, single-layer mode: only show the active layer
                return active_layers, True  # Filter zeros for clean output
            else:
                # Single-phase, multi-layer mode: show all layers
                return ['layer_1', 'layer_2', 'layer_3'], False
        else:
            # Default: show all layers
            return ['layer_1', 'layer_2', 'layer_3'], False
    
    def _filter_layer_metrics(self, metrics: Dict[str, Any], show_layers: list, filter_zeros: bool) -> Dict[str, Any]:
        """
        Filter layer-specific metrics based on intelligent display rules.
        
        Args:
            metrics: All available metrics
            show_layers: Which layers to display (from phase-aware logic)
            filter_zeros: Whether to filter out zero/near-zero values for cleaner output
        """
        layer_metrics = {}
        
        for metric_name, value in metrics.items():
            # Check if this is a layer metric and if we should show it
            for layer in show_layers:
                if (metric_name.startswith(f'{layer}_') or metric_name.startswith(f'val_{layer}_')):
                    # Apply zero filtering if specified (for cleaner output in single-layer modes)
                    if filter_zeros:
                        if isinstance(value, (int, float)) and value > 0.0001:  # Show only meaningful values
                            layer_metrics[metric_name] = value
                    else:
                        # Show all metrics for this layer (multi-layer modes)
                        layer_metrics[metric_name] = value
                    break
        
        return layer_metrics
    
    def _get_remaining_metrics(self, metrics: Dict[str, Any], core_metrics: list, 
                             layer_metrics: Dict[str, Any], ap_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics that don't fit other categories."""
        remaining = {}
        for metric_name, value in metrics.items():
            if (metric_name not in core_metrics and 
                metric_name not in layer_metrics and
                metric_name not in ap_metrics and
                not metric_name.startswith('layer_') and
                not metric_name.startswith('val_layer_') and
                not metric_name.startswith('val_ap_')):
                remaining[metric_name] = value
        return remaining
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get the latest metrics data."""
        return self.latest_metrics.copy()
    
    def get_latest_colored_metrics(self) -> Dict[str, Dict]:
        """Get the latest colored metrics data."""
        return self.latest_colored_metrics.copy()
    
    def get_metric_summary_for_ui(self) -> Dict[str, Any]:
        """Get a comprehensive summary for UI display with intelligent phase-aware filtering."""
        if not self.latest_colored_metrics:
            return {}
        
        # Apply intelligent phase-aware logic for UI display
        show_layers, filter_zeros = self._determine_layer_display(self.current_phase, self.latest_metrics)
        filtered_layer_metrics = self._filter_layer_metrics(self.latest_metrics, show_layers, filter_zeros)
        
        # Categorize metrics for UI display
        categories = {
            'loss_metrics': {},
            'accuracy_metrics': {},
            'map_metrics': {},
            'layer_metrics': {},
            'other_metrics': {}
        }
        
        for metric_name, color_data in self.latest_colored_metrics.items():
            # Apply phase-aware filtering for layer metrics
            if any(layer in metric_name.lower() for layer in ['layer_1', 'layer_2', 'layer_3']):
                # Only include layer metrics that passed the intelligent filtering
                if metric_name in filtered_layer_metrics:
                    categories['layer_metrics'][metric_name] = color_data
            elif 'loss' in metric_name.lower():
                categories['loss_metrics'][metric_name] = color_data
            elif 'accuracy' in metric_name.lower():
                categories['accuracy_metrics'][metric_name] = color_data
            elif 'map' in metric_name.lower():
                categories['map_metrics'][metric_name] = color_data
            else:
                categories['other_metrics'][metric_name] = color_data
        
        return {
            'categories': categories,
            'epoch': self.current_epoch,
            'phase': self.current_phase,
            'phase_info': {
                'active_layers': show_layers,
                'filter_zeros': filter_zeros,
                'display_mode': self._get_display_mode_description(self.current_phase, show_layers, filter_zeros)
            },
            'timestamp': None  # Could add timestamp if needed
        }
    
    def _get_display_mode_description(self, phase: str, show_layers: list, filter_zeros: bool) -> str:
        """Get a human-readable description of the current display mode."""
        if phase.lower() == 'training_phase_1':
            return "Phase 1: Single-layer focus (layer_1 only, filtered for clarity)"
        elif phase.lower() == 'training_phase_2':
            return "Phase 2: Multi-layer training (all layers visible)"
        elif phase.lower() == 'training_phase_single':
            if len(show_layers) == 1:
                return f"Single-phase: Single-layer mode ({show_layers[0]} only, filtered)"
            else:
                return "Single-phase: Multi-layer mode (all layers visible)"
        else:
            return f"Custom phase: {len(show_layers)} layers visible"


def create_ui_metrics_callback(verbose: bool = True, 
                             console_scheme: ColorScheme = ColorScheme.EMOJI,
                             ui_callback: Optional[Callable] = None) -> UIMetricsCallback:
    """
    Factory function to create a UI-enhanced metrics callback.
    
    Args:
        verbose: Whether to print console output
        console_scheme: Color scheme for console display
        ui_callback: Optional UI callback function
        
    Returns:
        UIMetricsCallback instance
    """
    callback = UIMetricsCallback(verbose, console_scheme)
    if ui_callback:
        callback.set_ui_callback(ui_callback)
    return callback


# Example UI callback function
def example_ui_callback(phase: str, epoch: int, metrics: Dict[str, Any], colored_metrics: Dict[str, Dict]):
    """
    Example UI callback function showing how to handle the color data.
    
    Args:
        phase: Training phase name
        epoch: Current epoch
        metrics: Original metrics dictionary
        colored_metrics: Enhanced metrics with color information
    """
    print(f"\n🎨 UI CALLBACK - {phase} Epoch {epoch}")
    print("Color data available for each metric:")
    
    for metric_name, color_data in colored_metrics.items():
        if isinstance(color_data.get('value'), (int, float)):
            value = color_data['value']
            status = color_data['status']
            html_color = color_data['colors']['html']
            emoji = color_data['colors']['emoji']
            
            print(f"  {metric_name}: {value:.4f} (status: {status}) [HTML: {html_color}] {emoji}")


if __name__ == "__main__":
    # Demo the UI metrics callback
    print("🔬 UI Metrics Callback Demo")
    print("=" * 50)
    
    # Create callback with example UI function
    callback = create_ui_metrics_callback(
        verbose=True,
        console_scheme=ColorScheme.EMOJI,
        ui_callback=example_ui_callback
    )
    
    # Simulate training metrics
    sample_metrics = {
        "train_loss": 0.7245,
        "val_loss": 2.3381,
        "val_accuracy": 0.25,
        "val_map50": 0.0,
        "layer_1_accuracy": 0.83,
        "layer_1_precision": 0.75,
        "layer_1_f1": 0.82
    }
    
    # Test the callback
    result = callback("training_phase_1", 1, sample_metrics, max_epochs=100)
    
    print("\n📋 Callback returned:")
    print(f"  Phase: {result['phase']}")
    print(f"  Epoch: {result['epoch']}")
    print(f"  Original metrics count: {len(result['original_metrics'])}")
    print(f"  Colored metrics count: {len(result['colored_metrics'])}")
    
    print("\n🎯 UI Summary:")
    ui_summary = callback.get_metric_summary_for_ui()
    for category, metrics in ui_summary['categories'].items():
        if metrics:
            print(f"  {category}: {len(metrics)} metrics")