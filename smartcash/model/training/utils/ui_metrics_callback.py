#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/ui_metrics_callback.py

Enhanced metrics callback that provides color information for both console and UI usage.
"""

from typing import Dict, Any, Callable, Optional, Tuple
from smartcash.model.training.utils.metric_color_utils import (
    MetricColorizer, ColorScheme, get_metrics_with_colors, get_metric_status
)
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


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
            **kwargs: Additional parameters (max_epochs, loss_breakdown, etc.)
            
        Returns:
            Dictionary containing original metrics plus color information and loss breakdown
        """
        # Store current state
        self.current_epoch = epoch
        self.current_phase = phase
        self.latest_metrics = metrics.copy()
        
        # Get max epochs for context-aware coloring
        max_epochs = kwargs.get('max_epochs', 100)
        
        # Extract phase number from phase string (e.g., "training_phase_2" -> 2)
        phase_num = 1  # Default to phase 1
        if isinstance(phase, str):
            if "phase_2" in phase.lower() or phase.lower() == "2":
                phase_num = 2
            elif "phase_1" in phase.lower() or phase.lower() == "1":
                phase_num = 1
        elif isinstance(phase, int):
            phase_num = phase
        
        # Generate comprehensive color information with phase awareness
        colored_metrics = get_metrics_with_colors(metrics, epoch, max_epochs, phase_num)
        self.latest_colored_metrics = colored_metrics
        
        # Extract loss breakdown before console output
        loss_breakdown = kwargs.get('loss_breakdown', {})
        
        # Debug: Log loss breakdown reception
        if loss_breakdown:
            logger.debug(f"UI Callback received loss_breakdown with {len(loss_breakdown)} components: {list(loss_breakdown.keys())}")
        else:
            logger.debug("UI Callback: No loss_breakdown received")
        
        # Console output if verbose
        if self.verbose:
            self._print_console_metrics(phase, epoch, metrics, colored_metrics, **kwargs)
            # Print loss breakdown if available
            if loss_breakdown:
                self._print_loss_breakdown(loss_breakdown)
        
        # Call UI callback if registered
        if self.ui_callback:
            try:
                self.ui_callback(phase, epoch, metrics, colored_metrics)
            except Exception as e:
                print(f"Warning: UI callback failed: {e}")
        
        # loss_breakdown already extracted above
        
        # Return enhanced metrics data
        return {
            'original_metrics': metrics,
            'colored_metrics': colored_metrics,
            'loss_breakdown': loss_breakdown,
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
        
        # Print phase-aware core metrics
        core_metrics = self._get_phase_appropriate_core_metrics(phase, metrics)
        
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
    
    def _print_loss_breakdown(self, loss_breakdown: Dict[str, Any]):
        """Print detailed loss breakdown information."""
        print("\n📊 LOSS BREAKDOWN:")
        
        # Core loss components - check both plain and prefixed versions
        core_losses = ['box_loss', 'obj_loss', 'cls_loss', 'total_loss']
        for loss_name in core_losses:
            # Check train_ and val_ prefixed versions
            for prefix in ['train_', 'val_', '']:
                prefixed_name = f"{prefix}{loss_name}"
                if prefixed_name in loss_breakdown:
                    value = loss_breakdown[prefixed_name]
                    if hasattr(value, 'item'):
                        value = value.item()
                    display_name = prefixed_name if prefix else loss_name
                    print(f"    {display_name}: {value:.6f}")
        
        # Multi-task loss components - check for both plain and prefixed versions
        has_layer_components = any(
            key.startswith('layer_') or key.startswith('train_layer_') or key.startswith('val_layer_') 
            for key in loss_breakdown.keys()
        )
        
        if has_layer_components:
            print("\n  Multi-task Loss Components:")
            
            # Group by layer, handling prefixes
            layer_losses = {}
            uncertainty_info = {}
            
            for key, value in loss_breakdown.items():
                if hasattr(value, 'item'):
                    value = value.item()
                
                # Handle both prefixed and non-prefixed layer keys
                if any(pattern in key for pattern in ['layer_', 'train_layer_', 'val_layer_']):
                    # Extract prefix and layer info
                    if key.startswith('train_'):
                        prefix = 'train_'
                        remainder = key[6:]  # Remove 'train_'
                    elif key.startswith('val_'):
                        prefix = 'val_'
                        remainder = key[4:]  # Remove 'val_'
                    else:
                        prefix = ''
                        remainder = key
                    
                    # Parse layer name and loss type
                    if remainder.startswith('layer_') and '_' in remainder:
                        parts = remainder.split('_')
                        if len(parts) >= 3:
                            layer_name = f"{prefix}{parts[0]}_{parts[1]}"  # e.g., 'train_layer_1'
                            loss_type = '_'.join(parts[2:])    # e.g., 'box_loss'
                            
                            if layer_name not in layer_losses:
                                layer_losses[layer_name] = {}
                            
                            if 'uncertainty' in loss_type:
                                uncertainty_info[layer_name] = value
                            elif 'weighted_loss' in loss_type or 'regularization' in loss_type:
                                layer_losses[layer_name][loss_type] = value
                            elif loss_type in ['box_loss', 'obj_loss', 'cls_loss', 'total_loss']:
                                layer_losses[layer_name][loss_type] = value
            
            # Print layer-wise losses
            for layer_name in sorted(layer_losses.keys()):
                print(f"\n    {layer_name.upper()}:")
                layer_loss_dict = layer_losses[layer_name]
                
                # Print individual loss components
                for loss_type in ['box_loss', 'obj_loss', 'cls_loss', 'total_loss']:
                    if loss_type in layer_loss_dict:
                        print(f"      {loss_type}: {layer_loss_dict[loss_type]:.6f}")
                
                # Print multi-task specific components
                if 'weighted_loss' in layer_loss_dict:
                    print(f"      weighted_loss: {layer_loss_dict['weighted_loss']:.6f}")
                if 'regularization' in layer_loss_dict:
                    print(f"      regularization: {layer_loss_dict['regularization']:.6f}")
                
                # Print uncertainty information
                if layer_name in uncertainty_info:
                    uncertainty = uncertainty_info[layer_name]
                    weight = 1.0 / (2.0 * uncertainty) if uncertainty > 0 else 0.0
                    print(f"      uncertainty (σ²): {uncertainty:.6f}")
                    print(f"      dynamic_weight: {weight:.6f}")
        
        # Additional loss information
        misc_info = {}
        for key, value in loss_breakdown.items():
            if not key.startswith('layer_') and key not in core_losses:
                if hasattr(value, 'item'):
                    value = value.item()
                misc_info[key] = value
        
        if misc_info:
            print("\n  Additional Loss Info:")
            for key, value in misc_info.items():
                if isinstance(value, (int, float)):
                    print(f"    {key}: {value:.6f}")
                else:
                    print(f"    {key}: {value}")
        
        print()  # Empty line for spacing
    
    def _print_metric(self, metric_name: str, value: Any, colored_metrics: Dict[str, Dict]):
        """Print a single metric with color coding."""
        if isinstance(value, (int, float)) and metric_name in colored_metrics:
            color_info = colored_metrics[metric_name]
            status_indicator = color_info['colors'][self.console_scheme.value]
            status_text = color_info['status']
            print(f"    {metric_name}: {status_indicator} {value:.4f} ({status_text})")
        else:
            print(f"    {metric_name}: {value}")
    
    def _get_phase_appropriate_core_metrics(self, phase: str, metrics: Dict[str, Any]) -> list:
        """
        Get core metrics appropriate for the current training phase.
        
        Args:
            phase: Training phase name
            metrics: Available metrics
            
        Returns:
            List of core metric names appropriate for this phase
        """
        # Extract phase number for proper logic
        phase_num = self._extract_phase_number(phase, metrics)
        
        if phase_num == 1:
            # Phase 1: Focus on core training metrics only
            core_metrics = [
                'train_loss', 'val_loss', 'learning_rate', 'epoch',
                'val_precision', 'val_recall', 'val_f1', 'val_accuracy'
            ]
            # mAP metrics disabled for performance - focusing on classification metrics
            # Note: to re-enable, uncomment the line below:
            # if 'val_map50' in metrics:
            #     core_metrics.insert(2, 'val_map50')  # Add after val_loss
        else:
            # Phase 2: Focus on core training metrics (mAP disabled for performance)
            core_metrics = [
                'train_loss', 'val_loss', 'learning_rate', 'epoch',
                'val_precision', 'val_recall', 'val_f1', 'val_accuracy'
            ]
            # No additional metrics needed - keep it clean
        
        # Filter to only include metrics that actually exist
        return [metric for metric in core_metrics if metric in metrics]
    
    def _extract_phase_number(self, phase: str, metrics: Dict[str, Any]) -> int:
        """
        Extract phase number from phase string or infer from metrics.
        
        Args:
            phase: Training phase name
            metrics: Available metrics for inference
            
        Returns:
            Phase number (1 or 2)
        """
        # Direct phase detection from string
        if isinstance(phase, str):
            if "phase_2" in phase.lower() or phase.lower() == "2":
                return 2
            elif "phase_1" in phase.lower() or phase.lower() == "1":
                return 1
        elif isinstance(phase, int):
            return phase
        
        # Infer phase from available metrics (fallback)
        # Phase 2 indicators: multiple active layers (simpler detection)
        phase_2_indicators = []  # No specific research metrics needed
        
        has_phase_2_metrics = False  # Use layer activity detection instead
        
        # Also check for multiple active layers (indication of Phase 2)
        active_layers = 0
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            has_layer_activity = any(
                metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            )
            if has_layer_activity:
                active_layers += 1
        
        # If we have Phase 2 specific metrics OR multiple active layers, it's Phase 2
        if has_phase_2_metrics or active_layers > 1:
            return 2
        else:
            return 1
    
    def _determine_layer_display(self, phase: str, metrics: Dict[str, Any]) -> Tuple[list, bool]:
        """
        Determine which layers to show and whether to filter zeros based on training phase.
        
        This implements intelligent phase-aware metrics logic aligned with new loss system:
        - Phase 1: Simple YOLO loss - focus on layer_1, filter zeros for clean output
        - Phase 2: Multi-task loss - show all layers with meaningful data
        - Auto-detect based on actual metrics and loss type
        """
        # Use the same phase detection logic
        phase_num = self._extract_phase_number(phase, metrics)
        
        if phase_num == 1:
            # Phase 1: Simple YOLO loss training
            # Focus on layer_1 since that's what simple YOLO optimizes
            # Filter zeros to show only meaningful metrics
            return ['layer_1'], True
        else:
            # Phase 2: Multi-task loss training
            # Show all layers that have meaningful data
            active_layers = []
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                # Check for any meaningful metrics for this layer
                has_activity = any(
                    metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                    metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                    for metric in ['accuracy', 'precision', 'recall', 'f1']
                )
                if has_activity:
                    active_layers.append(layer)
            
            # If no layers detected as active, default to all layers
            if not active_layers:
                active_layers = ['layer_1', 'layer_2', 'layer_3']
            
            # Don't filter zeros in Phase 2 - we want to see all multi-task metrics
            return active_layers, False
    
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
        """Get metrics that don't fit other categories, filtering out unwanted metrics."""
        # Patterns of metrics to exclude completely
        exclude_patterns = [
            'hierarchical_accuracy', 'research_primary_metric', 'denomination_accuracy',
            'multi_layer_benefit', '_contribution', 'layer_2_', 'layer_3_',
            'map50_95', 'map75', 'ap_'
        ]
        
        remaining = {}
        for metric_name, value in metrics.items():
            # Skip if already categorized
            if (metric_name in core_metrics or 
                metric_name in layer_metrics or
                metric_name in ap_metrics):
                continue
                    
            # Skip if matches exclude patterns
            should_exclude = any(pattern in metric_name.lower() for pattern in exclude_patterns)
            if should_exclude:
                continue
                
            # Skip layer metrics for inactive layers
            if (metric_name.startswith('layer_') or 
                metric_name.startswith('val_layer_') or
                metric_name.startswith('val_ap_')):
                continue
                
            # Only include genuinely useful remaining metrics
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
        phase_num = self._extract_phase_number(phase, self.latest_metrics)
        
        if phase_num == 1:
            return "Phase 1: Simple YOLO loss training (layer_1 focus, filtered for clarity)"
        elif phase_num == 2:
            layer_count = len(show_layers)
            return f"Phase 2: Multi-task loss training ({layer_count} active layers, all metrics visible)"
        else:
            return f"Custom phase: {len(show_layers)} layers visible (filter_zeros: {filter_zeros})"


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