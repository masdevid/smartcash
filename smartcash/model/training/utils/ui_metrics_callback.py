#!/usr/bin/env python3
"""
File: smartcash/model/training/utils/ui_metrics_callback.py

Enhanced metrics callback that provides color information for both console and UI usage.
"""

import json
from pathlib import Path
from typing import Dict, Any, Callable, Optional, Tuple
from smartcash.model.training.utils.metric_color_utils import (
    MetricColorizer, ColorScheme, get_metrics_with_colors
)
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class UIMetricsCallback:
    """
    Enhanced metrics callback that provides comprehensive color and formatting information
    for both console display and UI integration.
    """
    
    def _get_safe_color_scheme(self, preferred_scheme: Optional[ColorScheme] = None) -> ColorScheme:
        """
        Get a safe color scheme, falling back to terminal colors if emojis aren't supported.
        
        Args:
            preferred_scheme: The preferred color scheme, or None for auto-detect
            
        Returns:
            A supported color scheme
        """
        if preferred_scheme is not None:
            # If a specific scheme was requested, try to use it
            try:
                if preferred_scheme == ColorScheme.EMOJI:
                    # Test if emojis are supported
                    "".join(COLOR_MAPPINGS[ColorScheme.EMOJI].values())
                return preferred_scheme
            except (UnicodeEncodeError, UnicodeDecodeError, KeyError, AttributeError) as e:
                logger.warning(f"Preferred color scheme '{preferred_scheme}' not available: {e}")
        
        # Default to terminal colors for safety
        return ColorScheme.TERMINAL

    def __init__(self, verbose: bool = True, console_scheme: ColorScheme = None, 
                 training_logs_dir: str = "logs/training"):
        """
        Initialize the UI metrics callback. 
        
        Args:
            verbose: Whether to print verbose output
            console_scheme: Color scheme for console output. Defaults to EMOJI if supported, otherwise TERMINAL.
            training_logs_dir: Directory containing JSON metrics files
        """
        self.verbose = verbose
        
        # Set up color scheme with safe fallback
        self.console_scheme = self._get_safe_color_scheme(console_scheme)
        logger.debug(f"Using color scheme: {self.console_scheme}")
            
        self.colorizer = MetricColorizer(self.console_scheme)
        self.training_logs_dir = Path(training_logs_dir)
        
        # Storage for UI access
        self.latest_metrics = {}
        self.latest_colored_metrics = {}
        self.current_epoch = 0
        self.current_phase = ""
        
        # Model and data info for filename generation
        self.backbone = "unknown"
        self.data_name = "data"
        
        # Optional UI callback function
        self.ui_callback: Optional[Callable] = None
    
    def set_ui_callback(self, callback: Callable[[str, int, Dict, Dict, Dict], None]):
        """
        Set a UI callback function that will receive metrics data. 
        
        Args:
            callback: Function that receives (phase, epoch, metrics, colored_metrics, loss_breakdown)
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
        
        # Extract loss breakdown from kwargs and store for access in print methods
        loss_breakdown = kwargs.get('loss_breakdown', {})
        self._current_loss_breakdown = loss_breakdown
        
        # Debug: Log loss breakdown reception
        if loss_breakdown:
            logger.debug(f"UI Callback received loss_breakdown with {len(loss_breakdown)} components: {list(loss_breakdown.keys())}")
        else:
            logger.debug("UI Callback: No loss_breakdown received")
        
        # Console output if verbose
        if self.verbose:
            self._print_console_metrics(phase, epoch, metrics, colored_metrics, **kwargs)
            # Loss breakdown is already printed within _print_console_metrics
        
        # Call UI callback if registered (pass loss breakdown too)
        if self.ui_callback:
            try:
                # Enhanced UI callback with loss breakdown
                self.ui_callback(phase, epoch, metrics, colored_metrics, loss_breakdown)
            except Exception as e:
                print(f"Warning: UI callback failed: {e}")
        
        # Return enhanced metrics data
        return {
            'original_metrics': metrics,
            'colored_metrics': colored_metrics,
            'loss_breakdown': loss_breakdown,
            'phase': phase,
            'epoch': epoch,
            'max_epochs': max_epochs,
            'train_loss_available': 'train_loss' in metrics,
            'train_loss_value': metrics.get('train_loss', 'N/A')
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
        
        # Always show detailed loss breakdown if available (regardless of filtering)
        loss_breakdown = getattr(self, '_current_loss_breakdown', {})
        if loss_breakdown:
            self._print_loss_breakdown(loss_breakdown)
        
        # Print layer-specific metrics (include null values for debugging)
        layer_metrics = self._filter_layer_metrics(metrics, show_layers, filter_zeros=False)  # Don't filter zeros for debugging
        for metric_name, value in layer_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
        
        # Print AP metrics (don't filter by value - show all available)
        ap_metrics = {k: v for k, v in metrics.items() 
                     if k.startswith('val_ap_') and isinstance(v, (int, float))}
        for metric_name, value in ap_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
        
        # Print remaining metrics
        remaining_metrics = self._get_remaining_metrics(metrics, core_metrics, layer_metrics, ap_metrics)
        for metric_name, value in remaining_metrics.items():
            self._print_metric(metric_name, value, colored_metrics)
    
    def _print_loss_breakdown(self, loss_breakdown: Dict[str, Any]):
        """Print simplified loss breakdown information."""
        print("\n📊 LOSS BREAKDOWN:")
        
        # Core loss components
        core_losses = ['train_loss', 'val_loss', 'box_loss', 'obj_loss', 'cls_loss']
        for loss_name in core_losses:
            for prefix in ['train_', 'val_', '']:
                key = f"{prefix}{loss_name}" if prefix and not loss_name.startswith(prefix) else loss_name
                if key in loss_breakdown:
                    value = loss_breakdown[key]
                    if hasattr(value, 'item'):
                        value = value.item()
                    print(f"    {key}: {value:.6f}")
                    break
        
        # Layer-specific losses (simplified)
        layer_losses = {k: v for k, v in loss_breakdown.items() if 'layer_' in k}
        if layer_losses:
            print("  Layer Losses:")
            for key, value in sorted(layer_losses.items()):
                if hasattr(value, 'item'):
                    value = value.item()
                print(f"    {key}: {value:.6f}")
        
        print()  # Empty line for spacing
    
    def _extract_phase_number(self, phase: str, metrics: Dict[str, Any] = None) -> int:
        """Extract phase number from phase string or infer from metrics."""
        if isinstance(phase, int):
            return phase
        if isinstance(phase, str):
            if "phase_2" in phase.lower() or phase.lower() == "2":
                return 2
            elif "phase_1" in phase.lower() or phase.lower() == "1":
                return 1
        
        # Infer from metrics if available
        if metrics:
            active_layers = sum(1 for layer in ['layer_1', 'layer_2', 'layer_3']
                              if any(metrics.get(f'{layer}_{metric}', 0) > 0.0001
                                   for metric in ['accuracy', 'precision']))
            return 2 if active_layers > 1 else 1
        return 1
    
    def _print_metric(self, metric_name: str, value: Any, colored_metrics: Dict[str, Dict]):
        """Print a single metric with color coding, respecting TASK.md no-indicator requirements."""
        # TASK.md specifies these metrics should have no color/indicator
        no_indicator_metrics = [
            'train_loss', 'val_loss',  # Core losses
            # Loss component patterns (will be checked with endswith)
            '_box_loss', '_obj_loss', '_cls_loss'
        ]
        
        # Check if this metric should have no color indicator
        should_suppress_color = (
            metric_name in no_indicator_metrics or
            any(metric_name.endswith(pattern) for pattern in no_indicator_metrics if pattern.startswith('_'))
        )
        
        # Handle null values
        if value is None:
            print(f"    {metric_name}: null")
        elif should_suppress_color:
            # Print without color indicator as per TASK.md
            print(f"    {metric_name}: {value:.4f}" if isinstance(value, (int, float)) else f"    {metric_name}: {value}")
        elif isinstance(value, (int, float)) and metric_name in colored_metrics:
            # Print with color indicator for performance metrics
            color_info = colored_metrics[metric_name]
            status_indicator = color_info['colors'][self.console_scheme.value]
            status_text = color_info['status']
            print(f"    {metric_name}: {status_indicator} {value:.4f} ({status_text})")
        else:
            # Print plain for other metrics
            print(f"    {metric_name}: {value}")
    
    def _get_phase_appropriate_core_metrics(self, phase: str, metrics: Dict[str, Any]) -> list:
        """Get core metrics appropriate for the current training phase."""
        phase_num = self._extract_phase_number(phase, metrics)
        
        # Base core metrics for both phases
        core_metrics = ['train_loss', 'val_loss', 'val_accuracy', 'val_precision', 
                       'val_recall', 'val_f1', 'val_map50', 'learning_rate', 'epoch']
        
        # Phase 2: Add additional mAP metrics
        if phase_num == 2:
            core_metrics.extend(['val_map50_precision', 'val_map50_recall', 
                               'val_map50_f1', 'val_map50_accuracy'])
        
        # Return only metrics that exist
        return [m for m in core_metrics if m in metrics]
    
    def _determine_layer_display(self, phase: str, metrics: Dict[str, Any]) -> Tuple[list, bool]:
        """Determine which layers to show based on training phase."""
        phase_num = self._extract_phase_number(phase, metrics)
        
        if phase_num == 1:
            return ['layer_1'], False
        
        # Phase 2: Include all active layers
        active_layers = ['layer_1']
        for layer in ['layer_2', 'layer_3']:
            if any(metrics.get(f'{p}{layer}_{m}', 0) > 0 
                  for m in ['accuracy', 'precision'] 
                  for p in ['', 'val_']):
                active_layers.append(layer)
        
        return active_layers, False
    
    def _filter_layer_metrics(self, metrics: Dict[str, Any], show_layers: list, filter_zeros: bool) -> Dict[str, Any]:
        """
        Filter and order layer-specific metrics according to TASK.md specification.
        
        Args:
            metrics: All available metrics
            show_layers: Which layers to display (from phase-aware logic)
            filter_zeros: Whether to filter out zero/near-zero values for cleaner output
            
        Returns:
            Ordered dictionary of layer metrics following TASK.md order:
            - layer_*_accuracy, layer_*_precision, layer_*_recall, layer_*_f1
            - val_layer_*_box_loss, val_layer_*_obj_loss, val_layer_*_cls_loss
        """
        # TASK.md specified order for layer metrics
        layer_metric_types = [
            'accuracy', 'precision', 'recall', 'f1',  # Performance metrics first
            'box_loss', 'obj_loss', 'cls_loss'       # Loss components last (no color indicators)
        ]
        
        ordered_layer_metrics = {}
        
        # Process each layer in order
        for layer in show_layers:
            # Process each metric type in the specified order
            for metric_type in layer_metric_types:
                # Check for both layer_X_metric and val_layer_X_metric formats
                possible_names = [
                    f'{layer}_{metric_type}',           # e.g., layer_1_accuracy
                    f'val_{layer}_{metric_type}'        # e.g., val_layer_1_box_loss
                ]
                
                for metric_name in possible_names:
                    if metric_name in metrics:
                        value = metrics[metric_name]
                        
                        # Apply filtering if requested
                        should_include = True
                        if filter_zeros:
                            # Don't filter loss metrics - they can legitimately be zero or small
                            if 'loss' in metric_name.lower():
                                should_include = True
                            elif value is None:
                                should_include = True  # Include null values for debugging
                            elif isinstance(value, (int, float)) and value <= 0.0001:
                                should_include = False  # Skip very small performance metrics
                        
                        if should_include:
                            ordered_layer_metrics[metric_name] = value
        
        return ordered_layer_metrics
    
    def _get_remaining_metrics(self, metrics: Dict[str, Any], core_metrics: list, 
                             layer_metrics: Dict[str, Any], ap_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Get metrics that don't fit other categories, filtering out unwanted metrics."""
        # Patterns of metrics to exclude completely
        exclude_patterns = [
            'hierarchical_accuracy', 'research_primary_metric', 'denomination_accuracy',
            'multi_layer_benefit', '_contribution', 'layer_2_', 'layer_3_',
            'map50_95', 'map75', 'ap_', 'train_map50'  # Some mAP metrics disabled, but allow val_map50_*
        ]
        
        remaining = {}
        for metric_name, value in metrics.items():
            # Skip if already categorized
            if (
                metric_name in core_metrics or 
                metric_name in layer_metrics or
                metric_name in ap_metrics):
                continue
                    
            # Skip if matches exclude patterns
            should_exclude = any(pattern in metric_name.lower() for pattern in exclude_patterns)
            if should_exclude:
                continue
                
            # Skip layer metrics for inactive layers
            if (
                metric_name.startswith('layer_') or 
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
    
    def load_metrics_from_json(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Load metrics data from JSON history files.
        
        Args:
            session_id: Optional session ID to load specific session, otherwise loads latest
            
        Returns:
            Dictionary with metrics history and loss breakdown data
        """
        try:
            if session_id:
                # Fallback to old naming scheme for backward compatibility
                metrics_file = self.training_logs_dir / f"metrics_history_{session_id}.json"
            else:
                # Load latest session for this backbone
                latest_file = self.training_logs_dir / f"latest_metrics_{self.backbone}.json"
                if latest_file.exists():
                    with open(latest_file, 'r') as f:
                        latest_data = json.load(f)
                        metrics_file = Path(latest_data['file_paths']['metrics'])
                else:
                    # Look for new structured naming pattern
                    new_pattern = f"metrics_history_{self.backbone}_{self.data_name}_phase*.json"
                    metrics_files = list(self.training_logs_dir.glob(new_pattern))
                    if not metrics_files:
                        # Try old pattern as fallback
                        metrics_files = list(self.training_logs_dir.glob("metrics_history_*.json"))
                    if not metrics_files:
                        logger.warning(f"No metrics files found for backbone {self.backbone}")
                        return {}
                    metrics_file = max(metrics_files, key=lambda f: f.stat().st_mtime)
            
            if not metrics_file.exists():
                logger.warning(f"Metrics file not found: {metrics_file}")
                return {}
            
            with open(metrics_file, 'r') as f:
                metrics_history = json.load(f)
            
            logger.debug(f"Loaded {len(metrics_history)} epoch records from {metrics_file}")
            return {'metrics_history': metrics_history, 'file_path': str(metrics_file)}
            
        except Exception as e:
            logger.error(f"Failed to load metrics from JSON: {e}")
            return {}
    
    def get_latest_epoch_metrics(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics for the latest epoch from JSON files."""
        data = self.load_metrics_from_json(session_id)
        if not data or 'metrics_history' not in data or not data['metrics_history']:
            return {}
        
        latest_epoch = data['metrics_history'][-1]
        
        # Extract key metrics groups
        map_metrics = {k: v for k, v in latest_epoch.items() 
                      if k.startswith('val_map') and v is not None}
        
        loss_breakdown = {k: v for k, v in latest_epoch.items() 
                         if 'loss' in k.lower() and v is not None}
        
        layer_metrics = {k: v for k, v in latest_epoch.items() 
                        if k.startswith('layer_') and v is not None}
        
        return {
            'epoch': latest_epoch.get('epoch', 0),
            'phase': latest_epoch.get('phase', 1),
            'timestamp': latest_epoch.get('timestamp', ''),
            'core_metrics': {
                'train_loss': latest_epoch.get('train_loss', 0),
                'val_loss': latest_epoch.get('val_loss', 0),
                'learning_rate': latest_epoch.get('learning_rate', 0)
            },
            'map_metrics': map_metrics,
            'loss_breakdown': loss_breakdown,
            'layer_metrics': layer_metrics
        }
    
    def get_phase_metrics_history(self, phase: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Get metrics history for a specific phase."""
        data = self.load_metrics_from_json(session_id)
        if not data or 'metrics_history' not in data:
            return {}
        
        phase_data = [r for r in data['metrics_history'] if r['phase'] == phase]
        if not phase_data:
            return {'phase': phase, 'epochs': []}
        
        epochs = [r['epoch'] for r in phase_data]
        
        return {
            'phase': phase,
            'total_epochs': len(phase_data),
            'best_val_loss': min(r['val_loss'] for r in phase_data),
            'best_map50': max(r.get('val_map50', 0) for r in phase_data),
            'series': {
                'epochs': epochs,
                'train_loss': [r['train_loss'] for r in phase_data],
                'val_loss': [r['val_loss'] for r in phase_data],
                'val_map50': [r.get('val_map50', 0) for r in phase_data],
                'learning_rate': [r['learning_rate'] for r in phase_data]
            }
        }
    
    def get_metric_summary_for_ui(self) -> Dict[str, Any]:
        """Get a comprehensive summary for UI display."""
        # Try to load latest metrics from JSON first
        latest_data = self.get_latest_epoch_metrics()
        
        if latest_data:
            metrics = {**latest_data['core_metrics'], **latest_data['map_metrics'], **latest_data['layer_metrics']}
            phase, epoch, loss_breakdown = latest_data['phase'], latest_data['epoch'], latest_data['loss_breakdown']
        else:
            if not self.latest_colored_metrics:
                return {}
            metrics, phase, epoch, loss_breakdown = self.latest_metrics, self.current_phase, self.current_epoch, {}
        
        show_layers, filter_zeros = self._determine_layer_display(str(phase), metrics)
        filtered_layer_metrics = self._filter_layer_metrics(metrics, show_layers, filter_zeros)
        colored_metrics = get_metrics_with_colors(metrics, epoch, 100, int(phase))
        
        # Categorize metrics
        categories = {'loss_metrics': {}, 'accuracy_metrics': {}, 'map_metrics': {}, 'layer_metrics': {}, 'other_metrics': {}}
        
        for metric_name, color_data in colored_metrics.items():
            if any(layer in metric_name.lower() for layer in ['layer_1', 'layer_2', 'layer_3']):
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
            'epoch': epoch,
            'phase': phase,
            'loss_breakdown': loss_breakdown,
            'phase_info': {
                'active_layers': show_layers,
                'display_mode': self._get_display_mode_description(str(phase), show_layers, filter_zeros)
            },
            'data_source': 'json' if latest_data else 'memory'
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
            return f"Custom phase: {len(show_layers)} layers visible"


def create_ui_metrics_callback(verbose: bool = True, 
                             console_scheme: Optional[ColorScheme] = None,
                             ui_callback: Optional[Callable] = None,
                             training_logs_dir: str = "logs/training",
                             backbone: str = "unknown",
                             data_name: str = "data"):
    """
    Factory function to create a UI-enhanced metrics callback.
    
    Args:
        verbose: Whether to print console output
        console_scheme: Color scheme for console display. 
                      Defaults to EMOJI if supported, otherwise falls back to TERMINAL.
        ui_callback: Optional UI callback function
        training_logs_dir: Directory containing JSON metrics files
        backbone: Backbone model name for filename
        data_name: Dataset name for filename
        
    Returns:
        UIMetricsCallback instance
    """
    try:
        # Create the callback with safe color scheme handling
        callback = UIMetricsCallback(
            verbose=verbose,
            console_scheme=console_scheme,
            training_logs_dir=training_logs_dir
        )
        
        # Set model and data info for logging
        callback.backbone = backbone
        callback.data_name = data_name
        
        # Set UI callback if provided
        if ui_callback is not None:
            callback.set_ui_callback(ui_callback)
        
        return callback
        
    except Exception as e:
        logger.error(f"Failed to create metrics callback: {e}")
        # Return a minimal working callback with terminal colors as fallback
        return UIMetricsCallback(
            verbose=verbose,
            console_scheme=ColorScheme.TERMINAL,
            training_logs_dir=training_logs_dir
        )


# Example UI callback function
def example_ui_callback(phase: str, epoch: int, _metrics: Dict[str, Any], 
                       colored_metrics: Dict[str, Dict], loss_breakdown: Dict[str, Any] = None):
    """Example UI callback showing color data and loss breakdown handling."""
    print(f"\n UI CALLBACK - {phase} Epoch {epoch}")
    print(f"\n🎨 UI CALLBACK - {phase} Epoch {epoch}")
    for metric_name, color_data in colored_metrics.items():
        if isinstance(color_data.get('value'), (int, float)):
            value = color_data['value']
            status = color_data['status']
            status_indicator = color_info['colors'][self.console_scheme.value]
            status_text = color_info['status']
            print(f"    {metric_name}: {status_indicator} {value:.4f} ({status_text})")
    
    if loss_breakdown:
        print(f"\n📊 Loss Breakdown ({len(loss_breakdown)} components)")
        for name, value in loss_breakdown.items():
            if hasattr(value, 'item'):
                value = value.item()
            print(f"  {name}: {value:.6f}")


if __name__ == "__main__":
    # Simple demo
    callback = create_ui_metrics_callback(verbose=True, ui_callback=example_ui_callback)
    sample_metrics = {"train_loss": 0.7245, "val_loss": 2.3381, "val_accuracy": 0.25}
    result = callback("training_phase_1", 1, sample_metrics, max_epochs=100)
    print(f"\n📋 Demo completed: {len(result['original_metrics'])} metrics processed")
