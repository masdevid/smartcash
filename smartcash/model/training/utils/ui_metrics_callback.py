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
    
    def __init__(self, verbose: bool = True, console_scheme: ColorScheme = ColorScheme.EMOJI, 
                 training_logs_dir: str = "logs/training"):
        """
        Initialize the UI metrics callback. 
        
        Args:
            verbose: Whether to print verbose output
            console_scheme: Color scheme for console output
            training_logs_dir: Directory containing JSON metrics files
        """
        self.verbose = verbose
        self.console_scheme = console_scheme
        self.colorizer = MetricColorizer(console_scheme)
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
        
        # Print layer-specific metrics
        layer_metrics = self._filter_layer_metrics(metrics, show_layers, filter_zeros)
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
    
    def _print_loss_breakdown_summary(self, loss_breakdown: Dict[str, Any]):
        """Print a concise loss breakdown summary for console output."""
        if not loss_breakdown:
            return
            
        print("  📊 Loss Summary:", end=" ")
        
        # Show key loss components in a single line
        key_losses = []
        
        # Check for total/train loss
        for loss_key in ['train_loss', 'total_loss', 'loss']:
            if loss_key in loss_breakdown:
                value = loss_breakdown[loss_key]
                if hasattr(value, 'item'):
                    value = value.item()
                key_losses.append(f"Total: {value:.4f}")
                break
        
        # Show main loss components if available
        for component in ['box_loss', 'obj_loss', 'cls_loss']:
            if component in loss_breakdown:
                value = loss_breakdown[component]
                if hasattr(value, 'item'):
                    value = value.item()
                component_name = component.replace('_loss', '').title()
                key_losses.append(f"{component_name}: {value:.4f}")
        
        if key_losses:
            print(" | ".join(key_losses))
        else:
            print("Available in detailed view")
    
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
        
        if should_suppress_color:
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
        
        # Use consistent ordering as specified in TASK.md for both phases
        core_metrics = [
            # Core losses (no color/indicator needed as per TASK.md)
            'train_loss', 'val_loss', 
            # Core validation metrics (in TASK.md specified order)
            'val_accuracy', 'val_precision', 'val_recall', 'val_f1', 'val_map50',
            # Additional context metrics (not in TASK.md but useful)
            'learning_rate', 'epoch'
        ]
        
        # Phase-specific additions
        if phase_num == 2:
            # Phase 2: Add mAP-specific metrics after val_map50
            extended_metrics = core_metrics[:5]  # train_loss through val_map50
            extended_metrics.extend(['val_map50_precision', 'val_map50_recall', 'val_map50_f1', 'val_map50_accuracy'])
            extended_metrics.extend(core_metrics[5:])  # learning_rate, epoch
            core_metrics = extended_metrics
        
        # Filter to only include metrics that actually exist and handle special cases
        available_metrics = []
        for metric in core_metrics:
            if metric in metrics:
                # Don't filter out metrics with zero values - they may be legitimate
                available_metrics.append(metric)
        
        return available_metrics
    
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
        # Check for multiple active layers (indication of Phase 2)
        active_layers = 0
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            has_layer_activity = any(
                metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                for metric in ['accuracy', 'precision', 'recall', 'f1']
            )
            if has_layer_activity:
                active_layers += 1
        
        # If we have multiple active layers, it's Phase 2
        return 2 if active_layers > 1 else 1
    
    def _determine_layer_display(self, phase: str, metrics: Dict[str, Any]) -> Tuple[list, bool]:
        """
        Determine which layers to show and whether to filter zeros based on training phase.
        
        This implements intelligent phase-aware metrics logic aligned with new loss system:
        - Phase 1: Simple YOLO loss - show layer_1 metrics, but don't filter zeros to ensure visibility
        - Phase 2: Multi-task loss - show all layers with meaningful data
        - Auto-detect based on actual metrics and loss type
        """
        phase_num = self._extract_phase_number(phase, metrics)

        if phase_num == 1:
            # Phase 1: Show layer_1 metrics, but don't filter zeros to ensure all metrics are visible
            return ['layer_1'], False

        active_layers = ['layer_1']  # Always include layer_1
        for layer in ['layer_2', 'layer_3']:
            if any(metrics.get(f'{p}{layer}_{m}', 0) > 0 for m in ['accuracy', 'precision'] for p in ['', 'val_']):
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
        """
        Get metrics for the latest epoch from JSON files.
        
        Args:
            session_id: Optional session ID
            
        Returns:
            Latest epoch metrics with mAP data and loss breakdown
        """
        data = self.load_metrics_from_json(session_id)
        if not data or 'metrics_history' not in data:
            return {}
        
        metrics_history = data['metrics_history']
        if not metrics_history:
            return {}
        
        # Get latest epoch
        latest_epoch = metrics_history[-1]
        
        # Extract mAP metrics
        map_metrics = {}
        for key in ['val_map50', 'val_map50_95', 'val_precision', 'val_recall', 'val_f1', 
                   'val_map50_precision', 'val_map50_recall', 'val_map50_f1', 'val_map50_accuracy']:
            if latest_epoch.get(key) is not None:
                map_metrics[key] = latest_epoch[key]
        
        # Extract loss breakdown from multiple sources
        loss_breakdown = {}
        
        # Standard YOLO loss components (both training and validation)
        standard_loss_components = [
            'train_box_loss', 'train_obj_loss', 'train_cls_loss',
            'val_box_loss', 'val_obj_loss', 'val_cls_loss',
            'box_loss', 'obj_loss', 'cls_loss'
        ]
        
        # Extract from main metrics first
        for key in standard_loss_components:
            if latest_epoch.get(key) is not None:
                loss_breakdown[key] = latest_epoch[key]
        
        # Extract from additional_metrics if available
        if latest_epoch.get('additional_metrics'):
            for key, value in latest_epoch['additional_metrics'].items():
                if 'loss' in key.lower():
                    loss_breakdown[key] = value
                    
        # Also check for layer-specific loss components
        layer_loss_patterns = ['layer_1_', 'layer_2_', 'layer_3_']
        for layer_prefix in layer_loss_patterns:
            for suffix in ['box_loss', 'obj_loss', 'cls_loss', 'total_loss', 'weighted_loss']:
                key = f'{layer_prefix}{suffix}'
                if latest_epoch.get(key) is not None:
                    loss_breakdown[key] = latest_epoch[key]
        
        result = {
            'epoch': latest_epoch['epoch'],
            'phase': latest_epoch['phase'],
            'timestamp': latest_epoch['timestamp'],
            'core_metrics': {
                'train_loss': latest_epoch['train_loss'],
                'val_loss': latest_epoch['val_loss'],
                'learning_rate': latest_epoch['learning_rate']
            },
            'map_metrics': map_metrics,
            'loss_breakdown': loss_breakdown
        }
        
        # Add layer metrics
        layer_metrics = {}
        for key, value in latest_epoch.items():
            if key.startswith('layer_') and value is not None:
                layer_metrics[key] = value
        result['layer_metrics'] = layer_metrics
        
        return result
    
    def get_phase_metrics_history(self, phase: int, session_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics history for a specific phase.
        
        Args:
            phase: Phase number (1 or 2)
            session_id: Optional session ID
            
        Returns:
            Phase-specific metrics history with trends
        """
        data = self.load_metrics_from_json(session_id)
        if not data or 'metrics_history' not in data:
            return {}
        
        metrics_history = data['metrics_history']
        
        # Filter by phase
        phase_data = [record for record in metrics_history if record['phase'] == phase]
        
        if not phase_data:
            return {'phase': phase, 'epochs': []}
        
        # Extract time series for key metrics
        epochs = [record['epoch'] for record in phase_data]
        
        series = {
            'loss': {
                'epochs': epochs,
                'train_loss': [record['train_loss'] for record in phase_data],
                'val_loss': [record['val_loss'] for record in phase_data]
            },
            'loss_breakdown': {
                'epochs': epochs,
                'train_box_loss': [record.get('train_box_loss', 0) for record in phase_data],
                'train_obj_loss': [record.get('train_obj_loss', 0) for record in phase_data],
                'train_cls_loss': [record.get('train_cls_loss', 0) for record in phase_data],
                'val_box_loss': [record.get('val_box_loss', 0) for record in phase_data],
                'val_obj_loss': [record.get('val_obj_loss', 0) for record in phase_data],
                'val_cls_loss': [record.get('val_cls_loss', 0) for record in phase_data]
            },
            'map_metrics': {
                'epochs': epochs,
                'val_map50': [record.get('val_map50', 0) for record in phase_data],
                'val_precision': [record.get('val_precision', 0) for record in phase_data],
                'val_recall': [record.get('val_recall', 0) for record in phase_data],
                'val_map50_precision': [record.get('val_map50_precision', 0) for record in phase_data],
                'val_map50_recall': [record.get('val_map50_recall', 0) for record in phase_data],
                'val_map50_f1': [record.get('val_map50_f1', 0) for record in phase_data],
                'val_map50_accuracy': [record.get('val_map50_accuracy', 0) for record in phase_data]
            },
            'learning_rate': {
                'epochs': epochs,
                'values': [record['learning_rate'] for record in phase_data]
            }
        }
        
        # Add layer metrics if available
        layer_series = {}
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            layer_accuracy = []
            for record in phase_data:
                acc = record.get(f'{layer}_accuracy')
                layer_accuracy.append(acc if acc is not None else 0)
            
            if any(acc > 0 for acc in layer_accuracy):  # Only include if has data
                layer_series[f'{layer}_accuracy'] = {
                    'epochs': epochs,
                    'values': layer_accuracy
                }
        
        series['layer_metrics'] = layer_series
        
        return {
            'phase': phase,
            'total_epochs': len(phase_data),
            'best_val_loss': min(record['val_loss'] for record in phase_data),
            'best_map50': max(record.get('val_map50', 0) for record in phase_data),
            'best_map50_precision': max(record.get('val_map50_precision', 0) for record in phase_data),
            'best_map50_accuracy': max(record.get('val_map50_accuracy', 0) for record in phase_data),
            # Loss breakdown minimums (lower is better for loss components)
            'min_train_box_loss': min(record.get('train_box_loss', float('inf')) for record in phase_data if record.get('train_box_loss') is not None),
            'min_val_box_loss': min(record.get('val_box_loss', float('inf')) for record in phase_data if record.get('val_box_loss') is not None),
            'series': series
        }
    
    def get_metric_summary_for_ui(self) -> Dict[str, Any]:
        """Get a comprehensive summary for UI display with JSON-loaded data."""
        # Try to load latest metrics from JSON first
        latest_data = self.get_latest_epoch_metrics()
        
        if latest_data:
            # Use JSON data
            metrics = {}
            metrics.update(latest_data['core_metrics'])
            metrics.update(latest_data['map_metrics'])
            metrics.update(latest_data['layer_metrics'])
            
            phase = latest_data['phase']
            epoch = latest_data['epoch']
            loss_breakdown = latest_data['loss_breakdown']
        else:
            # Fallback to in-memory data
            if not self.latest_colored_metrics:
                return {}
            metrics = self.latest_metrics
            phase = self.current_phase
            epoch = self.current_epoch
            loss_breakdown = {}
        
        # Apply phase-aware filtering
        show_layers, filter_zeros = self._determine_layer_display(str(phase), metrics)
        filtered_layer_metrics = self._filter_layer_metrics(metrics, show_layers, filter_zeros)
        
        # Generate colored metrics for current data
        colored_metrics = get_metrics_with_colors(metrics, epoch, 100, int(phase))
        
        # Categorize metrics for UI display
        categories = {
            'loss_metrics': {},
            'accuracy_metrics': {},
            'map_metrics': {},
            'layer_metrics': {},
            'other_metrics': {}
        }
        
        for metric_name, color_data in colored_metrics.items():
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
            'epoch': epoch,
            'phase': phase,
            'loss_breakdown': loss_breakdown,
            'phase_info': {
                'active_layers': show_layers,
                'filter_zeros': filter_zeros,
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
            return f"Custom phase: {len(show_layers)} layers visible (filter_zeros: {filter_zeros})"


def create_ui_metrics_callback(verbose: bool = True, 
                             console_scheme: ColorScheme = ColorScheme.EMOJI,
                             ui_callback: Optional[Callable] = None,
                             training_logs_dir: str = "logs/training",
                             backbone: str = "unknown",
                             data_name: str = "data") -> UIMetricsCallback:
    """
    Factory function to create a UI-enhanced metrics callback.
    
    Args:
        verbose: Whether to print console output
        console_scheme: Color scheme for console display
        ui_callback: Optional UI callback function
        training_logs_dir: Directory containing JSON metrics files
        backbone: Backbone model name for filename
        data_name: Dataset name for filename
        
    Returns:
        UIMetricsCallback instance
    """
    callback = UIMetricsCallback(verbose, console_scheme, training_logs_dir)
    callback.backbone = backbone
    callback.data_name = data_name
    if ui_callback:
        callback.set_ui_callback(ui_callback)
    return callback


# Example UI callback function
def example_ui_callback(phase: str, epoch: int, _metrics: Dict[str, Any], colored_metrics: Dict[str, Dict], loss_breakdown: Dict[str, Any] = None):
    """
    Example UI callback function showing how to handle the color data and loss breakdown. 
    
    Args:
        phase: Training phase name
        epoch: Current epoch
        metrics: Original metrics dictionary
        colored_metrics: Enhanced metrics with color information
        loss_breakdown: Detailed loss breakdown information
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
    
    # Show loss breakdown if available
    if loss_breakdown:
        print(f"\n📊 Loss Breakdown ({len(loss_breakdown)} components):")
        for loss_name, loss_value in loss_breakdown.items():
            if hasattr(loss_value, 'item'):
                loss_value = loss_value.item()
            print(f"  {loss_name}: {loss_value:.6f}")


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
        "val_map50": 0.15,
        "val_map50_precision": 0.32,
        "val_map50_recall": 0.28,
        "val_map50_f1": 0.30,
        "val_map50_accuracy": 0.25,
        "layer_1_accuracy": 0.83,
        "layer_1_precision": 0.75,
        "layer_1_f1": 0.82,
        # Loss breakdown components
        "train_box_loss": 0.245,
        "train_obj_loss": 0.198,
        "train_cls_loss": 0.281,
        "val_box_loss": 0.892,
        "val_obj_loss": 0.736,
        "val_cls_loss": 0.710
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
