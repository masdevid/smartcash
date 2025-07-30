#!/usr/bin/env python3
"""
Progress management for the unified training pipeline.

This module handles progress tracking, UI callbacks, visualization updates,
and chart emissions for training monitoring.
"""

import numpy as np
from typing import Dict, Any, List

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ProgressManager:
    """Handles progress tracking, callbacks, and visualization updates."""
    
    def __init__(self, progress_tracker, emit_metrics_callback=None, 
                 emit_live_chart_callback=None, visualization_manager=None):
        """
        Initialize progress manager.
        
        Args:
            progress_tracker: Progress tracking instance
            emit_metrics_callback: Callback for metrics emission
            emit_live_chart_callback: Callback for live chart updates
            visualization_manager: Visualization manager instance
        """
        self.progress_tracker = progress_tracker
        self.emit_metrics_callback = emit_metrics_callback
        self.emit_live_chart_callback = emit_live_chart_callback
        self.visualization_manager = visualization_manager
        
        # State for visualization
        self._is_single_phase = False
    
    def emit_epoch_metrics(self, phase_num: int, epoch: int, metrics: Dict[str, Any]):
        """
        Emit metrics callback for UI updates.
        
        Args:
            phase_num: Current phase number
            epoch: Current epoch number
            metrics: Metrics dictionary
        """
        if not self.emit_metrics_callback:
            return
        
        phase_name = 'training_phase_single' if self._is_single_phase else f'training_phase_{phase_num}'
        self.emit_metrics_callback(phase_name, epoch, metrics)
    
    def emit_training_charts(self, epoch: int, phase_num: int, final_metrics: dict, layer_metrics: dict):
        """
        Emit live chart data for training visualization.
        
        Args:
            epoch: Current epoch number
            phase_num: Current phase number
            final_metrics: Final metrics dictionary
            layer_metrics: Layer-specific metrics dictionary
        """
        if not self.emit_live_chart_callback:
            return
            
        # Determine active layers using the same logic as the metrics callback
        show_layers = self._determine_active_layers_for_charts(phase_num, final_metrics)
        
        # Training curves chart - use primary active layer for accuracy
        self._emit_training_curves_chart(epoch, phase_num, final_metrics, show_layers)
        
        # Layer metrics chart - only include active layers
        self._emit_layer_metrics_chart(epoch, phase_num, final_metrics, show_layers)
    
    def _emit_training_curves_chart(self, epoch: int, phase_num: int, 
                                  final_metrics: dict, show_layers: List[str]):
        """Emit training curves chart data."""
        primary_layer = show_layers[0] if show_layers else 'layer_1'
        chart_data = {
            'epoch': epoch + 1,
            'train_loss': final_metrics.get('train_loss', 0),
            'val_loss': final_metrics.get('val_loss', 0),
            'train_accuracy': final_metrics.get(f'{primary_layer}_accuracy', 0),
            'val_accuracy': final_metrics.get('val_accuracy', 0),  # Use global val_accuracy, not layer-specific
            'phase': phase_num
        }
        
        phase_label = 'Single Phase' if self._is_single_phase else f'Phase {phase_num}'
        self.emit_live_chart_callback('training_curves', chart_data, {
            'title': f'Training Progress - {phase_label}',
            'xlabel': 'Epoch',
            'ylabel': 'Loss / Accuracy'
        })
    
    def _emit_layer_metrics_chart(self, epoch: int, phase_num: int, 
                                final_metrics: dict, show_layers: List[str]):
        """Emit layer metrics chart data."""
        layer_chart_data = {}
        for layer in show_layers:
            layer_chart_data[layer] = {
                'accuracy': final_metrics.get(f'{layer}_accuracy', 0),
                'precision': final_metrics.get(f'{layer}_precision', 0),
                'recall': final_metrics.get(f'{layer}_recall', 0),
                'f1': final_metrics.get(f'{layer}_f1', 0)
            }
        
        # Only emit chart if we have meaningful data for at least one layer
        if layer_chart_data and any(layer_chart_data[layer]['accuracy'] > 0 for layer in layer_chart_data):
            phase_label = "Single Phase" if self._is_single_phase else f"Phase {phase_num}"
            self.emit_live_chart_callback('layer_metrics', {
                'epoch': epoch + 1,
                'layers': layer_chart_data,
                'phase': phase_num
            }, {
                'title': f'Layer Performance - {phase_label}',
                'xlabel': 'Layer',
                'ylabel': 'Metric Value'
            })
    
    def _determine_active_layers_for_charts(self, phase_num: int, final_metrics: dict) -> List[str]:
        """Determine which layers should be included in charts using the same logic as metrics callback."""
        if self._is_single_phase:
            # Single-phase mode: Determine active layers from actual metrics
            layer_activity = {}
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                # Check if this layer has any meaningful metrics
                has_activity = any(
                    final_metrics.get(f'{layer}_{metric}', 0) > 0.0001 or 
                    final_metrics.get(f'val_{layer}_{metric}', 0) > 0.0001
                    for metric in ['accuracy', 'precision', 'recall', 'f1']
                )
                layer_activity[layer] = has_activity
            
            # Determine active layers
            active_layers = [layer for layer, active in layer_activity.items() if active]
            if len(active_layers) == 1:
                # Single-phase, single-layer mode: only show the active layer
                return active_layers
            else:
                # Single-phase, multi-layer mode: show all layers
                return ['layer_1', 'layer_2', 'layer_3']
        else:
            # Two-phase mode
            if phase_num == 1:
                # Phase 1: Only show layer_1
                return ['layer_1']
            else:
                # Phase 2: Show all layers
                return ['layer_1', 'layer_2', 'layer_3']
    
    def update_visualization_manager(self, epoch: int, phase_num: int, 
                                   final_metrics: dict, layer_metrics: dict):
        """
        Update visualization manager with current epoch data.
        
        Args:
            epoch: Current epoch number
            phase_num: Current phase number
            final_metrics: Final metrics dictionary
            layer_metrics: Layer-specific metrics dictionary
        """
        if not self.visualization_manager:
            return
            
        # Determine active layers using the same logic as charts
        show_layers = self._determine_active_layers_for_charts(phase_num, final_metrics)
        
        phase_name = f"phase_{phase_num}"
        # Create simulated predictions and targets for visualization - only for active layers
        viz_predictions = {}
        viz_targets = {}
        if layer_metrics:
            np.random.seed(epoch * 42)
            for layer in show_layers:
                # Use consistent class numbers based on layer
                num_classes = {'layer_1': 7, 'layer_2': 7, 'layer_3': 3}.get(layer, 7)
                viz_predictions[layer] = np.random.rand(50, num_classes)
                viz_targets[layer] = np.eye(num_classes)[np.random.randint(0, num_classes, 50)]
        
        # Use research-focused update method with phase context
        if hasattr(self.visualization_manager, 'update_with_research_metrics'):
            self.visualization_manager.update_with_research_metrics(
                epoch=epoch + 1,
                phase_num=phase_num,
                metrics=final_metrics,
                predictions=viz_predictions,
                ground_truth=viz_targets
            )
        else:
            # Fallback to original method with phase_num parameter
            self.visualization_manager.update_metrics(
                epoch=epoch + 1,
                phase=phase_name,
                metrics=final_metrics,
                predictions=viz_predictions,
                ground_truth=viz_targets,
                phase_num=phase_num
            )
    
    def set_single_phase_mode(self, is_single_phase: bool):
        """Set single phase mode flag for proper visualization."""
        self._is_single_phase = is_single_phase
    
    def start_epoch_tracking(self, total_epochs: int):
        """Start epoch progress tracking."""
        self.progress_tracker.start_epoch_tracking(total_epochs)
    
    def update_epoch_progress(self, epoch: int, total_epochs: int, message: str, progress_percentage: float = None):
        """Update epoch progress with message and optional custom percentage."""
        if progress_percentage is not None:
            # Use custom percentage calculation for resumed training
            # Convert percentage back to relative position for progress bar
            relative_current = int((progress_percentage / 100) * (total_epochs - 1))  
            self.progress_tracker.update_epoch_progress(epoch, total_epochs, message, relative_current)
        else:
            self.progress_tracker.update_epoch_progress(epoch, total_epochs, message)
    
    def complete_epoch_early_stopping(self, epoch: int, message: str):
        """Complete epoch tracking due to early stopping."""
        self.progress_tracker.complete_epoch_early_stopping(epoch, message)
    
    def handle_scheduler_step(self, scheduler, final_metrics: dict):
        """
        Handle learning rate scheduler step.
        
        Args:
            scheduler: Learning rate scheduler
            final_metrics: Final metrics dictionary
        """
        if scheduler:
            if hasattr(scheduler, 'step'):
                if 'ReduceLROnPlateau' in str(type(scheduler)):
                    monitor_metric = final_metrics.get('val_map50', 0)
                    scheduler.step(monitor_metric)
                else:
                    scheduler.step()
    
    def handle_early_stopping(self, early_stopping, final_metrics: dict, epoch: int, phase_num: int) -> bool:
        """
        Handle early stopping logic with phase-aware behavior.
        
        Args:
            early_stopping: Early stopping instance
            final_metrics: Final metrics dictionary
            epoch: Current epoch number
            phase_num: Current phase number - used to determine if early stopping should run
            
        Returns:
            True if training should stop, False otherwise
        """
        # Early stopping should only be active in Phase 2
        # Phase 1 focuses on detection layer training where val_map50 is not meaningful
        if phase_num == 1:
            logger.debug(f"Phase 1: Early stopping skipped (detection layer training only)")
            return False
        
        # Phase 2: Use val_map50 for early stopping as originally intended
        monitor_metric = final_metrics.get('val_map50', 0)
        should_stop = early_stopping(monitor_metric, None, epoch)  # Don't pass model for saving
        
        if should_stop:
            logger.info(f"🛑 Early stopping triggered at epoch {epoch + 1}")
            logger.info(f"   Monitoring val_map50: no improvement for {early_stopping.patience} epochs")
            logger.info(f"   Best val_map50: {early_stopping.best_score:.6f} at epoch {early_stopping.best_epoch + 1}")
            
            # Add early stopping info to final metrics
            final_metrics['early_stopped'] = True
            final_metrics['early_stop_epoch'] = epoch + 1
            final_metrics['early_stop_reason'] = f"No improvement in {early_stopping.metric} for {early_stopping.patience} epochs"
        
        return should_stop
    
    def cleanup(self):
        """Clean up progress manager resources."""
        try:
            logger.info("🧹 Cleaning up progress manager resources...")
            
            # Clean up progress tracker
            if hasattr(self.progress_tracker, 'cleanup'):
                self.progress_tracker.cleanup()
            
            # Clean up visualization manager
            if hasattr(self.visualization_manager, 'cleanup'):
                self.visualization_manager.cleanup()
            
            # Clear callback references
            self.emit_metrics_callback = None
            self.emit_live_chart_callback = None
            
            logger.info("✅ Progress manager resources cleaned up")
            
        except Exception as e:
            logger.warning(f"Error during progress manager cleanup: {e}")