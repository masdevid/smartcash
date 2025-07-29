"""
File: smartcash/ui/model/training/operations/unified_training_operation.py
Description: Unified training operation that directly uses the unified training pipeline.
"""

import time
from typing import Dict, Any, Callable, Optional
from smartcash.ui.logger import get_module_logger
from smartcash.model.api.core import run_full_training_pipeline


class UnifiedTrainingOperation:
    """
    Unified training operation handler that directly uses the training pipeline.
    
    This replaces the complex training operation system with a direct call to the
    training pipeline from smartcash.model.training.training_pipeline.
    """
    
    def __init__(self, ui_module, config: Dict[str, Any], callbacks: Dict[str, Callable]):
        """Initialize unified training operation.
        
        Args:
            ui_module: Reference to the UI module
            config: Training configuration
            callbacks: UI callback functions
        """
        self.ui_module = ui_module
        self.config = config
        self.callbacks = callbacks
        self.logger = get_module_logger("smartcash.ui.model.training.operations")
        
        # Training state
        self._is_training = False
        self._training_result = None
    
    def execute(self) -> Dict[str, Any]:
        """Execute unified training pipeline."""
        try:
            self._is_training = True
            self._log_info("🚀 Starting unified training pipeline...")
            
            # Extract training configuration
            training_config = self.config.get('training', {})
            
            # Validate resume functionality if enabled
            resume_checkpoint_path = training_config.get('resume_checkpoint_path', '').strip()
            if resume_checkpoint_path:
                validation_result = self._validate_resume_checkpoint(resume_checkpoint_path)
                if not validation_result['success']:
                    self._log_error(validation_result['message'])
                    return validation_result
                self._log_info(f"✅ Resume checkpoint validated: {validation_result['message']}")
            
            # Create progress callback that bridges to UI
            progress_callback = self._create_ui_progress_callback()
            
            # Map form configuration to unified training pipeline parameters
            pipeline_params = self._map_config_to_pipeline_params(training_config, progress_callback)
            
            self._log_info(f"Training configuration: {pipeline_params['backbone']} ({pipeline_params['training_mode']})")
            self._log_info(f"Epochs: Phase 1={pipeline_params['phase_1_epochs']}, Phase 2={pipeline_params['phase_2_epochs']}")
            
            if pipeline_params['resume_from_checkpoint']:
                self._log_info(f"🔄 Resume training from: {resume_checkpoint_path}")
            
            # Call the unified training pipeline directly
            result = run_full_training_pipeline(**pipeline_params)
            
            # Process results
            if result.get('success'):
                self._handle_training_success(result)
                return {'success': True, 'message': 'Training completed successfully', 'result': result}
            else:
                error_msg = result.get('error', 'Unknown training error')
                self._handle_training_failure(error_msg)
                return {'success': False, 'message': error_msg, 'result': result}
                
        except Exception as e:
            error_msg = f"Training execution failed: {e}"
            self.logger.error(error_msg, exc_info=True)
            self._handle_training_failure(error_msg)
            return {'success': False, 'message': error_msg}
        finally:
            self._is_training = False
    
    def _validate_resume_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Validate resume checkpoint before training starts."""
        try:
            import os
            from pathlib import Path
            
            # Check if file exists
            if not os.path.exists(checkpoint_path):
                return {
                    'success': False,
                    'message': f"❌ Checkpoint file not found: {checkpoint_path}"
                }
            
            # Check file size (should not be empty)
            file_size = os.path.getsize(checkpoint_path)
            if file_size == 0:
                return {
                    'success': False,
                    'message': f"❌ Checkpoint file is empty: {checkpoint_path}"
                }
            
            # Check file extension
            valid_extensions = ['.pt', '.pth', '.ckpt']
            if not any(checkpoint_path.endswith(ext) for ext in valid_extensions):
                return {
                    'success': False,
                    'message': f"❌ Invalid checkpoint format. Expected: {valid_extensions}"
                }
            
            # Try to load checkpoint metadata (without loading full model)
            try:
                import torch
                checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
                
                # Check for basic checkpoint structure
                if isinstance(checkpoint_data, dict):
                    epoch = checkpoint_data.get('epoch', 0)
                    phase = checkpoint_data.get('phase', 1)
                    metrics = checkpoint_data.get('metrics', {})
                    
                    return {
                        'success': True,
                        'message': f"Valid checkpoint (Epoch {epoch + 1}, Phase {phase})",
                        'epoch': epoch,
                        'phase': phase,
                        'metrics': metrics,
                        'file_size_mb': file_size / (1024 * 1024)
                    }
                else:
                    return {
                        'success': False,
                        'message': f"❌ Invalid checkpoint structure in: {checkpoint_path}"
                    }
                    
            except Exception as e:
                return {
                    'success': False,
                    'message': f"❌ Failed to load checkpoint: {str(e)}"
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f"❌ Checkpoint validation error: {str(e)}"
            }
    
    def _map_config_to_pipeline_params(self, config: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
        """Map UI configuration to unified training pipeline parameters."""
        training_mode = config.get('training_mode', 'two_phase')
        
        # For two-phase mode: disable early stopping in Phase 1, enable in Phase 2
        # For single-phase mode: use the user's early stopping setting
        early_stopping_enabled = config.get('early_stopping_enabled', True)
        if training_mode == 'two_phase':
            # Pass special flags to indicate phase-specific early stopping behavior
            early_stopping_phase_1_enabled = False
            early_stopping_phase_2_enabled = early_stopping_enabled
            self.logger.info("🎯 Two-phase mode: Early stopping disabled for Phase 1, " + 
                           ("enabled" if early_stopping_enabled else "disabled") + " for Phase 2")
        else:
            # For single-phase mode, use the same setting for both (though only one phase exists)
            early_stopping_phase_1_enabled = early_stopping_enabled
            early_stopping_phase_2_enabled = early_stopping_enabled
        
        # Handle resume functionality
        resume_checkpoint_path = config.get('resume_checkpoint_path', '').strip()
        resume_from_checkpoint = bool(resume_checkpoint_path)
        
        if resume_from_checkpoint:
            self.logger.info(f"🔄 Resume training enabled from: {resume_checkpoint_path}")
            # If a specific checkpoint path is provided, pass it as resume_info
            # The training pipeline will use this specific checkpoint
            resume_info = {'checkpoint_path': resume_checkpoint_path}
        else:
            resume_info = None
        
        return {
            # Core parameters
            'backbone': config.get('backbone', 'cspdarknet'),
            'phase_1_epochs': config.get('phase_1_epochs', 1),
            'phase_2_epochs': config.get('phase_2_epochs', 1),
            'checkpoint_dir': config.get('checkpoint_dir', 'data/checkpoints'),
            'progress_callback': progress_callback,
            'verbose': config.get('verbose', True),
            'force_cpu': config.get('force_cpu', False),
            'training_mode': training_mode,
            
            # Resume functionality parameters
            'resume_from_checkpoint': resume_from_checkpoint,
            'resume_info': resume_info,  # Pass specific checkpoint path if provided
            
            # Single-phase specific parameters
            'single_phase_layer_mode': config.get('single_phase_layer_mode', 'multi'),
            'single_phase_freeze_backbone': config.get('single_phase_freeze_backbone', False),
            
            # Training configuration parameters
            'loss_type': config.get('loss_type', 'uncertainty_multi_task'),
            'head_lr_p1': config.get('head_lr_p1', 1e-3),
            'head_lr_p2': config.get('head_lr_p2', 1e-4),
            'backbone_lr': config.get('backbone_lr', 1e-5),
            'batch_size': config.get('batch_size'),  # None for auto-detection
            
            # Phase-specific early stopping configuration
            'early_stopping_enabled': early_stopping_enabled,  # Overall setting (for single-phase)
            'early_stopping_phase_1_enabled': early_stopping_phase_1_enabled,  # Phase 1 specific
            'early_stopping_phase_2_enabled': early_stopping_phase_2_enabled,  # Phase 2 specific
            'early_stopping_patience': config.get('early_stopping_patience', 15),
            'early_stopping_metric': config.get('early_stopping_metric', 'val_map50'),
            'early_stopping_mode': config.get('early_stopping_mode', 'max'),
            'early_stopping_min_delta': config.get('early_stopping_min_delta', 0.001)
        }
    
    def _create_ui_progress_callback(self) -> Callable:
        """Create progress callback that bridges unified pipeline to UI."""
        def progress_callback(phase: str, current: int, total: int, message: str = "", **kwargs):
            """Progress callback that updates UI components."""
            try:
                # Calculate percentage
                percentage = (current / total) * 100 if total > 0 else 0
                
                # Ensure finalization phase reaches 100% completion
                if phase == 'finalize' and percentage >= 100:
                    percentage = 100
                    message = "Training Complete!"
                
                # Format phase name for display
                phase_display = phase.replace('_', ' ').title()
                
                # Handle different types of progress updates
                if phase in ['training_phase_1', 'training_phase_2']:
                    self._handle_training_phase_progress(phase, current, total, message, percentage, **kwargs)
                else:
                    self._handle_general_phase_progress(phase_display, percentage, message)
                
                # Call UI callbacks if available
                if 'on_progress' in self.callbacks:
                    # For finalization, use completion message
                    display_message = "🚀 Training Complete!" if phase == 'finalize' and percentage >= 100 else f"{phase_display}: {message}"
                    self.callbacks['on_progress'](int(percentage), display_message)
                    
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
        
        return progress_callback
    
    def update_progress(
        self,
        overall_progress: int = None,
        overall_message: str = "",
        phase_progress: int = None,
        phase_message: str = "",
        epoch_progress: int = None,
        epoch_message: str = ""
    ) -> None:
        """
        Update triple progress tracker for training operations.
        
        Args:
            overall_progress: Overall training progress (0-100)
            overall_message: Overall progress message
            phase_progress: Current phase progress (0-100)
            phase_message: Phase progress message  
            epoch_progress: Current epoch progress (0-100)
            epoch_message: Epoch progress message
        """
        try:
            # Get operation container from UI module
            if hasattr(self, 'ui_module') and self.ui_module:
                operation_container = self.ui_module.get_component('operation_container')
                
                if operation_container and hasattr(operation_container, 'get'):
                    update_func = operation_container.get('update_progress')
                    if update_func:
                        # Call with triple progress parameters
                        update_func(
                            progress=overall_progress if overall_progress is not None else 0,
                            message=overall_message,
                            secondary_progress=phase_progress,
                            secondary_message=phase_message,
                            tertiary_progress=epoch_progress,
                            tertiary_message=epoch_message
                        )
                        
        except Exception as e:
            # Fail silently on progress update errors
            if hasattr(self, 'logger'):
                self.logger.debug(f"Progress update failed: {e}")
    
    def _log_info(self, message: str):
        """Log info message to both logger and UI callbacks."""
        self.logger.info(message)
        if 'log' in self.callbacks:
            self.callbacks['log']('info', message)
    
    def _log_error(self, message: str):
        """Log error message to both logger and UI callbacks."""
        self.logger.error(message)
        if 'log' in self.callbacks:
            self.callbacks['log']('error', message)
    
    def _handle_training_phase_progress(self, phase: str, current: int, total: int, 
                                      message: str, percentage: float, **kwargs):
        """Handle progress updates for training phases."""
        phase_num = "1" if phase == 'training_phase_1' else "2"
        
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
            epochs_total = kwargs.get('total_epochs', total)
            
            # Calculate overall training progress (across both phases)
            overall_progress = self._calculate_overall_progress(phase, current, total, **kwargs)
            
            # Update triple progress tracker
            self.update_progress(
                overall_progress=overall_progress,
                overall_message=f"Training Progress",
                phase_progress=int(percentage),
                phase_message=f"Phase {phase_num}",
                epoch_progress=int((current / total) * 100) if total > 0 else 0,
                epoch_message=f"Epoch {epoch}/{epochs_total}"
            )
            
            # Log epoch progress
            epoch_msg = f"Phase {phase_num} - Epoch {epoch}/{epochs_total}"
            self._log_info(f"🔄 {epoch_msg}: {percentage:.0f}%")
            
            # Handle epoch completion with metrics
            if current == total and kwargs.get('metrics'):
                metrics = kwargs['metrics']
                self._handle_epoch_completion(phase_num, epoch, metrics)
        else:
            # General phase progress - update phase level only
            overall_progress = self._calculate_overall_progress(phase, current, total, **kwargs)
            
            self.update_progress(
                overall_progress=overall_progress,
                overall_message=f"Training Progress",
                phase_progress=int(percentage),
                phase_message=f"Phase {phase_num}"
            )
            
            self._log_info(f"🔄 Phase {phase_num}: {percentage:.0f}%")
    
    def _handle_general_phase_progress(self, phase_display: str, percentage: float, message: str):
        """Handle progress updates for non-training phases."""
        # Update overall progress for non-training phases
        self.update_progress(
            overall_progress=int(percentage),
            overall_message=f"{phase_display}: {message}" if message else phase_display
        )
        
        if percentage >= 100:
            self._log_success(f"✅ {phase_display}: Complete")
        elif message:
            self._log_info(f"🔄 {phase_display} ({percentage:.0f}%): {message}")
        else:
            self._log_info(f"🔄 {phase_display}: {percentage:.0f}%")
    
    def _calculate_overall_progress(self, phase: str, current: int, total: int, **kwargs) -> int:
        """Calculate overall training progress across both phases."""
        try:
            # Get phase configuration
            phase_1_epochs = self.config.get('training', {}).get('phase_1_epochs', 1)
            phase_2_epochs = self.config.get('training', {}).get('phase_2_epochs', 1)
            total_epochs = phase_1_epochs + phase_2_epochs
            
            if phase == 'training_phase_1':
                # Phase 1: 0% to 50% of overall progress
                phase_progress = (current / total) if total > 0 else 0
                return int((phase_progress * phase_1_epochs / total_epochs) * 100)
            elif phase == 'training_phase_2':
                # Phase 2: 50% to 100% of overall progress
                phase_1_weight = phase_1_epochs / total_epochs
                phase_progress = (current / total) if total > 0 else 0
                return int((phase_1_weight + (phase_progress * phase_2_epochs / total_epochs)) * 100)
            else:
                # Non-training phases or unknown phases
                return int((current / total) * 100) if total > 0 else 0
                
        except Exception as e:
            self.logger.debug(f"Error calculating overall progress: {e}")
            return int((current / total) * 100) if total > 0 else 0
    
    def _handle_epoch_completion(self, phase_num: str, epoch: int, metrics: Dict[str, Any]):
        """Handle epoch completion with metrics display."""
        self._log_success(f"📊 Phase {phase_num} - Epoch {epoch} Complete:")
        
        # Display key metrics
        train_loss = metrics.get('train_loss', 0)
        val_loss = metrics.get('val_loss', 0)
        if train_loss > 0 or val_loss > 0:
            self._log_info(f"   Loss: Train={train_loss:.4f} Val={val_loss:.4f}")
        
        # Display layer metrics
        for layer in ['layer_1', 'layer_2', 'layer_3']:
            acc = metrics.get(f'{layer}_accuracy', 0)
            f1 = metrics.get(f'{layer}_f1', 0)
            if acc > 0 or f1 > 0:
                self._log_info(f"   {layer.upper()}: Acc={acc:.3f} F1={f1:.3f}")
        
        # Execute additional callbacks
        self._execute_metrics_callback(phase_num, epoch, metrics)
        self._execute_chart_callback(phase_num, metrics)
    
    def _handle_training_success(self, result: Dict[str, Any]):
        """Handle successful training completion."""
        self._log_success("🎉 Training completed successfully!")
        
        # Display final results
        training_result = result.get('final_training_result', {})
        if training_result.get('success'):
            best_metrics = training_result.get('best_metrics', {})
            self._log_success(f"Best mAP@0.5: {best_metrics.get('val_map50', 0):.4f}")
            
            # Display layer performance
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                acc = best_metrics.get(f'{layer}_accuracy', 0)
                if acc > 0:
                    f1 = best_metrics.get(f'{layer}_f1', 0)
                    self._log_info(f"{layer}: Accuracy={acc:.4f} F1={f1:.4f}")
        
        # Display visualization info
        viz_result = result.get('visualization_result', {})
        if viz_result.get('success'):
            session_id = viz_result.get('session_id', 'N/A')
            charts_count = viz_result.get('charts_count', 0)
            self._log_info(f"📊 Generated {charts_count} visualization charts")
            self._log_info(f"📁 Charts saved to: data/visualization/{session_id}/")
        
        # Execute completion callbacks
        self._execute_success_callback(result)
        
        # Update final progress
        self.update_progress(
            overall_progress=100,
            overall_message="Training completed successfully!",
            phase_progress=100,
            phase_message="Complete",
            epoch_progress=100,
            epoch_message="Complete"
        )
    
    def _handle_training_failure(self, error_msg: str):
        """Handle training failure."""
        self._log_error(f"❌ Training failed: {error_msg}")
        
        # Call failure callback
        if 'on_failure' in self.callbacks:
            self.callbacks['on_failure'](error_msg)
    
    def stop(self) -> Dict[str, Any]:
        """Stop training operation."""
        # Note: The unified training pipeline doesn't currently support stopping mid-training
        # This would require keyboard interrupt simulation or process termination
        self._log_warning("⚠️ Training stop not implemented for unified pipeline")
        return {'success': False, 'message': 'Stop functionality not available'}
    
    def is_training(self) -> bool:
        """Check if training is currently active."""
        return self._is_training
    
    def get_training_result(self) -> Optional[Dict[str, Any]]:
        """Get the last training result."""
        return self._training_result
    
    # Logging helper methods
    def _log_info(self, message: str):
        """Log info message to UI."""
        self.logger.info(message)
        if hasattr(self.ui_module, 'log_info'):
            self.ui_module.log_info(message)
        self._execute_log_callback(message, 'info')
    
    def _log_success(self, message: str):
        """Log success message to UI."""
        self.logger.info(message)
        if hasattr(self.ui_module, 'log_success'):
            self.ui_module.log_success(message)
        self._execute_log_callback(message, 'success')
    
    def _log_warning(self, message: str):
        """Log warning message to UI."""
        self.logger.warning(message)
        if hasattr(self.ui_module, 'log_warning'):
            self.ui_module.log_warning(message)
        self._execute_log_callback(message, 'warning')
    
    def _log_error(self, message: str):
        """Log error message to UI."""
        self.logger.error(message)
        if hasattr(self.ui_module, 'log_error'):
            self.ui_module.log_error(message)
        self._execute_log_callback(message, 'error')
    
    # Enhanced callback system methods
    
    def _execute_metrics_callback(self, phase_num: str, epoch: int, metrics: Dict[str, Any]) -> None:
        """Execute metrics callback for epoch completion with intelligent layer filtering."""
        try:
            if 'on_metrics_update' in self.callbacks:
                # Apply intelligent layer filtering to metrics before display
                filtered_metrics = self._filter_metrics_by_active_layers(phase_num, metrics)
                
                callback_data = {
                    'phase': phase_num,
                    'epoch': epoch,
                    'metrics': filtered_metrics,
                    'timestamp': time.time()
                }
                self.callbacks['on_metrics_update'](callback_data)
        except Exception as e:
            self.logger.debug(f"Metrics callback error: {e}")
    
    def _filter_metrics_by_active_layers(self, phase: str, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Filter metrics to show only active layers using intelligent detection."""
        # Determine which layers to show based on phase and actual metrics presence
        show_layers = []
        filter_zeros = False
        
        if phase.lower() == 'training_phase_1':
            # Two-phase mode, phase 1: Only show layer_1
            show_layers = ['layer_1']
            filter_zeros = True  # Filter zeros to reduce clutter
        elif phase.lower() == 'training_phase_2':
            # Two-phase mode, phase 2: Show all layers
            show_layers = ['layer_1', 'layer_2', 'layer_3']
            filter_zeros = False
        elif phase.lower() == 'training_phase_single':
            # Single-phase mode: Determine active layers from actual metrics
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
                show_layers = active_layers
                filter_zeros = True
            else:
                # Single-phase, multi-layer mode: show all layers
                show_layers = ['layer_1', 'layer_2', 'layer_3']
                filter_zeros = False
        else:
            # Default: show all layers
            show_layers = ['layer_1', 'layer_2', 'layer_3']
            filter_zeros = False
        
        # Filter metrics based on active layers
        filtered_metrics = {}
        
        # Always include core metrics
        core_metrics = ['train_loss', 'val_loss', 'val_map50', 'val_map50_95', 
                       'val_precision', 'val_recall', 'val_f1', 'val_accuracy']
        for metric in core_metrics:
            if metric in metrics:
                filtered_metrics[metric] = metrics[metric]
        
        # Include layer-specific metrics for active layers only
        for metric_name, value in metrics.items():
            # Check if this is a layer metric and if we should show it
            for layer in show_layers:
                if (metric_name.startswith(f'{layer}_') or metric_name.startswith(f'val_{layer}_')):
                    # Apply zero filtering if specified
                    if filter_zeros:
                        if isinstance(value, (int, float)) and value > 0.0001:
                            filtered_metrics[metric_name] = value
                    else:
                        filtered_metrics[metric_name] = value
                    break
        
        # Include class-specific AP metrics if they have meaningful values
        for metric_name, value in metrics.items():
            if metric_name.startswith('val_ap_') and isinstance(value, (int, float)) and value > 0.0001:
                filtered_metrics[metric_name] = value
        
        return filtered_metrics
    
    def _execute_chart_callback(self, phase_num: str, metrics: Dict[str, Any]) -> None:
        """Execute chart update callback with intelligent layer filtering."""
        try:
            if 'on_chart_update' in self.callbacks:
                # Use the same filtering logic for charts as for metrics display
                filtered_metrics = self._filter_metrics_by_active_layers(phase_num, metrics)
                self.callbacks['on_chart_update'](filtered_metrics)
        except Exception as e:
            self.logger.debug(f"Chart callback error: {e}")
    
    def _execute_success_callback(self, result: Dict[str, Any]) -> None:
        """Execute success callback with full result data."""
        try:
            if 'on_success' in self.callbacks:
                self.callbacks['on_success'](result)
            
            # Also call the simple success callback for backward compatibility
            if 'on_training_complete' in self.callbacks:
                self.callbacks['on_training_complete']("Training completed successfully")
        except Exception as e:
            self.logger.debug(f"Success callback error: {e}")
    
    def _execute_log_callback(self, message: str, level: str = 'info') -> None:
        """Execute log callback for real-time log updates."""
        try:
            if 'on_log_update' in self.callbacks:
                log_data = {
                    'message': message,
                    'level': level,
                    'timestamp': time.time(),
                    'module': 'training'
                }
                self.callbacks['on_log_update'](log_data)
        except Exception as e:
            self.logger.debug(f"Log callback error: {e}")
    
    def _execute_live_chart_callback(self, chart_data: Dict[str, Any]) -> None:
        """Execute live chart callback for real-time chart updates with intelligent filtering."""
        try:
            if 'on_live_chart_update' in self.callbacks:
                # Apply intelligent filtering to live chart data if it contains metrics
                if 'layers' in chart_data and 'phase' in chart_data:
                    phase_num = chart_data.get('phase', 1)
                    phase_name = f'training_phase_{phase_num}' if isinstance(phase_num, int) else str(phase_num)
                    
                    # Filter layer data in chart_data
                    layers_data = chart_data['layers']
                    show_layers = self._determine_active_layers_for_charts(phase_name, layers_data)
                    
                    # Only include active layers in chart data
                    filtered_layers_data = {layer: data for layer, data in layers_data.items() if layer in show_layers}
                    chart_data['layers'] = filtered_layers_data
                
                self.callbacks['on_live_chart_update'](chart_data)
        except Exception as e:
            self.logger.debug(f"Live chart callback error: {e}")
    
    def _determine_active_layers_for_charts(self, phase_name: str, layers_data: Dict[str, Any]) -> list:
        """Determine which layers should be included in charts using the same logic as metrics callback."""
        if 'training_phase_1' in phase_name.lower():
            # Phase 1: Only show layer_1
            return ['layer_1']
        elif 'training_phase_2' in phase_name.lower():
            # Phase 2: Show all layers
            return ['layer_1', 'layer_2', 'layer_3']
        elif 'training_phase_single' in phase_name.lower():
            # Single-phase mode: Determine active layers from actual data
            layer_activity = {}
            for layer in ['layer_1', 'layer_2', 'layer_3']:
                # Check if this layer has any meaningful metrics
                layer_data = layers_data.get(layer, {})
                has_activity = any(
                    layer_data.get(metric, 0) > 0.0001
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
            # Default: show all layers
            return ['layer_1', 'layer_2', 'layer_3']
    
    def register_additional_callbacks(self, additional_callbacks: Dict[str, Callable]) -> None:
        """Register additional callbacks for extended functionality.
        
        Args:
            additional_callbacks: Dictionary of callback names and functions
                Supported callbacks:
                - on_metrics_update: Called after each epoch with metrics
                - on_chart_update: Called when charts need updating
                - on_log_update: Called for real-time log updates
                - on_live_chart_update: Called for live chart updates
                - on_training_complete: Called when training completes
        """
        self.callbacks.update(additional_callbacks)