"""
File: smartcash/ui/model/training/operations/unified_training_operation.py
Description: Unified training operation that directly uses the unified training pipeline.
"""

from typing import Dict, Any, Callable, Optional
from smartcash.ui.logger import get_module_logger
from smartcash.model.api.core import run_full_training_pipeline


class UnifiedTrainingOperation:
    """
    Unified training operation handler that directly uses the unified training pipeline.
    
    This replaces the complex training operation system with a direct call to the
    unified training pipeline from smartcash.model.training.unified_training_pipeline.
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
            
            # Create progress callback that bridges to UI
            progress_callback = self._create_ui_progress_callback()
            
            # Map form configuration to unified training pipeline parameters
            pipeline_params = self._map_config_to_pipeline_params(training_config, progress_callback)
            
            self._log_info(f"Training configuration: {pipeline_params['backbone']} ({pipeline_params['training_mode']})")
            self._log_info(f"Epochs: Phase 1={pipeline_params['phase_1_epochs']}, Phase 2={pipeline_params['phase_2_epochs']}")
            
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
    
    def _map_config_to_pipeline_params(self, config: Dict[str, Any], progress_callback: Callable) -> Dict[str, Any]:
        """Map UI configuration to unified training pipeline parameters."""
        return {
            # Core parameters
            'backbone': config.get('backbone', 'cspdarknet'),
            'phase_1_epochs': config.get('phase_1_epochs', 1),
            'phase_2_epochs': config.get('phase_2_epochs', 1),
            'checkpoint_dir': config.get('checkpoint_dir', 'data/checkpoints'),
            'progress_callback': progress_callback,
            'verbose': config.get('verbose', True),
            'force_cpu': config.get('force_cpu', False),
            'training_mode': config.get('training_mode', 'two_phase'),
            
            # Single-phase specific parameters
            'single_phase_layer_mode': config.get('single_phase_layer_mode', 'multi'),
            'single_phase_freeze_backbone': config.get('single_phase_freeze_backbone', False),
            
            # Training configuration parameters
            'loss_type': config.get('loss_type', 'uncertainty_multi_task'),
            'head_lr_p1': config.get('head_lr_p1', 1e-3),
            'head_lr_p2': config.get('head_lr_p2', 1e-4),
            'backbone_lr': config.get('backbone_lr', 1e-5),
            'batch_size': config.get('batch_size'),  # None for auto-detection
            
            # Early stopping configuration parameters
            'early_stopping_enabled': config.get('early_stopping_enabled', True),
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
                
                # Format phase name for display
                phase_display = phase.replace('_', ' ').title()
                
                # Handle different types of progress updates
                if phase in ['training_phase_1', 'training_phase_2']:
                    self._handle_training_phase_progress(phase, current, total, message, percentage, **kwargs)
                else:
                    self._handle_general_phase_progress(phase_display, percentage, message)
                
                # Call UI callbacks if available
                if 'on_progress' in self.callbacks:
                    self.callbacks['on_progress'](int(percentage), f"{phase_display}: {message}")
                    
            except Exception as e:
                self.logger.warning(f"Progress callback error: {e}")
        
        return progress_callback
    
    def _handle_training_phase_progress(self, phase: str, current: int, total: int, 
                                      message: str, percentage: float, **kwargs):
        """Handle progress updates for training phases."""
        phase_num = "1" if phase == 'training_phase_1' else "2"
        
        if 'epoch' in kwargs:
            epoch = kwargs['epoch']
            
            # Update UI with epoch progress
            epoch_msg = f"Phase {phase_num} - Epoch {epoch}/{total}"
            self._log_info(f"🔄 {epoch_msg}: {percentage:.0f}%")
            
            # Handle epoch completion with metrics
            if current == total and kwargs.get('metrics'):
                metrics = kwargs['metrics']
                self._handle_epoch_completion(phase_num, epoch, metrics)
        else:
            # General phase progress
            self._log_info(f"🔄 Phase {phase_num}: {percentage:.0f}%")
    
    def _handle_general_phase_progress(self, phase_display: str, percentage: float, message: str):
        """Handle progress updates for non-training phases."""
        if percentage >= 100:
            self._log_success(f"✅ {phase_display}: Complete")
        elif message:
            self._log_info(f"🔄 {phase_display} ({percentage:.0f}%): {message}")
        else:
            self._log_info(f"🔄 {phase_display}: {percentage:.0f}%")
    
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
        
        # Update charts if callback available
        if 'on_chart_update' in self.callbacks:
            self.callbacks['on_chart_update'](metrics)
    
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
        
        # Call success callback
        if 'on_success' in self.callbacks:
            self.callbacks['on_success']("Training completed successfully")
    
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
    
    def _log_success(self, message: str):
        """Log success message to UI."""
        self.logger.info(message)
        if hasattr(self.ui_module, 'log_success'):
            self.ui_module.log_success(message)
    
    def _log_warning(self, message: str):
        """Log warning message to UI."""
        self.logger.warning(message)
        if hasattr(self.ui_module, 'log_warning'):
            self.ui_module.log_warning(message)
    
    def _log_error(self, message: str):
        """Log error message to UI."""
        self.logger.error(message)
        if hasattr(self.ui_module, 'log_error'):
            self.ui_module.log_error(message)