"""
File: smartcash/ui/model/train/operations/train_operation_manager.py
Operation manager for train module extending OperationHandler.
"""

from typing import Dict, Any, Optional
import asyncio
import threading
from smartcash.ui.core.handlers.operation_handler import OperationHandler
from smartcash.ui.logger import get_module_logger

# Import service for backend integration
from ..services.training_service import TrainingService


class TrainOperationManager(OperationHandler):
    """
    Operation manager for training module.
    
    Features:
    - 🚀 Training operations (start, stop, resume, validate)
    - 📊 Live dual chart updates (loss and mAP)
    - 🔄 Real-time progress tracking
    - 🛡️ Error handling with user feedback
    - 🎯 Button management with disable/enable functionality
    - 📋 Training status tracking and reporting
    - 🔗 Backend training service integration
    - 💾 Best model automatic saving with naming convention
    """
    
    def __init__(self, config: Dict[str, Any], operation_container: Any):
        """
        Initialize training operation manager.
        
        Args:
            config: Configuration dictionary
            operation_container: UI operation container for logging and progress
        """
        super().__init__(
            module_name='train',
            parent_module='model',
            operation_container=operation_container
        )
        
        self.config = config
        self.logger = get_module_logger("smartcash.ui.model.train.operations")
        
        # Initialize service and state
        self._service = None
        self._training_thread = None
        self._is_training = False
        self._training_metrics = {}
        self._chart_callbacks = {}
        
        # Initialize service instance
        self._initialize_service()
        
        # Track button states for restore
        self._button_states = {}
    
    def _initialize_service(self) -> None:
        """Initialize training service instance."""
        try:
            self._service = TrainingService()
            self.logger.debug("✅ Training service initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize service: {e}")
    
    def initialize(self) -> None:
        """Initialize the operation manager."""
        try:
            super().initialize()
            self.log("🔧 Training operation manager initialized", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize training operation manager: {e}")
            self.log(f"❌ Initialization failed: {e}", 'error')
    
    def get_operations(self) -> Dict[str, str]:
        """
        Get available operations.
        
        Returns:
            Dictionary of operation names and descriptions
        """
        return {
            'start': 'Start training with current configuration',
            'stop': 'Stop current training and save best model',
            'resume': 'Resume training from last checkpoint',
            'validate': 'Run validation on current best model'
        }
    
    def set_chart_callbacks(self, loss_chart_callback: callable, map_chart_callback: callable) -> None:
        """
        Set callbacks for updating live charts.
        
        Args:
            loss_chart_callback: Callback function for loss chart updates
            map_chart_callback: Callback function for mAP chart updates
        """
        self._chart_callbacks = {
            'loss': loss_chart_callback,
            'map': map_chart_callback
        }
        self.logger.debug("✅ Chart callbacks registered")
    
    # ==================== TRAINING OPERATIONS ====================
    
    def execute_start(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training start operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Operation result dictionary
        """
        try:
            if self._is_training:
                return {'success': False, 'message': 'Training already in progress'}
            
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("🚀 Starting model training...", 'info')
            # Note: Using disable_all_buttons for now, individual button control would need implementation
            self._button_states = self.disable_all_buttons("🚀 Training...")
            
            # Update progress
            self.update_progress(0, "Initializing training...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute training start operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Training service not available - cannot proceed with training")
            
            # Start training in background thread
            self._training_thread = threading.Thread(
                target=self._execute_training_async,
                args=(operation_config,),
                daemon=True
            )
            self._training_thread.start()
            self._is_training = True
            
            return {
                'success': True,
                'message': 'Training started successfully',
                'training_status': 'started'
            }
            
        except Exception as e:
            self.logger.error(f"Training start error: {e}")
            self.log(f"❌ Training start error: {e}", 'error')
            self.update_progress(0, "Training start failed")
            if self._button_states:
                self.enable_all_buttons(self._button_states)
            return {'success': False, 'message': str(e)}
    
    def execute_stop(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training stop operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Stop result dictionary
        """
        try:
            if not self._is_training:
                return {'success': False, 'message': 'No training in progress'}
            
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("🛑 Stopping training...", 'info')
            self._button_states = self.disable_all_buttons("🛑 Stopping...")
            
            # Update progress
            self.update_progress(90, "Stopping training and saving best model...")
            
            # Stop training via service
            if self._service:
                stop_result = self._service.stop_training()
                
                if stop_result.get('success'):
                    self.update_progress(100, "Training stopped successfully")
                    self.log("✅ Training stopped and best model saved", 'success')
                else:
                    self.update_progress(0, "Training stop failed")
                    self.log(f"❌ Training stop failed: {stop_result.get('message', 'Unknown error')}", 'error')
                
                self._is_training = False
                if self._button_states:
                    self.enable_all_buttons(self._button_states)
                
                return stop_result
            else:
                raise RuntimeError("Training service not available")
            
        except Exception as e:
            self.logger.error(f"Training stop error: {e}")
            self.log(f"❌ Training stop error: {e}", 'error')
            if self._button_states:
                self.enable_all_buttons(self._button_states)
            return {'success': False, 'message': str(e)}
    
    def execute_resume(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute training resume operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Resume result dictionary
        """
        try:
            if self._is_training:
                return {'success': False, 'message': 'Training already in progress'}
            
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("🔄 Resuming training from checkpoint...", 'info')
            # Note: Using disable_all_buttons for now, individual button control would need implementation
            self._button_states = self.disable_all_buttons("🚀 Training...")
            
            # Update progress
            self.update_progress(0, "Loading checkpoint...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute resume operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Training service not available - cannot proceed with resume")
            
            # Start resume training in background thread
            self._training_thread = threading.Thread(
                target=self._execute_resume_async,
                args=(operation_config,),
                daemon=True
            )
            self._training_thread.start()
            self._is_training = True
            
            return {
                'success': True,
                'message': 'Training resumed successfully',
                'training_status': 'resumed'
            }
            
        except Exception as e:
            self.logger.error(f"Training resume error: {e}")
            self.log(f"❌ Training resume error: {e}", 'error')
            self.update_progress(0, "Training resume failed")
            if self._button_states:
                self.enable_all_buttons(self._button_states)
            return {'success': False, 'message': str(e)}
    
    def execute_validate(self, config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute model validation operation.
        
        Args:
            config: Optional configuration override
            
        Returns:
            Validation result dictionary
        """
        try:
            # Clear logs from previous operations
            self.clear_logs()
            
            self.log("📊 Running model validation...", 'info')
            self._button_states = self.disable_all_buttons("📊 Validating...")
            
            # Update progress
            self.update_progress(0, "Initializing validation...")
            
            # Use provided config or current config
            operation_config = config or self.config
            
            # Execute validation operation - fail fast if service not available
            if not self._service:
                raise RuntimeError("Training service not available - cannot proceed with validation")
            
            result = self._execute_validate_with_service(operation_config)
            
            # Update progress based on result
            if result.get('success'):
                self.update_progress(100, "Validation completed successfully")
                self.log("✅ Model validation completed successfully", 'success')
            else:
                self.update_progress(0, "Validation failed")
                self.log(f"❌ Validation failed: {result.get('message', 'Unknown error')}", 'error')
            
            return result
            
        except Exception as e:
            self.logger.error(f"Validation operation error: {e}")
            self.log(f"❌ Validation operation error: {e}", 'error')
            self.update_progress(0, "Validation failed")
            return {'success': False, 'message': str(e)}
        
        finally:
            if self._button_states:
                self.enable_all_buttons(self._button_states)
    
    # ==================== ASYNC TRAINING EXECUTION ====================
    
    def _execute_training_async(self, config: Dict[str, Any]) -> None:
        """Execute training in background thread with live updates."""
        try:
            # Progress callback for training updates
            def progress_callback(epoch: int, total_epochs: int, metrics: Dict[str, Any]):
                progress = int((epoch / total_epochs) * 100)
                self.update_progress(progress, f"Training epoch {epoch}/{total_epochs}")
                
                # Update live charts
                self._update_live_charts(metrics)
                
                # Store metrics
                self._training_metrics = metrics
            
            # Log callback for training logs
            def log_callback(message: str, level: str = 'info'):
                self.log(message, level)
            
            # Load backbone configuration
            self.update_progress(10, "Loading backbone configuration...")
            backbone_config = self._load_backbone_config(config)
            
            # Build model from backbone
            self.update_progress(20, "Building model from backbone...")
            
            # Start training with service
            self.update_progress(30, "Starting training loop...")
            training_result = self._service.start_training(
                config=config,
                backbone_config=backbone_config,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if training_result.get('success'):
                self.update_progress(100, "Training completed successfully")
                self.log("✅ Training completed successfully", 'success')
                
                # Get best model info
                best_model_info = training_result.get('best_model_info', {})
                model_name = best_model_info.get('model_name', 'Unknown')
                self.log(f"💾 Best model saved as: {model_name}", 'success')
                
            else:
                self.update_progress(0, "Training failed")
                self.log(f"❌ Training failed: {training_result.get('message', 'Unknown error')}", 'error')
            
        except Exception as e:
            self.logger.error(f"Async training error: {e}")
            self.log(f"❌ Training error: {e}", 'error')
            self.update_progress(0, "Training failed")
        
        finally:
            self._is_training = False
            if self._button_states:
                self.enable_all_buttons(self._button_states)
    
    def _execute_resume_async(self, config: Dict[str, Any]) -> None:
        """Execute resume training in background thread."""
        try:
            # Progress callback for resume updates
            def progress_callback(epoch: int, total_epochs: int, metrics: Dict[str, Any]):
                progress = int((epoch / total_epochs) * 100)
                self.update_progress(progress, f"Resuming epoch {epoch}/{total_epochs}")
                
                # Update live charts
                self._update_live_charts(metrics)
                
                # Store metrics
                self._training_metrics = metrics
            
            # Log callback for resume logs
            def log_callback(message: str, level: str = 'info'):
                self.log(message, level)
            
            # Resume training with service
            self.update_progress(20, "Restoring training state...")
            resume_result = self._service.resume_training(
                config=config,
                progress_callback=progress_callback,
                log_callback=log_callback
            )
            
            if resume_result.get('success'):
                self.update_progress(100, "Training resumed successfully")
                self.log("✅ Training resumed successfully", 'success')
            else:
                self.update_progress(0, "Resume failed")
                self.log(f"❌ Resume failed: {resume_result.get('message', 'Unknown error')}", 'error')
            
        except Exception as e:
            self.logger.error(f"Async resume error: {e}")
            self.log(f"❌ Resume error: {e}", 'error')
            self.update_progress(0, "Resume failed")
        
        finally:
            self._is_training = False
            if self._button_states:
                self.enable_all_buttons(self._button_states)
    
    # ==================== SERVICE INTEGRATION ====================
    
    def _execute_validate_with_service(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Execute validation operation with service integration."""
        try:
            training_config = config.get('training', {})
            
            # Load validation data
            self.update_progress(30, "Loading validation data...")
            
            # Run validation
            self.update_progress(60, "Running model validation...")
            validation_result = self._service.validate_model(
                config=training_config,
                progress_callback=self.update_progress,
                log_callback=self.log
            )
            
            return {
                'success': validation_result.get('success', False),
                'message': 'Model validation completed',
                'validation_results': validation_result,
                'metrics': validation_result.get('metrics', {})
            }
            
        except Exception as e:
            return {'success': False, 'message': f'Service validation failed: {e}'}
    
    def _load_backbone_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Load backbone configuration for training continuation."""
        try:
            # Get backbone integration config
            backbone_integration = config.get('backbone_integration', {})
            
            if backbone_integration.get('load_from_backbone', True):
                # Try to load from shared methods
                try:
                    from smartcash.ui.core.shared_methods import get_shared_method
                    get_backbone_config = get_shared_method('backbone.get_config')
                    
                    if get_backbone_config:
                        backbone_config = get_backbone_config()
                        self.log("✅ Loaded backbone configuration from backbone module", 'info')
                        return backbone_config
                    
                except Exception as e:
                    self.logger.warning(f"Could not load backbone config from shared methods: {e}")
            
            # Fallback to embedded config
            return backbone_integration.get('backbone_config', {})
            
        except Exception as e:
            self.logger.warning(f"Failed to load backbone config: {e}")
            return {}
    
    def _update_live_charts(self, metrics: Dict[str, Any]) -> None:
        """Update live charts with training metrics."""
        try:
            # Update loss chart
            if 'loss' in self._chart_callbacks and 'train_loss' in metrics:
                loss_data = {
                    'train_loss': metrics.get('train_loss', 0.0),
                    'val_loss': metrics.get('val_loss', 0.0)
                }
                self._chart_callbacks['loss'](loss_data)
            
            # Update mAP chart
            if 'map' in self._chart_callbacks and 'val_map50' in metrics:
                map_data = {
                    'val_map50': metrics.get('val_map50', 0.0),
                    'val_map75': metrics.get('val_map75', 0.0)
                }
                self._chart_callbacks['map'](map_data)
                
        except Exception as e:
            self.logger.warning(f"Failed to update live charts: {e}")
    
    # ==================== FAIL-FAST APPROACH ====================
    # No fallback simulations - service must be available for operations
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current operation manager status.
        
        Returns:
            Status dictionary
        """
        return {
            'initialized': True,
            'service_ready': self._service is not None,
            'training_active': self._is_training,
            'available_operations': list(self.get_operations().keys()),
            'current_metrics': self._training_metrics.copy(),
            'module_name': self.module_name,
            'parent_module': self.parent_module
        }
    
    def cleanup(self) -> None:
        """Cleanup operation manager resources."""
        try:
            # Stop training if active
            if self._is_training and self._service:
                self._service.stop_training()
                self._is_training = False
            
            # Wait for training thread to finish
            if self._training_thread and self._training_thread.is_alive():
                self._training_thread.join(timeout=5.0)
            
            # Cleanup service instance
            if self._service and hasattr(self._service, 'cleanup'):
                self._service.cleanup()
            
            # Clear references
            self._service = None
            self._training_thread = None
            self._chart_callbacks.clear()
            self._training_metrics.clear()
            
            super().cleanup()
            
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")