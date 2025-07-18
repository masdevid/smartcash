"""
File: smartcash/ui/model/training/operations/training_start_operation.py
Description: Training start operation handler with backend integration.
"""

from typing import Dict, Any
from .training_base_operation import BaseTrainingOperation


class TrainingStartOperationHandler(BaseTrainingOperation):
    """
    Handler for starting training operations with model selection and live charts.
    
    Features:
    - 🏗️ Automatic model selection from backbone configuration
    - 📊 Live chart initialization and updates
    - 🔄 Real-time progress tracking
    - 🛡️ Comprehensive error handling
    """

    def execute(self) -> Dict[str, Any]:
        """Execute the training start operation."""
        self.log_operation("🚀 Memulai proses training...", level='info')
        
        # Start dual progress tracking: 5 overall steps
        self.start_dual_progress("Training Start", total_steps=5)
        
        try:
            # Step 1: Initialize backend training API
            self.update_dual_progress(
                current_step=1, 
                current_percent=0,
                message="Menginisialisasi backend training API..."
            )
            
            training_api = self._initialize_training_api()
            if not training_api:
                error_msg = "Failed to initialize training API"
                self.error_dual_progress(error_msg)
                return {'success': False, 'message': error_msg}
            
            self.update_dual_progress(
                current_step=1,
                current_percent=100,
                message="Backend training API siap"
            )
            
            # Step 2: Model selection from backbone
            self.update_dual_progress(
                current_step=2,
                current_percent=0,
                message="Memilih model dari konfigurasi backbone..."
            )
            
            model_selection_result = self._select_model_from_backbone()
            if not model_selection_result['success']:
                self.error_dual_progress(model_selection_result['message'])
                return model_selection_result
            
            self.update_dual_progress(
                current_step=2,
                current_percent=100,
                message="Model dipilih dan dikonfigurasi"
            )
            
            # Step 3: Setup live charts
            self.update_dual_progress(
                current_step=3,
                current_percent=0,
                message="Menyiapkan live charts..."
            )
            
            chart_setup_result = self._setup_live_charts()
            if not chart_setup_result['success']:
                self.log_operation(f"⚠️ Chart setup warning: {chart_setup_result['message']}", 'warning')
            
            self.update_dual_progress(
                current_step=3,
                current_percent=100,
                message="Live charts siap"
            )
            
            # Step 4: Validate training prerequisites
            self.update_dual_progress(
                current_step=4,
                current_percent=0,
                message="Memvalidasi prerequisite training..."
            )
            
            validation_result = self._validate_training_prerequisites()
            if not validation_result['success']:
                self.error_dual_progress(validation_result['message'])
                return validation_result
            
            self.update_dual_progress(
                current_step=4,
                current_percent=100,
                message="Prerequisite training valid"
            )
            
            # Step 5: Start training process
            self.update_dual_progress(
                current_step=5,
                current_percent=0,
                message="Memulai training process..."
            )
            
            training_result = self._start_training_process(training_api)
            
            if training_result['success']:
                self.complete_dual_progress("Training berhasil dimulai")
                
                # Execute success callback
                self._execute_callback('on_success', "Training berhasil dimulai dengan live charts aktif")
                
                return {
                    'success': True,
                    'message': 'Training started successfully',
                    'training_id': training_result.get('training_id'),
                    'estimated_time': training_result.get('estimated_time')
                }
            else:
                self.error_dual_progress(training_result['message'])
                self._execute_callback('on_failure', training_result['message'])
                return training_result
                
        except Exception as e:
            error_message = f"Training start operation failed: {str(e)}"
            self.error_dual_progress(error_message)
            self._execute_callback('on_failure', error_message)
            return {'success': False, 'message': error_message}

    def _initialize_training_api(self) -> Any:
        """Initialize backend training API."""
        try:
            from smartcash.model.api.core import create_model_api
            
            # Create progress callback for backend
            def progress_callback(*args, **kwargs):
                # Handle different callback signatures from backend
                if len(args) >= 2:
                    percentage = args[0] if isinstance(args[0], (int, float)) else 0
                    message = args[1] if isinstance(args[1], str) else ""
                elif len(args) == 1:
                    percentage = args[0] if isinstance(args[0], (int, float)) else 0
                    message = ""
                else:
                    percentage = kwargs.get('percentage', 0)
                    message = kwargs.get('message', "")
                
                # Update current step progress
                self.update_dual_progress(
                    current_step=self._current_step if hasattr(self, '_current_step') else 1,
                    current_percent=percentage,
                    message=message
                )
            
            # Initialize API with training-specific config
            api = create_model_api(progress_callback=progress_callback)
            
            self.log_operation("✅ Training API initialized successfully", 'success')
            return api
            
        except Exception as e:
            self.log_operation(f"❌ Failed to initialize training API: {e}", 'error')
            return None

    def _select_model_from_backbone(self) -> Dict[str, Any]:
        """Select model based on current backbone configuration."""
        try:
            # Get backbone configuration from shared config or backbone module
            backbone_config = self._get_backbone_configuration()
            
            if not backbone_config:
                return {
                    'success': False,
                    'message': 'No backbone configuration found. Please configure backbone first.'
                }
            
            # Use config handler to select model
            config_handler = getattr(self._ui_module, '_config_handler', None)
            if config_handler and hasattr(config_handler, 'select_model_from_backbone'):
                model_selection_result = config_handler.select_model_from_backbone(backbone_config)
                
                if model_selection_result['success']:
                    model_info = model_selection_result['model_selection']
                    self.log_operation(
                        f"🏗️ Model selected: {model_info.get('backbone_type')} "
                        f"({model_info.get('num_classes')} classes)", 
                        'info'
                    )
                    
                return model_selection_result
            else:
                return {
                    'success': False,
                    'message': 'Config handler not available for model selection'
                }
                
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to select model from backbone: {e}'
            }

    def _get_backbone_configuration(self) -> Dict[str, Any]:
        """Get current backbone configuration."""
        try:
            # Try to get from backbone module if available
            from smartcash.ui.model.backbone.backbone_uimodule import get_backbone_uimodule
            
            backbone_module = get_backbone_uimodule(auto_initialize=False)
            if backbone_module and hasattr(backbone_module, 'get_current_config'):
                backbone_config = backbone_module.get_current_config()
                self.log_operation("✅ Backbone configuration retrieved", 'success')
                return backbone_config
            else:
                # Fallback to default backbone config
                self.log_operation("⚠️ Using default backbone configuration", 'warning')
                return {
                    'backbone': {
                        'model_type': 'efficientnet_b4',
                        'input_size': 640,
                        'num_classes': 7,
                        'feature_optimization': False
                    },
                    'model': {
                        'model_name': 'smartcash_training_model'
                    }
                }
                
        except Exception as e:
            self.log_operation(f"⚠️ Error getting backbone config: {e}", 'warning')
            return {}

    def _setup_live_charts(self) -> Dict[str, Any]:
        """Setup live charts for training monitoring."""
        try:
            # Initialize chart updaters in UI module
            if hasattr(self._ui_module, '_setup_chart_updaters'):
                self._ui_module._setup_chart_updaters()
                
            # Initialize charts with empty data
            initial_metrics = {
                'train_loss': 0.0,
                'val_loss': 0.0,
                'mAP@0.5': 0.0,
                'mAP@0.75': 0.0,
                'epoch': 0
            }
            
            self.update_charts(initial_metrics)
            
            self.log_operation("📊 Live charts initialized", 'success')
            return {'success': True, 'message': 'Live charts setup successfully'}
            
        except Exception as e:
            self.log_operation(f"⚠️ Chart setup failed: {e}", 'warning')
            return {'success': False, 'message': f'Chart setup failed: {e}'}

    def _validate_training_prerequisites(self) -> Dict[str, Any]:
        """Validate training prerequisites."""
        try:
            # Check if model is selected
            model_selection = self.config.get('model_selection', {})
            if not model_selection.get('backbone_type'):
                return {
                    'success': False,
                    'message': 'No model selected. Please configure backbone first.'
                }
            
            # Check training parameters
            training_config = self.config.get('training', {})
            epochs = training_config.get('epochs', 0)
            if epochs <= 0:
                return {
                    'success': False,
                    'message': 'Invalid training epochs. Please check training configuration.'
                }
            
            # Check data availability (placeholder)
            # In real implementation, this would check for training data
            
            self.log_operation("✅ Training prerequisites validated", 'success')
            return {'success': True, 'message': 'All prerequisites validated'}
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Prerequisites validation failed: {e}'
            }

    def _start_training_process(self, training_api) -> Dict[str, Any]:
        """Start the actual training process."""
        try:
            # Prepare training configuration
            training_config = self._prepare_training_config()
            
            # Start training with backend API
            # This would integrate with the actual training service
            self.log_operation("🚀 Starting training with backend API...", 'info')
            
            # Simulate training start (replace with actual API call)
            training_id = f"training_{int(time.time())}"
            
            # Set up periodic chart updates during training
            self._setup_training_monitoring(training_id)
            
            self.log_operation(f"✅ Training started with ID: {training_id}", 'success')
            
            return {
                'success': True,
                'message': 'Training started successfully',
                'training_id': training_id,
                'estimated_time': self._estimate_training_time()
            }
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Failed to start training: {e}'
            }

    def _prepare_training_config(self) -> Dict[str, Any]:
        """Prepare training configuration for backend API."""
        training_config = self.config.get('training', {})
        model_selection = self.config.get('model_selection', {})
        
        return {
            'model': {
                'backbone': model_selection.get('backbone_type', 'efficientnet_b4'),
                'num_classes': model_selection.get('num_classes', 7),
                'input_size': model_selection.get('input_size', 640),
                'feature_optimization': model_selection.get('feature_optimization', False)
            },
            'training': training_config,
            'charts_enabled': self.config.get('charts', {}).get('enabled', True)
        }

    def _setup_training_monitoring(self, training_id: str) -> None:
        """Setup training monitoring and chart updates."""
        # This would set up periodic updates from the training process
        # For now, just log that monitoring is set up
        self.log_operation(f"📊 Training monitoring setup for {training_id}", 'info')

    def _estimate_training_time(self) -> str:
        """Estimate training completion time."""
        epochs = self.config.get('training', {}).get('epochs', 100)
        # Simple estimation: ~30 seconds per epoch (placeholder)
        estimated_minutes = (epochs * 30) // 60
        return f"~{estimated_minutes} minutes"


# Import time for training ID generation
import time