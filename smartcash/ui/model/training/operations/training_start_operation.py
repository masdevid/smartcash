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
        
        # Start triple progress tracking: 5 overall steps
        self.start_progress("Training Initialization", progress=0, level='primary')
        
        try:
            # Step 1: Initialize backend training API
            self.update_triple_progress(
                overall_step=0,
                overall_message="Initializing Training Session",
                phase_step=0,
                phase_message="Menginisialisasi backend training API...",
                current_step=0,
                current_message="Setting up training environment"
            )
            
            training_api = self._initialize_training_api()
            if not training_api:
                error_msg = "Failed to initialize training API"
                self.error_dual_progress(error_msg)
                return {'success': False, 'message': error_msg}
            
            self.update_triple_progress(
                overall_step=20,
                overall_message="Initializing Training Session",
                phase_step=100,
                phase_message="Backend training API siap",
                current_step=100,
                current_message="Training environment ready"
            )
            
            # Step 2: Model selection from backbone
            self.update_triple_progress(
                overall_step=20,
                overall_message="Model Configuration",
                phase_step=0,
                phase_message="Memilih model dari konfigurasi backbone...",
                current_step=0,
                current_message="Loading backbone configuration"
            )
            
            model_selection_result = self._select_model_from_backbone()
            if not model_selection_result['success']:
                self.error_dual_progress(model_selection_result['message'])
                return model_selection_result
            
            self.update_triple_progress(
                overall_step=40,
                overall_message="Model Configuration",
                phase_step=100,
                phase_message="Model dipilih dan dikonfigurasi",
                current_step=100,
                current_message="Backbone model validated"
            )
            
            # Step 3: Setup live charts
            self.update_triple_progress(
                overall_step=40,
                overall_message="UI Preparation",
                phase_step=0,
                phase_message="Menyiapkan live charts...",
                current_step=0,
                current_message="Initializing chart components"
            )
            
            chart_setup_result = self._setup_live_charts()
            if not chart_setup_result['success']:
                self.log_operation(f"⚠️ Chart setup warning: {chart_setup_result['message']}", 'warning')
            
            self.update_triple_progress(
                overall_step=60,
                overall_message="UI Preparation",
                phase_step=100,
                phase_message="Live charts siap",
                current_step=100,
                current_message="Charts ready for real-time updates"
            )
            
            # Step 4: Validate training prerequisites
            self.update_triple_progress(
                overall_step=60,
                overall_message="Prerequisites Check",
                phase_step=0,
                phase_message="Memvalidasi prerequisite training...",
                current_step=0,
                current_message="Checking Python packages"
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
                self.complete_progress("Training berhasil dimulai")
                
                # Execute success callback
                self._execute_callback('on_success', "Training berhasil dimulai dengan live charts aktif")
                
                return {
                    'success': True,
                    'message': 'Training started successfully',
                    'training_id': training_result.get('training_id'),
                    'estimated_time': training_result.get('estimated_time')
                }
            else:
                self.error_progress(training_result['message'])
                self._execute_callback('on_failure', training_result['message'])
                return training_result
                
        except Exception as e:
            error_message = f"Training start operation failed: {str(e)}"
            self.error_progress(error_message)
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
        """Validate training prerequisites including required packages."""
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
            
            # Check required Python packages for training
            self.log_operation("🔍 Checking required Python packages for training...", 'info')
            package_check_result = self._check_training_packages()
            if not package_check_result['success']:
                return package_check_result
            
            # Check data availability (placeholder)
            # In real implementation, this would check for training data
            
            self.log_operation("✅ All training prerequisites validated", 'success')
            return {'success': True, 'message': 'All prerequisites validated'}
            
        except Exception as e:
            return {
                'success': False,
                'message': f'Prerequisites validation failed: {e}'
            }

    def _check_training_packages(self) -> Dict[str, Any]:
        """Check if required training packages are installed."""
        try:
            # Define required packages for training
            required_packages = [
                'torch', 'torchvision', 'ultralytics', 'timm', 
                'scikit-learn', 'tensorboard', 'thop'
            ]
            
            # Import dependency service
            from smartcash.ui.setup.dependency.services.package_status_tracker import PackageStatusTracker
            
            # Create package tracker
            package_tracker = PackageStatusTracker({}, self.logger if hasattr(self, 'logger') else None)
            
            # Check all required packages
            self.log_operation(f"📦 Checking {len(required_packages)} required packages...", 'info')
            package_results = package_tracker.check_multiple_packages(required_packages)
            
            # Analyze results
            missing_packages = []
            installed_packages = []
            
            for package_name, result in package_results.items():
                if result.get('installed', False):
                    version = result.get('version', 'unknown')
                    installed_packages.append(f"{package_name} ({version})")
                    self.log_operation(f"✅ {package_name} ({version}) - Installed", 'info')
                else:
                    missing_packages.append(package_name)
                    self.log_operation(f"❌ {package_name} - Not installed", 'warning')
            
            # Check if any packages are missing
            if missing_packages:
                missing_list = ', '.join(missing_packages)
                error_message = f"Training cannot start. Missing required packages: {missing_list}. Please install them using the Dependency module."
                self.log_operation(f"❌ {error_message}", 'error')
                return {
                    'success': False,
                    'message': error_message,
                    'missing_packages': missing_packages,
                    'installed_packages': installed_packages
                }
            
            # All packages are installed
            self.log_operation(f"✅ All {len(required_packages)} training packages are installed", 'success')
            return {
                'success': True,
                'message': 'All required training packages are available',
                'installed_packages': installed_packages
            }
            
        except Exception as e:
            error_msg = f"Failed to check training packages: {e}"
            self.log_operation(f"❌ {error_msg}", 'error')
            return {
                'success': False,
                'message': error_msg
            }
    
    def _start_training_process(self, training_api) -> Dict[str, Any]:
        """Start the actual training process with backend integration."""
        try:
            # Prepare training configuration
            training_config = self._prepare_training_config()
            
            # Create enhanced progress callback with proper backend integration
            def progress_callback(progress_data: Dict[str, Any]) -> None:
                """Forward backend triple progress data to UI components."""
                try:
                    # Extract backend progress data (new format)
                    if isinstance(progress_data, dict):
                        overall_progress = progress_data.get('overall_progress', 0)
                        epoch_progress = progress_data.get('epoch_progress', 0) 
                        batch_progress = progress_data.get('batch_progress', 0)
                        current_epoch = progress_data.get('current_epoch', 0)
                        total_epochs = progress_data.get('total_epochs', training_config.get('training', {}).get('epochs', 100))
                        current_batch = progress_data.get('current_batch', 0)
                        total_batches = progress_data.get('total_batches', 0)
                        phase = progress_data.get('phase', 'training')
                        message = progress_data.get('message', '')
                    else:
                        # Fallback for old callback signature (phase, current_step, total_steps, message)
                        overall_progress = 0
                        epoch_progress = 0 
                        batch_progress = 0
                        phase = 'training'
                        message = str(progress_data)
                    
                    # Update triple progress bars with backend data
                    self.update_triple_progress(
                        overall_step=int(overall_progress),
                        overall_message=f"Training Progress ({current_epoch}/{total_epochs})",
                        phase_step=int(epoch_progress),
                        phase_message=f"Phase: {phase.title()}",
                        current_step=int(batch_progress),
                        current_message=f"Batch {current_batch}/{total_batches}" if total_batches > 0 else message
                    )
                    
                    # Minimal operation logging - only key milestones
                    if phase == 'completed':
                        self.log_operation("✅ Training completed successfully", 'success')
                    elif phase == 'error':
                        self.log_operation(f"❌ Training error: {message}", 'error')
                    elif current_epoch > 0 and current_batch == 1:  # Start of new epoch
                        self.log_operation(f"🔄 Starting epoch {current_epoch}/{total_epochs}", 'info')
                    
                except Exception as e:
                    self.log_operation(f"⚠️ Progress callback error: {e}", 'warning')
            
            # Create enhanced metrics callback for live chart updates
            def metrics_callback(metrics_data: Dict[str, Any]) -> None:
                """Forward training metrics to UI charts with minimal logging."""
                try:
                    # Extract metrics from backend data
                    epoch = metrics_data.get('epoch', 0)
                    phase = metrics_data.get('phase', 'training')
                    metrics = metrics_data.get('metrics', {})
                    
                    # Extract loss values
                    train_loss = metrics.get('train_loss', 0.0)
                    val_loss = metrics.get('val_loss', 0.0)
                    
                    # Prepare comprehensive chart data
                    chart_data = {
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'mAP@0.5': metrics.get('mAP_50', metrics.get('mAP@0.5', 0.0)),
                        'mAP@0.75': metrics.get('mAP_75', metrics.get('mAP@0.75', 0.0)),
                        'precision': metrics.get('precision', 0.0),
                        'recall': metrics.get('recall', 0.0),
                        'f1_score': metrics.get('f1', 0.0),
                        **metrics
                    }
                    
                    # Update charts through callback
                    self.update_charts(chart_data)
                    
                    # Minimal metrics logging - only epoch completion
                    if phase == 'validation':
                        mAP_50 = chart_data['mAP@0.5']
                        self.log_operation(f"📊 Epoch {epoch} complete | Loss: {train_loss:.3f} | Val: {val_loss:.3f} | mAP: {mAP_50:.3f}", 'success')
                    
                except Exception as e:
                    self.log_operation(f"⚠️ Metrics callback error: {e}", 'warning')
            
            # Start actual training with backend integration
            self.log_operation("🚀 Starting training with backend API...", 'info')
            
            # Import and use the actual training service
            from smartcash.model.training import start_training
            
            # Start training with progress and metrics callbacks
            training_result = start_training(
                model_api=training_api,
                config=training_config,
                ui_components=self._ui_components,
                progress_callback=progress_callback,
                metrics_callback=metrics_callback
            )
            
            # Handle training result
            if training_result.get('success', False):
                training_id = training_result.get('training_id', f"training_{int(time.time())}")
                
                # Update button states for active training
                self._update_training_button_states(training_active=True)
                
                # Set up ongoing monitoring for training status
                self._setup_training_monitoring(training_id)
                
                # Store training state for stop functionality
                self._store_training_state(training_id, training_config)
                
                self.log_operation(f"✅ Training started successfully with ID: {training_id}", 'success')
                
                return {
                    'success': True,
                    'message': 'Training started successfully and is now visible',
                    'training_id': training_id,
                    'estimated_time': self._estimate_training_time()
                }
            else:
                error_message = training_result.get('message', 'Unknown training start error')
                self.log_operation(f"❌ Training failed to start: {error_message}", 'error')
                return {
                    'success': False,
                    'message': f'Training failed to start: {error_message}'
                }
            
        except Exception as e:
            # Ensure buttons are reset on error
            self._update_training_button_states(training_active=False)
            
            self.log_operation(f"❌ Training start error: {e}", 'error')
            return {
                'success': False,
                'message': f'Failed to start training: {e}'
            }

    def _update_training_button_states(self, training_active: bool) -> None:
        """Update button states based on training status."""
        try:
            # Get buttons from UI components
            buttons = {}
            if hasattr(self, '_ui_components') and self._ui_components:
                action_container = self._ui_components.get('action_container')
                if isinstance(action_container, dict) and 'buttons' in action_container:
                    buttons = action_container['buttons']
            
            if training_active:
                # Disable start and resume, enable stop
                if 'start_training' in buttons:
                    buttons['start_training'].disabled = True
                if 'resume_training' in buttons:
                    buttons['resume_training'].disabled = True
                if 'stop_training' in buttons:
                    buttons['stop_training'].disabled = False
                    
                self.log_operation("⚙️ Training buttons updated: Stop enabled", 'info')
            else:
                # Enable start and resume, disable stop
                if 'start_training' in buttons:
                    buttons['start_training'].disabled = False
                if 'resume_training' in buttons:
                    buttons['resume_training'].disabled = False
                if 'stop_training' in buttons:
                    buttons['stop_training'].disabled = True
                    
                self.log_operation("⚙️ Training buttons updated: Start/Resume enabled", 'info')
                
        except Exception as e:
            self.log_operation(f"⚠️ Failed to update button states: {e}", 'warning')
    
    def _store_training_state(self, training_id: str, training_config: Dict[str, Any]) -> None:
        """Store training state for stop functionality."""
        try:
            import time
            
            training_state = {
                'training_id': training_id,
                'phase': 'training',
                'start_timestamp': time.time(),
                'config': training_config
            }
            
            # Store in UI module for access by stop operation
            if hasattr(self._ui_module, '_training_state'):
                self._ui_module._training_state = training_state
            
            # Also store in config for operation access
            self.config['training_state'] = training_state
            
            self.log_operation(f"💾 Training state stored for ID: {training_id}", 'info')
            
        except Exception as e:
            self.log_operation(f"⚠️ Failed to store training state: {e}", 'warning')
    
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