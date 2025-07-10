"""
File: smartcash/ui/model/train/handlers/training_ui_handler.py
Main UI handler for training module.
"""

import asyncio
from typing import Dict, Any, Optional, Callable

from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..constants import TrainingPhase, DEFAULT_CONFIG
from ..services.training_service import TrainingService
from ..operations.start_training_operation import StartTrainingOperation
from ..operations.stop_training_operation import StopTrainingOperation
from ..operations.resume_training_operation import ResumeTrainingOperation


class TrainingUIHandler(ModuleUIHandler):
    """Main UI handler for training module."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """Initialize the training UI handler."""
        super().__init__(module_name="train", parent_module="model")
        self.ui_components = ui_components
        
        # Initialize services and operations
        self.training_service = TrainingService()
        self.start_operation = StartTrainingOperation()
        self.stop_operation = StopTrainingOperation(self.training_service)
        self.resume_operation = ResumeTrainingOperation(self.training_service)
        
        # Current state
        self.current_config = DEFAULT_CONFIG.copy()
        self.training_active = False
        self.metrics_update_task = None
        
        # Setup event handlers
        self._setup_event_handlers()
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        try:
            # Get action container buttons
            action_container = self.ui_components.get('containers', {}).get('actions', {})
            
            # Start training button
            start_btn = action_container.get('start')
            if start_btn and hasattr(start_btn, 'on_click'):
                start_btn.on_click(self._on_start_training)
            
            # Stop training button
            stop_btn = action_container.get('stop')
            if stop_btn and hasattr(stop_btn, 'on_click'):
                stop_btn.on_click(self._on_stop_training)
            
            # Resume training button
            resume_btn = action_container.get('resume')
            if resume_btn and hasattr(resume_btn, 'on_click'):
                resume_btn.on_click(self._on_resume_training)
            
            # Configuration form changes
            self._setup_config_handlers()
            
            # Update button states based on initial state
            self._update_button_states()
            
        except Exception as e:
            self.logger.error(f"Error setting up event handlers: {str(e)}")
            raise
    
    def _setup_config_handlers(self) -> None:
        """Setup handlers for configuration form changes."""
        try:
            # Get form components
            form_components = self.ui_components.get('input_options', {})
            
            # Epochs input
            if 'epochs_input' in form_components:
                form_components['epochs_input'].observe(
                    self._on_config_change, names='value'
                )
            
            # Batch size input
            if 'batch_size_input' in form_components:
                form_components['batch_size_input'].observe(
                    self._on_config_change, names='value'
                )
            
            # Learning rate input
            if 'learning_rate_input' in form_components:
                form_components['learning_rate_input'].observe(
                    self._on_config_change, names='value'
                )
            
            # Optimizer selection
            if 'optimizer_dropdown' in form_components:
                form_components['optimizer_dropdown'].observe(
                    self._on_config_change, names='value'
                )
            
        except Exception as e:
            self.logger.error(f"Error setting up config handlers: {str(e)}")
    
    def _on_config_change(self, change) -> None:
        """Handle configuration form changes."""
        try:
            # Extract current config from UI
            config = self._extract_config_from_ui()
            
            # Update current config
            self.current_config.update(config)
            
            # Update config summary
            self._update_config_summary()
            
        except Exception as e:
            self.logger.error(f"Error handling config change: {str(e)}")
    
    def _extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract training configuration from UI components."""
        config = {"training": {}, "optimizer": {}, "scheduler": {}}
        
        try:
            form_components = self.ui_components.get('input_options', {})
            
            # Training parameters
            if 'epochs_input' in form_components:
                config["training"]["epochs"] = form_components['epochs_input'].value
            
            if 'batch_size_input' in form_components:
                config["training"]["batch_size"] = form_components['batch_size_input'].value
            
            if 'learning_rate_input' in form_components:
                config["training"]["learning_rate"] = form_components['learning_rate_input'].value
            
            if 'validation_interval_input' in form_components:
                config["training"]["validation_interval"] = form_components['validation_interval_input'].value
            
            # Optimizer parameters
            if 'optimizer_dropdown' in form_components:
                config["optimizer"]["type"] = form_components['optimizer_dropdown'].value
            
            if 'weight_decay_input' in form_components:
                config["optimizer"]["weight_decay"] = form_components['weight_decay_input'].value
            
            # Scheduler parameters  
            if 'scheduler_dropdown' in form_components:
                config["scheduler"]["type"] = form_components['scheduler_dropdown'].value
            
            if 'warmup_epochs_input' in form_components:
                config["scheduler"]["warmup_epochs"] = form_components['warmup_epochs_input'].value
            
            # Early stopping
            early_stopping_config = {}
            if 'early_stopping_enabled' in form_components:
                early_stopping_config["enabled"] = form_components['early_stopping_enabled'].value
            
            if 'early_stopping_patience' in form_components:
                early_stopping_config["patience"] = form_components['early_stopping_patience'].value
            
            if early_stopping_config:
                config["training"]["early_stopping"] = early_stopping_config
            
            # Mixed precision
            if 'mixed_precision_enabled' in form_components:
                config["mixed_precision"] = {
                    "enabled": form_components['mixed_precision_enabled'].value
                }
            
        except Exception as e:
            self.logger.error(f"Error extracting config from UI: {str(e)}")
        
        return config
    
    def _update_config_summary(self) -> None:
        """Update the training configuration summary display."""
        try:
            if 'config_summary' in self.ui_components:
                summary_html = self._generate_config_summary_html()
                self.ui_components['config_summary'].value = summary_html
        except Exception as e:
            self.logger.error(f"Error updating config summary: {str(e)}")
    
    def _generate_config_summary_html(self) -> str:
        """Generate HTML for configuration summary."""
        training_config = self.current_config.get("training", {})
        optimizer_config = self.current_config.get("optimizer", {})
        scheduler_config = self.current_config.get("scheduler", {})
        
        return f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 12px;
            padding: 20px;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        ">
            <h4 style="margin: 0 0 15px 0; font-size: 1.2rem; font-weight: 600;">
                🚀 Training Configuration
            </h4>
            
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px;">
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Training</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">
                        {training_config.get('epochs', 100)} epochs
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">
                        Batch: {training_config.get('batch_size', 16)} | 
                        LR: {training_config.get('learning_rate', 0.001)}
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Optimizer</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">
                        {optimizer_config.get('type', 'adam').upper()}
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">
                        Weight Decay: {optimizer_config.get('weight_decay', 0.0005)}
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Scheduler</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">
                        {scheduler_config.get('type', 'cosine').title()}
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">
                        Warmup: {scheduler_config.get('warmup_epochs', 5)} epochs
                    </div>
                </div>
                
                <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                    <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Features</div>
                    <div style="font-size: 1.1rem; font-weight: 600;">
                        {'✅' if training_config.get('early_stopping', {}).get('enabled', True) else '❌'} Early Stop
                    </div>
                    <div style="font-size: 0.8rem; opacity: 0.7;">
                        {'✅' if self.current_config.get('mixed_precision', {}).get('enabled', True) else '❌'} Mixed Precision
                    </div>
                </div>
            </div>
        </div>
        """
    
    def _on_start_training(self, button) -> None:
        """Handle start training button click."""
        if self.training_active:
            self._log_message("⚠️ Training is already in progress")
            return
        
        # Start training operation asynchronously
        asyncio.create_task(self._execute_start_training())
    
    def _on_stop_training(self, button) -> None:
        """Handle stop training button click."""
        if not self.training_active:
            self._log_message("⚠️ No training in progress to stop")
            return
        
        # Stop training operation asynchronously
        asyncio.create_task(self._execute_stop_training())
    
    def _on_resume_training(self, button) -> None:
        """Handle resume training button click."""
        if self.training_active:
            self._log_message("⚠️ Training is already in progress")
            return
        
        # Get checkpoint path from UI
        checkpoint_path = self._get_checkpoint_path_from_ui()
        if not checkpoint_path:
            self._log_message("❌ Please select a checkpoint file to resume from")
            return
        
        # Resume training operation asynchronously
        asyncio.create_task(self._execute_resume_training(checkpoint_path))
    
    async def _execute_start_training(self) -> None:
        """Execute start training operation."""
        try:
            self.training_active = True
            self._update_button_states()
            
            # Extract current config
            config = self._extract_config_from_ui()
            
            # Execute operation
            result = await self.start_operation.execute_operation(
                config=config,
                progress_callback=self._get_progress_callback(),
                log_callback=self._get_log_callback(),
                metrics_callback=self._get_metrics_callback()
            )
            
            if result.get("success", False):
                self._log_message("✅ Training started successfully")
                self._start_metrics_updates()
            else:
                self._log_message(f"❌ Training start failed: {result.get('message', 'Unknown error')}")
                self.training_active = False
                self._update_button_states()
                
        except Exception as e:
            self._log_message(f"❌ Error starting training: {str(e)}")
            self.training_active = False
            self._update_button_states()
    
    async def _execute_stop_training(self) -> None:
        """Execute stop training operation."""
        try:
            result = await self.stop_operation.execute_operation(
                progress_callback=self._get_progress_callback(),
                log_callback=self._get_log_callback()
            )
            
            self.training_active = False
            self._update_button_states()
            self._stop_metrics_updates()
            
            if result.get("success", False):
                self._log_message("✅ Training stopped successfully")
            else:
                self._log_message(f"⚠️ Training stop: {result.get('message', 'Unknown error')}")
                
        except Exception as e:
            self._log_message(f"❌ Error stopping training: {str(e)}")
    
    async def _execute_resume_training(self, checkpoint_path: str) -> None:
        """Execute resume training operation."""
        try:
            self.training_active = True
            self._update_button_states()
            
            result = await self.resume_operation.execute_operation(
                checkpoint_path=checkpoint_path,
                additional_epochs=50,  # Default additional epochs
                progress_callback=self._get_progress_callback(),
                log_callback=self._get_log_callback()
            )
            
            if result.get("success", False):
                self._log_message("✅ Training resumed successfully")
                self._start_metrics_updates()
            else:
                self._log_message(f"❌ Training resume failed: {result.get('message', 'Unknown error')}")
                self.training_active = False
                self._update_button_states()
                
        except Exception as e:
            self._log_message(f"❌ Error resuming training: {str(e)}")
            self.training_active = False
            self._update_button_states()
    
    def _update_button_states(self) -> None:
        """Update button states based on training status."""
        try:
            if 'start_button' in self.ui_components:
                self.ui_components['start_button'].disabled = self.training_active
            
            if 'stop_button' in self.ui_components:
                self.ui_components['stop_button'].disabled = not self.training_active
            
            if 'resume_button' in self.ui_components:
                self.ui_components['resume_button'].disabled = self.training_active
                
        except Exception as e:
            self.logger.error(f"Error updating button states: {str(e)}")
    
    def _start_metrics_updates(self) -> None:
        """Start periodic metrics updates for charts."""
        if self.metrics_update_task:
            self.metrics_update_task.cancel()
        
        self.metrics_update_task = asyncio.create_task(self._metrics_update_loop())
    
    def _stop_metrics_updates(self) -> None:
        """Stop periodic metrics updates."""
        if self.metrics_update_task:
            self.metrics_update_task.cancel()
            self.metrics_update_task = None
    
    async def _metrics_update_loop(self) -> None:
        """Periodic loop to update metrics charts."""
        try:
            while self.training_active:
                # Get current metrics from service
                metrics_data = self.training_service.get_metrics_for_charts()
                
                # Update charts
                self._update_metrics_charts(metrics_data)
                
                # Wait before next update
                await asyncio.sleep(2.0)
                
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in metrics update loop: {str(e)}")
    
    def _update_metrics_charts(self, metrics_data: Dict[str, Any]) -> None:
        """Update the metrics charts with new data."""
        try:
            if 'chart_container' in self.ui_components:
                chart_container = self.ui_components['chart_container']
                
                # Update loss chart (left)
                loss_data = metrics_data.get("loss_chart", {})
                if loss_data.get("data"):
                    # Use train_loss as primary metric for loss chart
                    train_loss_data = loss_data["data"].get("train_loss", [])
                    if train_loss_data:
                        chart_container.update_chart(
                            "chart_1",
                            train_loss_data,
                            {
                                "title": "Training Loss",
                                "color": "#ff6b6b",
                                "type": "line"
                            }
                        )
                
                # Update performance chart (right)
                perf_data = metrics_data.get("performance_chart", {})
                if perf_data.get("data"):
                    # Use val_map50 as primary metric for performance chart
                    map_data = perf_data["data"].get("val_map50", [])
                    if map_data:
                        chart_container.update_chart(
                            "chart_2",
                            map_data,
                            {
                                "title": "Validation mAP@0.5",
                                "color": "#4ecdc4",
                                "type": "line"
                            }
                        )
                        
        except Exception as e:
            self.logger.error(f"Error updating metrics charts: {str(e)}")
    
    def _get_checkpoint_path_from_ui(self) -> Optional[str]:
        """Get checkpoint path from UI input."""
        try:
            form_components = self.ui_components.get('input_options', {})
            if 'checkpoint_path_input' in form_components:
                return form_components['checkpoint_path_input'].value
        except Exception:
            pass
        return None
    
    def _get_progress_callback(self) -> Optional[Callable]:
        """Get progress callback for operations."""
        if 'progress_tracker' in self.ui_components:
            tracker = self.ui_components['progress_tracker']
            return lambda percent, message: tracker.update(percent, message)
        return None
    
    def _get_log_callback(self) -> Optional[Callable]:
        """Get log callback for operations."""
        if 'log_output' in self.ui_components:
            log_output = self.ui_components['log_output']
            return lambda message: self._log_message(message)
        return None
    
    def _get_metrics_callback(self) -> Optional[Callable]:
        """Get metrics callback for operations."""
        return lambda metrics: self._handle_metrics_update(metrics)
    
    def _handle_metrics_update(self, metrics: Dict[str, Any]) -> None:
        """Handle metrics updates from training operations."""
        try:
            # Update current metrics in service
            if hasattr(self.training_service, '_handle_metrics_callback'):
                self.training_service._handle_metrics_callback(metrics)
        except Exception as e:
            self.logger.error(f"Error handling metrics update: {str(e)}")
    
    def _log_message(self, message: str) -> None:
        """Log message to UI log component."""
        try:
            if 'log_output' in self.ui_components:
                log_component = self.ui_components['log_output']
                if hasattr(log_component, 'add_log'):
                    log_component.add_log(message)
                elif hasattr(log_component, 'append_stdout'):
                    log_component.append_stdout(message + '\n')
        except Exception as e:
            self.logger.error(f"Error logging message: {str(e)}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status."""
        return {
            "training_active": self.training_active,
            "current_config": self.current_config,
            "service_status": self.training_service.get_current_status()
        }
    
    def initialize(self) -> None:
        """Initialize the training UI handler.
        
        Implements the abstract method required by ModuleUIHandler.
        This method is called during handler setup to perform any
        necessary initialization of the training UI components.
        """
        self.logger.info("🚀 Initializing Training UI Handler")
        
        # Initialize training service if not already done
        if not hasattr(self, 'training_service') or self.training_service is None:
            self.training_service = TrainingService()
        
        # Set initial training state
        self.training_active = False
        
        # Load default configuration
        self.current_config = DEFAULT_CONFIG.copy()
        
        self.logger.info("✅ Training UI Handler initialized successfully")