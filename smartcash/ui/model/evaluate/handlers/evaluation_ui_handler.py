"""
File: smartcash/ui/model/evaluate/handlers/evaluation_ui_handler.py
Description: UI handler for evaluation management following dependency pattern
"""

import asyncio
from typing import Dict, Any, Optional, Callable
from smartcash.ui.core.handlers.ui_handler import ModuleUIHandler
from ..constants import (
    DEFAULT_CONFIG, EvaluationOperation, EvaluationPhase,
    AVAILABLE_SCENARIOS, AVAILABLE_MODELS, UI_CONFIG,
    AVAILABLE_METRICS, DEFAULT_ENABLED_METRICS, METRIC_CONFIGS
)
from ..services.evaluation_service import EvaluationService
from ..operations import (
    ScenarioEvaluationOperation,
    ComprehensiveEvaluationOperation,
    CheckpointOperation
)


class EvaluationUIHandler(ModuleUIHandler):
    """Handler for evaluation UI management following dependency pattern."""
    
    def __init__(self, module_name: str = 'evaluate', parent_module: str = 'model', **kwargs):
        """Initialize evaluation UI handler.
        
        Args:
            module_name: Module name
            parent_module: Parent module name
            **kwargs: Additional arguments
        """
        super().__init__(module_name, parent_module)
        
        # Initialize services
        self.evaluation_service = EvaluationService()
        
        # Initialize operations
        self.scenario_operation = ScenarioEvaluationOperation(self.evaluation_service)
        self.comprehensive_operation = ComprehensiveEvaluationOperation(self.evaluation_service)
        self.checkpoint_operation = CheckpointOperation(self.evaluation_service)
        
        # State management
        self.current_config = DEFAULT_CONFIG.copy()
        self.evaluation_active = False
        self.current_results = {}
        self.selected_scenarios = set(AVAILABLE_SCENARIOS)
        self.selected_models = set(AVAILABLE_MODELS)
        self.selected_metrics = set(DEFAULT_ENABLED_METRICS)
        
        # UI component references
        self._ui_components = {}
    
    def extract_config_from_ui(self) -> Dict[str, Any]:
        """Extract configuration from UI components."""
        try:
            config = self.current_config.copy()
            
            # Extract scenario selections
            if 'position_variation_checkbox' in self._ui_components:
                pos_enabled = self._ui_components['position_variation_checkbox'].value
                config['evaluation']['scenarios']['position_variation']['enabled'] = pos_enabled
                if pos_enabled:
                    self.selected_scenarios.add('position_variation')
                else:
                    self.selected_scenarios.discard('position_variation')
            
            if 'lighting_variation_checkbox' in self._ui_components:
                light_enabled = self._ui_components['lighting_variation_checkbox'].value
                config['evaluation']['scenarios']['lighting_variation']['enabled'] = light_enabled
                if light_enabled:
                    self.selected_scenarios.add('lighting_variation')
                else:
                    self.selected_scenarios.discard('lighting_variation')
            
            # Extract model selections
            if 'cspdarknet_checkbox' in self._ui_components:
                csp_enabled = self._ui_components['cspdarknet_checkbox'].value
                if csp_enabled:
                    self.selected_models.add('cspdarknet')
                else:
                    self.selected_models.discard('cspdarknet')
            
            if 'efficientnet_checkbox' in self._ui_components:
                eff_enabled = self._ui_components['efficientnet_checkbox'].value
                if eff_enabled:
                    self.selected_models.add('efficientnet_b4')
                else:
                    self.selected_models.discard('efficientnet_b4')
            
            # Extract evaluation settings
            if 'confidence_threshold_slider' in self._ui_components:
                confidence = self._ui_components['confidence_threshold_slider'].value
                config['inference']['confidence_threshold'] = confidence
            
            if 'iou_threshold_slider' in self._ui_components:
                iou = self._ui_components['iou_threshold_slider'].value
                config['inference']['iou_threshold'] = iou
            
            # Extract augmentation settings
            if 'num_variations_slider' in self._ui_components:
                num_vars = int(self._ui_components['num_variations_slider'].value)
                for scenario_config in config['evaluation']['scenarios'].values():
                    if 'augmentation_config' in scenario_config:
                        scenario_config['augmentation_config']['num_variations'] = num_vars
            
            # Extract metric selections
            for metric in AVAILABLE_METRICS:
                checkbox_name = f'{metric}_metric_checkbox'
                if checkbox_name in self._ui_components:
                    enabled = self._ui_components[checkbox_name].value
                    if enabled:
                        self.selected_metrics.add(metric)
                    else:
                        self.selected_metrics.discard(metric)
            
            # Update config with selected metrics
            config['evaluation']['selected_metrics'] = list(self.selected_metrics)
            
            return config
            
        except Exception as e:
            self.logger.error(f"❌ Error extracting config from UI: {e}")
            return self.current_config
    
    def update_ui_from_config(self, config: Dict[str, Any]) -> None:
        """Update UI components from configuration."""
        try:
            eval_config = config.get('evaluation', {})
            inference_config = config.get('inference', {})
            
            # Update scenario checkboxes
            scenarios = eval_config.get('scenarios', {})
            if 'position_variation_checkbox' in self._ui_components:
                enabled = scenarios.get('position_variation', {}).get('enabled', True)
                self._ui_components['position_variation_checkbox'].value = enabled
            
            if 'lighting_variation_checkbox' in self._ui_components:
                enabled = scenarios.get('lighting_variation', {}).get('enabled', True)
                self._ui_components['lighting_variation_checkbox'].value = enabled
            
            # Update inference settings
            if 'confidence_threshold_slider' in self._ui_components:
                confidence = inference_config.get('confidence_threshold', 0.25)
                self._ui_components['confidence_threshold_slider'].value = confidence
            
            if 'iou_threshold_slider' in self._ui_components:
                iou = inference_config.get('iou_threshold', 0.45)
                self._ui_components['iou_threshold_slider'].value = iou
            
            # Update augmentation settings
            pos_config = scenarios.get('position_variation', {}).get('augmentation_config', {})
            if 'num_variations_slider' in self._ui_components:
                num_vars = pos_config.get('num_variations', 5)
                self._ui_components['num_variations_slider'].value = float(num_vars)
            
            # Update metric checkboxes
            selected_metrics = set(config.get('evaluation', {}).get('selected_metrics', DEFAULT_ENABLED_METRICS))
            for metric in AVAILABLE_METRICS:
                checkbox_name = f'{metric}_metric_checkbox'
                if checkbox_name in self._ui_components:
                    self._ui_components[checkbox_name].value = metric in selected_metrics
            
            # Update evaluation summary
            if 'evaluation_summary' in self._ui_components:
                self._update_evaluation_summary(config)
            
            self.logger.info("✅ UI successfully updated from config")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating UI from config: {e}")
    
    def _update_evaluation_summary(self, config: Dict[str, Any]) -> None:
        """Update evaluation summary widget."""
        try:
            scenarios = list(self.selected_scenarios)
            models = list(self.selected_models)
            metrics = list(self.selected_metrics)
            total_tests = len(scenarios) * len(models)
            
            summary_html = f"""
            <div style="background: linear-gradient(135deg, #28a745, #20c997); padding: 15px; border-radius: 10px; color: white; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
                <div style="display: flex; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 24px; margin-right: 10px;">🎯</div>
                    <div>
                        <div style="font-size: 1.2rem; font-weight: 600; margin-bottom: 2px;">{UI_CONFIG['title']}</div>
                        <div style="font-size: 0.9rem; opacity: 0.9;">{UI_CONFIG['subtitle']}</div>
                    </div>
                </div>
                
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin-top: 15px;">
                    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Scenarios</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{len(scenarios)}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">{', '.join(scenarios) if scenarios else 'None selected'}</div>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Models</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{len(models)}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">{', '.join(models) if models else 'None selected'}</div>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Total Tests</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{total_tests}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">Est. {total_tests * 3}-{total_tests * 5} min</div>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Metrics</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">{len(metrics)}</div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">{', '.join(m.upper() for m in metrics[:3])}{' +' + str(len(metrics)-3) if len(metrics) > 3 else ''}</div>
                    </div>
                    
                    <div style="background: rgba(255,255,255,0.1); padding: 12px; border-radius: 8px;">
                        <div style="font-size: 0.9rem; opacity: 0.8; margin-bottom: 5px;">Status</div>
                        <div style="font-size: 1.1rem; font-weight: 600;">
                            {'🔄 Active' if self.evaluation_active else '⏸️ Ready'}
                        </div>
                        <div style="font-size: 0.8rem; opacity: 0.7;">
                            {'In Progress' if self.evaluation_active else 'Waiting to start'}
                        </div>
                    </div>
                </div>
            </div>
            """
            
            if 'evaluation_summary' in self._ui_components:
                self._ui_components['evaluation_summary'].value = summary_html
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error updating evaluation summary: {e}")
    
    def setup(self, ui_components: Dict[str, Any]) -> None:
        """Set up the handler with UI components.
        
        Args:
            ui_components: Dictionary of UI components to be managed by this handler
        """
        self.logger.info("🖥️ Setting up UI components for Evaluation UI Handler")
        self._ui_components = ui_components
        
        # Setup event handlers
        self._setup_event_handlers()
        
        # Initialize with default config
        self.sync_ui_with_config()
        
        self.logger.info("✅ UI components setup complete for Evaluation UI Handler")
    
    def _setup_event_handlers(self) -> None:
        """Setup event handlers for UI components."""
        # Scenario selection handlers
        scenario_widgets = ['position_variation_checkbox', 'lighting_variation_checkbox']
        for widget_name in scenario_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_scenario_change(w, change),
                    names='value'
                )
        
        # Model selection handlers
        model_widgets = ['cspdarknet_checkbox', 'efficientnet_checkbox']
        for widget_name in model_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_model_change(w, change),
                    names='value'
                )
        
        # Settings change handlers
        settings_widgets = [
            'confidence_threshold_slider', 'iou_threshold_slider', 'num_variations_slider'
        ]
        
        # Metric selection handlers
        metric_widgets = [f'{metric}_metric_checkbox' for metric in AVAILABLE_METRICS]
        for widget_name in settings_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_settings_change(w, change),
                    names='value'
                )
        
        for widget_name in metric_widgets:
            if widget_name in self._ui_components:
                widget = self._ui_components[widget_name]
                widget.observe(
                    lambda change, w=widget_name: self._on_metrics_change(w, change),
                    names='value'
                )
        
        # Button handlers
        button_handlers = {
            'run_scenario_btn': self._handle_run_scenario,
            'run_comprehensive_btn': self._handle_run_comprehensive,
            'load_checkpoint_btn': self._handle_load_checkpoint,
            'list_checkpoints_btn': self._handle_list_checkpoints,
            'stop_evaluation_btn': self._handle_stop_evaluation,
            'save_config_btn': self._handle_save_config,
            'reset_config_btn': self._handle_reset_config
        }
        
        for button_name, handler in button_handlers.items():
            if button_name in self._ui_components:
                button = self._ui_components[button_name]
                button.on_click(lambda b, h=handler: h())
        
        self.logger.info("✅ Event handlers setup complete")
    
    def _on_scenario_change(self, widget_name: str, change) -> None:
        """Handle scenario selection changes."""
        try:
            current_config = self.extract_config_from_ui()
            self.current_config.update(current_config)
            
            # Update evaluation summary
            if 'evaluation_summary' in self._ui_components:
                self._update_evaluation_summary(self.current_config)
            
            self.logger.debug(f"📋 Scenario selection updated: {widget_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Scenario change handling error: {e}")
    
    def _on_model_change(self, widget_name: str, change) -> None:
        """Handle model selection changes."""
        try:
            current_config = self.extract_config_from_ui()
            self.current_config.update(current_config)
            
            # Update evaluation summary
            if 'evaluation_summary' in self._ui_components:
                self._update_evaluation_summary(self.current_config)
            
            self.logger.debug(f"🤖 Model selection updated: {widget_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Model change handling error: {e}")
    
    def _on_settings_change(self, widget_name: str, change) -> None:
        """Handle settings changes."""
        try:
            current_config = self.extract_config_from_ui()
            self.current_config.update(current_config)
            
            self.logger.debug(f"⚙️ Settings updated: {widget_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Settings change handling error: {e}")
    
    def _on_metrics_change(self, widget_name: str, change) -> None:
        """Handle metric selection changes."""
        try:
            current_config = self.extract_config_from_ui()
            self.current_config.update(current_config)
            
            # Update evaluation summary
            if 'evaluation_summary' in self._ui_components:
                self._update_evaluation_summary(self.current_config)
            
            self.logger.debug(f"📊 Metrics selection updated: {widget_name}")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Metrics change handling error: {e}")
    
    def _handle_run_scenario(self) -> None:
        """Handle run single scenario evaluation."""
        if self.evaluation_active:
            self.track_status("⚠️ Evaluation is already in progress", "warning")
            return
        
        # Get first selected scenario and model
        scenarios = list(self.selected_scenarios)
        models = list(self.selected_models)
        
        if not scenarios or not models:
            self.track_status("❌ Please select at least one scenario and one model", "error")
            return
        
        scenario = scenarios[0]
        model = models[0]
        
        config = {
            "scenario": scenario,
            "model": model,
            "selected_metrics": list(self.selected_metrics)
        }
        
        self._run_async_operation(
            operation=self.scenario_operation,
            config=config,
            operation_name=f"Scenario Evaluation ({scenario}, {model})"
        )
    
    def _handle_run_comprehensive(self) -> None:
        """Handle run comprehensive evaluation."""
        if self.evaluation_active:
            self.track_status("⚠️ Evaluation is already in progress", "warning")
            return
        
        if not self.selected_scenarios or not self.selected_models:
            self.track_status("❌ Please select at least one scenario and one model", "error")
            return
        
        config = {
            "scenarios": list(self.selected_scenarios),
            "models": list(self.selected_models),
            "selected_metrics": list(self.selected_metrics)
        }
        
        self._run_async_operation(
            operation=self.comprehensive_operation,
            config=config,
            operation_name="Comprehensive Evaluation"
        )
    
    def _handle_load_checkpoint(self) -> None:
        """Handle load checkpoint operation."""
        models = list(self.selected_models)
        if not models:
            self.track_status("❌ Please select at least one model for checkpoint loading", "error")
            return
        
        config = {
            "action": "load",
            "model": models[0]  # Use first selected model
        }
        
        self._run_async_operation(
            operation=self.checkpoint_operation,
            config=config,
            operation_name="Load Checkpoint"
        )
    
    def _handle_list_checkpoints(self) -> None:
        """Handle list checkpoints operation."""
        config = {
            "action": "list"
        }
        
        self._run_async_operation(
            operation=self.checkpoint_operation,
            config=config,
            operation_name="List Checkpoints"
        )
    
    def _handle_stop_evaluation(self) -> None:
        """Handle stop evaluation operation."""
        if not self.evaluation_active:
            self.track_status("⚠️ No evaluation is currently running", "warning")
            return
        
        self.evaluation_active = False
        self.track_status("🛑 Evaluation stopped by user", "info")
    
    def _run_async_operation(self, operation, config: Dict[str, Any], operation_name: str) -> None:
        """Run an async operation in a thread to avoid blocking the UI."""
        try:
            self.evaluation_active = True
            self.track_status(f"🚀 Starting {operation_name}...", "info")
            
            # Create a new event loop for the thread
            import threading
            
            def run_operation():
                # Create new event loop for this thread
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                try:
                    # Run the async operation
                    result = loop.run_until_complete(
                        operation.execute(
                            config=config,
                            progress_callback=self._create_progress_callback(),
                            log_callback=self._create_log_callback()
                        )
                    )
                    
                    # Handle result
                    self._handle_operation_result(operation_name, result)
                    
                except Exception as e:
                    self.logger.error(f"Operation {operation_name} failed: {e}")
                    self.track_status(f"❌ {operation_name} failed: {str(e)}", "error")
                finally:
                    self.evaluation_active = False
                    loop.close()
            
            # Start operation in a separate thread
            thread = threading.Thread(target=run_operation, daemon=True)
            thread.start()
            
        except Exception as e:
            self.evaluation_active = False
            self.logger.error(f"Failed to start operation {operation_name}: {e}")
            self.track_status(f"❌ Failed to start {operation_name}: {str(e)}", "error")
    
    def _create_progress_callback(self):
        """Create progress callback for operations."""
        def callback(progress: int, message: str):
            # Update progress if operation container is available
            if 'progress_tracker' in self._ui_components:
                progress_tracker = self._ui_components['progress_tracker']
                if hasattr(progress_tracker, 'update'):
                    progress_tracker.update(progress, message)
        return callback
    
    def _create_log_callback(self):
        """Create log callback for operations."""
        def callback(message: str, level: str):
            self.track_status(message, level.lower())
        return callback
    
    def _handle_operation_result(self, operation_name: str, result: Dict[str, Any]) -> None:
        """Handle the result of an operation."""
        try:
            if result.get('success', False):
                self.track_status(f"✅ {operation_name} completed successfully", "success")
                
                # Store results if available
                if 'result' in result:
                    operation_result = result['result']
                    if 'results' in operation_result:
                        self.current_results.update(operation_result['results'])
                    elif 'checkpoints' in operation_result:
                        # Handle checkpoint listing results
                        checkpoints = operation_result['checkpoints']
                        self.track_status(f"📁 Found {len(checkpoints)} checkpoints", "info")
                
                # Update UI with results
                if 'evaluation_summary' in self._ui_components:
                    self._update_evaluation_summary(self.current_config)
                    
            else:
                error = result.get('error', 'Unknown error')
                self.track_status(f"❌ {operation_name} failed: {error}", "error")
                    
        except Exception as e:
            self.logger.error(f"Error handling operation result: {e}")
            self.track_status(f"❌ Error processing {operation_name} result", "error")
    
    def _handle_save_config(self) -> None:
        """Handle configuration save."""
        try:
            config = self.extract_config_from_ui()
            self.current_config.update(config)
            self.track_status("💾 Configuration saved", "success")
        except Exception as e:
            self.track_status(f"❌ Save failed: {str(e)}", "error")
    
    def _handle_reset_config(self) -> None:
        """Handle configuration reset."""
        try:
            self.current_config = DEFAULT_CONFIG.copy()
            self.selected_scenarios = set(AVAILABLE_SCENARIOS)
            self.selected_models = set(AVAILABLE_MODELS)
            self.selected_metrics = set(DEFAULT_ENABLED_METRICS)
            self.update_ui_from_config(self.current_config)
            self.track_status("🔄 Configuration reset to defaults", "info")
        except Exception as e:
            self.track_status(f"❌ Reset failed: {str(e)}", "error")
    
    def sync_config_with_ui(self) -> None:
        """Sync configuration with UI state."""
        try:
            current_config = self.extract_config_from_ui()
            self.current_config.update(current_config)
            self.logger.info("✅ Config successfully synced with UI")
        except Exception as e:
            self.logger.error(f"❌ Error syncing config with UI: {e}")
    
    def sync_ui_with_config(self) -> None:
        """Sync UI with configuration."""
        try:
            self.update_ui_from_config(self.current_config)
            self.logger.info("✅ UI successfully synced with config")
        except Exception as e:
            self.logger.error(f"❌ Error syncing UI with config: {e}")
    
    def initialize(self) -> None:
        """Initialize the evaluation UI handler.
        
        Implements the abstract method required by ModuleUIHandler.
        """
        self.logger.info("🚀 Initializing Evaluation UI Handler")
        
        # Initialize evaluation service if not already done
        if not hasattr(self, 'evaluation_service') or self.evaluation_service is None:
            self.evaluation_service = EvaluationService()
        
        # Set initial evaluation state
        self.evaluation_active = False
        
        # Load default configuration
        self.current_config = DEFAULT_CONFIG.copy()
        
        self.logger.info("✅ Evaluation UI Handler initialized successfully")
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get current evaluation status."""
        return {
            "evaluation_active": self.evaluation_active,
            "selected_scenarios": list(self.selected_scenarios),
            "selected_models": list(self.selected_models),
            "selected_metrics": list(self.selected_metrics),
            "current_config": self.current_config,
            "num_results": len(self.current_results),
            "service_status": self.evaluation_service.get_current_status()
        }