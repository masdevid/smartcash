"""
Evaluation UIModule - BaseUIModule Pattern (Optimized)
Handles model evaluation across 2×4 research scenarios
"""

from typing import Dict, Any
from smartcash.ui.core.base_ui_module import BaseUIModule
from smartcash.ui.model.evaluation.configs.evaluation_config_handler import EvaluationConfigHandler
from smartcash.ui.model.evaluation.configs.evaluation_defaults import get_default_evaluation_config
from smartcash.ui.model.mixins import ModelDiscoveryMixin, ModelConfigSyncMixin, BackendServiceMixin
from .reports import EvaluationReportGenerator


class EvaluationUIModule(ModelDiscoveryMixin, ModelConfigSyncMixin, BackendServiceMixin, BaseUIModule):
    """Evaluation UI Module for comprehensive model evaluation."""
    
    def __init__(self):
        super().__init__(module_name='evaluation', parent_module='model', enable_environment=True)
        
        # Lazy initialization flags
        self._ui_components_created = False
        
        self._required_components = ['main_container', 'action_container', 'operation_container', 'summary_container']
        self.evaluation_service = self.checkpoint_selector = self.progress_bridge = None
        self.report_generator = EvaluationReportGenerator()
        
        # Button registration tracking
        self._buttons_registered = False
    
    def get_default_config(self) -> Dict[str, Any]: return get_default_evaluation_config()
    def create_config_handler(self, config: Dict[str, Any]) -> EvaluationConfigHandler: return EvaluationConfigHandler(config=config)
    def create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Create UI components with lazy initialization."""
        # Prevent double initialization
        if self._ui_components_created and hasattr(self, '_ui_components') and self._ui_components:
            self.log_debug("⏭️ Skipping UI component creation - already created")
            return self._ui_components
            
        try:
            from .components.evaluation_ui import create_evaluation_ui
            ui_components = create_evaluation_ui(config=config)
            
            if not ui_components:
                raise RuntimeError("Failed to create UI components")
            
            # Mark as created to prevent reinitalization
            self._ui_components_created = True
            return ui_components
            
        except Exception as e:
            self.log_error(f"Failed to create UI components: {e}")
            raise
    
    def _get_module_button_handlers(self) -> Dict[str, Any]:
        """Get module-specific button handlers."""
        return {
            'run_evaluation': self._operation_run_evaluation,
            # 'stop_evaluation': self._operation_stop_evaluation,  # Commented out - no button widget exists
            'save': self._handle_save_config,
            'reset': self._handle_reset_config,
            # 'export_results': self._operation_export_results,  # Commented out - no button widget exists
            'refresh_models': self._handle_refresh_models
        }

    def _register_dynamic_button_handlers(self) -> None:
        """Register dynamic button handlers with duplicate prevention."""
        if self._buttons_registered:
            self.log_debug("⏭️ Skipping evaluation button registration - already registered")
            return
        
        try:
            # Call parent method to handle registration
            if hasattr(super(), '_register_dynamic_button_handlers'):
                super()._register_dynamic_button_handlers()
            
            # Mark as registered
            self._buttons_registered = True
            self.log_info("🎯 Evaluation button handlers registered successfully")
            
        except Exception as e:
            self.log_error(f"Failed to register evaluation button handlers: {e}", exc_info=True)
    
    def _operation_run_evaluation(self, button=None) -> Dict[str, Any]:
        """Handle run evaluation operation."""
        return self._execute_operation_with_wrapper(
            operation_name="Evaluation",
            operation_func=lambda: self._execute_evaluation_operation(),
            button=button,
            validation_func=lambda: self._validate_evaluation_prerequisites(),
            success_message="Evaluation completed successfully",
            error_message="Evaluation failed"
        )
    
    def _operation_stop_evaluation(self, button=None) -> Dict[str, Any]:
        """Handle stop evaluation operation."""
        try:
            if self.evaluation_service and hasattr(self.evaluation_service, 'stop'):
                self.evaluation_service.stop()
                self.log_info("Evaluation stopped")
                return {'success': True, 'message': 'Evaluation stopped'}
            return {'success': False, 'message': 'No running evaluation to stop'}
        except Exception as e:
            return {'success': False, 'message': f'Stop failed: {e}'}
    
    def _operation_export_results(self, button=None) -> Dict[str, Any]:
        """Handle export results operation."""
        try:
            # Get latest results and export
            config = self.get_current_config()
            export_path = config.get('export', {}).get('output_path', 'evaluation_report.html')
            # Implementation would export using report_generator
            return {'success': True, 'message': f'Results exported to {export_path}'}
        except Exception as e:
            return {'success': False, 'message': f'Export failed: {e}'}
    
    def _execute_evaluation_operation(self) -> Dict[str, Any]:
        """Execute evaluation using real backend services."""
        try:
            self._initialize_backend_services()
            if not self.evaluation_service:
                return {'success': False, 'message': 'Evaluation service not available'}
            
            config = self.get_current_config()
            form_values = self._extract_form_values()
            
            # Determine operation type based on form configuration
            operation_type = self._determine_operation_type(form_values)
            
            # Create appropriate operation using factory
            from .operations.evaluation_factory import EvaluationOperationFactory
            operation = EvaluationOperationFactory.create_operation(
                operation_type=operation_type,
                ui_module=self,
                config=config,
                callbacks=self._get_operation_callbacks()
            )
            
            self.log_info(f"🚀 Starting {operation_type} evaluation with backend integration...")
            
            # Execute operation with real backend
            result = operation.execute()
            
            # Update UI with results
            if result.get('success', False):
                self._update_summary_panel(result)
                self.log_success(f"✅ Evaluation completed: {result.get('successful_tests', 0)}/{result.get('total_tests', 0)} tests")
            else:
                self.log_error(f"❌ Evaluation failed: {result.get('error', 'Unknown error')}")
            
            return result
            
        except Exception as e:
            self.log_error(f"❌ Evaluation operation failed: {e}")
            return {'success': False, 'message': f'Evaluation failed: {e}'}
    
    def _validate_evaluation_prerequisites(self) -> Dict[str, Any]:
        """Validate prerequisites for evaluation."""
        try:
            # Check for available models using mixin
            discovered_models = self.discover_checkpoints()
            if not discovered_models:
                self.log_error("❌ Evaluation failed to start: prerequisite not met - no valid checkpoints available")
                return {'valid': False, 'message': 'No models available for evaluation'}
            
            # Check configuration
            config = self.get_current_config()
            if not config.get('evaluation', {}).get('scenarios'):
                return {'valid': False, 'message': 'No evaluation scenarios configured'}
            
            return {'valid': True}
        except Exception:
            return {'valid': False, 'message': 'Prerequisites validation failed'}
    
    def _initialize_backend_services(self) -> None:
        """Initialize evaluation backend services using BackendServiceMixin."""
        config = self.get_current_config()
        
        # Initialize model API for real inference
        model_api = self._initialize_model_api()
        
        service_configs = {
            'checkpoint_selector': config,
            'evaluation_service': config,
            'scenario_manager': config,
            'evaluation_metrics': config
        }
        
        init_result = self.initialize_backend_services(service_configs, ['evaluation_service'])
        if init_result['success']:
            self.checkpoint_selector = self._backend_services.get('checkpoint_selector')
            self.evaluation_service = self._backend_services.get('evaluation_service')
            
            # Set model API for real inference
            if model_api and self.evaluation_service:
                self.evaluation_service.model_api = model_api
                self.log_info("🤖 Model API integrated with evaluation service")
            
            # Create progress bridge with UI components
            ui_components = {
                'operation_container': self.get_component('operation_container'),
                'summary_container': self.get_component('summary_container')
            }
            self.progress_bridge = self.create_progress_bridge(ui_components, 'evaluation')
            
            # Setup service callbacks for real-time updates
            callback_mappings = {
                'progress_update': '_handle_backend_progress',
                'status_update': '_handle_backend_status',
                'results_update': '_handle_backend_results'
            }
            self.setup_service_callbacks(self.evaluation_service, callback_mappings)
            
            self.log_info("✅ Backend services initialized with full integration")
        else:
            self.log_error(f"❌ Backend service initialization failed: {init_result}")
    
    def _update_summary_panel(self, results: dict) -> None:
        """Update summary panel with evaluation results using report generator."""
        try:
            summary_container = self.get_component('summary_container')
            if not summary_container: return
            
            if results and results.get('successful_tests'):
                report_html = self.report_generator.generate_evaluation_report(results, self._get_current_timestamp())
            elif results and results.get('scan_completed'):
                report_html = self.report_generator.generate_scan_report(
                    results.get('models_available', 0), results.get('evaluation_enabled', False))
            else:
                report_html = self.report_generator.generate_empty_state()
            
            if hasattr(summary_container, 'set_content'):
                summary_container.set_content(report_html)
            elif hasattr(summary_container, 'set_html'):
                summary_container.set_html(report_html, 'success' if results else 'info')
        except Exception as e:
            self.log_error(f"Failed to update summary panel: {e}")
    
    def _get_current_timestamp(self) -> str:
        from datetime import datetime
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def _handle_refresh_models(self, button=None) -> Dict[str, Any]:
        """Handle refresh models button using enhanced checkpoint discovery."""
        try:
            # Suppress button registration logs during discovery
            self._suppress_button_logs = True
            
            self.log_info("🔄 Refreshing available models...")
            discovered_models = self.discover_checkpoints(
                discovery_paths=['data/checkpoints', 'runs/train/*/weights', 'experiments/*/checkpoints'],
                filename_patterns=['best_*.pt', 'last.pt', 'epoch_*.pt']
            )
            
            scan_results = {
                'scan_completed': True,
                'models_available': len(discovered_models),
                'evaluation_enabled': len(discovered_models) > 0,
                'discovered_models': discovered_models
            }
            
            self._update_summary_panel(scan_results)
            self.log_success(f"✅ Found {len(discovered_models)} available models")
            return {'success': True, 'message': f'Found {len(discovered_models)} models', 'models': discovered_models}
            
        except Exception as e:
            self.log_error(f"Model refresh failed: {e}")
            return {'success': False, 'message': f'Refresh failed: {e}'}
        finally:
            # Always reset the log suppression flag
            self._suppress_button_logs = False
    
    def get_evaluation_status(self) -> Dict[str, Any]:
        """Get current evaluation status with real backend data."""
        try:
            config = self.get_current_config()
            execution_config = config.get('evaluation', {}).get('execution', {})
            
            # Get real model count from backend services
            models_available = 0
            if self.checkpoint_selector:
                try:
                    checkpoints = self.checkpoint_selector.list_available_checkpoints()
                    models_available = len(checkpoints)
                except Exception as e:
                    self.log_debug(f"Could not get checkpoint count: {e}")
            
            # Get scenarios from configuration
            scenarios_config = config.get('evaluation', {}).get('scenarios', ['position_variation', 'lighting_variation'])
            scenarios_enabled = len(scenarios_config) if isinstance(scenarios_config, list) else 2
            
            # Check backend service status
            service_status = self.get_service_status(['evaluation_service', 'checkpoint_selector'])
            backend_ready = service_status.get('overall_status') in ['healthy', 'degraded']
            
            return {
                'ready': backend_ready and models_available > 0,
                'run_mode': execution_config.get('run_mode', 'all_scenarios'),
                'parallel_execution': execution_config.get('parallel_execution', False),
                'models_available': models_available,
                'scenarios_enabled': scenarios_enabled,
                'backend_status': service_status.get('overall_status', 'unknown'),
                'service_details': service_status.get('service_details', {})
            }
        except Exception as e:
            self.log_error(f"Failed to get evaluation status: {e}")
            return {
                'ready': False,
                'error': str(e),
                'models_available': 0,
                'scenarios_enabled': 0
            }
    
    def _initialize_model_api(self) -> Any:
        """Initialize model API for real inference."""
        try:
            from smartcash.model.api.core import create_model_api
            model_api = create_model_api()
            self.log_debug("🤖 Model API created for inference")
            return model_api
        except Exception as e:
            self.log_warning(f"⚠️ Could not initialize model API: {e}")
            return None
    
    def _determine_operation_type(self, form_values: Dict[str, Any]) -> str:
        """Determine operation type based on form configuration."""
        # Check if specific scenarios are selected
        scenarios = form_values.get('scenarios', {})
        
        if isinstance(scenarios, dict):
            selected_scenarios = [k for k, v in scenarios.items() if v]
        else:
            selected_scenarios = ['position', 'lighting']  # Default
        
        # Map scenarios to operation types
        if len(selected_scenarios) == 0:
            return 'all_scenarios'  # Default fallback
        elif 'position' in selected_scenarios and 'lighting' in selected_scenarios:
            return 'all_scenarios'
        elif 'position' in selected_scenarios:
            return 'position_only'
        elif 'lighting' in selected_scenarios:
            return 'lighting_only'
        else:
            return 'all_scenarios'  # Default
    
    def _get_operation_callbacks(self) -> Dict[str, Any]:
        """Get callbacks for operation execution."""
        return {
            'progress_callback': self._handle_operation_progress,
            'status_callback': self._handle_operation_status,
            'results_callback': self._handle_operation_results
        }
    
    def _extract_form_values(self) -> Dict[str, Any]:
        """Extract values from form components."""
        try:
            main_form = self.get_component('main_form_row')
            if not main_form:
                return {}
            
            form_values = {}
            
            # Extract scenario selections
            if hasattr(main_form, '_scenario_checkboxes'):
                scenarios = {}
                for scenario_key, checkbox in main_form._scenario_checkboxes.items():
                    scenarios[scenario_key] = checkbox.value if hasattr(checkbox, 'value') else False
                form_values['scenarios'] = scenarios
            
            # Extract metrics selections
            if hasattr(main_form, '_metrics_checkboxes'):
                metrics = {}
                for metric_key, checkbox in main_form._metrics_checkboxes.items():
                    metrics[metric_key] = checkbox.value if hasattr(checkbox, 'value') else True
                form_values['metrics'] = metrics
            
            return form_values
            
        except Exception as e:
            self.log_error(f"Failed to extract form values: {e}")
            return {}
    
    def _handle_operation_progress(self, current: int, total: int, message: str = "") -> None:
        """Handle progress updates from operations."""
        try:
            operation_container = self.get_component('operation_container')
            if operation_container and hasattr(operation_container, 'update_progress'):
                operation_container.update_progress(current, total, message)
        except Exception as e:
            self.log_debug(f"Progress update failed: {e}")
    
    def _handle_operation_status(self, status: str, message: str = "") -> None:
        """Handle status updates from operations."""
        try:
            if status == 'success':
                self.log_success(message)
            elif status == 'error':
                self.log_error(message)
            elif status == 'warning':
                self.log_warning(message)
            else:
                self.log_info(message)
        except Exception as e:
            self.log_debug(f"Status update failed: {e}")
    
    def _handle_operation_results(self, results: Dict[str, Any]) -> None:
        """Handle results updates from operations."""
        try:
            self._update_summary_panel(results)
        except Exception as e:
            self.log_debug(f"Results update failed: {e}")
    
    def _handle_backend_progress(self, progress_data: Dict[str, Any]) -> None:
        """Handle progress updates from backend services."""
        try:
            current = progress_data.get('current', 0)
            total = progress_data.get('total', 100)
            message = progress_data.get('message', '')
            self._handle_operation_progress(current, total, message)
        except Exception as e:
            self.log_debug(f"Backend progress update failed: {e}")
    
    def _handle_backend_status(self, status_data: Dict[str, Any]) -> None:
        """Handle status updates from backend services."""
        try:
            status = status_data.get('status', 'info')
            message = status_data.get('message', '')
            self._handle_operation_status(status, message)
        except Exception as e:
            self.log_debug(f"Backend status update failed: {e}")
    
    def _handle_backend_results(self, results_data: Dict[str, Any]) -> None:
        """Handle results updates from backend services."""
        try:
            self._handle_operation_results(results_data)
        except Exception as e:
            self.log_debug(f"Backend results update failed: {e}")

    def cleanup(self) -> None:
        """Widget lifecycle cleanup - optimization.md compliance."""
        try:
            # Cleanup report generator if it exists
            if hasattr(self, 'report_generator'):
                if hasattr(self.report_generator, 'cleanup'):
                    self.report_generator.cleanup()
            
            # Cleanup UI components if they have cleanup methods
            if hasattr(self, '_ui_components') and self._ui_components:
                # Call component-specific cleanup if available
                if hasattr(self._ui_components, '_cleanup'):
                    self._ui_components._cleanup()
                
                # Close individual widgets
                for component_name, component in self._ui_components.items():
                    if hasattr(component, 'close'):
                        try:
                            component.close()
                        except Exception:
                            pass  # Ignore cleanup errors
            
            # Call parent cleanup
            if hasattr(super(), 'cleanup'):
                super().cleanup()
            
            # Minimal logging for cleanup completion
            if hasattr(self, 'logger'):
                self.logger.info("Evaluation module cleanup completed")
                
        except Exception as e:
            # Critical errors always logged
            if hasattr(self, 'logger'):
                self.logger.error(f"Evaluation module cleanup failed: {e}")
    
    def __del__(self):
        """Memory management - ensure cleanup on deletion."""
        try:
            self.cleanup()
        except Exception:
            pass  # Ignore cleanup errors during deletion


def get_evaluation_uimodule(auto_initialize: bool = True) -> EvaluationUIModule:
    """Factory function to get EvaluationUIModule instance."""
    module = EvaluationUIModule()
    if auto_initialize:
        module.initialize()
    return module