"""
Evaluation UIModule - New Core Pattern
Handles model evaluation across 2×4 research scenarios (2 scenarios × 4 models = 8 tests)
"""

from typing import Dict, Any, Optional
from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.decorators import suppress_ui_init_logs
from smartcash.ui.model.evaluation.operations.evaluation_operation_manager import EvaluationOperationManager
from smartcash.ui.model.evaluation.configs.evaluation_config_handler import EvaluationConfigHandler
from smartcash.ui.model.evaluation.configs.evaluation_defaults import get_default_evaluation_config
from smartcash.ui.model.evaluation.constants import UI_CONFIG
from typing import Optional, Dict, Any
from IPython.display import display

def initialize_evaluation_ui(config: Optional[Dict[str, Any]] = None, display: bool = True):
    """
    Initialize and optionally display the evaluation UI.
    
    Args:
        config: Optional configuration dictionary to override defaults
        display: If True, display the UI immediately (default: True)
        
    Returns:
        If display=True: Returns the displayed widget
        If display=False: Returns the EvaluationUIModule instance
        
    Raises:
        Exception: If initialization fails
    """
    # Get logger first to ensure we can log any issues
    try:
        logger = get_module_logger("smartcash.ui.model.evaluation")
        logger.info("🎯 Initializing evaluation UI...")
    except Exception as e:
        print(f"[WARNING] Failed to initialize logger: {e}")
        logger = None
    
    try:
        from IPython.display import display as ipython_display, HTML, clear_output
        from ipywidgets import Output
        
        # Create an output widget to capture the UI
        output = Output()
        
        # Initialize the evaluation module with error handling for read-only file systems
        evaluation_module = None
        
        try:
            # First try to initialize with the provided config
            evaluation_module = EvaluationUIModule()
            
            # Initialize with the provided or loaded config
            if logger:
                logger.info("⚙️ Initializing evaluation module with config...")
            
            # Try to initialize with the provided config
            try:
                evaluation_module.initialize(config=config)
            except OSError as e:
                if 'Read-only file system' in str(e):
                    if logger:
                        logger.warning("Running in read-only environment, using in-memory config only")
                    # If we can't write to disk, use a minimal in-memory config
                    from smartcash.ui.model.evaluation.configs.evaluation_defaults import get_default_evaluation_config
                    in_memory_config = get_default_evaluation_config()
                    if config:
                        # Update with any provided config values
                        if 'evaluation' in in_memory_config and 'evaluation' in config:
                            in_memory_config['evaluation'].update(config['evaluation'])
                        else:
                            in_memory_config.update(config)
                    
                    # Try initializing again with the in-memory config
                    evaluation_module.initialize(config=in_memory_config)
                else:
                    raise
            
        except Exception as e:
            error_msg = f"❌ Failed to initialize evaluation module: {str(e)}"
            if logger:
                logger.error(error_msg, exc_info=True)
            
            # Try to display the error in the notebook
            try:
                clear_output(wait=True)
                display(HTML(f"<div style='color:red;'>{error_msg}</div>"))
            except:
                print(error_msg)
            
            raise Exception(error_msg) from e
        
        # Get the UI components
        ui_components = evaluation_module.get_ui_components()
        if not ui_components:
            error_msg = "No UI components were created"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Get the main container
        main_ui = ui_components.get('main_container')
        if main_ui is None:
            error_msg = "Main container not found in UI components"
            if logger:
                logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Display the UI if requested
        if display and ui_components:
            from IPython import get_ipython
            from IPython.display import display as ipython_display, clear_output, HTML
            
            # Clear any existing output
            if get_ipython() is not None:
                clear_output(wait=True)
            
            # Get the main UI container and display it
            main_ui = ui_components.get('main_container')
            if main_ui is not None:
                try:
                    # Try using show() method if available
                    if hasattr(main_ui, 'show'):
                        ui_widget = main_ui.show()
                        ipython_display(ui_widget)
                    else:
                        # Fallback to direct display
                        ipython_display(main_ui)
                    return None  # Don't return data when display=True
                except Exception as e:
                    # Fallback to simple display if anything goes wrong
                    logger = get_module_logger("smartcash.ui.model.evaluation")
                    logger.error(f"Error displaying UI: {str(e)}")
                    try:
                        ipython_display(main_ui)
                    except Exception as inner_e:
                        logger.error(f"Failed to display fallback UI: {inner_e}", exc_info=True)
                        ipython_display(HTML(f"<div style='color:red; padding: 10px; border: 1px solid #f5c6cb; background-color: #f8d7da; border-radius: 4px;'>"
                                          f"<h3 style='margin-top: 0; color: #721c24;'>❌ Fatal Error</h3>"
                                          f"<p><strong>Error:</strong> {str(inner_e)}</p>"
                                          "<p>Please check the logs for more details.</p>"
                                          "</div>"))
                    return None  # Don't return data when display=True
        
        # If not displaying, return the module
        return evaluation_module
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize evaluation UI: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        
        try:
            # Try to display the error in the notebook
            clear_output(wait=True)
            display(HTML(f"<div style='color:red;'>{error_msg}</div>"))
            from IPython.display import display, HTML
            display(HTML(f"<div style='color:red'>{error_msg}</div>"))
        except:
            print(error_msg)
        
        raise Exception(error_msg) from e

class EvaluationUIModule(UIModule):
    """
    Evaluation UI Module for comprehensive model evaluation.
    
    Handles 2×4 evaluation matrix:
    - 2 scenarios: position_variation, lighting_variation
    - 4 model combinations: 2 backbones × 2 layer modes
    - Total: 8 evaluation tests
    """
    
    def __init__(self):
        """Initialize evaluation UI module."""
        # Initialize logger first
        self.logger = get_module_logger("smartcash.ui.model.evaluation")
        
        try:
            # Then initialize parent
            super().__init__(
                module_name=UI_CONFIG['module_name'],
                parent_module=UI_CONFIG['parent_module']
            )
            
            self._operation_manager = None
            self._config_handler = None
            self._ui_components = None
            self._merged_config = None
            
            self.logger.debug("✅ EvaluationUIModule initialized")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize EvaluationUIModule: {e}", exc_info=True)
            raise
        
    @suppress_ui_init_logs(duration=3.0)
    def initialize(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> None:
        """
        Initialize evaluation module with configuration.
        
        Args:
            config: Optional configuration override
            **kwargs: Additional initialization parameters
        """
        try:
            self.log("🎯 Initializing evaluation module...", 'info')
            
            # Initialize configuration handler
            self._initialize_config_handler(config)
            
            # Create UI components
            self._ui_components = self._create_ui_components(self._merged_config)
            
            # Setup button handlers for evaluation operations
            self._setup_button_handlers()
            
            # Initialize operation manager
            self._initialize_operation_manager()
            
            self.log("✅ Evaluation module initialized successfully", 'info')
            self.log(f"📊 Ready to test 8 model combinations (2 scenarios × 4 models)", 'info')
            
            # Call parent initialization
            super().initialize()
            
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation module: {e}")
            self.log(f"❌ Evaluation initialization failed: {e}", 'error')
            raise RuntimeError("Failed to initialize evaluation module")
    
    def _initialize_config_handler(self, config: Optional[Dict[str, Any]] = None) -> None:
        """Initialize configuration handler.
        
        Args:
            config: Optional configuration dictionary to override defaults
        """
        try:
            # Get default configuration
            default_config = get_default_evaluation_config()
            
            # Merge with provided config if any
            if config:
                # Deep merge the configs to preserve nested structures
                merged_config = default_config.copy()
                for key, value in config.items():
                    if key in merged_config and isinstance(merged_config[key], dict) and isinstance(value, dict):
                        merged_config[key].update(value)
                    else:
                        merged_config[key] = value
                self._merged_config = merged_config
            else:
                self._merged_config = default_config
                
            # Create config handler with the merged config
            self._config_handler = EvaluationConfigHandler(config=self._merged_config)
            
            # If we have a merged config, update the handler
            if hasattr(self, '_merged_config') and self._merged_config:
                self._config_handler.update_config(self._merged_config)
            
            self.log("⚙️ Configuration handler initialized", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize config handler: {e}", exc_info=True)
            self.log(f"❌ Failed to initialize config: {str(e)}", 'error')
            raise
    
    def _create_ui_components(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create UI components for evaluation module.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Dictionary containing UI components
        """
        try:
            from smartcash.ui.model.evaluation.components.evaluation_ui import create_evaluation_ui
            
            # Create evaluation UI components
            ui_components = create_evaluation_ui(config)
            
            self.log(f"🎨 Created {len(ui_components)} UI components", 'info')
            return ui_components
            
        except Exception as e:
            self.logger.error(f"Failed to create UI components: {e}")
            raise
    
    def _setup_button_handlers(self) -> None:
        """Setup button event handlers for evaluation operations."""
        try:
            # Get action container which contains the buttons
            action_container = self._ui_components.get('action_container')
            if not action_container:
                self.logger.warning("Action container not found, skipping button handlers")
                return
            
            # Standard action container returns a dict with 'buttons' key
            if isinstance(action_container, dict) and 'buttons' in action_container:
                buttons = action_container['buttons']
                
                # Bind button handlers
                for button_id, button_widget in buttons.items():
                    if button_id == 'run_all_scenarios':
                        button_widget.on_click(self._handle_run_all_scenarios)
                    elif button_id == 'run_position_scenario':
                        button_widget.on_click(self._handle_run_position_scenario)
                    elif button_id == 'run_lighting_scenario':
                        button_widget.on_click(self._handle_run_lighting_scenario)
                    elif button_id == 'load_checkpoint':
                        button_widget.on_click(self._handle_load_checkpoint)
                    elif button_id == 'export_results':
                        button_widget.on_click(self._handle_export_results)
                
                self.log("🔗 Button handlers configured", 'info')
            else:
                self.logger.warning("Invalid action container structure")
            
        except Exception as e:
            self.logger.error(f"Failed to setup button handlers: {e}")
            raise
    
    def _initialize_operation_manager(self) -> None:
        """Initialize the evaluation operation manager."""
        try:
            operation_container = self._ui_components.get('operation_container')
            
            # Ensure config is a dictionary and has the expected structure
            if not isinstance(self._merged_config, dict):
                self.logger.warning(f"Config is not a dictionary (got {type(self._merged_config).__name__}), converting to dict")
                config = {"evaluation": self._merged_config} if self._merged_config is not None else {}
            else:
                config = self._merged_config
            
            # Ensure the config has the required structure
            if "evaluation" not in config:
                self.logger.warning("Config missing 'evaluation' key, adding it")
                config = {"evaluation": config if config is not None else {}}
            
            self._operation_manager = EvaluationOperationManager(
                config=config,
                operation_container=operation_container
            )
            self._operation_manager.initialize()
            
            self.log("🎯 Operation manager initialized", 'info')
            
        except Exception as e:
            self.logger.error(f"Failed to initialize operation manager: {e}", exc_info=True)
            raise
    
    def log(self, message: str, level: str = 'info') -> None:
        """
        Log message to operation container.
        
        Args:
            message: Message to log
            level: Log level (info, success, warning, error)
        """
        try:
            operation_container = self._ui_components.get('operation_container')
            if operation_container:
                # Standard operation container should have log method or log_accordion
                if hasattr(operation_container, 'log'):
                    operation_container.log(message, level)
                    return
                elif hasattr(operation_container, 'log_accordion'):
                    # Use log accordion if available
                    log_accordion = operation_container.log_accordion
                    if hasattr(log_accordion, 'log'):
                        log_accordion.log(message, level)
                        return
            
            # Fallback to logger
            getattr(self.logger, level, self.logger.info)(message)
            
        except Exception as e:
            self.logger.error(f"Failed to log message: {e}")
            getattr(self.logger, level, self.logger.info)(message)
    
    # Button Handler Methods
    async def _handle_run_all_scenarios(self, button) -> None:
        """Handle run all scenarios button click."""
        try:
            self.log("🚀 Starting comprehensive evaluation (8 tests)...", 'info')
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not initialized")
            
            result = await self._operation_manager.execute_all_scenarios()
            
            if result.get('success'):
                successful = result.get('successful_tests', 0)
                total = result.get('total_tests', 0)
                self.log(f"🎉 Comprehensive evaluation completed: {successful}/{total} tests successful", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Comprehensive evaluation failed: {error_msg}", 'error')
                
        except Exception as e:
            self.logger.error(f"All scenarios evaluation failed: {e}")
            self.log(f"❌ Evaluation error: {e}", 'error')
    
    async def _handle_run_position_scenario(self, button) -> None:
        """Handle run position scenario button click."""
        try:
            self.log("📐 Starting position variation scenario (4 models)...", 'info')
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not initialized")
            
            result = await self._operation_manager.execute_position_scenario()
            
            if result.get('success'):
                successful = result.get('successful_tests', 0)
                total = result.get('total_tests', 0)
                self.log(f"✅ Position scenario completed: {successful}/{total} models successful", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Position scenario failed: {error_msg}", 'error')
                
        except Exception as e:
            self.logger.error(f"Position scenario failed: {e}")
            self.log(f"❌ Position scenario error: {e}", 'error')
    
    async def _handle_run_lighting_scenario(self, button) -> None:
        """Handle run lighting scenario button click."""
        try:
            self.log("💡 Starting lighting variation scenario (4 models)...", 'info')
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not initialized")
            
            result = await self._operation_manager.execute_lighting_scenario()
            
            if result.get('success'):
                successful = result.get('successful_tests', 0)
                total = result.get('total_tests', 0)
                self.log(f"✅ Lighting scenario completed: {successful}/{total} models successful", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Lighting scenario failed: {error_msg}", 'error')
                
        except Exception as e:
            self.logger.error(f"Lighting scenario failed: {e}")
            self.log(f"❌ Lighting scenario error: {e}", 'error')
    
    async def _handle_load_checkpoint(self, button) -> None:
        """Handle load checkpoint button click."""
        try:
            self.log("📂 Loading best model checkpoints...", 'info')
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not initialized")
            
            result = await self._operation_manager.load_best_checkpoints()
            
            if result.get('success'):
                loaded_count = result.get('loaded_count', 0)
                self.log(f"✅ Loaded {loaded_count} model checkpoints", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Checkpoint loading failed: {error_msg}", 'error')
                
        except Exception as e:
            self.logger.error(f"Checkpoint loading failed: {e}")
            self.log(f"❌ Checkpoint loading error: {e}", 'error')
    
    async def _handle_export_results(self, button) -> None:
        """Handle export results button click."""
        try:
            self.log("📊 Exporting evaluation results...", 'info')
            
            if not self._operation_manager:
                raise RuntimeError("Operation manager not initialized")
            
            # Get current results (would normally come from completed evaluations)
            results = {"results": {}, "summary": "No evaluations completed yet"}
            
            result = await self._operation_manager.export_results(results)
            
            if result.get('success'):
                export_path = result.get('export_path', 'Unknown path')
                self.log(f"✅ Results exported to {export_path}", 'success')
            else:
                error_msg = result.get('error', 'Unknown error')
                self.log(f"❌ Export failed: {error_msg}", 'error')
                
        except Exception as e:
            self.logger.error(f"Export failed: {e}")
            self.log(f"❌ Export error: {e}", 'error')
    
    def get_ui_components(self) -> Dict[str, Any]:
        """
        Get UI components.
        
        Returns:
            Dictionary containing UI components
        """
        return self._ui_components or {}
    
    def get_operation_manager(self) -> Optional[EvaluationOperationManager]:
        """
        Get operation manager.
        
        Returns:
            Evaluation operation manager instance
        """
        return self._operation_manager
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get current configuration.
        
        Returns:
            Current configuration dictionary
        """
        return self._merged_config or {}
    
    def update_config(self, new_config: Dict[str, Any]) -> None:
        """
        Update configuration.
        
        Args:
            new_config: New configuration values
        """
        try:
            if self._merged_config:
                self._merged_config.update(new_config)
                self.log("⚙️ Configuration updated", 'info')
            else:
                self.logger.warning("No configuration to update")
                
        except Exception as e:
            self.logger.error(f"Failed to update configuration: {e}")
            self.log(f"❌ Configuration update failed: {e}", 'error')
    
    def display(self) -> None:
        """
        Display the UI components in the notebook.
        This method is called when the module is displayed in a notebook cell.
        """
        try:
            from IPython.display import display as ipython_display, clear_output, HTML
            from IPython import get_ipython
            
            # Get the UI components
            ui_components = self.get_ui_components()
            if not ui_components:
                self.logger.warning("⚠️ No UI components to display")
                return
            
            self.logger.info(f"📋 UI Components available: {', '.join(ui_components.keys())}")
            
            # Clear any existing output
            if get_ipython() is not None:
                clear_output(wait=True)
            
            # Get the main UI container
            main_ui = ui_components.get('main_container')
            if main_ui is None:
                error_msg = "❌ Main container not found in UI components"
                self.logger.error(error_msg)
                ipython_display(HTML(f"<div style='color:red;'>{error_msg}</div>"))
                return
            
            try:
                # Try using show() method if available
                if hasattr(main_ui, 'show'):
                    try:
                        # Some widgets have a show() method that returns a widget
                        ui_widget = main_ui.show()
                        ipython_display(ui_widget)
                    except Exception as e:
                        self.logger.warning(f"show() method failed, falling back to direct display: {e}")
                        ipython_display(main_ui)
                else:
                    # Fallback to direct display
                    ipython_display(main_ui)
                
                self.logger.info("✅ Evaluation UI displayed successfully")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to display UI: {e}", exc_info=True)
                
                # Fallback to error display
                error_html = f"""
                <div style='color: #721c24; padding: 15px; margin: 10px 0;
                           border: 1px solid #f5c6cb; background-color: #f8d7da;
                           border-radius: 4px;'>
                    <h3 style='margin-top: 0; color: #721c24;'>❌ Error Displaying UI</h3>
                    <p><strong>Error:</strong> {error}</p>
                    <p>Please check the logs for more details.</p>
                </div>
                """.format(error=str(e))
                
                try:
                    ipython_display(HTML(error_html))
                except:
                    print(f"Failed to display UI: {e}")
                
                # As a last resort, try to display all components
                self._display_all_components(ui_components)
                    
        except Exception as e:
            self.logger.error(f"❌ Fatal error in display(): {e}", exc_info=True)
            try:
                from IPython.display import display, HTML
                display(HTML(f"<div style='color:red; padding: 10px; border: 1px solid #f5c6cb; background-color: #f8d7da; border-radius: 4px;'><h3>❌ Fatal Error</h3><p><strong>Error:</strong> {str(e)}</p><p>Please check the logs for more details.</p></div>"))
            except:
                print(f"Fatal error in display(): {e}")
    
    def _display_all_components(self, ui_components: Dict[str, Any]) -> None:
        """Helper method to display all available UI components for debugging"""
        from IPython.display import display, HTML
        
        display(HTML("<h3>Debug: Available UI Components</h3>"))
        
        for name, component in ui_components.items():
            try:
                display(HTML(f"<h4>Component: {name}</h4>"))
                display(component)
                if hasattr(component, '__dict__'):
                    display(HTML(f"<pre>Attributes: {list(component.__dict__.keys())}</pre>"))
            except Exception as e:
                display(HTML(f"<div style='color:orange'>Failed to display {name}: {e}</div>"))