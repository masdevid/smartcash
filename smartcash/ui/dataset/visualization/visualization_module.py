"""
File: smartcash/ui/dataset/visualization/visualization_module.py
Description: Visualization module using the new UIModule pattern.
"""

from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.core.ui_module import UIModule
from smartcash.ui.core.ui_module_factory import UIModuleFactory, ModuleTemplate
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors import SmartCashUIError

# Import visualization components and handlers
from .handlers.visualization_ui_handler import VisualizationUIHandler
from .configs.visualization_config_handler import VisualizationConfigHandler
from .constants import DEFAULT_CONFIG

# Import dataset processing modules
from smartcash.dataset.preprocessor import PreprocessingService
from smartcash.dataset.augmentor import AugmentationService


class VisualizationUIModule(UIModule):
    """Visualization module for dataset analysis and comparison.
    
    Provides visualization capabilities for analyzing and comparing datasets,
    including raw, preprocessed, and augmented data.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs):
        """Initialize the visualization module.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional keyword arguments
        """
        # Merge default config with provided config
        final_config = DEFAULT_CONFIG.copy()
        if config:
            final_config.update(config)
            
        # Initialize the base UIModule
        super().__init__(
            module_name="visualization",
            parent_module="dataset",
            config=final_config,
            **kwargs
        )
        
        # Initialize services
        self.preprocessor = PreprocessingService()
        self.augmentor = AugmentationService()
        self._ui_handler = None
        
        # Set up logger
        self.logger = get_module_logger(f"smartcash.ui.dataset.visualization")
        
    def _setup_components(self):
        """Set up UI components and handlers."""
        try:
            self.logger.info("🔧 Setting up visualization UI components")
            
            # Create UI components
            from .components.visualization_ui import create_visualization_ui
            ui_components = create_visualization_ui(self._config)
            
            # Initialize UI handler
            self._ui_handler = VisualizationUIHandler(
                ui_components=ui_components,
                logger=self.logger
            )
            
            # Register components with the module
            for name, component in ui_components.items():
                self.register_component(name, component)
                
            # Register operations
            self.register_operation("analyze", self._analyze_dataset)
            self.register_operation("export", self._export_visualization)
            self.register_operation("compare", self._compare_datasets)
            
            # Setup UI logging bridge and progress display
            operation_container = ui_components.get('operation_container')
            if operation_container:
                self._setup_ui_logging_bridge(operation_container)
                self._initialize_progress_display()
            
            self.logger.info("✅ Visualization UI components set up successfully")
            
        except Exception as e:
            error_msg = f"Failed to set up visualization UI: {str(e)}"
            self.logger.error(f"❌ {error_msg}", exc_info=True)
            raise SmartCashUIError(
                message=error_msg,
                error_code="UI_SETUP_ERROR"
            ) from e
    
    def _analyze_dataset(self, **kwargs) -> Dict[str, Any]:
        """Analyze dataset and update visualizations.
        
        Args:
            **kwargs: Analysis parameters
            
        Returns:
            Dictionary with analysis results
        """
        try:
            self.logger.info("🔍 Starting dataset analysis")
            self.update_status("Analyzing dataset...", "info")
            
            # Get dataset path from config or UI
            dataset_path = self.get_config("dataset_path")
            if not dataset_path:
                raise ValueError("No dataset path specified")
                
            # TODO: Implement actual analysis logic
            # This is a placeholder for the analysis pipeline
            
            result = {
                "status": "success",
                "message": "Analysis completed successfully",
                "metrics": {}
            }
            
            self.update_status("Analysis completed", "success")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Dataset analysis failed: {e}", exc_info=True)
            self.update_status(f"Analysis failed: {str(e)}", "error")
            raise
    
    def _export_visualization(self, **kwargs) -> Dict[str, Any]:
        """Export visualization results.
        
        Args:
            **kwargs: Export parameters
            
        Returns:
            Dictionary with export results
        """
        try:
            self.logger.info("💾 Exporting visualization")
            self.update_status("Exporting visualization...", "info")
            
            # TODO: Implement actual export logic
            # This is a placeholder for the export functionality
            
            result = {
                "status": "success",
                "message": "Export completed successfully",
                "export_path": "/path/to/export"
            }
            
            self.update_status("Export completed", "success")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Export failed: {e}", exc_info=True)
            self.update_status(f"Export failed: {str(e)}", "error")
            raise
    
    def _compare_datasets(self, **kwargs) -> Dict[str, Any]:
        """Compare multiple datasets.
        
        Args:
            **kwargs: Comparison parameters
            
        Returns:
            Dictionary with comparison results
        """
        try:
            self.logger.info("🔄 Comparing datasets")
            self.update_status("Comparing datasets...", "info")
            
            # TODO: Implement actual comparison logic
            # This is a placeholder for the comparison functionality
            
            result = {
                "status": "success",
                "message": "Comparison completed successfully",
                "comparison_metrics": {}
            }
            
            self.update_status("Comparison completed", "success")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Comparison failed: {e}", exc_info=True)
            self.update_status(f"Comparison failed: {str(e)}", "error")
            raise
    
    def _setup_ui_logging_bridge(self, operation_container: Any) -> None:
        """Setup UI logging bridge to capture backend service logs."""
        try:
            import logging
            
            # Create custom handler for backend services
            class BackendUILogHandler(logging.Handler):
                def __init__(self, log_func):
                    super().__init__()
                    self.log_func = log_func
                    self.setFormatter(logging.Formatter('%(name)s: %(message)s'))
                
                def emit(self, record):
                    try:
                        msg = self.format(record)
                        level = 'info' if record.levelno == logging.INFO else 'error'
                        self.log_func(msg, level)
                    except Exception:
                        pass  # Silently fail to avoid recursive errors
            
            # Get log function from operation container
            if hasattr(operation_container, 'log_message'):
                log_func = operation_container.log_message
            elif hasattr(operation_container, 'log'):
                log_func = operation_container.log
            else:
                # Fallback to internal logging
                log_func = self._log_to_ui
            
            # Create handler
            ui_handler = BackendUILogHandler(log_func)
            ui_handler.setLevel(logging.INFO)
            
            # Target specific backend service loggers that might log during visualization operations
            target_loggers = [
                'smartcash.dataset',
                'smartcash.model', 
                'smartcash.ui.dataset.visualization',
                'smartcash.core',
                'matplotlib',
                'plotly'
            ]
            
            # Remove existing console handlers and add UI handlers
            for logger_name in target_loggers:
                logger = logging.getLogger(logger_name)
                
                # Remove existing console handlers
                for handler in logger.handlers[:]:
                    if isinstance(handler, logging.StreamHandler):
                        logger.removeHandler(handler)
                
                # Add UI handler
                logger.addHandler(ui_handler)
            
            # Store handler for cleanup
            if not hasattr(self, '_ui_handlers'):
                self._ui_handlers = []
            self._ui_handlers.append(ui_handler)
            
            self.logger.debug("🌉 UI logging bridge setup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to setup UI logging bridge: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress tracker display."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'progress_tracker'):
                progress_tracker = operation_container.progress_tracker
                if hasattr(progress_tracker, 'initialize') and not getattr(progress_tracker, '_initialized', False):
                    progress_tracker.initialize()
                if hasattr(progress_tracker, 'show'):
                    progress_tracker.show()
                    
            self.logger.debug("📊 Progress display initialized")
            
        except Exception as e:
            self.logger.debug(f"Progress display initialization failed: {e}")
    
    def _log_to_ui(self, message: str, level: str = "info") -> None:
        """Log message to UI components using operation container's log_accordion."""
        try:
            # Get operation container and use its log_accordion
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'log'):
                # Map log levels to LogLevel enum if needed
                from smartcash.ui.components.log_accordion import LogLevel
                level_map = {
                    'info': LogLevel.INFO,
                    'success': LogLevel.INFO,
                    'warning': LogLevel.WARNING,
                    'error': LogLevel.ERROR,
                    'debug': LogLevel.DEBUG
                }
                log_level = level_map.get(level, LogLevel.INFO)
                operation_container.log(message, log_level)
            else:
                # Fallback to logger if operation container not available
                getattr(self.logger, level, self.logger.info)(message)
        except Exception as e:
            self.logger.debug(f"UI logging failed: {e}")
    
    def _update_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """Update progress tracker using operation container."""
        try:
            operation_container = self.get_component("operation_container")
            if operation_container and hasattr(operation_container, 'update_progress'):
                operation_container.update_progress(progress, message, level)
            else:
                # Fallback to logging
                self._log_to_ui(f"Progress {progress}%: {message}", "info")
        except Exception as e:
            self.logger.debug(f"Progress update failed: {e}")
    
    def _cleanup_ui_logging_bridge(self) -> None:
        """Cleanup UI logging bridge handlers."""
        try:
            if hasattr(self, '_ui_handlers'):
                import logging
                for handler in self._ui_handlers:
                    # Remove handler from all loggers
                    for logger_name in logging.Logger.manager.loggerDict:
                        logger = logging.getLogger(logger_name)
                        if handler in logger.handlers:
                            logger.removeHandler(handler)
                self._ui_handlers.clear()
                
            self.logger.debug("🧹 UI logging bridge cleanup completed")
            
        except Exception as e:
            self.logger.debug(f"UI logging bridge cleanup failed: {e}")

    def cleanup(self) -> None:
        """Cleanup resources."""
        try:
            self._cleanup_ui_logging_bridge()
            if hasattr(super(), 'cleanup'):
                super().cleanup()
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

    def display(self):
        """Display the visualization UI.
        
        Returns:
            The main container widget if display is successful, None otherwise
        """
        try:
            main_container = self.get_component("main_container")
            if main_container is None:
                self.logger.warning("Main container not found in visualization module")
                return None
                
            # Import display here to avoid circular imports
            from IPython.display import display as ipy_display
            
            try:
                # Try displaying with the standard IPython display
                ipy_display(main_container)
                return main_container
                
            except TypeError as e:
                if "unexpected keyword argument 'display'" in str(e):
                    # Fallback for ZMQDisplayPublisher issue
                    self.logger.debug("Using fallback display method due to ZMQDisplayPublisher issue")
                    from IPython.core.display import publish_display_data
                    publish_display_data(data={
                        'text/plain': str(main_container),
                        'text/html': main_container._repr_html_() if hasattr(main_container, '_repr_html_') else str(main_container)
                    })
                    return main_container
                raise
                
        except Exception as e:
            self.logger.error(f"Failed to display visualization UI: {e}", exc_info=True)
            return None


# Register the module template with the factory
VISUALIZATION_MODULE_TEMPLATE = ModuleTemplate(
    module_name="visualization",
    parent_module="dataset",
    default_config=DEFAULT_CONFIG,
    required_components=[
        "header_container",
        "form_container",
        "action_container",
        "summary_container",
        "operation_container",
        "footer_container",
        "main_container"
    ],
    required_operations=["analyze", "export", "compare"],
    description="Dataset visualization and comparison module"
)

# Register the template with the factory
UIModuleFactory.register_template(VISUALIZATION_MODULE_TEMPLATE)


def create_visualization_module(config: Optional[Dict[str, Any]] = None, **kwargs) -> VisualizationUIModule:
    """Create a new visualization module instance.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
        
    Returns:
        VisualizationUIModule: New visualization module instance
    """
    return VisualizationUIModule(config=config, **kwargs)


def get_visualization_module() -> Optional[VisualizationUIModule]:
    """Get the current visualization module instance.
    
    Returns:
        Optional[VisualizationUIModule]: The current visualization module or None
    """
    return UIModuleFactory.get_module("visualization", "dataset")


def initialize_visualization_ui(
    config: Optional[Dict[str, Any]] = None,
    display: bool = True,
    **kwargs
) -> Optional[Dict[str, Any]]:
    """Initialize Visualization UI using new UIModule pattern.
    
    Args:
        config: Optional configuration dictionary
        display: Whether to display the UI immediately (True) or return components (False)
        **kwargs: Additional keyword arguments for module creation
        
    Returns:
        None if display=True, dict of components if display=False
    """
    try:
        from IPython.display import display as ipython_display
        
        # Get the module and UI components
        module = create_visualization_module(config, **kwargs)
        
        # Setup components and initialize module
        if not hasattr(module, '_components') or not module._components:
            module._setup_components()
        
        ui_components = {
            component_type: module.get_component(component_type)
            for component_type in module.list_components()
        }
        
        main_ui = ui_components.get('main_container')
        
        # Setup UI logging bridge to capture backend service logs
        operation_container = ui_components.get('operation_container')
        if operation_container and hasattr(module, '_setup_ui_logging_bridge'):
            module._setup_ui_logging_bridge(operation_container)
        
        # Initialize progress display
        if hasattr(module, '_initialize_progress_display'):
            module._initialize_progress_display()
        
        if display and main_ui:
            ipython_display(main_ui)
            return None
        
        # Return components without displaying
        result = {
            'success': True,
            'module': module,
            'ui_components': ui_components,
            'main_ui': main_ui
        }
        
        return result
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'module': None,
            'ui_components': {},
            'main_ui': None
        }
        
        if display:
            get_module_logger("smartcash.ui.dataset.visualization").error(f"Failed to initialize visualization UI: {e}")
            return None
        
        return error_result


def display_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Display the visualization UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
    """
    initialize_visualization_ui(config, display=True, **kwargs)
