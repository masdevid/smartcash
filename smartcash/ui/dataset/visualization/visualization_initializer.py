"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Visualization module initializer following ModuleInitializer pattern.

This module provides visualization capabilities for dataset analysis,
including comparison between raw, preprocessed, and augmented data.
"""

# import asyncio  # Removed async operations
from typing import Dict, Any, Optional, List, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.core.errors import SmartCashUIError
from smartcash.ui.logger import get_module_logger
from smartcash.ui.core.errors.handlers import create_error_response
from smartcash.ui.core.errors.error_component import create_error_component

# Import visualization components and handlers
from .handlers.visualization_ui_handler import VisualizationUIHandler
from .components.visualization_ui import create_visualization_ui
from .configs.visualization_config_handler import VisualizationConfigHandler
from .constants import DEFAULT_CONFIG

# Import dataset processing modules
from smartcash.dataset.preprocessor import PreprocessingService
from smartcash.dataset.augmentor import AugmentationService


class VisualizationInitializer(ModuleInitializer):
    """Visualization module initializer with dataset comparison capabilities.
    
    Provides visualization for dataset analysis including comparison between
    raw, preprocessed, and augmented data.
    """
    
    def __init__(self, **kwargs):
        """Initialize the visualization module with shared config support.
        
        Args:
            **kwargs: Additional arguments passed to parent initializer
        """
        # Ensure enable_shared_config is True for shared config support
        kwargs['enable_shared_config'] = kwargs.get('enable_shared_config', True)
        
        # Initialize with proper config handler and parent module
        super().__init__(
            module_name="visualization",
            config_handler_class=VisualizationConfigHandler,
            parent_module="dataset",
            **kwargs
        )
        
        # Initialize services
        self.preprocessor = PreprocessingService()
        self.augmentor = AugmentationService()
        self._ui_handler = None
        
        # Ensure config handler is properly initialized
        if not hasattr(self.config_handler, '_shared_manager') and hasattr(self.config_handler, 'initialize'):
            self.config_handler.initialize()
        
    def create_ui_components(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Create visualization UI components.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional arguments
            
        Returns:
            Dictionary of UI components
        """
        try:
            self.logger.info("🔧 Creating visualization UI components")
            
            # Get default config and merge with provided config
            final_config = DEFAULT_CONFIG.copy()
            if config:
                final_config.update(config)
            
            # Validate configuration
            validated_config = self.config_handler.validate_config(final_config)
            
            # Create UI components
            ui_components = create_visualization_ui(validated_config, **kwargs)
            
            # Initialize UI handler if not already set
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                # The handler is initialized with the UI components directly in __init__
                self._ui_handler = VisualizationUIHandler(ui_components=ui_components, logger=self.logger)
            
            # Update UI from config
            self.config_handler.update_ui_from_config(ui_components, validated_config)
            
            # Schedule post-init check
            self._schedule_post_init_check(ui_components, validated_config)
            
            self.logger.info(f"✅ Created {len(ui_components)} visualization UI components")
            return ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create visualization UI components: {e}", exc_info=True)
            raise RuntimeError(f"Failed to create visualization UI: {str(e)}") from e

    def _initialize_handlers(self, ui_components: Dict[str, Any], **kwargs) -> bool:
        """Initialize visualization UI handlers.
        
        Args:
            ui_components: Dictionary of UI components
            **kwargs: Additional initialization parameters
            
        Returns:
            bool: True if initialization was successful, False otherwise
        """
        try:
            self.logger.info("🔧 Initializing visualization UI handlers")
            
            # Initialize UI handler if not already done
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self._ui_handler = VisualizationUIHandler(ui_components=ui_components)
                
                # Setup event handlers
                self._ui_handler.setup(ui_components=ui_components)
            
            # Add comparison chart handlers
            self._setup_comparison_handlers(ui_components)
            
            self.logger.info("✅ Visualization handlers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize visualization handlers: {e}", exc_info=True)
            return False
            
    def _setup_comparison_handlers(self, ui_components: Dict[str, Any]) -> None:
        """Setup handlers for comparison charts.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            # Get the action buttons
            actions = ui_components.get('containers', {}).get('actions', {})
            
            # Add handler for raw vs preprocessed comparison
            if 'compare_raw_preprocessed' in actions:
                actions['compare_raw_preprocessed'].on_click(
                    lambda _: self._compare_raw_vs_preprocessed(ui_components)
                )
            
            # Add handler for raw vs augmented comparison
            if 'compare_raw_augmented' in actions:
                actions['compare_raw_augmented'].on_click(
                    lambda _: self._compare_raw_vs_augmented(ui_components)
                )
                
        except Exception as e:
            self.logger.error(f"❌ Failed to setup comparison handlers: {e}", exc_info=True)
            
    def _compare_raw_vs_preprocessed(self, ui_components: Dict[str, Any]) -> None:
        """Generate comparison between raw and preprocessed data.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            self.logger.info("🔄 Generating raw vs preprocessed comparison")
            
            # Get the chart container
            chart_container = ui_components.get('chart_container')
            if not chart_container:
                self.logger.warning("Chart container not found in UI components")
                return
                
            # Get the raw data (example - replace with actual data loading)
            raw_data = self._load_raw_data()
            
            # Preprocess the data
            preprocessed_data = self.preprocessor.preprocess(raw_data)
            
            # Create comparison chart
            self._render_comparison_chart(
                chart_container,
                raw_data,
                preprocessed_data,
                "Raw Data vs Preprocessed Data"
            )
            
            self.logger.info("✅ Raw vs preprocessed comparison completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in raw vs preprocessed comparison: {e}", exc_info=True)
            self._show_error(ui_components, f"Comparison failed: {str(e)}")
            
    def _compare_raw_vs_augmented(self, ui_components: Dict[str, Any]) -> None:
        """Generate comparison between raw and augmented data.
        
        Args:
            ui_components: Dictionary of UI components
        """
        try:
            self.logger.info("🔄 Generating raw vs augmented comparison")
            
            # Get the chart container
            chart_container = ui_components.get('chart_container')
            if not chart_container:
                self.logger.warning("Chart container not found in UI components")
                return
                
            # Get the raw data (example - replace with actual data loading)
            raw_data = self._load_raw_data()
            
            # Augment the data
            augmented_data = self.augmentor.augment(raw_data)
            
            # Create comparison chart
            self._render_comparison_chart(
                chart_container,
                raw_data,
                augmented_data,
                "Raw Data vs Augmented Data"
            )
            
            self.logger.info("✅ Raw vs augmented comparison completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error in raw vs augmented comparison: {e}", exc_info=True)
            self._show_error(ui_components, f"Comparison failed: {str(e)}")
            
    def _load_raw_data(self):
        """Load raw data for visualization.
        
        Returns:
            Raw data in the required format
            
        Note: This is a placeholder. Replace with actual data loading logic.
        """
        # Example: Load sample data
        # In a real implementation, this would load data from your dataset
        import numpy as np
        return np.random.rand(100, 5)
        
    def _render_comparison_chart(self, container, data1, data2, title: str) -> None:
        """Render a comparison chart in the specified container.
        
        Args:
            container: The container widget to render the chart in
            data1: First dataset to compare
            data2: Second dataset to compare
            title: Chart title
        """
        try:
            # Clear previous content
            container.clear_output()
            
            # Create a figure with subplots
            import matplotlib.pyplot as plt
            import numpy as np
            
            with container:
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))
                
                # Plot first dataset
                axes[0].hist(data1.flatten() if hasattr(data1, 'flatten') else data1, bins=20, alpha=0.7, color='blue')
                axes[0].set_title('Before')
                axes[0].set_xlabel('Value')
                axes[0].set_ylabel('Frequency')
                
                # Plot second dataset
                axes[1].hist(data2.flatten() if hasattr(data2, 'flatten') else data2, bins=20, alpha=0.7, color='green')
                axes[1].set_title('After')
                axes[1].set_xlabel('Value')
                axes[1].set_ylabel('Frequency')
                
                # Add main title
                plt.suptitle(title, fontsize=14, fontweight='bold')
                plt.tight_layout()
                
                # Display the plot
                display(fig)
                
        except Exception as e:
            self.logger.error(f"❌ Error rendering comparison chart: {e}", exc_info=True)
            with container:
                display(HTML(f'<div class="alert alert-danger">Error rendering chart: {str(e)}</div>'))
    
    def _show_error(self, ui_components: Dict[str, Any], message: str) -> None:
        """Display an error message in the UI.
        
        Args:
            ui_components: Dictionary of UI components
            message: Error message to display
        """
        log_output = ui_components.get('log_output')
        if log_output and hasattr(log_output, 'append_stdout'):
            log_output.append_stdout(f"❌ {message}\n")
            
    def _schedule_post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Schedule post-initialization checks synchronously.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            # Run post-init check synchronously
            self._post_init_check(ui_components, config)
        except Exception as e:
            self.logger.warning(f"Could not run post-init check: {str(e)}")
    
    def _post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Perform post-initialization checks synchronously.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            # Check if we have a valid UI handler
            if not hasattr(self, '_ui_handler') or self._ui_handler is None:
                self.logger.warning("No UI handler available for post-init check")
                return
                
            # Check if we have any data to visualize
            # This is a placeholder - implement actual data availability check
            has_data = True  # Replace with actual check
            
            if not has_data:
                log = ui_components.get('log_output')
                if log and hasattr(log, 'append_stdout'):
                    log.append_stdout("ℹ️ No data available for visualization. Please load a dataset first.\n")
        
        except Exception as e:
            self.logger.error(f"❌ Error during post-init check: {e}", exc_info=True)
            
    def _initialize_impl(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Implementation of visualization module initialization.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing UI components
        """
        try:
            # Create UI components
            ui_components = self.create_ui_components(config=config, **kwargs)
            
            # Initialize handlers
            if not self._initialize_handlers(ui_components, **kwargs):
                raise RuntimeError("Failed to initialize visualization handlers")
            
            # Get the main container from UI components
            main_container = ui_components.get('main_container')
            if main_container:
                from IPython.display import display
                display(main_container, **kwargs)
            else:
                self.logger.warning("Main container not found in UI components")
            
            return ui_components
                
        except Exception as e:
            error_msg = f"Failed to display visualization UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            # Include error_type when raising SmartCashUIError
            raise SmartCashUIError(
                message=error_msg,
                error_code="UI_INIT_ERROR"
            ) from e


# Global instances
_visualization_initializer = None
_visualization_display_initializer = None


class VisualizationDisplayInitializer(DisplayInitializer):
    """DisplayInitializer wrapper for visualization module"""
    
    def __init__(self):
        super().__init__(module_name="visualization", parent_module="dataset")
        self._visualization_initializer = VisualizationInitializer()
    
    def _initialize_impl(self, **kwargs):
        """Implementation using existing VisualizationInitializer"""
        return self._visualization_initializer._initialize_impl(**kwargs)
        
    def display(self, **kwargs):
        """Display the visualization UI.
        
        Args:
            **kwargs: Additional arguments to pass to the initializer
        """
        try:
            # Get the UI components
            components = self._initialize_impl(**kwargs)
            
            # Display the main container if it exists
            if 'container' in components:
                from IPython.display import display as ipy_display
                ipy_display(components['container'])
                
            return components
            
        except Exception as e:
            error_msg = f"❌ Failed to display visualization UI: {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            
            # Display error message
            from IPython.display import display as ipy_display, HTML
            ipy_display(HTML(f'<div class="alert alert-danger">{error_msg}</div>'))
            
            return {
                'status': 'error',
                'error': str(e),
                'message': error_msg
            }


def get_visualization_initializer() -> VisualizationInitializer:
    """Get the global visualization initializer instance.
    
    Returns:
        VisualizationInitializer: The global visualization initializer instance
    """
    global _visualization_initializer
    if _visualization_initializer is None:
        _visualization_initializer = VisualizationInitializer()
    return _visualization_initializer


def get_visualization_display_initializer() -> VisualizationDisplayInitializer:
    """Get the global visualization display initializer instance.
    
    Returns:
        VisualizationDisplayInitializer: The global display initializer instance
    """
    global _visualization_display_initializer
    if _visualization_display_initializer is None:
        _visualization_display_initializer = VisualizationDisplayInitializer()
    return _visualization_display_initializer


def initialize_visualization_ui(env=None, config=None, **kwargs):
    """Initialize and display visualization UI using DisplayInitializer
    
    This function initializes the visualization module with proper configuration
    and error handling, ensuring the shared config manager is properly set up.
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
        
    Note:
        This function displays the UI directly and returns None.
        Use get_visualization_components() if you need access to the components dictionary.
    """
    try:
        # Set up environment and config in kwargs
        if env is not None:
            kwargs['env'] = env
        if config is not None:
            kwargs['config'] = config
        
        # Get the display initializer
        display_initializer = get_visualization_display_initializer()
        
        # Ensure we have a valid config handler
        if hasattr(display_initializer, '_visualization_initializer') and \
           hasattr(display_initializer._visualization_initializer, 'config_handler'):
            handler = display_initializer._visualization_initializer.config_handler
            if hasattr(handler, 'initialize') and not hasattr(handler, '_shared_manager'):
                handler.initialize()
        
        # Initialize and display the UI
        display_initializer.initialize_and_display(**kwargs)
        
    except Exception as e:
        # Create a clean error display
        error_msg = f"Failed to display visualization UI: {str(e)}"
        error_component = create_error_component(
            error_message=error_msg,
            title="🚨 Visualization Error",
            error_type="error",
            show_traceback=True  # Show traceback for debugging
        )
        
        # Display the error using IPython display
        from IPython.display import display as ipy_display
        ipy_display(error_component)
        
        # Log the full error for debugging
        import traceback
        print("\nFull error details:")
        traceback.print_exc()
    
    return None


def get_visualization_components(env=None, config=None, **kwargs) -> Dict[str, Any]:
    """Get visualization components dictionary without displaying UI
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments

    Returns:
        Dictionary of UI components
    """
    if env is not None:
        kwargs['env'] = env
    if config is not None:
        kwargs['config'] = config
    
    return get_visualization_display_initializer().get_components(**kwargs)


def display_visualization_ui(env=None, config=None, **kwargs):
    """Display visualization UI (alias for initialize_visualization_ui)
    
    Args:
        env: Optional environment context
        config: Optional configuration dictionary
        **kwargs: Additional arguments
    """
    initialize_visualization_ui(env=env, config=config, **kwargs)


# Main entry point function for cell execution
def init_visualization_ui(config: Optional[Dict[str, Any]] = None, display: bool = True, **kwargs):
    """Initialize and optionally display the visualization UI.
    
    Args:
        config: Optional configuration dictionary to override defaults
        display: If True, display the UI immediately (default: True)
        **kwargs: Additional keyword arguments for UI customization
        
    Returns:
        If display=True: Returns None after displaying UI
        If display=False: Returns a dictionary with UI components and initializer
        
    Raises:
        Exception: If initialization fails
    """
    try:
        from IPython.display import display as ipython_display, clear_output, HTML
        from ipywidgets import VBox, Output
        
        # Initialize the visualization module
        initializer = get_visualization_initializer()
        
        # Create UI components
        components = {}
        if hasattr(initializer, 'create_ui_components'):
            components = initializer.create_ui_components(config=config, **kwargs)
        elif hasattr(initializer, 'initialize_module_ui'):
            # Fallback to legacy method if create_ui_components doesn't exist
            ui = initializer.initialize_module_ui(
                module_name='visualization', 
                parent_module='dataset', 
                config=config, 
                **kwargs
            )
            if ui is not None:
                if display:
                    clear_output(wait=True)
                    ipython_display(ui)
                    initializer.logger.info("✅ Visualization UI displayed successfully")
                return ui if display else initializer
        else:
            error_msg = "❌ No supported UI initialization method found"
            if display:
                ipython_display(HTML(f'<div style="color: red; padding: 10px; border: 1px solid red;">{error_msg}</div>'))
            raise ValueError(error_msg)
        
        # Get the main container from components
        main_container = None
        ui_components = components.get('ui_components', {})
        
        # Try to get the main container from different possible locations
        for container_key in ['ui', 'main_container', 'container']:
            if container_key in components:
                main_container = components[container_key]
                break
        
        if not main_container and 'containers' in components and 'main' in components['containers']:
            main_container = components['containers']['main']
        if not main_container and 'containers' in ui_components and 'main' in ui_components['containers']:
            main_container = ui_components['containers']['main']
        
        # If still no main container, try to create one from available components
        if not main_container:
            all_components = {**components, **(ui_components or {})}
            displayable = [
                comp for comp in all_components.values() 
                if hasattr(comp, '_ipython_display_') or hasattr(comp, '_repr_html_')
            ]
            
            if displayable:
                main_container = VBox(displayable)
            else:
                error_msg = "❌ No displayable UI components found in visualization module"
                if display:
                    ipython_display(HTML(f'<div style="color: red; padding: 10px; border: 1px solid red;">{error_msg}</div>'))
                raise ValueError(error_msg)
        
        if display:
            try:
                clear_output(wait=True)
                
                # Try using show() method if available
                if hasattr(main_container, 'show'):
                    try:
                        # Some widgets have a show() method that returns a widget
                        ui_widget = main_container.show()
                        ipython_display(ui_widget)
                    except Exception as show_error:
                        initializer.logger.warning(f"show() method failed, falling back to direct display: {show_error}")
                        ipython_display(main_container)
                else:
                    # Fallback to direct display
                    ipython_display(main_container)
                
                initializer.logger.info("✅ Visualization UI displayed successfully")
                
            except Exception as display_error:
                error_msg = f"❌ Failed to display visualization UI: {str(display_error)}"
                initializer.logger.error(error_msg, exc_info=True)
                
                # Fallback to error display
                error_html = (
                    "<div style='color: #721c24; padding: 15px; margin: 10px 0; "
                    "border: 1px solid #f5c6cb; background-color: #f8d7da; "
                    "border-radius: 4px;'>"
                    "<h3 style='margin-top: 0; color: #721c24;'>❌ Error Displaying Visualization UI</h3>"
                    f"<p><strong>Error:</strong> {str(display_error)}</p>"
                    "<p>Please check the logs for more details.</p>"
                    "</div>"
                )
                
                if display:
                    ipython_display(HTML(error_html))
                raise
        
        # Return the appropriate result based on display flag
        if display:
            return None  # Return None when displaying, consistent with other modules
        
        # Return a result dictionary when not displaying
        result = {
            'status': 'success',
            'ui': main_container,
            'main_container': main_container,
            'components': components,
            'initializer': initializer
        }
        
        # Include all containers and widgets if available
        if 'containers' in ui_components:
            result['containers'] = ui_components['containers']
        if 'widgets' in ui_components:
            result['widgets'] = ui_components['widgets']
            
        return result
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize visualization UI: {str(e)}"
        print(error_msg)
        import traceback
        traceback.print_exc()
        
        if display and 'ipython_display' in locals():
            ipython_display(HTML(f'<div style="color: red; padding: 10px; border: 1px solid red;">{error_msg}</div>'))
        
        return {
            'status': 'error',
            'message': error_msg,
            'error': str(e)
        }
