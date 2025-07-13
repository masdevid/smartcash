"""
File: smartcash/ui/dataset/visualization/visualization_initializer.py
Visualization module initializer following ModuleInitializer pattern.

This module provides visualization capabilities for dataset analysis,
including comparison between raw, preprocessed, and augmented data.
"""

import asyncio
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
        """Schedule post-initialization checks.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._post_init_check(ui_components, config))
            else:
                loop.run_until_complete(self._post_init_check(ui_components, config))
        except Exception as e:
            self.logger.warning(f"Could not schedule post-init check: {str(e)}")
    
    async def _post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """Perform post-initialization checks.
        
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
def init_visualization_ui(**kwargs):
    """Initialize and display visualization UI.
    
    This is the main entry point function that should be called from notebook cells.
    It creates the visualization initializer and displays the UI directly.
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results and UI components
    """
    try:
        # Get or create the global initializer
        initializer = get_visualization_initializer()
        
        # Initialize and get UI components
        components = initializer.initialize_full(**kwargs)
        
        # Display the UI
        if 'container' in components:
            from IPython.display import display
            display(components['container'])
        
        # Log success
        initializer.logger.info("✅ Visualization UI initialized successfully")
        
        return {
            'status': 'success',
            'components': components,
            'initializer': initializer
        }
        
    except Exception as e:
        error_msg = f"❌ Failed to initialize visualization UI: {str(e)}"
        get_module_logger('visualization').error(error_msg, exc_info=True)
        
        # Display error message
        from IPython.display import display, HTML
        display(HTML(f'<div class="alert alert-danger">{error_msg}</div>'))
        
        return {
            'status': 'error',
            'error': str(e),
            'message': error_msg
        }
