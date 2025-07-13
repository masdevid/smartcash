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
            
            self.logger.info("✅ Visualization UI components set up successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to set up visualization UI: {e}", exc_info=True)
            raise SmartCashUIError(f"Failed to set up visualization UI: {str(e)}") from e
    
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
    
    def display(self):
        """Display the visualization UI."""
        main_container = self.get_component("main_container")
        if main_container is None:
            self.logger.warning("Main container not found in visualization module")
            return None
            
        # Import display here to avoid circular imports
        from IPython.display import display as ipy_display
        
        # Display the main container
        ipy_display(main_container)
        return main_container


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
    return UIModuleFactory.create_module(
        module_name="visualization",
        parent_module="dataset",
        config=config,
        module_class=VisualizationUIModule,
        **kwargs
    )


def get_visualization_module() -> Optional[VisualizationUIModule]:
    """Get the current visualization module instance.
    
    Returns:
        Optional[VisualizationUIModule]: The current visualization module or None
    """
    return UIModuleFactory.get_module("visualization", "dataset")


def display_visualization_ui(config: Optional[Dict[str, Any]] = None, **kwargs):
    """Display the visualization UI.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional keyword arguments
    """
    module = create_visualization_module(config, **kwargs)
    module.display()


# For backward compatibility
initialize_visualization_ui = display_visualization_ui
