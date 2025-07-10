"""
File: smartcash/ui/model/pretrained/pretrained_initializer.py
Pretrained models initializer following core UI structure with DisplayInitializer pattern.
"""

import asyncio
from typing import Dict, Any, Optional

from smartcash.ui.core.initializers.module_initializer import ModuleInitializer
from smartcash.ui.core.initializers.display_initializer import create_ui_display_function

from .components.pretrained_ui import create_pretrained_ui
from .handlers.pretrained_ui_handler import PretrainedUIHandler
from .configs.pretrained_config_handler import PretrainedConfigHandler
from .services.pretrained_service import PretrainedService
from .constants import DEFAULT_CONFIG


class PretrainedInitializer(ModuleInitializer):
    """
    Pretrained models initializer following core UI structure standard.
    Manages YOLOv5s and EfficientNet-B4 pretrained model downloads.
    """
    
    def __init__(self):
        """Initialize the pretrained initializer."""
        super().__init__(
            module_name="pretrained",
            parent_module="model",
            config_handler_class=PretrainedConfigHandler
        )
        self.service = PretrainedService()
    
    def _initialize_impl(self, config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Implementation of pretrained module initialization.
        
        Args:
            config: Optional configuration dictionary
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing UI components
        """
        try:
            # Get default config and merge with provided config
            final_config = DEFAULT_CONFIG.copy()
            if config:
                final_config.update(config)
            
            # Validate configuration
            validated_config = self.config_handler.validate_config(final_config)
            
            # Create UI components
            ui_components = create_pretrained_ui(validated_config, **kwargs)
            
            # Initialize UI handler
            ui_handler = PretrainedUIHandler(ui_components)
            ui_components['ui_handler'] = ui_handler
            
            # Update UI from config
            self.config_handler.update_ui_from_config(ui_components, validated_config)
            
            # Perform post-initialization checks in a non-blocking way
            try:
                # Create an event loop if none exists
                loop = asyncio.get_event_loop()
                if loop.is_running():
                    # If loop is running, schedule the task
                    loop.create_task(self._post_init_check(ui_components, validated_config))
                else:
                    # If no loop is running, run the coroutine directly
                    loop.run_until_complete(self._post_init_check(ui_components, validated_config))
            except Exception as e:
                logger.warning(f"Could not schedule post-init check: {str(e)}")
            
            return ui_components
            
        except Exception as e:
            error_msg = f"Failed to initialize pretrained module: {str(e)}"
            return self._create_error_response(error_msg, str(e))
    
    async def _post_init_check(self, ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
        """
        Perform post-initialization checks for existing pretrained models.
        
        Args:
            ui_components: UI components dictionary
            config: Configuration dictionary
        """
        try:
            # Get UI handler
            ui_handler = ui_components.get('ui_handler')
            if not ui_handler:
                return
            
            # Check models status
            models_status = await ui_handler.check_models_status()
            
            # Update log with status
            log_output = ui_components.get('log_output')
            if log_output and hasattr(log_output, 'log'):
                total_found = models_status.get('total_found', 0)
                total_models = 2  # YOLOv5s + EfficientNet-B4
                
                if models_status.get('all_present', False):
                    log_output.log("✅ All pretrained models are available and ready to use!")
                elif total_found > 0:
                    log_output.log(f"📋 Found {total_found}/{total_models} pretrained models. Use download button to get missing models.")
                else:
                    log_output.log("📋 No pretrained models found. Use download button to download YOLOv5s and EfficientNet-B4.")
                
                # List found models
                for model in models_status.get('models_found', []):
                    size_mb = model.get('file_size_mb', 0)
                    log_output.log(f"  ✓ {model['name']}: {size_mb} MB")
                
                # List missing models
                for model in models_status.get('models_missing', []):
                    log_output.log(f"  ❌ {model['name']}: Not found")
                    
        except Exception as e:
            # Don't fail initialization for post-init check errors
            log_output = ui_components.get('log_output')
            if log_output and hasattr(log_output, 'log'):
                log_output.log(f"⚠️ Warning: Could not check models status: {str(e)}")
    
    def _create_error_response(self, error_msg: str, details: str) -> Dict[str, Any]:
        """
        Create error response with error component.
        
        Args:
            error_msg: Main error message
            details: Detailed error information
            
        Returns:
            Dictionary containing error UI component
        """
        try:
            from smartcash.ui.components.error.error_component import create_error_component
            error_ui = create_error_component(
                error_msg, 
                details, 
                "Pretrained Models Error"
            )
            return {
                'ui': error_ui,
                'error': True,
                'error_message': error_msg
            }
        except Exception:
            # Fallback error display
            import ipywidgets as widgets
            error_ui = widgets.HTML(f"<div style='color: red; padding: 20px;'><h3>❌ {error_msg}</h3><p>{details}</p></div>")
            return {
                'ui': error_ui,
                'error': True,
                'error_message': error_msg
            }


def _pretrained_initialize_legacy(config: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Legacy function wrapper for pretrained module initialization.
    
    Args:
        config: Optional configuration dictionary
        **kwargs: Additional parameters
        
    Returns:
        Dictionary containing UI components
    """
    initializer = PretrainedInitializer()
    return initializer._initialize_impl(config, **kwargs)


# Create the standard UI display function using DisplayInitializer pattern
initialize_pretrained_ui = create_ui_display_function(
    module_name='pretrained',
    parent_module='model',
    initializer_class=PretrainedInitializer,
    legacy_function=_pretrained_initialize_legacy
)