"""
File: smartcash/ui/model/evaluate/evaluation_initializer.py
Description: Evaluation module initializer following DisplayInitializer pattern
"""

from typing import Dict, Any, Optional
from IPython.display import display

from smartcash.ui.core.initializers.display_initializer import DisplayInitializer
from smartcash.ui.logger import get_module_logger
from .components.evaluation_ui import create_evaluation_ui
from .handlers.evaluation_ui_handler import EvaluationUIHandler
from .constants import UI_CONFIG


class EvaluationInitializer(DisplayInitializer):
    """Initializer for evaluation module following UI standards."""
    
    def __init__(self):
        """Initialize evaluation initializer."""
        super().__init__(
            module_name="evaluate",
            parent_module="model"
        )
        self.logger = get_module_logger(__name__)
        self.ui_handler: Optional[EvaluationUIHandler] = None
        self._ui_components: Dict[str, Any] = {}
    
    def create_ui_components(self) -> Dict[str, Any]:
        """Create evaluation UI components.
        
        Returns:
            Dictionary containing all UI components
        """
        self.logger.info("🎯 Creating evaluation UI components")
        
        try:
            # Create UI components
            self._ui_components = create_evaluation_ui()
            
            self.logger.info(f"✅ Created {len(self._ui_components)} evaluation UI components")
            return self._ui_components
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create evaluation UI components: {e}")
            self.handle_error(f"Failed to create evaluation UI: {str(e)}", exc_info=True)
            return {}
    
    def initialize_handlers(self, ui_components: Dict[str, Any]) -> bool:
        """Initialize evaluation UI handlers.
        
        Args:
            ui_components: UI components dictionary
            
        Returns:
            True if initialization successful, False otherwise
        """
        self.logger.info("🔧 Initializing evaluation UI handlers")
        
        try:
            # Create UI handler
            self.ui_handler = EvaluationUIHandler()
            
            # Setup handler with UI components
            self.ui_handler.setup(ui_components)
            
            self.logger.info("✅ Evaluation UI handlers initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize evaluation handlers: {e}")
            self.handle_error(f"Failed to initialize evaluation handlers: {str(e)}", exc_info=True)
            return False
    
    def display_ui(self, ui_components: Dict[str, Any]) -> None:
        """Display the evaluation UI.
        
        Args:
            ui_components: UI components dictionary
        """
        self.logger.info("🖥️ Displaying evaluation UI")
        
        try:
            # Get the main container from the UI components
            main_container = None
            
            # Check if main_container is in the root
            if 'main_container' in ui_components:
                main_container = ui_components['main_container']
            # Check if it's in the ui_components dictionary
            elif 'ui_components' in ui_components and 'main_container' in ui_components['ui_components']:
                main_container = ui_components['ui_components']['main_container']
            # Check if it's in the containers
            elif 'containers' in ui_components and 'main' in ui_components['containers']:
                main_container = ui_components['containers']['main']
            
            if main_container is not None:
                display(main_container.container if hasattr(main_container, 'container') else main_container)
                self.logger.info("✅ Evaluation UI displayed successfully")
            else:
                # For debugging: Log available keys
                available_keys = list(ui_components.keys())
                if 'ui_components' in ui_components:
                    available_keys.extend(f"ui_components['{k}']" for k in ui_components['ui_components'].keys())
                self.logger.warning(f"Available UI component keys: {available_keys}")
                raise ValueError("Main container not found in UI components")
                
        except Exception as e:
            self.logger.error(f"❌ Failed to display evaluation UI: {e}")
            self.handle_error(f"Failed to display evaluation UI: {str(e)}", exc_info=True)
            raise
    
    def _initialize_impl(self, **kwargs) -> Dict[str, Any]:
        """Implementation of initialization logic for DisplayInitializer.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            Dictionary containing initialization results
        """
        return self.initialize_full(**kwargs)
    
    def initialize_full(self, **kwargs) -> Dict[str, Any]:
        """Initialize the complete evaluation module.
        
        Args:
            **kwargs: Additional initialization parameters
            
        Returns:
            Dictionary containing initialization results
        """
        # Check if already initialized and return cached result
        if self._is_initialized and self._initialization_result:
            self.logger.debug("Evaluation module already initialized, returning cached result")
            return self._initialization_result
            
        self.logger.info("🚀 Initializing evaluation module")
        
        try:
            # Step 1: Create UI components
            ui_components = self.create_ui_components()
            if not ui_components:
                raise Exception("Failed to create UI components")
            
            # Step 2: Initialize handlers
            if not self.initialize_handlers(ui_components):
                raise Exception("Failed to initialize handlers")
            
            # Step 3: Display UI
            self.display_ui(ui_components)
            
            # Step 4: Return success result
            result = {
                "success": True,
                "module": "evaluate",
                "ui_components": ui_components,
                "ui_handler": self.ui_handler,
                "message": "Evaluation module initialized successfully"
            }
            
            # Cache the result
            self._initialization_result = result
            self._is_initialized = True
            
            self.logger.info("🎉 Evaluation module initialization completed successfully")
            return result
            
        except Exception as e:
            error_result = {
                "success": False,
                "module": "evaluate",
                "error": str(e),
                "message": f"Evaluation module initialization failed: {str(e)}"
            }
            
            self.logger.error(f"❌ Evaluation module initialization failed: {e}")
            self.handle_error(f"Evaluation module initialization failed: {str(e)}", exc_info=True)
            return error_result
    
    def get_ui_handler(self) -> Optional[EvaluationUIHandler]:
        """Get the evaluation UI handler instance.
        
        Returns:
            EvaluationUIHandler instance or None if not initialized
        """
        return self.ui_handler
    
    def get_ui_components(self) -> Dict[str, Any]:
        """Get the UI components dictionary.
        
        Returns:
            Dictionary of UI components
        """
        return self._ui_components.copy()
    
    def cleanup(self) -> None:
        """Cleanup evaluation module resources."""
        try:
            if self.ui_handler:
                # Cleanup handler resources
                if hasattr(self.ui_handler, 'cleanup'):
                    self.ui_handler.cleanup()
                self.ui_handler = None
            
            # Clear UI components
            self._ui_components.clear()
            
            self.logger.info("🧹 Evaluation module cleanup completed")
            
        except Exception as e:
            self.logger.error(f"❌ Error during evaluation module cleanup: {e}")
    
    def reset_state(self) -> None:
        """Reset the evaluation initializer state."""
        self._is_initialized = False
        self._initialization_result = None
        self.ui_handler = None
        self._ui_components.clear()


# Global initializer instance
_evaluation_initializer: Optional[EvaluationInitializer] = None

def get_evaluation_initializer() -> EvaluationInitializer:
    """Get or create evaluation initializer instance.
    
    Returns:
        EvaluationInitializer instance
    """
    global _evaluation_initializer
    if _evaluation_initializer is None:
        _evaluation_initializer = EvaluationInitializer()
    return _evaluation_initializer

def reset_evaluation_initializer() -> bool:
    """Reset the evaluation initializer instance.
    
    This is useful for testing or when you need to force reinitialization.
    
    Returns:
        bool: True if reset was successful, False otherwise
    """
    global _evaluation_initializer
    if _evaluation_initializer is not None:
        _evaluation_initializer.reset_state()
        _evaluation_initializer = None
        return True
    return False

def initialize_evaluation_ui(**kwargs) -> Dict[str, Any]:
    """Initialize evaluation UI module.
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results
    """
    initializer = get_evaluation_initializer()
    return initializer.initialize(**kwargs)


# Legacy function for backward compatibility
def initialize_evaluate_ui(**kwargs) -> Dict[str, Any]:
    """Initialize evaluation UI (legacy function name).
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results
    """
    return initialize_evaluation_ui(**kwargs)


# Main entry point function for cell execution
def init_evaluation_ui(**kwargs):
    """Initialize and display evaluation UI.
    
    This is the main entry point function that should be called from notebook cells.
    It creates the evaluation initializer and displays the UI directly.
    
    Args:
        **kwargs: Additional initialization parameters
        
    Returns:
        Dictionary containing initialization results and UI components
    """
    try:
        # Suppress early logging
        import logging
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(logging.CRITICAL)
        
        smartcash_logger = logging.getLogger('smartcash')
        original_smartcash_level = smartcash_logger.level
        smartcash_logger.setLevel(logging.CRITICAL)
        
        try:
            # Create and initialize
            initializer = EvaluationInitializer()
            ui_result = initializer.initialize_full(**kwargs)
            
            # Display UI components if available
            if ui_result and 'ui_components' in ui_result:
                ui_components = ui_result['ui_components']
                if 'main_container' in ui_components:
                    display(ui_components['main_container'])
                    
            return ui_result
                    
        except Exception as e:
            # Create error result
            error_result = {
                "success": False,
                "module": "evaluate",
                "error": str(e),
                "message": f"Failed to initialize evaluation UI: {str(e)}"
            }
            
            # Display error
            from IPython.display import HTML
            error_html = f"""
            <div style="color: #d32f2f; padding: 15px; border-left: 4px solid #d32f2f; 
                        margin: 10px 0; background: rgba(244, 67, 54, 0.05); border-radius: 4px;">
                <strong>🚨 Evaluation Initialization Error</strong><br>
                <div style="margin-top: 8px; font-family: monospace; font-size: 13px;">
                    {str(e)}
                </div>
            </div>
            """
            display(HTML(error_html))
            return error_result
            
        finally:
            # Always restore logging levels
            root_logger.setLevel(original_level)
            smartcash_logger.setLevel(original_smartcash_level)
            
    except Exception as e:
        print(f"❌ Critical error in evaluation initialization: {e}")
        return {
            "success": False,
            "module": "evaluate",
            "error": str(e),
            "message": f"Critical error in evaluation initialization: {str(e)}"
        }