# smartcash/ui/core/shared/ui_component_manager.py
"""
UI Component Manager for managing shared UI components across modules.
Provides centralized management of status panel, summary container, and other shared components.
"""
from typing import Dict, Any, Optional, List
import logging
import ipywidgets as widgets

from smartcash.ui.core.shared.logger import get_ui_logger
from smartcash.ui.components.status_panel import create_status_panel, update_status_panel
from smartcash.ui.components.summary_container import create_summary_container, SummaryContainer
from smartcash.ui.decorators import safe_ui_operation


class UIComponentManager:
    """
    Manager for shared UI components across modules.
    
    This class provides centralized management of status panel, summary container,
    and other shared UI components that are used across multiple modules.
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the UI component manager.
        
        Args:
            ui_components: Dictionary of UI components
            logger: Optional logger instance
        """
        self.ui_components = ui_components
        self.logger = logger or get_ui_logger(__name__)
    
    @safe_ui_operation(operation_name="update_status_panel", log_level="error")
    def update_status_panel(self, message: str, status_type: str = "info") -> Dict[str, Any]:
        """
        Update the status panel with the given message and status type.
        
        Args:
            message: Message to display in the status panel
            status_type: Type of status (info, success, warning, error)
            
        Returns:
            Dict with update status
        """
        try:
            if 'status_panel' not in self.ui_components:
                # Create a new status panel if it doesn't exist
                self.ui_components['status_panel'] = create_status_panel(
                    message=message,
                    status_type=status_type
                )
            else:
                # Update existing status panel
                update_status_panel(
                    panel=self.ui_components['status_panel'],
                    message=message,
                    status_type=status_type
                )
            
            return {
                "status": True,
                "message": "Status panel updated",
                "component": "status_panel"
            }
        except Exception as e:
            self.logger.error(f"Error updating status panel: {str(e)}")
            return {
                "status": False,
                "message": f"Error updating status panel: {str(e)}",
                "error": str(e),
                "component": "status_panel"
            }
    
    @safe_ui_operation(operation_name="clear_status_panel", log_level="error")
    def clear_status_panel(self) -> Dict[str, Any]:
        """
        Clear the status panel.
        
        Returns:
            Dict with clear status
        """
        try:
            if 'status_panel' in self.ui_components:
                update_status_panel(
                    panel=self.ui_components['status_panel'],
                    message="",
                    status_type="info"
                )
            else:
                self.ui_components['status_panel'] = create_status_panel(
                    message="",
                    status_type="info"
                )
            
            return {
                "status": True,
                "message": "Status panel cleared",
                "component": "status_panel"
            }
        except Exception as e:
            self.logger.error(f"Error clearing status panel: {str(e)}")
            return {
                "status": False,
                "message": f"Error clearing status panel: {str(e)}",
                "error": str(e),
                "component": "status_panel"
            }
    
    @safe_ui_operation(operation_name="update_summary_container", log_level="error")
    def update_summary_container(self, content: Any, title: str = "", message_type: str = "info", icon: str = "") -> Dict[str, Any]:
        """
        Update the summary container with the given content.
        
        Args:
            content: Content to display in the summary container (can be HTML string or dict for status items)
            title: Optional title for the summary container
            message_type: Type of message (info, success, warning, danger, primary)
            icon: Optional icon to display
            
        Returns:
            Dict with update status
        """
        try:
            # Create summary container if it doesn't exist
            if 'summary_container' not in self.ui_components or not isinstance(self.ui_components['summary_container'], SummaryContainer):
                self.ui_components['summary_container'] = create_summary_container(theme=message_type, title=title, icon=icon)
            
            summary_container = self.ui_components['summary_container']
            
            # Update content based on type
            if isinstance(content, dict):
                summary_container.show_status(content, title=title, icon=icon)
            elif isinstance(content, str):
                if title:
                    summary_container.show_message(title=title, message=content, message_type=message_type, icon=icon)
                else:
                    summary_container.set_html(content, theme=message_type)
            else:
                summary_container.set_content(str(content))
            
            return {
                "status": True,
                "message": "Summary container updated",
                "component": "summary_container"
            }
        except Exception as e:
            self.logger.error(f"Error updating summary container: {str(e)}")
            return {
                "status": False,
                "message": f"Error updating summary container: {str(e)}",
                "error": str(e),
                "component": "summary_container"
            }
    
    @safe_ui_operation(operation_name="clear_summary_container", log_level="error")
    def clear_summary_container(self) -> Dict[str, Any]:
        """
        Clear the summary container.
        
        Returns:
            Dict with clear status
        """
        try:
            if 'summary_container' not in self.ui_components:
                # Create an empty summary container if it doesn't exist
                self.ui_components['summary_container'] = create_summary_container()
                return {
                    "status": True,
                    "message": "Empty summary container created",
                    "component": "summary_container"
                }
            
            # Use the clear method if it's a SummaryContainer
            if isinstance(self.ui_components['summary_container'], SummaryContainer):
                self.ui_components['summary_container'].clear()
            # Fallback for other widget types
            elif hasattr(self.ui_components['summary_container'], 'clear_output'):
                self.ui_components['summary_container'].clear_output()
            elif hasattr(self.ui_components['summary_container'], 'value'):
                self.ui_components['summary_container'].value = ""
            else:
                self.logger.warning("Summary container does not have clear method or compatible attributes")
                return {
                    "status": False,
                    "message": "Summary container does not have clear method or compatible attributes",
                    "component": "summary_container"
                }
            
            return {
                "status": True,
                "message": "Summary container cleared",
                "component": "summary_container"
            }
        except Exception as e:
            self.logger.error(f"Error clearing summary container: {str(e)}")
            return {
                "status": False,
                "message": f"Error clearing summary container: {str(e)}",
                "error": str(e),
                "component": "summary_container"
            }
    
    def reset_components(self, components: List[str] = None) -> Dict[str, Any]:
        """
        Reset specified UI components.
        
        Args:
            components: List of component names to reset. If None, reset all managed components.
                Valid values: 'status_panel', 'summary_container'
                
        Returns:
            Dict with reset status for each component
        """
        all_components = ['status_panel', 'summary_container']
        to_reset = components if components is not None else all_components
        
        results = {}
        
        if 'status_panel' in to_reset:
            results['status_panel'] = self.clear_status_panel()
        
        if 'summary_container' in to_reset:
            results['summary_container'] = self.clear_summary_container()
        
        return {
            "status": all(result.get("status", False) for result in results.values()),
            "message": "Components reset",
            "components": results
        }


# Create a singleton instance for global access
_ui_component_manager = None


def get_ui_component_manager(ui_components: Dict[str, Any] = None, 
                           logger: Optional[logging.Logger] = None) -> UIComponentManager:
    """
    Get or create the UI component manager singleton.
    
    Args:
        ui_components: Dictionary of UI components
        logger: Optional logger instance
        
    Returns:
        UIComponentManager instance
    """
    global _ui_component_manager
    
    if _ui_component_manager is None and ui_components is not None:
        _ui_component_manager = UIComponentManager(ui_components, logger)
    
    if _ui_component_manager is None:
        raise ValueError("UI component manager not initialized. Please provide ui_components.")
    
    return _ui_component_manager
