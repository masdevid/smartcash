# smartcash/ui/core/handlers/operation_handler.py
"""Operation handler class for managing long-running operations in SmartCash UI.
Extends BaseHandler with operation-specific capabilities."""
from typing import Dict, Any, Optional, Callable, List, Set
import logging
from enum import Enum, auto
from abc import ABC, abstractmethod

from smartcash.ui.core.handlers.base_handler import BaseHandler
from smartcash.ui.core.shared.logger import UILogger, get_ui_logger
from smartcash.ui.core.shared.error_handler import UIErrorHandler, get_ui_error_handler
from smartcash.ui.core.shared.ui_component_manager import UIComponentManager, get_ui_component_manager
from smartcash.ui.decorators import safe_ui_operation, safe_progress_operation

# UI Component imports
from smartcash.ui.components.progress_tracker.factory import (
    create_single_progress_tracker,
    create_dual_progress_tracker,
    create_triple_progress_tracker,
)
from smartcash.ui.components.progress_tracker.types import ProgressConfig, ProgressLevel
from smartcash.ui.components.dialog.confirmation_dialog import show_confirmation_dialog, create_confirmation_area
from smartcash.ui.components.log_accordion import create_log_accordion, LogLevel


class ProgressStages(Enum):
    """Minimal progress stages for operations that can be extended by subclasses."""
    INIT = auto()
    COMPLETE = auto()
    ERROR = auto()
    
    @classmethod
    def get_default_stages(cls) -> List[str]:
        """Get default stage names as strings."""
        return [stage.name.capitalize() for stage in cls]
    
    @classmethod
    def get_default_weights(cls) -> Dict[str, int]:
        """Get default stage weights."""
        stages = cls.get_default_stages()
        return {stage: 1 for stage in stages}
        
    @classmethod
    def extend_with_custom_stages(cls, custom_stages: Dict[str, int]) -> Dict[str, int]:
        """Extend default stages with custom stages.
        
        Args:
            custom_stages: Dictionary of custom stage names and weights
            
        Returns:
            Combined dictionary of default and custom stages with weights
        """
        stages = cls.get_default_weights()
        stages.update(custom_stages)
        return stages


class OperationHandler(BaseHandler, ABC):
    """
    Handler for managing long-running operations in SmartCash UI.
    
    This class extends BaseHandler with operation-specific capabilities,
    including progress tracking, dialog management, and summary generation.
    
    Features:
    - Integrated progress tracking (single, dual, triple levels)
    - Dialog management for confirmations and notifications
    - Status panel updates
    - Log accordion integration
    - UI state management (enable/disable buttons, reset states)
    """
    
    def __init__(
        self,
        ui_components: Dict[str, Any],
        operation_name: str,
        parent_module: str,
        module_name: str,
        logger: Optional[logging.Logger] = None,
        progress_level: ProgressLevel = ProgressLevel.SINGLE,
        auto_setup: bool = True
    ):
        """
        Initialize the operation handler.
        
        Args:
            ui_components: Dictionary containing UI components
            operation_name: Name of the operation
            parent_module: Parent module name (e.g., 'dataset', 'setup')
            module_name: Module name (e.g., 'downloader', 'env_config')
            logger: Optional logger instance
            progress_level: Level of progress tracking (SINGLE, DUAL, TRIPLE)
            auto_setup: Whether to automatically set up UI components
        """
        self.operation_name = operation_name
        self.parent_module = parent_module
        self.module_name = module_name
        
        # Operation state
        self.is_running = False
        self.is_completed = False
        self.is_cancelled = False
        self.has_error = False
        
        # UI components
        self.progress_tracker = None
        self.progress_level = progress_level
        self.progress_stages = ProgressStages
        self.controlled_buttons = set()
        
        # UI component manager for shared components
        self.ui_component_manager = get_ui_component_manager(ui_components, logger)
        
        # Call parent constructor with centralized logger
        super().__init__(ui_components, logger)
        
        # Auto setup if requested
        if auto_setup:
            self.setup()
    
    def setup(self) -> None:
        """Set up the operation handler after initialization."""
        # Ensure log components exist
        self._ensure_log_components()
        
        # Set up UI components
        self._setup_progress_tracker()
        self._setup_confirmation_area()
        
        # Status panel is managed by UIComponentManager
        self.ui_component_manager.update_status_panel("Ready", "info")
        
        # Log setup completion
        self.logger.info(f"Operation handler for '{self.operation_name}' has been set up")
    
    def _ensure_log_components(self) -> None:
        """Ensure log output and accordion components exist."""
        if 'log_output' not in self.ui_components:
            self.ui_components['log_output'] = create_log_accordion(
                module_name=f"{self.parent_module}.{self.module_name}",
                height='250px',
                auto_scroll=True,
                enable_deduplication=True
            )['log_output']
        
        if 'log_accordion' not in self.ui_components:
            self.ui_components['log_accordion'] = create_log_accordion(
                module_name=f"{self.parent_module}.{self.module_name}",
                height='250px',
                auto_scroll=True,
                enable_deduplication=True
            )['log_accordion']
    
    def _setup_progress_tracker(self) -> None:
        """Set up the progress tracker based on the configured level."""
        if self.progress_level == ProgressLevel.SINGLE:
            self.progress_tracker = create_single_progress_tracker(
                operation=self.operation_name,
                auto_hide=False
            )
        elif self.progress_level == ProgressLevel.DUAL:
            self.progress_tracker = create_dual_progress_tracker(
                operation=self.operation_name,
                auto_hide=False
            )
        elif self.progress_level == ProgressLevel.TRIPLE:
            # Use default stages from ProgressStages enum
            stages = self.progress_stages.get_default_stages()
            weights = self.progress_stages.get_default_weights()
            
            self.progress_tracker = create_triple_progress_tracker(
                operation=self.operation_name,
                steps=stages,
                step_weights=weights,
                auto_hide=False
            )
        
        # Add to UI components
        if self.progress_tracker:
            self.ui_components['progress_tracker'] = self.progress_tracker.get_ui_component()
    
    # Dialog management methods
    def show_confirmation_dialog(self, title: str, message: str, 
                              on_confirm: Optional[Callable] = None,
                              on_cancel: Optional[Callable] = None,
                              confirm_text: str = "Confirm",
                              cancel_text: str = "Cancel",
                              danger_mode: bool = False) -> None:
        """Show a confirmation dialog with the given title and message."""
        show_confirmation_dialog(
            ui_components=self.ui_components,
            title=title,
            message=message,
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text=confirm_text,
            cancel_text=cancel_text,
            danger_mode=danger_mode
        )
    
    # Progress tracker methods
    def update_progress(self, step: str = None, progress: float = None, 
                      message: str = None, level: int = 0) -> None:
        """Update the progress tracker."""
        if self.progress_tracker:
            self.progress_tracker.update(
                step=step,
                progress=progress,
                message=message,
                level=level
            )
    
    def complete_progress(self, message: str = "Operation completed successfully") -> None:
        """Mark the progress tracker as complete."""
        if self.progress_tracker:
            self.progress_tracker.complete(message=message)
    
    # Status panel methods (delegated to UI component manager)
    def update_status(self, message: str, status_type: str = "info") -> None:
        """Update the status panel with the given message and status type."""
        self.ui_component_manager.update_status_panel(message, status_type)
    
    def clear_status(self) -> None:
        """Clear the status panel."""
        self.ui_component_manager.clear_status_panel()
        
    # Summary container methods (delegated to UI component manager)
    def update_summary(self, content: Any, title: str = "", message_type: str = "info", icon: str = "") -> None:
        """Update the summary container with the given content."""
        self.ui_component_manager.update_summary_container(content, title, message_type, icon)
    
    def clear_summary(self) -> None:
        """Clear the summary container."""
        self.ui_component_manager.clear_summary_container()
    
    # Log accordion methods
    def clear_logs(self) -> None:
        """Clear the log accordion."""
        if 'log_output' in self.ui_components and hasattr(self.ui_components['log_output'], 'clear_output'):
            self.ui_components['log_output'].clear_output()
        
        if 'log_accordion' in self.ui_components and hasattr(self.ui_components['log_accordion'], 'clear_output'):
            self.ui_components['log_accordion'].clear_output()
    
    # UI state management
    def register_button(self, button_name: str) -> None:
        """Register a button to be controlled by this handler."""
        self.controlled_buttons.add(button_name)
    
    def enable_buttons(self) -> None:
        """Enable all registered buttons."""
        for button_name in self.controlled_buttons:
            if button_name in self.ui_components and hasattr(self.ui_components[button_name], 'disabled'):
                self.ui_components[button_name].disabled = False
    
    def disable_buttons(self) -> None:
        """Disable all registered buttons."""
        for button_name in self.controlled_buttons:
            if button_name in self.ui_components and hasattr(self.ui_components[button_name], 'disabled'):
                self.ui_components[button_name].disabled = True
    
    def reset_ui_state(self, components: List[str] = None, exclude: List[str] = None) -> None:
        """Reset the UI state to its initial state.
        
        Args:
            components: Specific components to reset. If None, reset all components.
                Valid values: 'logs', 'status', 'progress', 'buttons', 'dialog', 'summary', 'operation_state'
            exclude: Components to exclude from reset. Ignored if components is provided.
                Valid values: same as components
        """
        # Default components to reset
        all_components = ['logs', 'status', 'progress', 'buttons', 'dialog', 'summary', 'operation_state']
        
        # Determine which components to reset
        to_reset = components if components is not None else all_components
        if components is None and exclude is not None:
            to_reset = [comp for comp in all_components if comp not in exclude]
        
        # Reset selected components
        if 'logs' in to_reset:
            self.clear_logs()
        
        if 'status' in to_reset:
            self.clear_status()
        
        if 'progress' in to_reset and self.progress_tracker:
            self.progress_tracker.reset()
        
        if 'buttons' in to_reset:
            self.enable_buttons()
        
        if 'dialog' in to_reset and 'confirmation_area' in self.ui_components:
            # Clear any confirmation dialogs
            if hasattr(self.ui_components['confirmation_area'], 'clear'):
                self.ui_components['confirmation_area'].clear()
        
        if 'summary' in to_reset:
            # Use UI component manager to clear summary
            self.ui_component_manager.clear_summary_container()
        
        if 'operation_state' in to_reset:
            # Reset operation state
            self.is_running = False
            self.is_completed = False
            self.is_cancelled = False
            self.has_error = False
    
    @safe_ui_operation(operation_name="pre_operation_checks", log_level="error", fallback_return=False)
    def pre_operation_checks(self) -> bool:
        """
        Perform checks before starting the operation.
        
        Returns:
            True if checks pass, False otherwise
        """
        # This method should be implemented by subclasses
        return True
    
    @safe_ui_operation(operation_name="post_operation_cleanup", log_level="error")
    def post_operation_cleanup(self) -> None:
        """Perform cleanup after the operation completes."""
        # Reset UI state
        self.enable_buttons()
        
        # Additional cleanup can be implemented by subclasses
        pass
    
    @safe_progress_operation(operation_name="run", log_level="error")
    def run(self, **kwargs) -> Dict[str, Any]:
        """
        Run the operation.
        
        Args:
            **kwargs: Additional arguments for the operation
            
        Returns:
            Dictionary with operation results
        """
        try:
            # Update state
            self.is_running = True
            self.is_completed = False
            self.is_cancelled = False
            self.has_error = False
            
            # Disable buttons during operation
            self.disable_buttons()
            
            # Update status
            self.update_status(f"Running {self.operation_name}...", "info")
            
            # Pre-operation checks
            if not self.pre_operation_checks():
                self.logger.warning("Pre-operation checks failed")
                self.update_status("Operation cancelled: pre-checks failed", "warning")
                self.has_error = True
                self.is_running = False
                self.enable_buttons()
                return {
                    "status": False,
                    "message": f"Pre-operation checks failed for '{self.operation_name}'",
                    "handler": self.__class__.__name__
                }
            
            # Run the operation
            result = self._execute_operation(**kwargs)
            
            # Update state based on result
            self.is_running = False
            self.is_completed = result.get("status", False)
            self.has_error = not result.get("status", False)
            
            # Update status
            if self.is_completed:
                self.update_status(result.get("message", "Operation completed successfully"), "success")
                self.complete_progress(result.get("message", "Operation completed successfully"))
            else:
                self.update_status(result.get("message", "Operation failed"), "error")
            
            # Perform post-operation cleanup
            self.post_operation_cleanup()
            
            return result
        except Exception as e:
            self.is_running = False
            self.has_error = True
            self.logger.error(f"Error during operation: {str(e)}")
            self.update_status(f"Operation failed: {str(e)}", "error")
            self.enable_buttons()
            return {
                "status": False, 
                "message": f"Operation failed: {str(e)}", 
                "error": str(e),
                "handler": self.__class__.__name__
            }
    
    @abstractmethod
    def _execute_operation(self, **kwargs) -> Dict[str, Any]:
        """
        Execute the operation.
        
        This method must be implemented by subclasses.
        
        Args:
            **kwargs: Additional arguments for the operation
            
        Returns:
            Dict with operation status and results
        """
        pass
    
    def cancel(self) -> Dict[str, Any]:
        """
        Cancel the operation.
        
        Returns:
            Dict with operation status
        """
        if not self.is_running:
            return {
                "status": False,
                "message": f"Operation '{self.operation_name}' is not running",
                "handler": self.__class__.__name__
            }
        
        try:
            self.is_running = False
            self.is_cancelled = True
            
            # Perform post-operation cleanup
            self.post_operation_cleanup()
            
            return {
                "status": True,
                "message": f"Operation '{self.operation_name}' cancelled",
                "handler": self.__class__.__name__
            }
        except Exception as e:
            self.logger.error(f"Error cancelling operation: {str(e)}")
            return {
                "status": False,
                "message": f"Error cancelling operation: {str(e)}",
                "error": str(e),
                "handler": self.__class__.__name__
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the current status of the operation.
        
        Returns:
            Dict with operation status
        """
        return {
            "is_running": self.is_running,
            "is_completed": self.is_completed,
            "is_cancelled": self.is_cancelled,
            "has_error": self.has_error,
            "operation_name": self.operation_name,
            "handler": self.__class__.__name__
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get the summary of the operation.
        
        Returns:
            Dict with operation summary
        """
        if not self.is_completed:
            return {
                "success": False,
                "error": f"Operation '{self.operation_name}' is not completed",
                "handler": self.__class__.__name__
            }
        
        if self.summary:
            return {
                "success": True,
                "summary": self.summary,
                "handler": self.__class__.__name__
            }
        
        return {
            "success": False,
            "error": f"No summary available for operation '{self.operation_name}'",
            "handler": self.__class__.__name__
        }
