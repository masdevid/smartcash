"""
Confirmation dialog handlers for dataset downloader operations.
"""
from typing import Dict, Any, Callable, Optional, Tuple
from smartcash.ui.utils.ui_logger import UILogger
from smartcash.ui.dataset.downloader.utils.button_manager import get_button_manager


class ConfirmationHandler:
    """Handler for confirmation dialogs and related operations."""

    @staticmethod
    def show_confirmation_dialog(
        ui_components: Dict[str, Any],
        title: str,
        message: str,
        confirm_callback: Callable,
        confirm_args: Tuple = (),
        confirm_kwargs: Optional[Dict] = None,
        cancel_callback: Optional[Callable] = None,
        cancel_args: Tuple = (),
        cancel_kwargs: Optional[Dict] = None,
        danger_mode: bool = False
    ) -> None:
        """Show a confirmation dialog with consistent styling and behavior.
        
        Args:
            ui_components: Dictionary containing UI components
            title: Dialog title
            message: Dialog message (can be HTML)
            confirm_callback: Function to call when confirmed
            confirm_args: Positional arguments for confirm callback
            confirm_kwargs: Keyword arguments for confirm callback
            cancel_callback: Function to call when cancelled
            cancel_args: Positional arguments for cancel callback
            cancel_kwargs: Keyword arguments for cancel callback
            danger_mode: Whether to show danger styling (red for destructive actions)
        """
        from smartcash.ui.components.dialog.confirmation_dialog import (
            show_confirmation_dialog as show_dialog,
            clear_dialog_area,
            is_dialog_visible
        )
        
        try:
            # Clear any existing dialog first
            if is_dialog_visible(ui_components):
                clear_dialog_area(ui_components)
            
            # Show confirmation area
            ConfirmationHandler._show_confirmation_area(ui_components)
            
            # Default cancel callback if not provided
            if cancel_callback is None:
                cancel_callback = ConfirmationHandler._handle_default_cancel
                cancel_args = (ui_components, title.lower())
            
            # Wrap callbacks to ensure proper cleanup and error handling
            def wrapped_confirm(*args, **kwargs):
                try:
                    return confirm_callback(*args, **kwargs)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"⚠️ Error in confirmation handler: {str(e)}")
                    ConfirmationHandler._hide_confirmation_area(ui_components)
            
            def wrapped_cancel(*args, **kwargs):
                try:
                    return cancel_callback(*args, **kwargs)
                except Exception as e:
                    logger = ui_components.get('logger')
                    if logger:
                        logger.error(f"⚠️ Error in cancellation handler: {str(e)}")
                    ConfirmationHandler._hide_confirmation_area(ui_components)
            
            # Show the dialog with consistent styling
            show_dialog(
                ui_components=ui_components,
                title=title,
                message=message,
                on_confirm=wrapped_confirm,
                on_cancel=wrapped_cancel,
                confirm_text="Konfirmasi",
                cancel_text="Batal",
                danger_mode=danger_mode
            )
            
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.error(f"⚠️ Error showing confirmation dialog: {str(e)}")
            ConfirmationHandler._hide_confirmation_area(ui_components)
            raise

    @staticmethod
    def _show_confirmation_area(ui_components: Dict[str, Any]) -> None:
        """Show confirmation area with visibility management."""
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].layout.display = 'block'

    @staticmethod
    def _hide_confirmation_area(ui_components: Dict[str, Any]) -> None:
        """Hide confirmation area with visibility management."""
        if 'confirmation_area' in ui_components:
            ui_components['confirmation_area'].layout.display = 'none'

    @staticmethod
    def _handle_default_cancel(ui_components: Dict[str, Any], operation_type: str) -> None:
        """Default cancel handler that logs the cancellation and hides the dialog."""
        ConfirmationHandler._hide_confirmation_area(ui_components)
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"🚫 {' '.join(operation_type.split('_')).title()} dibatalkan oleh user")
        button_manager = get_button_manager(ui_components)
        if button_manager:
            button_manager.enable_buttons()


# Singleton instance for easy access
confirmation_handler = ConfirmationHandler()
