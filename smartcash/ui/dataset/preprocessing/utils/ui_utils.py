"""
File: smartcash/ui/dataset/preprocessing/utils/ui_utils.py
Deskripsi: Optimized UI utilities untuk preprocessing handlers dengan centralized error handling
"""

from typing import Dict, Any, Optional
from smartcash.ui.handlers.error_handler import handle_ui_errors

@handle_ui_errors(error_component_title="Summary Panel Update Error", log_error=True)
def update_summary_panel(ui_components: Dict[str, Any], message: str, status_type: str = "info", title: Optional[str] = None) -> None:
    """Update summary panel dengan message dan status.
    
    Args:
        ui_components: Dictionary containing UI components
        message: Message to display
        status_type: Status type (info, success, warning, error)
        title: Optional title for the summary
    """
    summary_container = ui_components.get('summary_container')
    if not summary_container:
        return
        
    if hasattr(summary_container, 'update_status'):
        summary_container.update_status(message, status_type, title)

# Note: All UI utility functions have been moved to BaseHandler
# Use the handler methods instead:
# - handler.clear_ui_outputs() - For clearing UI outputs
# - handler.set_buttons_state() - For enabling/disabling buttons
# - handler.disable_all_buttons() - For disabling all buttons
# - handler.enable_all_buttons() - For enabling all buttons
# - handler.setup_progress_tracker() - For setting up progress tracking
# - handler.update_progress_tracker() - For updating progress
# - handler.complete_progress_tracker() - For completing progress
# - handler.error_progress_tracker() - For setting error state on progress
