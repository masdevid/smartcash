"""
Progress tracking mixin for UI modules.

Provides standard progress tracking functionality with operation container integration.
"""

from typing import Dict, Any, Optional
# Removed problematic import for now


class ProgressTrackingMixin:
    """
    Mixin providing common progress tracking functionality.
    
    This mixin provides:
    - Progress updates to operation container
    - Fallback progress tracking methods
    - Progress state management
    - Standard progress formatting
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._progress_state: Dict[str, Any] = {
            'current': 0,
            'total': 100,
            'message': '',
            'level': 'primary'
        }
    
    def update_progress(self, progress: int, message: str = "", level: str = "primary") -> None:
        """
        Update progress display.
        
        Args:
            progress: Progress value (0-100)
            message: Progress message
            level: Progress level (primary, info, warning, error)
        """
        try:
            # Update internal state
            self._progress_state.update({
                'current': progress,
                'message': message,
                'level': level
            })
            
            # Try operation manager first
            if self._operation_manager and hasattr(self._operation_manager, 'update_progress'):
                self._operation_manager.update_progress(progress, message, level)
                return
            
            # Try operation container directly
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container:
                    # Handle dict-style operation container
                    if isinstance(operation_container, dict) and 'update_progress' in operation_container:
                        operation_container['update_progress'](progress, message, level)
                        return
                    # Handle object-style operation container
                    elif hasattr(operation_container, 'update_progress'):
                        operation_container.update_progress(progress, message, level)
                        return
                
                # Try progress tracker component
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker:
                    # Handle dict-style progress tracker
                    if isinstance(progress_tracker, dict) and 'update' in progress_tracker:
                        progress_tracker['update'](progress, message)
                        return
                    # Handle object-style progress tracker
                    elif hasattr(progress_tracker, 'update'):
                        progress_tracker.update(progress, message)
                        return
            
            # NO FALLBACK to console logger - store as pending progress instead
            if not hasattr(self, '_pending_progress'):
                self._pending_progress = []
            self._pending_progress.append({'progress': progress, 'message': message, 'level': level})
                
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to update progress: {e}")
    
    def start_progress(self, message: str = "Processing...", total: int = 100) -> None:
        """
        Start progress tracking.
        
        Args:
            message: Initial progress message
            total: Total progress value
        """
        self._progress_state.update({
            'current': 0,
            'total': total,
            'message': message,
            'level': 'primary'
        })
        
        # Make progress tracker visible before updating
        if hasattr(self, '_ui_components') and self._ui_components:
            progress_tracker = self._ui_components.get('progress_tracker')
            if progress_tracker:
                # Ensure initialization
                if hasattr(progress_tracker, 'initialize') and not getattr(progress_tracker, '_initialized', False):
                    progress_tracker.initialize()
                
                # Show the tracker
                if hasattr(progress_tracker, 'show'):
                    progress_tracker.show(operation=message)
                
                # Also try operation container progress
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'update_progress'):
                    operation_container.update_progress(0, message, 'primary')
        
        self.update_progress(0, message, 'primary')
    
    def complete_progress(self, message: str = "Completed") -> None:
        """
        Complete progress tracking.
        
        Args:
            message: Completion message
        """
        total = self._progress_state.get('total', 100)
        self.update_progress(total, message, 'success')
        
        # Optionally hide progress tracker after completion (with delay)
        if hasattr(self, '_ui_components') and self._ui_components:
            progress_tracker = self._ui_components.get('progress_tracker')
            if progress_tracker and hasattr(progress_tracker, 'hide'):
                # Hide after a short delay to let user see completion
                import threading
                def delayed_hide():
                    import time
                    time.sleep(2)  # Wait 2 seconds
                    progress_tracker.hide()
                threading.Thread(target=delayed_hide, daemon=True).start()
    
    def error_progress(self, message: str = "Error occurred") -> None:
        """
        Set progress to error state.
        
        Args:
            message: Error message
        """
        self.update_progress(self._progress_state.get('current', 0), message, 'error')
    
    def increment_progress(self, increment: int = 1, message: str = None) -> None:
        """
        Increment progress by specified amount.
        
        Args:
            increment: Amount to increment progress
            message: Optional message to update
        """
        current = self._progress_state.get('current', 0)
        total = self._progress_state.get('total', 100)
        new_progress = min(current + increment, total)
        
        update_message = message or self._progress_state.get('message', '')
        self.update_progress(new_progress, update_message)
    
    def get_progress_state(self) -> Dict[str, Any]:
        """
        Get current progress state.
        
        Returns:
            Progress state dictionary
        """
        return self._progress_state.copy()
    
    def reset_progress(self) -> None:
        """Reset progress to initial state."""
        self._progress_state = {
            'current': 0,
            'total': 100,
            'message': '',
            'level': 'primary'
        }
        
        try:
            # Try to reset UI components
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'reset_progress'):
                    operation_container.reset_progress()
                    return
                
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'reset'):
                    progress_tracker.reset()
                    return
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to reset progress UI: {e}")
    
    def hide_progress(self) -> None:
        """Hide progress display."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'hide_progress'):
                    operation_container.hide_progress()
                    return
                
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'hide'):
                    progress_tracker.hide()
                    return
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to hide progress: {e}")
    
    def show_progress(self) -> None:
        """Show progress display."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'show_progress'):
                    operation_container.show_progress()
                    return
                
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker and hasattr(progress_tracker, 'show'):
                    progress_tracker.show()
                    return
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to show progress: {e}")
    
    def _initialize_progress_display(self) -> None:
        """Initialize progress display with welcome message."""
        try:
            if hasattr(self, '_ui_components') and self._ui_components:
                # Initialize progress tracker first
                progress_tracker = self._ui_components.get('progress_tracker')
                if progress_tracker:
                    # Ensure progress tracker is initialized
                    if hasattr(progress_tracker, 'initialize') and not getattr(progress_tracker, '_initialized', False):
                        progress_tracker.initialize()
                    
                    # Show progress tracker by default for operation modules
                    if hasattr(progress_tracker, 'show'):
                        progress_tracker.show()
                
                # Initialize operation container progress
                operation_container = self._ui_components.get('operation_container')
                if operation_container and hasattr(operation_container, 'initialize_progress'):
                    operation_container.initialize_progress()
                    
                # Add initialization logs if logging is available
                if hasattr(self, 'log'):
                    if hasattr(self, 'module_name'):
                        self.log(f"🚀 {self.module_name.title()} module initialized", 'info')
                        self.log(f"📊 Configuration loaded successfully", 'info')
                        self.log(f"🔧 Operation container ready", 'info')
                    else:
                        self.log("🚀 Module initialized successfully", 'info')
                    
        except Exception as e:
            if hasattr(self, 'logger'):
                self.logger.debug(f"Failed to initialize progress display: {e}")