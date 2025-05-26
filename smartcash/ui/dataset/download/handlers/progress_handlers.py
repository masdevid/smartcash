"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Progress handlers dengan direct ProgressTracker integration
"""

from typing import Dict, Any, Optional

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking handlers dengan ProgressTracker integration."""
    
    logger = ui_components.get('logger')
    
    try:
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        
        if 'progress_container' not in ui_components:
            progress_components = create_progress_tracking_container()
            ui_components.update(progress_components)
            logger and logger.info("📊 Progress tracking container created")
        
        _setup_progress_observers(ui_components)
        
        ui_components['progress_setup'] = True
        logger and logger.info("📊 Progress handlers ready")
        
    except Exception as e:
        logger and logger.error(f"❌ Error setup progress: {str(e)}")
    
    return ui_components

def _setup_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers dengan BaseObserver inheritance."""
    try:
        from smartcash.components.observer import EventDispatcher, BaseObserver
        
        progress_observer = ProgressObserver(ui_components)
        
        download_events = [
            'DOWNLOAD_START', 'DOWNLOAD_PROGRESS', 'DOWNLOAD_COMPLETE', 'DOWNLOAD_ERROR',
            'DOWNLOAD_STEP_START', 'DOWNLOAD_STEP_PROGRESS', 'DOWNLOAD_STEP_COMPLETE',
            'DOWNLOAD_FILE_START', 'DOWNLOAD_FILE_PROGRESS', 'DOWNLOAD_FILE_COMPLETE'
        ]
        
        registered_events = []
        for event in download_events:
            try:
                EventDispatcher.register(event, progress_observer)
                registered_events.append(event)
            except Exception as e:
                logger = ui_components.get('logger')
                logger and logger.debug(f"⚠️ Could not register event {event}: {str(e)}")
        
        ui_components['_progress_observer'] = progress_observer
        ui_components['_registered_events'] = registered_events
        
        logger = ui_components.get('logger')
        logger and ui_components.get('log_output') and logger.info(f"📡 Observers registered for {len(registered_events)} events")
        
    except ImportError:
        logger = ui_components.get('logger')
        logger and ui_components.get('log_output') and logger.warning("⚠️ Observer system tidak tersedia")
    except Exception as e:
        logger = ui_components.get('logger')
        logger and ui_components.get('log_output') and logger.warning(f"⚠️ Observer setup error: {str(e)}")

class ProgressObserver:
    """BaseObserver implementation untuk progress tracking."""
    
    def __init__(self, ui_components):
        try:
            from smartcash.components.observer import BaseObserver
            super().__init__()  # Only if BaseObserver is available
        except ImportError:
            pass  # Fallback tanpa BaseObserver
            
        self.ui_components = ui_components
        self.tracker = ui_components.get('tracker')
        self.logger = ui_components.get('logger')
        self.current_operation = None
        self.operation_start_time = None
    
    def update(self, event_type: str, sender, **kwargs):
        """Handle progress events."""
        try:
            if not event_type:
                return
            
            if event_type == 'DOWNLOAD_START':
                self._handle_download_start(**kwargs)
            elif event_type == 'DOWNLOAD_PROGRESS':
                self._handle_download_progress(**kwargs)
            elif event_type == 'DOWNLOAD_STEP_START':
                self._handle_step_start(**kwargs)
            elif event_type == 'DOWNLOAD_STEP_PROGRESS':
                self._handle_step_progress(**kwargs)
            elif event_type == 'DOWNLOAD_FILE_PROGRESS':
                self._handle_file_progress(**kwargs)
            elif event_type == 'DOWNLOAD_COMPLETE':
                self._handle_download_complete(**kwargs)
            elif event_type == 'DOWNLOAD_ERROR':
                self._handle_download_error(**kwargs)
                
        except Exception as e:
            self.logger and self.logger.error(f"❌ Observer error for {event_type}: {str(e)}")
    
    def _handle_download_start(self, **kwargs):
        """Handle download start."""
        message = kwargs.get('message', 'Memulai proses download')
        operation = kwargs.get('operation', 'download')
        
        self.current_operation = operation
        import time
        self.operation_start_time = time.time()
        
        if 'show_for_operation' in self.ui_components:
            self.ui_components['show_for_operation'](operation)
        
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('overall', 0, message)
    
    def _handle_download_progress(self, **kwargs):
        """Handle download progress."""
        if 'step_progress' in kwargs:
            step_progress = kwargs.get('step_progress', 0)
            step_message = kwargs.get('message', '')
            step_name = kwargs.get('step_name', 'Processing')
            current_step = kwargs.get('current_step', 1)
            total_steps = kwargs.get('total_steps', 3)
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('step', step_progress, f"Step {current_step}/{total_steps}: {step_name}")
                
        elif 'current_progress' in kwargs:
            current_progress = kwargs.get('current_progress', 0)
            current_message = kwargs.get('message', '')
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('current', current_progress, current_message)
                
        else:
            overall_progress = kwargs.get('progress', 0)
            overall_message = kwargs.get('message', 'Processing...')
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('overall', overall_progress, overall_message)
    
    def _handle_step_start(self, **kwargs):
        """Handle step start events."""
        step_name = kwargs.get('step_name', 'Step')
        message = f"Starting {step_name}"
        
        if self.tracker:
            self.tracker.update('step', 0, message)
    
    def _handle_step_progress(self, **kwargs):
        """Handle step progress events."""
        progress = kwargs.get('progress', 0)
        message = kwargs.get('message', 'Step progress')
        
        if self.tracker:
            self.tracker.update('step', progress, message)
    
    def _handle_file_progress(self, **kwargs):
        """Handle file-level progress events."""
        progress = kwargs.get('progress', 0)
        filename = kwargs.get('filename', 'file')
        message = f"Downloading {filename}"
        
        if self.tracker:
            self.tracker.update('current', progress, message)
    
    def _handle_download_complete(self, **kwargs):
        """Handle download completion."""
        message = kwargs.get('message', 'Download selesai')
        
        if self.operation_start_time:
            import time
            duration = time.time() - self.operation_start_time
            self.logger and self.logger.info(f"✅ Operation completed in {duration:.2f}s")
        
        if 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation'](message)
    
    def _handle_download_error(self, **kwargs):
        """Handle download error."""
        message = kwargs.get('message', 'Terjadi error')
        error_details = kwargs.get('error', '')
        
        if self.logger and error_details:
            self.logger.error(f"❌ Download error details: {error_details}")
        
        if 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](message)

# Direct progress control functions

def start_progress(ui_components: Dict[str, Any], message: str, operation: str = 'download') -> None:
    """Start progress dengan ProgressTracker."""
    try:
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 0, message)
        
        logger = ui_components.get('logger')
        logger and logger.info(f"🚀 Started {operation}: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Start progress error: {str(e)}")

def update_overall_progress(ui_components: Dict[str, Any], progress: int, total: int, message: str):
    """Update overall progress."""
    try:
        percentage = min(100, int((progress / max(total, 1)) * 100))
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', percentage, message)
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update overall progress error: {str(e)}")

def update_step_progress(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str, progress: int = None, message: str = None):
    """Update step progress."""
    try:
        if 'update_progress' in ui_components:
            if progress is not None:
                step_message = message or f"Step {step}/{total_steps}: {step_name} ({progress}%)"
                ui_components['update_progress']('step', progress, step_message)
            else:
                percentage = min(100, int((step / max(total_steps, 1)) * 100))
                step_message = message or f"Step {step}/{total_steps}: {step_name}"
                ui_components['update_progress']('step', percentage, step_message)
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update step progress error: {str(e)}")

def update_current_progress(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update current progress."""
    try:
        percentage = min(100, int((current / max(total, 1)) * 100))
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('current', percentage, message)
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update current progress error: {str(e)}")

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress."""
    try:
        if 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
        
        logger = ui_components.get('logger')
        logger and logger.info(f"✅ Operation completed: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Complete progress error: {str(e)}")

def error_progress(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state."""
    try:
        if 'error_operation' in ui_components:
            ui_components['error_operation'](message)
        
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Operation error: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Error progress error: {str(e)}")

def show_progress_for_operation(ui_components: Dict[str, Any], operation: str):
    """Show progress untuk operation."""
    try:
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
        
        logger = ui_components.get('logger')
        logger and logger.info(f"📊 Showing progress for operation: {operation}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Show progress error: {str(e)}")

def complete_progress_operation(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress operation."""
    complete_progress(ui_components, message)

def error_progress_operation(ui_components: Dict[str, Any], message: str = "Error"):
    """Error state untuk progress."""
    error_progress(ui_components, message)