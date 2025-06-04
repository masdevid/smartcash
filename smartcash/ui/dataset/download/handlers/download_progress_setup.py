"""
File: smartcash/ui/dataset/download/handlers/download_progress_setup.py
Deskripsi: Setup progress handlers dengan observer integration dan ProgressTracker compatibility
"""

from typing import Dict, Any
try:
    from smartcash.components.observer.manager_observer import get_observer_manager
    from smartcash.components.observer.base_observer import BaseObserver
except ImportError:
    # Fallback jika observer system tidak tersedia
    get_observer_manager = lambda: None
    BaseObserver = object

def setup_download_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking handlers dengan ProgressTracker integration."""
    logger = ui_components.get('logger')
    
    try:
        # Ensure observer_manager tersedia
        if 'observer_manager' not in ui_components:
            ui_components['observer_manager'] = get_observer_manager()
            logger and logger.debug("🔄 Observer manager initialized")
        
        # Ensure progress tracking container tersedia
        if 'progress_container' not in ui_components:
            from smartcash.ui.components.progress_tracking import create_progress_tracking_container
            progress_components = create_progress_tracking_container()
            ui_components.update(progress_components)
            logger and logger.debug("📊 Progress tracking container created")
        
        # Setup progress observers
        _setup_download_progress_observers(ui_components)
        
        # Mark progress setup sebagai complete
        ui_components['_progress_setup_complete'] = True
        logger and logger.debug("✅ Progress handlers setup berhasil")
        
    except Exception as e:
        logger and logger.warning(f"⚠️ Error setup progress: {str(e)}")
        ui_components['_progress_setup_error'] = str(e)
    
    return ui_components

def _setup_download_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk progress tracking dengan BaseObserver pattern."""
    try:
        # Create DownloadProgressObserver
        progress_observer = DownloadProgressObserver(ui_components)
        
        # Get observer manager
        observer_manager = ui_components.get('observer_manager')
        if not observer_manager:
            ui_components.get('logger', print)("⚠️ Observer manager not available")
            return
            
        # Register observer for progress events
        events = [
            'download.start',
            'download.progress',
            'download.step.start',
            'download.step.progress',
            'download.step.complete',
            'download.complete',
            'download.error',
            # Legacy event names for backward compatibility
            'DOWNLOAD_START', 
            'DOWNLOAD_PROGRESS', 
            'DOWNLOAD_COMPLETE', 
            'DOWNLOAD_ERROR',
            'DOWNLOAD_STEP_START', 
            'DOWNLOAD_STEP_PROGRESS', 
            'DOWNLOAD_STEP_COMPLETE'
        ]
        
        # Register all events with the observer
        registered_events = []
        for event in events:
            try:
                if hasattr(observer_manager, 'register_observer'):
                    observer_manager.register_observer(event, progress_observer)
                    registered_events.append(event)
            except Exception as e:
                ui_components.get('logger', print)(f"⚠️ Failed to register observer for {event}: {str(e)}")
        
        if registered_events:
            ui_components.get('logger', print)(f"👀 Registered {len(registered_events)} progress observers")
        else:
            ui_components.get('logger', print)("⚠️ No progress observers were registered")
            
    except Exception as e:
        ui_components.get('logger', print)(f"⚠️ Error setting up progress observers: {str(e)}")

class DownloadProgressObserver(BaseObserver):
    """BaseObserver implementation untuk download progress tracking."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        # Initialize BaseObserver dengan nama dan priority
        super().__init__(name="DownloadProgressObserver", priority=0)
        
        self.ui_components = ui_components
        self.tracker = ui_components.get('tracker')
        self.logger = ui_components.get('logger')
        self.current_operation = None
        self.operation_start_time = None
    
    def update(self, event_type: str, sender, **kwargs):
        """Handle progress events dengan comprehensive tracking."""
        try:
            if not event_type or not event_type.startswith('DOWNLOAD_'):
                return
            
            if event_type == 'DOWNLOAD_START':
                self._handle_download_start(**kwargs)
            elif event_type == 'DOWNLOAD_PROGRESS':
                self._handle_download_progress(**kwargs)
            elif event_type == 'DOWNLOAD_STEP_START':
                self._handle_step_start(**kwargs)
            elif event_type == 'DOWNLOAD_STEP_PROGRESS':
                self._handle_step_progress(**kwargs)
            elif event_type == 'DOWNLOAD_STEP_COMPLETE':
                self._handle_step_complete(**kwargs)
            elif event_type == 'DOWNLOAD_COMPLETE':
                self._handle_download_complete(**kwargs)
            elif event_type == 'DOWNLOAD_ERROR':
                self._handle_download_error(**kwargs)
                
        except Exception as e:
            self.logger and self.logger.debug(f"❌ Progress observer error: {str(e)}")
    
    def should_process_event(self, event_type: str) -> bool:
        """Implementasi method dari BaseObserver."""
        return event_type and event_type.startswith('DOWNLOAD_')
    
    def _handle_download_start(self, **kwargs):
        """Handle download start dengan operation setup."""
        message = kwargs.get('message', 'Memulai proses download')
        operation = kwargs.get('operation', 'download')
        
        self.current_operation = operation
        import time
        self.operation_start_time = time.time()
        
        # Show progress untuk operation
        if 'show_for_operation' in self.ui_components:
            self.ui_components['show_for_operation'](operation)
        elif self.tracker:
            self.tracker.show(operation)
        
        # Update initial progress
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('overall', 0, message)
        elif self.tracker:
            self.tracker.update('overall', 0, message)
    
    def _handle_download_progress(self, **kwargs):
        """Handle download progress dengan multi-level tracking."""
        # Overall progress
        if 'progress' in kwargs:
            overall_progress = kwargs.get('progress', 0)
            overall_message = kwargs.get('message', 'Processing...')
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('overall', overall_progress, overall_message)
            elif self.tracker:
                self.tracker.update('overall', overall_progress, overall_message)
        
        # Step progress
        if 'step_progress' in kwargs:
            step_progress = kwargs.get('step_progress', 0)
            step_message = kwargs.get('step_message', '')
            current_step = kwargs.get('current_step', 1)
            total_steps = kwargs.get('total_steps', 3)
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('step', step_progress, f"Step {current_step}/{total_steps}: {step_message}")
            elif self.tracker:
                self.tracker.update('step', step_progress, f"Step {current_step}/{total_steps}: {step_message}")
        
        # Current progress (file-level)
        if 'current_progress' in kwargs:
            current_progress = kwargs.get('current_progress', 0)
            current_message = kwargs.get('current_message', '')
            
            if 'update_progress' in self.ui_components:
                self.ui_components['update_progress']('current', current_progress, current_message)
            elif self.tracker:
                self.tracker.update('current', current_progress, current_message)
    
    def _handle_step_start(self, **kwargs):
        """Handle step start events."""
        step_name = kwargs.get('step_name', 'Step')
        message = f"Starting {step_name}"
        
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('step', 0, message)
        elif self.tracker:
            self.tracker.update('step', 0, message)
    
    def _handle_step_progress(self, **kwargs):
        """Handle step progress events."""
        progress = kwargs.get('progress', 0)
        message = kwargs.get('message', 'Step progress')
        
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('step', progress, message)
        elif self.tracker:
            self.tracker.update('step', progress, message)
    
    def _handle_step_complete(self, **kwargs):
        """Handle step completion."""
        step_name = kwargs.get('step_name', 'Step')
        message = kwargs.get('message', f"{step_name} complete")
        
        if 'update_progress' in self.ui_components:
            self.ui_components['update_progress']('step', 100, f"✅ {message}")
        elif self.tracker:
            self.tracker.update('step', 100, f"✅ {message}")
    
    def _handle_download_complete(self, **kwargs):
        """Handle download completion."""
        message = kwargs.get('message', 'Download selesai')
        
        if self.operation_start_time:
            import time
            duration = time.time() - self.operation_start_time
            self.logger and self.logger.debug(f"✅ Download completed in {duration:.2f}s")
        
        if 'complete_operation' in self.ui_components:
            self.ui_components['complete_operation'](message)
        elif self.tracker:
            self.tracker.complete(message)
    
    def _handle_download_error(self, **kwargs):
        """Handle download error."""
        message = kwargs.get('message', 'Terjadi error')
        error_details = kwargs.get('error_details', '')
        
        if self.logger and error_details:
            self.logger.debug(f"❌ Download error details: {error_details}")
        
        if 'error_operation' in self.ui_components:
            self.ui_components['error_operation'](message)
        elif self.tracker:
            self.tracker.error(message)

# Direct progress control functions untuk backward compatibility
def start_download_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress dengan ProgressTracker integration."""
    try:
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation']('download')
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 0, message)
        
        logger = ui_components.get('logger')
        logger and logger.debug(f"🚀 Started download progress: {message}")
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.debug(f"❌ Start progress error: {str(e)}")

def update_download_progress(ui_components: Dict[str, Any], progress_type: str, value: int, message: str):
    """Update progress dengan type-specific handling."""
    try:
        if 'update_progress' in ui_components:
            ui_components['update_progress'](progress_type, value, message)
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.debug(f"❌ Update progress error: {str(e)}")

def complete_download_progress(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress."""
    try:
        if 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
        
        logger = ui_components.get('logger')
        logger and logger.debug(f"✅ Download progress completed: {message}")
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.debug(f"❌ Complete progress error: {str(e)}")

def error_download_progress(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state."""
    try:
        if 'error_operation' in ui_components:
            ui_components['error_operation'](message)
        
        logger = ui_components.get('logger')
        logger and logger.debug(f"❌ Download progress error: {message}")
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.debug(f"❌ Error progress error: {str(e)}")

def get_progress_setup_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get status progress setup untuk debugging."""
    return {
        'setup_complete': ui_components.get('_progress_setup_complete', False),
        'setup_error': ui_components.get('_progress_setup_error'),
        'observer_manager_available': 'observer_manager' in ui_components,
        'progress_observer_available': '_progress_observer' in ui_components,
        'registered_events': ui_components.get('_registered_progress_events', []),
        'progress_methods': {
            'show_for_operation': 'show_for_operation' in ui_components,
            'update_progress': 'update_progress' in ui_components,
            'complete_operation': 'complete_operation' in ui_components,
            'error_operation': 'error_operation' in ui_components,
            'tracker': 'tracker' in ui_components
        },
        'progress_container_available': 'progress_container' in ui_components
    }