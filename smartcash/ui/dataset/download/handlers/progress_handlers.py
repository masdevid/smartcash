"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Enhanced progress handlers dengan integration ke progress_tracking dan business logic preservation
"""

from typing import Dict, Any, Optional

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup enhanced progress tracking handlers dengan dynamic controls dan ProgressTracker integration.
    Maintains existing business logic while adding enhanced functionality.
    """
    
    logger = ui_components.get('logger')
    
    try:
        # Import dan setup enhanced progress tracking
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        
        # Jika belum ada progress container, buat yang baru dengan enhanced features
        if 'progress_container' not in ui_components:
            progress_components = create_progress_tracking_container()
            ui_components.update(progress_components)
            logger and logger.info("📊 Enhanced progress tracking container created with ProgressTracker")
        
        # Get tracker instance untuk enhanced functionality
        tracker = ui_components.get('tracker')
        if not tracker:
            logger and logger.warning("⚠️ ProgressTracker not found, using fallback handlers")
        
        # Setup enhanced observer system untuk progress events
        _setup_enhanced_progress_observers(ui_components)
        
        # Register enhanced control functions dengan business logic preservation
        ui_components['_progress_controls'] = {
            'start': lambda msg, operation='download': start_enhanced_progress(ui_components, msg, operation),
            'update_overall': lambda prog, total, msg: update_overall_progress_enhanced(ui_components, prog, total, msg),
            'update_step': lambda step, total, name, prog=None, msg=None: update_step_progress_enhanced(ui_components, step, total, name, prog, msg),
            'update_current': lambda curr, total, msg: update_current_progress_enhanced(ui_components, curr, total, msg),
            'complete': lambda msg='Selesai': complete_enhanced_progress(ui_components, msg),
            'error': lambda msg='Error': error_enhanced_progress(ui_components, msg),
            'reset': lambda: reset_enhanced_progress(ui_components)
        }
        
        # Enhanced download-specific progress handlers
        ui_components.update({
            'show_download_progress': lambda operation='download': show_progress_for_operation_enhanced(ui_components, operation),
            'update_download_progress': lambda **kwargs: update_download_progress_enhanced(ui_components, **kwargs),
            'complete_download_progress': lambda msg='Download selesai': complete_progress_operation_enhanced(ui_components, msg),
            'error_download_progress': lambda msg='Download error': error_progress_operation_enhanced(ui_components, msg)
        })
        
        ui_components['progress_setup'] = True
        
        logger and logger.info("📊 Enhanced download progress handlers ready with ProgressTracker integration")
        
    except Exception as e:
        logger and logger.error(f"❌ Error setup enhanced progress: {str(e)}")
        # Fallback ke basic handlers jika enhanced setup gagal
        _setup_fallback_handlers(ui_components)
    
    return ui_components

def _setup_enhanced_progress_observers(ui_components: Dict[str, Any]) -> None:
    """
    Setup enhanced observers untuk progress events dengan ProgressTracker integration.
    Maintains existing observer patterns while adding enhanced functionality.
    """
    try:
        from smartcash.components.observer import EventDispatcher
        
        progress_observer = _create_enhanced_progress_observer(ui_components)
        
        # Enhanced event mapping untuk download operations
        download_events = [
            'DOWNLOAD_START', 'DOWNLOAD_PROGRESS', 'DOWNLOAD_COMPLETE', 'DOWNLOAD_ERROR',
            'DOWNLOAD_STEP_START', 'DOWNLOAD_STEP_PROGRESS', 'DOWNLOAD_STEP_COMPLETE',
            'DOWNLOAD_FILE_START', 'DOWNLOAD_FILE_PROGRESS', 'DOWNLOAD_FILE_COMPLETE'
        ]
        
        # Register observers dengan error handling untuk business continuity
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
        logger and logger.info(f"📡 Enhanced observers registered for {len(registered_events)} events")
        
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Enhanced observer setup error: {str(e)}")

def _create_enhanced_progress_observer(ui_components: Dict[str, Any]):
    """
    Create enhanced observer untuk progress events dengan business logic preservation.
    Integrates ProgressTracker while maintaining existing event handling patterns.
    """
    
    class EnhancedProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.tracker = ui_components.get('tracker')
            self.logger = ui_components.get('logger')
            
            # Business logic preservation - track operation state
            self.current_operation = None
            self.operation_start_time = None
            self.last_progress = {}
        
        def update(self, event_type: str, sender, **kwargs):
            """
            Handle progress events dengan enhanced controls dan business logic preservation.
            Maintains existing event handling while adding ProgressTracker integration.
            """
            try:
                # Preserve existing business logic - event validation
                if not event_type or not isinstance(event_type, str):
                    return
                
                # Enhanced event handling dengan ProgressTracker integration
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
                    
                # Maintain existing business logic - event logging
                if self.logger:
                    self.logger.debug(f"🔄 Processed event: {event_type}")
                    
            except Exception as e:
                # Preserve error handling business logic
                if self.logger:
                    self.logger.error(f"❌ Observer error for {event_type}: {str(e)}")
        
        def _handle_download_start(self, **kwargs):
            """Handle download start dengan enhanced tracking."""
            message = kwargs.get('message', 'Memulai proses download')
            operation = kwargs.get('operation', 'download')
            
            # Preserve business logic - operation state tracking
            self.current_operation = operation
            import time
            self.operation_start_time = time.time()
            
            self.start_progress(message, operation)
        
        def _handle_download_progress(self, **kwargs):
            """Handle download progress dengan multi-level tracking."""
            if 'step_progress' in kwargs:
                # Step-level progress
                step_progress = kwargs.get('step_progress', 0)
                step_message = kwargs.get('message', '')
                step_name = kwargs.get('step_name', 'Processing')
                current_step = kwargs.get('current_step', 1)
                total_steps = kwargs.get('total_steps', 3)
                self.update_step_progress(current_step, total_steps, step_name, step_progress, step_message)
                
            elif 'current_progress' in kwargs:
                # Current operation progress
                current_progress = kwargs.get('current_progress', 0)
                current_message = kwargs.get('message', '')
                self.update_current_progress(current_progress, 100, current_message)
                
            else:
                # Overall progress
                overall_progress = kwargs.get('progress', 0)
                overall_message = kwargs.get('message', 'Processing...')
                self.update_overall_progress(overall_progress, 100, overall_message)
        
        def _handle_step_start(self, **kwargs):
            """Handle step start events."""
            step_name = kwargs.get('step_name', 'Step')
            step_number = kwargs.get('step_number', 1)
            message = f"Starting {step_name} (Step {step_number})"
            
            if self.tracker:
                self.tracker.update('step', 0, message)
        
        def _handle_step_progress(self, **kwargs):
            """Handle step progress events."""
            progress = kwargs.get('progress', 0)
            step_name = kwargs.get('step_name', 'Step')
            message = kwargs.get('message', f"{step_name} progress")
            
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
            """Handle download completion dengan business logic preservation."""
            message = kwargs.get('message', 'Download selesai')
            
            # Preserve business logic - completion time calculation
            if self.operation_start_time:
                import time
                duration = time.time() - self.operation_start_time
                if self.logger:
                    self.logger.info(f"✅ Operation completed in {duration:.2f}s")
            
            self.complete_progress(message)
        
        def _handle_download_error(self, **kwargs):
            """Handle download error dengan business logic preservation."""
            message = kwargs.get('message', 'Terjadi error')
            error_details = kwargs.get('error', '')
            
            # Preserve business logic - error detail logging
            if self.logger and error_details:
                self.logger.error(f"❌ Download error details: {error_details}")
            
            self.error_progress(message)
        
        # Enhanced progress control methods dengan ProgressTracker integration
        def start_progress(self, message: str, operation: str = 'download') -> None:
            start_enhanced_progress(self.ui_components, message, operation)
        
        def update_overall_progress(self, progress: int, total: int, message: str) -> None:
            update_overall_progress_enhanced(self.ui_components, progress, total, message)
        
        def update_step_progress(self, step: int, total_steps: int, step_name: str, progress: int = None, message: str = None) -> None:
            update_step_progress_enhanced(self.ui_components, step, total_steps, step_name, progress, message)
            
        def update_current_progress(self, current: int, total: int, message: str) -> None:
            update_current_progress_enhanced(self.ui_components, current, total, message)
        
        def complete_progress(self, message: str) -> None:
            complete_enhanced_progress(self.ui_components, message)
        
        def error_progress(self, message: str) -> None:
            error_enhanced_progress(self.ui_components, message)
    
    return EnhancedProgressObserver(ui_components)

# Enhanced control functions dengan ProgressTracker integration dan business logic preservation

def start_enhanced_progress(ui_components: Dict[str, Any], message: str, operation: str = 'download') -> None:
    """
    Start progress dengan enhanced controls dan ProgressTracker integration.
    Maintains existing business logic while adding enhanced functionality.
    """
    try:
        # Show container menggunakan ProgressTracker
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
        elif 'show_container' in ui_components:
            ui_components['show_container']()
        
        # Reset untuk clean state
        if 'reset_all' in ui_components:
            ui_components['reset_all']()
        
        # Initial progress update
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', 0, message)
        
        # Preserve business logic - operation start logging
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"🚀 Started {operation}: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Start progress error: {str(e)}")

def update_overall_progress_enhanced(ui_components: Dict[str, Any], progress: int, total: int, message: str):
    """
    Update overall progress dengan enhanced controls dan business logic preservation.
    """
    try:
        # Preserve business logic - input validation
        progress = max(0, progress)
        total = max(1, total)
        percentage = min(100, int((progress / total) * 100))
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('overall', percentage, message)
        
        # Preserve business logic - milestone logging
        logger = ui_components.get('logger')
        if logger and percentage in [25, 50, 75, 100]:
            logger.debug(f"📊 Overall progress milestone: {percentage}%")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update overall progress error: {str(e)}")

def update_step_progress_enhanced(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str, progress: int = None, message: str = None):
    """
    Update step progress dengan enhanced controls dan flexible progress calculation.
    """
    try:
        if 'update_progress' in ui_components:
            if progress is not None:
                # Direct progress percentage update
                progress = max(0, min(100, progress))
                step_message = message or f"Step {step}/{total_steps}: {step_name} ({progress}%)"
                ui_components['update_progress']('step', progress, step_message)
            else:
                # Step completion based progress
                total_steps = max(1, total_steps)
                step = max(0, step)
                percentage = min(100, int((step / total_steps) * 100))
                step_message = message or f"Step {step}/{total_steps}: {step_name}"
                ui_components['update_progress']('step', percentage, step_message)
        
        # Preserve business logic - step progress logging
        logger = ui_components.get('logger')
        if logger:
            logger.debug(f"🔄 Step progress: {step}/{total_steps} - {step_name}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update step progress error: {str(e)}")

def update_current_progress_enhanced(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """
    Update current progress dengan enhanced controls dan business logic preservation.
    """
    try:
        # Preserve business logic - input validation
        current = max(0, current)
        total = max(1, total)
        percentage = min(100, int((current / total) * 100))
        
        if 'update_progress' in ui_components:
            ui_components['update_progress']('current', percentage, message)
        
        # Preserve business logic - current operation logging
        logger = ui_components.get('logger')
        if logger and percentage == 100:
            logger.debug(f"✅ Current operation completed: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Update current progress error: {str(e)}")

def complete_enhanced_progress(ui_components: Dict[str, Any], message: str = "Selesai"):
    """
    Complete progress dengan enhanced controls dan business logic preservation.
    """
    try:
        if 'complete_operation' in ui_components:
            ui_components['complete_operation'](message)
        
        # Preserve business logic - completion logging
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"✅ Operation completed: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Complete progress error: {str(e)}")

def error_enhanced_progress(ui_components: Dict[str, Any], message: str = "Error"):
    """
    Set error state dengan enhanced controls dan business logic preservation.
    """
    try:
        if 'error_operation' in ui_components:
            ui_components['error_operation'](message)
        
        # Preserve business logic - error logging
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Operation error: {message}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Error progress error: {str(e)}")

def reset_enhanced_progress(ui_components: Dict[str, Any]):
    """Reset progress dengan enhanced controls."""
    try:
        if 'reset_all' in ui_components:
            ui_components['reset_all']()
        
        logger = ui_components.get('logger')
        if logger:
            logger.info("🔄 Progress reset")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Reset progress error: {str(e)}")

def update_download_progress_enhanced(ui_components: Dict[str, Any], **kwargs):
    """
    Enhanced download progress update dengan flexible parameter handling.
    Maintains existing business logic while supporting multiple progress types.
    """
    try:
        # Extract progress parameters dengan backward compatibility
        progress_type = kwargs.get('type', kwargs.get('progress_type', 'overall'))
        value = kwargs.get('value', kwargs.get('progress', 0))
        total = kwargs.get('total', 100)
        message = kwargs.get('message', 'Processing...')
        
        # Handle different progress types
        if progress_type in ['overall', 'general']:
            update_overall_progress_enhanced(ui_components, value, total, message)
        elif progress_type in ['step', 'stage']:
            step = kwargs.get('step', kwargs.get('current_step', 1))
            total_steps = kwargs.get('total_steps', kwargs.get('max_steps', 3))
            step_name = kwargs.get('step_name', kwargs.get('stage_name', 'Step'))
            update_step_progress_enhanced(ui_components, step, total_steps, step_name, value, message)
        elif progress_type in ['current', 'file', 'item']:
            update_current_progress_enhanced(ui_components, value, total, message)
        else:
            # Default ke overall progress
            update_overall_progress_enhanced(ui_components, value, total, message)
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Enhanced download progress update error: {str(e)}")

# Enhanced legacy compatibility functions dengan ProgressTracker integration
def show_progress_for_operation_enhanced(ui_components: Dict[str, Any], operation: str):
    """Enhanced show progress sesuai operation type dengan ProgressTracker."""
    try:
        if 'show_for_operation' in ui_components:
            ui_components['show_for_operation'](operation)
        elif 'show_container' in ui_components:
            ui_components['show_container']()
            
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"📊 Showing progress for operation: {operation}")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Show progress error: {str(e)}")

def complete_progress_operation_enhanced(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Enhanced complete progress operation dengan business logic preservation."""
    complete_enhanced_progress(ui_components, message)

def error_progress_operation_enhanced(ui_components: Dict[str, Any], message: str = "Error"):
    """Enhanced error state untuk progress dengan business logic preservation."""
    error_enhanced_progress(ui_components, message)

def _setup_fallback_handlers(ui_components: Dict[str, Any]):
    """Setup fallback handlers jika enhanced setup gagal."""
    try:
        # Basic fallback functions
        ui_components.update({
            'show_download_progress': lambda operation='download': None,
            'update_download_progress': lambda **kwargs: None,
            'complete_download_progress': lambda msg='Download selesai': None,
            'error_download_progress': lambda msg='Download error': None
        })
        
        logger = ui_components.get('logger')
        if logger:
            logger.warning("⚠️ Using fallback progress handlers")
            
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.error(f"❌ Fallback handler setup error: {str(e)}")

# Legacy compatibility functions (preserved for backward compatibility)
def show_progress_for_operation(ui_components: Dict[str, Any], operation: str):
    """Legacy show progress function."""
    show_progress_for_operation_enhanced(ui_components, operation)

def complete_progress_operation(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Legacy complete progress function."""
    complete_progress_operation_enhanced(ui_components, message)

def error_progress_operation(ui_components: Dict[str, Any], message: str = "Error"):
    """Legacy error progress function."""
    error_progress_operation_enhanced(ui_components, message)