"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Enhanced progress handlers dengan integration ke progress_tracking yang sudah diperbaiki
"""

from typing import Dict, Any

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup enhanced progress tracking handlers dengan dynamic controls."""
    
    logger = ui_components.get('logger')
    
    try:
        # Import dan setup enhanced progress tracking
        from smartcash.ui.components.progress_tracking import create_progress_tracking_container
        
        # Jika belum ada progress container, buat yang baru
        if 'progress_container' not in ui_components:
            progress_components = create_progress_tracking_container()
            ui_components.update(progress_components)
            logger and logger.info("📊 Enhanced progress tracking container created")
        
        # Setup observer system untuk progress events
        _setup_progress_observers(ui_components)
        
        # Register enhanced control functions
        ui_components['_progress_controls'] = {
            'start': lambda msg: start_enhanced_progress(ui_components, msg),
            'update_overall': lambda prog, total, msg: update_overall_progress_enhanced(ui_components, prog, total, msg),
            'update_step': lambda step, total, name: update_step_progress_enhanced(ui_components, step, total, name),
            'update_current': lambda curr, total, msg: update_current_progress_enhanced(ui_components, curr, total, msg),
            'complete': lambda msg: complete_enhanced_progress(ui_components, msg),
            'error': lambda msg: error_enhanced_progress(ui_components, msg)
        }
        
        ui_components['progress_setup'] = True
        
        logger and logger.info("📊 Enhanced progress handlers ready")
        
    except Exception as e:
        logger and logger.error(f"❌ Error setup enhanced progress: {str(e)}")
    
    return ui_components

def _setup_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk progress events dengan enhanced controls."""
    try:
        from smartcash.components.observer import EventDispatcher
        
        progress_observer = _create_enhanced_progress_observer(ui_components)
        
        download_events = [
            'DOWNLOAD_START', 'DOWNLOAD_PROGRESS', 'DOWNLOAD_COMPLETE', 'DOWNLOAD_ERROR'
        ]
        
        for event in download_events:
            try:
                EventDispatcher.register(event, progress_observer)
            except Exception:
                pass
        
        ui_components['_progress_observer'] = progress_observer
        
    except Exception as e:
        logger = ui_components.get('logger')
        logger and logger.warning(f"⚠️ Enhanced observer setup error: {str(e)}")

def _create_enhanced_progress_observer(ui_components: Dict[str, Any]):
    """Create enhanced observer untuk progress events."""
    
    class EnhancedProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
        
        def update(self, event_type: str, sender, **kwargs):
            """Handle progress events dengan enhanced controls."""
            try:
                if event_type == 'DOWNLOAD_START':
                    message = kwargs.get('message', 'Memulai proses')
                    self.start_progress(message)
                    
                elif event_type == 'DOWNLOAD_PROGRESS':
                    if 'step_progress' in kwargs:
                        step_progress = kwargs.get('step_progress', 0)
                        step_message = kwargs.get('message', '')
                        step_name = kwargs.get('step_name', 'Processing')
                        current_step = kwargs.get('current_step', 1)
                        total_steps = kwargs.get('total_steps', 3)
                        self.update_step_progress(current_step, total_steps, step_name, step_progress, step_message)
                    elif 'current_progress' in kwargs:
                        current_progress = kwargs.get('current_progress', 0)
                        current_message = kwargs.get('message', '')
                        self.update_current_progress(current_progress, 100, current_message)
                    else:
                        overall_progress = kwargs.get('progress', 0)
                        overall_message = kwargs.get('message', 'Processing...')
                        self.update_overall_progress(overall_progress, 100, overall_message)
                
                elif event_type == 'DOWNLOAD_COMPLETE':
                    message = kwargs.get('message', 'Proses selesai')
                    self.complete_progress(message)
                
                elif event_type == 'DOWNLOAD_ERROR':
                    message = kwargs.get('message', 'Terjadi error')
                    self.error_progress(message)
                    
            except Exception:
                pass
        
        def start_progress(self, message: str) -> None:
            start_enhanced_progress(self.ui_components, message)
        
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

# Enhanced control functions yang menggunakan dynamic progress controls
def start_enhanced_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress dengan enhanced controls."""
    if 'show_container' in ui_components:
        ui_components['show_container']()
    
    if 'reset_all' in ui_components:
        ui_components['reset_all']()
    
    if 'update_progress' in ui_components:
        ui_components['update_progress']('overall', 0, message, 'info')

def update_overall_progress_enhanced(ui_components: Dict[str, Any], progress: int, total: int, message: str):
    """Update overall progress dengan enhanced controls."""
    if 'update_progress' in ui_components:
        percentage = int((progress / max(total, 1)) * 100)
        color = 'success' if percentage >= 100 else 'info' if percentage > 0 else ''
        ui_components['update_progress']('overall', percentage, message, color)

def update_step_progress_enhanced(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str, progress: int = None, message: str = None):
    """Update step progress dengan enhanced controls."""
    if 'update_progress' in ui_components:
        if progress is not None:
            # Update dengan progress percentage langsung
            color = 'success' if progress >= 100 else 'info' if progress > 0 else ''
            step_message = message or f"Step {step}/{total_steps}: {step_name} ({progress}%)"
            ui_components['update_progress']('step', progress, step_message, color)
        else:
            # Update berdasarkan step completion
            percentage = int((step / max(total_steps, 1)) * 100)
            color = 'success' if percentage >= 100 else 'info' if percentage > 0 else ''
            step_message = message or f"Step {step}/{total_steps}: {step_name}"
            ui_components['update_progress']('step', percentage, step_message, color)

def update_current_progress_enhanced(ui_components: Dict[str, Any], current: int, total: int, message: str):
    """Update current progress dengan enhanced controls."""
    if 'update_progress' in ui_components:
        percentage = int((current / max(total, 1)) * 100)
        color = 'success' if percentage >= 100 else 'warning' if percentage > 0 else ''
        ui_components['update_progress']('current', percentage, message, color)

def complete_enhanced_progress(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress dengan enhanced controls."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](message)

def error_enhanced_progress(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state dengan enhanced controls."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)

# Legacy compatibility functions
def show_progress_for_operation(ui_components: Dict[str, Any], operation: str):
    """Show progress sesuai operation type."""
    if 'show_for_operation' in ui_components:
        ui_components['show_for_operation'](operation)

def complete_progress_operation(ui_components: Dict[str, Any], message: str = "Selesai"):
    """Complete progress operation."""
    if 'complete_operation' in ui_components:
        ui_components['complete_operation'](message)

def error_progress_operation(ui_components: Dict[str, Any], message: str = "Error"):
    """Set error state untuk progress."""
    if 'error_operation' in ui_components:
        ui_components['error_operation'](message)