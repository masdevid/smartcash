"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Updated progress handlers untuk match struktur progress_tracking.py yang baru
"""

from typing import Dict, Any, Callable

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking handlers yang match dengan progress_tracking.py."""
    
    logger = ui_components.get('logger')
    
    try:
        # Setup observer system untuk progress events
        _setup_progress_observers(ui_components)
        
        # Register manual control functions
        ui_components['_progress_controls'] = {
            'start': lambda msg: start_progress(ui_components, msg),
            'update_overall': lambda val, msg: update_overall_progress(ui_components, val, msg),
            'update_step': lambda val, msg, step: update_step_progress(ui_components, val, msg, step),
            'update_current': lambda val, msg: update_current_progress(ui_components, val, msg),
            'complete': lambda msg: complete_progress(ui_components, msg),
            'error': lambda msg: error_progress(ui_components, msg)
        }
        
        ui_components['progress_setup'] = True
        
        if logger:
            logger.info("📊 Progress handlers ready")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup progress: {str(e)}")
    
    return ui_components

def _setup_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk progress events."""
    try:
        from smartcash.components.observer import EventDispatcher
        
        progress_observer = _create_progress_observer(ui_components)
        
        download_events = [
            'DOWNLOAD_START', 'DOWNLOAD_PROGRESS', 'DOWNLOAD_COMPLETE', 'DOWNLOAD_ERROR'
        ]
        
        for event in download_events:
            try:
                EventDispatcher.register(event, progress_observer)
            except Exception:
                pass
        
        ui_components['_progress_observer'] = progress_observer
        ui_components['_observers'] = {
            'progress': {
                'start_handler': progress_observer.start_progress,
                'overall_progress': progress_observer.update_overall_progress,
                'step_progress': progress_observer.update_step_progress, 
                'current_progress': progress_observer.update_current_progress,
                'complete_handler': progress_observer.complete_progress,
                'error_handler': progress_observer.error_progress
            }
        }
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Observer setup error: {str(e)}")

def _create_progress_observer(ui_components: Dict[str, Any]):
    """Create observer untuk progress events."""
    
    class ProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
        
        def update(self, event_type: str, sender, **kwargs):
            """Handle progress events."""
            try:
                if event_type == 'DOWNLOAD_START':
                    message = kwargs.get('message', 'Memulai proses')
                    self.start_progress(message)
                    
                elif event_type == 'DOWNLOAD_PROGRESS':
                    # Determine progress type berdasarkan kwargs
                    if 'step_progress' in kwargs:
                        # Step progress update
                        step_progress = kwargs.get('step_progress', 0)
                        step_message = kwargs.get('message', '')
                        step_name = kwargs.get('step_name', 'Processing')
                        self.update_step_progress(step_progress, step_message, step_name)
                    elif 'current_progress' in kwargs:
                        # Current progress update
                        current_progress = kwargs.get('current_progress', 0)
                        current_message = kwargs.get('message', '')
                        self.update_current_progress(current_progress, current_message)
                    else:
                        # Overall progress update
                        overall_progress = kwargs.get('progress', 0)
                        overall_message = kwargs.get('message', 'Processing...')
                        self.update_overall_progress(overall_progress, overall_message)
                
                elif event_type == 'DOWNLOAD_COMPLETE':
                    message = kwargs.get('message', 'Proses selesai')
                    self.complete_progress(message)
                
                elif event_type == 'DOWNLOAD_ERROR':
                    message = kwargs.get('message', 'Terjadi error')
                    self.error_progress(message)
                    
            except Exception:
                pass
        
        def start_progress(self, message: str) -> None:
            start_progress(self.ui_components, message)
        
        def update_overall_progress(self, value: int, message: str) -> None:
            update_overall_progress(self.ui_components, value, message)
        
        def update_step_progress(self, value: int, message: str, step_name: str = "Step") -> None:
            update_step_progress(self.ui_components, value, message, step_name)
            
        def update_current_progress(self, value: int, message: str) -> None:
            update_current_progress(self.ui_components, value, message)
        
        def complete_progress(self, message: str) -> None:
            complete_progress(self.ui_components, message)
        
        def error_progress(self, message: str) -> None:
            error_progress(self.ui_components, message)
    
    return ProgressObserver(ui_components)

# Manual control functions yang match dengan progress_tracking.py
def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking."""
    if 'progress_container' in ui_components:
        ui_components['progress_container']['show_container']()
    
    # Reset semua progress
    _update_progress_widgets(ui_components, 
                           overall=0, step=0, current=0,
                           overall_msg=message, step_msg="Siap memulai", current_msg="-")

def update_overall_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update overall progress (keseluruhan proses)."""
    from smartcash.ui.components.progress_tracking import update_overall_progress as update_overall
    update_overall(ui_components, progress, 100, message)

def update_step_progress(ui_components: Dict[str, Any], progress: int, message: str, step_name: str = "Step") -> None:
    """Update step progress (per split/tahap)."""
    from smartcash.ui.components.progress_tracking import update_step_progress as update_step
    # Assuming total steps context - bisa di-enhance nanti
    update_step(ui_components, 1, 3, step_name)  # 1 dari 3 steps sebagai contoh

def update_current_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update current progress (per file/batch)."""
    from smartcash.ui.components.progress_tracking import update_current_progress as update_current
    update_current(ui_components, progress, 100, message)

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai") -> None:
    """Complete progress tracking."""
    _update_progress_widgets(ui_components,
                           overall=100, step=100, current=100,
                           overall_msg=message, step_msg="Semua tahap selesai", current_msg="Selesai")

def error_progress(ui_components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Set error state."""
    _update_progress_widgets(ui_components,
                           overall=0, step=0, current=0,
                           overall_msg=f"❌ {message}", step_msg="Error", current_msg="Error")

def _update_progress_widgets(ui_components: Dict[str, Any], overall: int, step: int, current: int,
                           overall_msg: str, step_msg: str, current_msg: str) -> None:
    """Update semua progress widgets."""
    # Update overall progress
    if 'overall_progress' in ui_components and ui_components['overall_progress']:
        ui_components['overall_progress'].value = overall
    if 'overall_label' in ui_components and ui_components['overall_label']:
        ui_components['overall_label'].value = f"<div style='margin-bottom: 5px; color: #495057; font-weight: bold;'>📊 Overall: {overall_msg}</div>"
    
    # Update step progress
    if 'step_progress' in ui_components and ui_components['step_progress']:
        ui_components['step_progress'].value = step
    if 'step_label' in ui_components and ui_components['step_label']:
        ui_components['step_label'].value = f"<div style='margin: 10px 0 5px 0; color: #6c757d; font-size: 12px;'>🔄 {step_msg}</div>"
    
    # Update current progress  
    if 'current_progress' in ui_components and ui_components['current_progress']:
        ui_components['current_progress'].value = current
    if 'current_label' in ui_components and ui_components['current_label']:
        ui_components['current_label'].value = f"<div style='margin: 4px 0 5px 0; color: #868e96; font-size: 12px;'>⚡ {current_msg}</div>"