"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Fixed progress handlers dengan proper progress_tracking integration
"""

from typing import Dict, Any, Callable

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking handlers dengan proper integration ke progress_tracking."""
    
    logger = ui_components.get('logger')
    
    try:
        # Ensure progress_container exists dan visible untuk testing
        _ensure_progress_container_ready(ui_components)
        
        # Setup observer system untuk progress events
        _setup_progress_observers(ui_components)
        
        # Register manual control functions yang sesuai dengan progress_tracking
        ui_components['_progress_controls'] = {
            'start': lambda msg: start_progress(ui_components, msg),
            'update_overall': lambda prog, total, msg: update_overall_progress(ui_components, prog, total, msg),
            'update_step': lambda step, total, name: update_step_progress(ui_components, step, total, name),
            'update_current': lambda curr, total, msg: update_current_progress(ui_components, curr, total, msg),
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

def _ensure_progress_container_ready(ui_components: Dict[str, Any]) -> None:
    """Ensure progress container exists dan dalam state yang tepat."""
    try:
        # Check apakah progress_container ada
        if 'progress_container' not in ui_components:
            if ui_components.get('logger'):
                ui_components['logger'].warning("⚠️ progress_container tidak ditemukan")
            return
        
        container = ui_components['progress_container']
        
        # Jika container adalah dict dengan method show_container
        if isinstance(container, dict):
            if 'show_container' in container and 'hide_container' in container:
                # Ready - ini dari progress_tracking.py
                pass
            else:
                if ui_components.get('logger'):
                    ui_components['logger'].warning("⚠️ progress_container dict tidak memiliki show/hide methods")
        
        # Jika container adalah widget
        elif hasattr(container, 'layout'):
            # Ready - widget biasa
            pass
        else:
            if ui_components.get('logger'):
                ui_components['logger'].warning("⚠️ progress_container format tidak dikenali")
                
    except Exception as e:
        if ui_components.get('logger'):
            ui_components['logger'].debug(f"📦 Error checking progress container: {str(e)}")

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
                    if 'step_progress' in kwargs:
                        step_progress = kwargs.get('step_progress', 0)
                        step_message = kwargs.get('message', '')
                        step_name = kwargs.get('step_name', 'Processing')
                        current_step = kwargs.get('current_step', 1)
                        total_steps = kwargs.get('total_steps', 3)
                        self.update_step_progress(current_step, total_steps, step_name)
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
            start_progress(self.ui_components, message)
        
        def update_overall_progress(self, progress: int, total: int, message: str) -> None:
            update_overall_progress(self.ui_components, progress, total, message)
        
        def update_step_progress(self, step: int, total_steps: int, step_name: str = "Step") -> None:
            update_step_progress(self.ui_components, step, total_steps, step_name)
            
        def update_current_progress(self, current: int, total: int, message: str) -> None:
            update_current_progress(self.ui_components, current, total, message)
        
        def complete_progress(self, message: str) -> None:
            complete_progress(self.ui_components, message)
        
        def error_progress(self, message: str) -> None:
            error_progress(self.ui_components, message)
    
    return ProgressObserver(ui_components)

# Manual control functions yang match dengan progress_tracking.py structure
def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking dengan proper container showing."""
    # Show progress container menggunakan method dari progress_tracking
    if 'progress_container' in ui_components:
        container = ui_components['progress_container']
        if isinstance(container, dict) and 'show_container' in container:
            container['show_container']()
        elif hasattr(container, 'layout'):
            container.layout.visibility = 'visible'
            container.layout.display = 'block'
    
    # Reset semua progress menggunakan progress_tracking functions
    try:
        from smartcash.ui.components.progress_tracking import (
            update_overall_progress, update_step_progress, update_current_progress
        )
        
        # Start dengan progress 0 dan message
        update_overall_progress(ui_components, 0, 100, message)
        update_step_progress(ui_components, 0, 3, "Siap memulai")  
        update_current_progress(ui_components, 0, 100, "")
        
    except ImportError:
        # Fallback manual update
        _update_progress_widgets(ui_components, overall=0, step=0, current=0,
                               overall_msg=message, step_msg="Siap memulai", current_msg="-")

def update_overall_progress(ui_components: Dict[str, Any], progress: int, total: int, message: str) -> None:
    """Update overall progress menggunakan progress_tracking functions."""
    try:
        from smartcash.ui.components.progress_tracking import update_overall_progress as update_overall
        update_overall(ui_components, progress, total, message)
    except ImportError:
        # Fallback manual update
        _safe_update_widget(ui_components, 'overall_progress', progress, f"Overall: {progress}%")
        _safe_update_widget(ui_components, 'progress_bar', progress, f"Overall: {progress}%")  # Alias
        _safe_update_label(ui_components, 'overall_label', message)

def update_step_progress(ui_components: Dict[str, Any], step: int, total_steps: int, step_name: str = "Step") -> None:
    """Update step progress menggunakan progress_tracking functions."""
    try:
        from smartcash.ui.components.progress_tracking import update_step_progress as update_step
        update_step(ui_components, step, total_steps, step_name)
    except ImportError:
        # Fallback manual update
        percentage = int((step / max(total_steps, 1)) * 100)
        _safe_update_widget(ui_components, 'step_progress', percentage, f"Step: {step}/{total_steps}")
        _safe_update_label(ui_components, 'step_label', f"Step {step}/{total_steps}: {step_name}")

def update_current_progress(ui_components: Dict[str, Any], current: int, total: int, message: str) -> None:
    """Update current progress menggunakan progress_tracking functions."""
    try:
        from smartcash.ui.components.progress_tracking import update_current_progress as update_current
        update_current(ui_components, current, total, message)
    except ImportError:
        # Fallback manual update
        percentage = int((current / max(total, 1)) * 100)
        _safe_update_widget(ui_components, 'current_progress', percentage, f"Current: {percentage}%")
        _safe_update_label(ui_components, 'current_label', message)

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
    """Update semua progress widgets dengan mapping yang tepat."""
    # Update overall progress (dan alias)
    _safe_update_widget(ui_components, 'overall_progress', overall, f"Overall: {overall}%")
    _safe_update_widget(ui_components, 'progress_bar', overall, f"Overall: {overall}%")  # Alias
    _safe_update_label(ui_components, 'overall_label', overall_msg)
    
    # Update step progress
    _safe_update_widget(ui_components, 'step_progress', step, f"Step: {step}%")
    _safe_update_label(ui_components, 'step_label', step_msg)
    
    # Update current progress  
    _safe_update_widget(ui_components, 'current_progress', current, f"Current: {current}%")
    _safe_update_label(ui_components, 'current_label', current_msg)

def _safe_update_widget(ui_components: Dict[str, Any], key: str, value: int, description: str) -> None:
    """Safely update progress widget."""
    try:
        if key in ui_components and ui_components[key]:
            widget = ui_components[key]
            if hasattr(widget, 'value'):
                widget.value = value
            if hasattr(widget, 'description'):
                widget.description = description
    except Exception:
        pass

def _safe_update_label(ui_components: Dict[str, Any], key: str, value: str) -> None:
    """Safely update label widget."""
    try:
        if key in ui_components and ui_components[key]:
            widget = ui_components[key]
            if hasattr(widget, 'value'):
                widget.value = value
    except Exception:
        pass