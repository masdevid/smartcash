"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Fixed progress handlers dengan observer pattern yang sederhana dan reliable
"""

from typing import Dict, Any, Callable

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking dengan observer pattern yang disederhanakan."""
    
    try:
        # Create simple observer manager replacement
        ui_components['_observers'] = {}
        
        # Setup progress observers
        _setup_progress_observers(ui_components)
        
        # Mark progress as setup
        ui_components['progress_setup'] = True
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ Error setup progress handlers: {str(e)}")
    
    return ui_components

def _setup_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup progress observers dengan implementasi sederhana."""
    
    # Progress callback function yang akan dipanggil dari service layer
    def progress_callback(step: str, current: int, total: int, message: str) -> None:
        """Callback untuk update progress dari service layer."""
        try:
            # Update overall progress
            _update_overall_progress(ui_components, current, message)
            
            # Update step progress jika ada info step
            if step:
                _update_step_progress(ui_components, step, current, total, message)
                
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.debug(f"Progress callback error: {str(e)}")
    
    # Store callback untuk digunakan service layer
    ui_components['_progress_callback'] = progress_callback
    
    # Simple event handlers untuk compatibility
    ui_components['_observers']['progress'] = {
        'overall_progress': _create_overall_progress_handler(ui_components),
        'step_progress': _create_step_progress_handler(ui_components),
        'start_handler': _create_start_handler(ui_components),
        'complete_handler': _create_complete_handler(ui_components),
        'error_handler': _create_error_handler(ui_components)
    }

def _create_overall_progress_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk overall progress."""
    def handler(progress: int, message: str = "Processing...") -> None:
        _update_overall_progress(ui_components, progress, message)
    return handler

def _create_step_progress_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk step progress."""
    def handler(step: str, progress: int, total: int, message: str) -> None:
        _update_step_progress(ui_components, step, progress, total, message)
    return handler

def _create_start_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk start event."""
    def handler(message: str = "Memulai proses...") -> None:
        # Show progress container
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.display = 'block'
            ui_components['progress_container'].layout.visibility = 'visible'
        
        _update_overall_progress(ui_components, 0, message)
        
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"🚀 {message}")
    return handler

def _create_complete_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk complete event.""" 
    def handler(message: str = "Proses selesai", duration: float = 0) -> None:
        _update_overall_progress(ui_components, 100, message)
        
        logger = ui_components.get('logger')
        if logger:
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
            logger.success(f"✅ {message}{duration_str}")
    return handler

def _create_error_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk error event."""
    def handler(message: str = "Terjadi error") -> None:
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ {message}")
    return handler

def _update_overall_progress(ui_components: Dict[str, Any], progress: int, message: str) -> None:
    """Update overall progress bar."""
    progress = max(0, min(100, progress))
    
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress
        ui_components['progress_bar'].description = f"Progress: {progress}%"
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    if 'overall_label' in ui_components and message:
        ui_components['overall_label'].value = message
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'

def _update_step_progress(ui_components: Dict[str, Any], step: str, progress: int, total: int, message: str) -> None:
    """Update step progress bar."""
    step_progress = int((progress / total) * 100) if total > 0 else progress
    step_progress = max(0, min(100, step_progress))
    
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = step_progress
        ui_components['current_progress'].description = f"Step: {step}"
        if hasattr(ui_components['current_progress'], 'layout'):
            ui_components['current_progress'].layout.visibility = 'visible'
    
    if 'step_label' in ui_components and message:
        ui_components['step_label'].value = f"{step}: {message}"
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'visible'

# Helper functions untuk manual progress control
def show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan progress container dan set pesan awal."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['start_handler'](message)

def update_progress(ui_components: Dict[str, Any], value: int, message: str = None) -> None:
    """Update progress bar secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['overall_progress'](value, message or f"Progress: {value}%")

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai", duration: float = 0) -> None:
    """Complete progress dengan pesan."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['complete_handler'](message, duration)

def error_progress(ui_components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Show error state."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['error_handler'](message)

def reset_progress(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicator."""
    for widget_key in ['progress_bar', 'current_progress']:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'hidden'
    
    for label_key in ['overall_label', 'step_label']:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'hidden'
    
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'none'