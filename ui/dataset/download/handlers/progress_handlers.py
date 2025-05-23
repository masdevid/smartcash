"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Progress handlers yang diperbaiki dengan observer integration yang reliable dan sederhana
"""

from typing import Dict, Any, Callable
from smartcash.components.observer import notify, EventTopics

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking dengan observer pattern yang sederhana dan reliable."""
    
    logger = ui_components.get('logger')
    
    try:
        # 📊 Create progress observer system
        ui_components['_progress_system'] = _create_progress_system(ui_components)
        
        # 🔗 Setup progress observers untuk berbagai event
        _setup_download_progress_observers(ui_components)
        
        # ✅ Mark progress as setup
        ui_components['progress_setup'] = True
        
        if logger:
            logger.debug("📊 Progress handlers berhasil disetup")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup progress handlers: {str(e)}")
    
    return ui_components

def _create_progress_system(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create sistem progress tracking yang terintegrasi."""
    return {
        'current_progress': 0,
        'current_message': '',
        'current_step': '',
        'step_progress': 0,
        'is_active': False,
        'handlers': {
            'start': _create_start_handler(ui_components),
            'update': _create_update_handler(ui_components),
            'step': _create_step_handler(ui_components),
            'complete': _create_complete_handler(ui_components),
            'error': _create_error_handler(ui_components)
        }
    }

def _setup_download_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk download progress events."""
    try:
        from smartcash.components.observer import EventDispatcher
        
        # Create progress observer
        progress_observer = _create_progress_observer(ui_components)
        
        # Register untuk download events
        download_events = [
            'DOWNLOAD_START',
            'DOWNLOAD_PROGRESS', 
            'DOWNLOAD_STEP_PROGRESS',
            'DOWNLOAD_COMPLETE',
            'DOWNLOAD_ERROR'
        ]
        
        for event in download_events:
            try:
                EventDispatcher.register(event, progress_observer)
            except Exception:
                # Fallback manual registration
                pass
        
        # Store observer reference
        ui_components['_progress_observer'] = progress_observer
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Observer setup error: {str(e)}")

def _create_progress_observer(ui_components: Dict[str, Any]):
    """Create observer object untuk progress events."""
    
    class ProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.progress_system = ui_components.get('_progress_system', {})
        
        def update(self, event_type: str, sender, **kwargs):
            """Handle progress events."""
            try:
                handlers = self.progress_system.get('handlers', {})
                
                if event_type in ['DOWNLOAD_START']:
                    handlers['start'](kwargs.get('message', 'Memulai proses'))
                
                elif event_type in ['DOWNLOAD_PROGRESS']:
                    progress = kwargs.get('progress', 0)
                    message = kwargs.get('message', 'Processing...')
                    handlers['update'](progress, message)
                
                elif event_type in ['DOWNLOAD_STEP_PROGRESS']:
                    step_name = kwargs.get('step_name', 'Step')
                    progress = kwargs.get('step_progress', 0)
                    message = kwargs.get('step_message', '')
                    handlers['step'](step_name, progress, message)
                
                elif event_type in ['DOWNLOAD_COMPLETE']:
                    message = kwargs.get('message', 'Proses selesai')
                    duration = kwargs.get('duration', 0)
                    handlers['complete'](message, duration)
                
                elif event_type in ['DOWNLOAD_ERROR']:
                    message = kwargs.get('message', 'Terjadi error')
                    handlers['error'](message)
                    
            except Exception as e:
                logger = self.ui_components.get('logger')
                if logger:
                    logger.debug(f"Progress observer error: {str(e)}")
    
    return ProgressObserver(ui_components)

def _create_start_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk start progress."""
    def handler(message: str = "Memulai proses...") -> None:
        # Show progress container
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.display = 'block'
            ui_components['progress_container'].layout.visibility = 'visible'
        
        # Reset dan show progress bars
        _update_progress_widgets(ui_components, 0, message)
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': True,
            'current_progress': 0,
            'current_message': message
        })
        
        logger = ui_components.get('logger')
        if logger:
            logger.info(f"🚀 {message}")
    
    return handler

def _create_update_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk progress update."""
    def handler(progress: int, message: str = "Processing...") -> None:
        # Clamp progress 0-100
        progress = max(0, min(100, progress))
        
        # Update UI widgets
        _update_progress_widgets(ui_components, progress, message)
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'current_progress': progress,
            'current_message': message
        })
    
    return handler

def _create_step_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk step progress.""" 
    def handler(step_name: str, progress: int, message: str) -> None:
        # Update step progress bar
        if 'current_progress' in ui_components:
            progress = max(0, min(100, progress))
            ui_components['current_progress'].value = progress
            ui_components['current_progress'].description = f"{step_name}: {progress}%"
            
            if hasattr(ui_components['current_progress'], 'layout'):
                ui_components['current_progress'].layout.visibility = 'visible'
        
        # Update step label
        if 'step_label' in ui_components:
            ui_components['step_label'].value = f"{step_name}: {message}"
            if hasattr(ui_components['step_label'], 'layout'):
                ui_components['step_label'].layout.visibility = 'visible'
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'current_step': step_name,
            'step_progress': progress
        })
    
    return handler

def _create_complete_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk complete progress."""
    def handler(message: str = "Proses selesai", duration: float = 0) -> None:
        # Update ke 100%
        _update_progress_widgets(ui_components, 100, message)
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': False,
            'current_progress': 100,
            'current_message': message
        })
        
        logger = ui_components.get('logger')
        if logger:
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
            logger.success(f"✅ {message}{duration_str}")
    
    return handler

def _create_error_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk error progress."""
    def handler(message: str = "Terjadi error") -> None:
        # Update progress dengan error state
        _update_progress_widgets(ui_components, 0, f"❌ {message}", error=True)
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': False,
            'current_progress': 0,
            'current_message': message
        })
        
        logger = ui_components.get('logger')
        if logger:
            logger.error(f"❌ {message}")
    
    return handler

def _update_progress_widgets(ui_components: Dict[str, Any], 
                            progress: int, message: str, error: bool = False) -> None:
    """Update semua progress widgets dengan nilai dan pesan baru."""
    # Update main progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = progress
        description = "Error" if error else f"Progress: {progress}%"
        ui_components['progress_bar'].description = description
        
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    # Update overall label
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = message
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'

# Helper functions untuk manual control
def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking secara manual."""
    progress_system = ui_components.get('_progress_system', {})
    if 'handlers' in progress_system:
        progress_system['handlers']['start'](message)

def update_progress(ui_components: Dict[str, Any], value: int, message: str = None) -> None:
    """Update progress secara manual."""
    progress_system = ui_components.get('_progress_system', {})
    if 'handlers' in progress_system:
        progress_system['handlers']['update'](value, message or f"Progress: {value}%")

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai", duration: float = 0) -> None:
    """Complete progress secara manual."""
    progress_system = ui_components.get('_progress_system', {})
    if 'handlers' in progress_system:
        progress_system['handlers']['complete'](message, duration)

def error_progress(ui_components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Set error state secara manual."""
    progress_system = ui_components.get('_progress_system', {})
    if 'handlers' in progress_system:
        progress_system['handlers']['error'](message)

def reset_progress(ui_components: Dict[str, Any]) -> None:
    """Reset semua progress indicators."""
    # Reset progress widgets
    progress_widgets = ['progress_bar', 'current_progress']
    for widget_key in progress_widgets:
        if widget_key in ui_components:
            ui_components[widget_key].value = 0
            ui_components[widget_key].description = "Progress: 0%"
            if hasattr(ui_components[widget_key], 'layout'):
                ui_components[widget_key].layout.visibility = 'hidden'
    
    # Reset labels
    label_widgets = ['overall_label', 'step_label']
    for label_key in label_widgets:
        if label_key in ui_components:
            ui_components[label_key].value = ""
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'hidden'
    
    # Hide progress container
    if 'progress_container' in ui_components:
        ui_components['progress_container'].layout.display = 'none'
    
    # Reset system state
    progress_system = ui_components.get('_progress_system', {})
    progress_system.update({
        'current_progress': 0,
        'current_message': '',
        'current_step': '',
        'step_progress': 0,
        'is_active': False
    })

def get_progress_status(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Get current progress status untuk debugging."""
    progress_system = ui_components.get('_progress_system', {})
    return {
        'is_active': progress_system.get('is_active', False),
        'current_progress': progress_system.get('current_progress', 0),
        'current_message': progress_system.get('current_message', ''),
        'current_step': progress_system.get('current_step', ''),
        'step_progress': progress_system.get('step_progress', 0)
    }