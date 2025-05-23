"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Fixed progress handlers dengan proper observer integration dan callback availability
"""

from typing import Dict, Any, Callable

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking dengan fixed observer integration."""
    
    logger = ui_components.get('logger')
    
    try:
        # Create progress system dengan proper callback availability
        ui_components['_progress_system'] = _create_progress_system(ui_components)
        
        # Setup progress observers dengan fixed integration
        _setup_download_progress_observers(ui_components)
        
        # Mark progress as setup
        ui_components['progress_setup'] = True
        
        if logger:
            logger.info("📊 Progress system aktif")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup progress: {str(e)}")
    
    return ui_components

def _create_progress_system(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create sistem progress tracking yang fixed dan terintegrasi."""
    progress_system = {
        'current_progress': 0,
        'current_message': '',
        'current_step': '',
        'step_progress': 0,
        'is_active': False
    }
    
    # Create handlers dengan proper UI component access
    progress_system['handlers'] = {
        'start': _create_start_handler(ui_components),
        'update': _create_update_handler(ui_components),
        'step': _create_step_handler(ui_components),
        'complete': _create_complete_handler(ui_components),
        'error': _create_error_handler(ui_components)
    }
    
    return progress_system

def _setup_download_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk download progress events dengan fixed integration."""
    try:
        from smartcash.components.observer import EventDispatcher
        
        # Create progress observer yang fixed
        progress_observer = _create_fixed_progress_observer(ui_components)
        
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
                pass  # Continue dengan fallback
        
        # Store observer reference untuk callback availability
        ui_components['_progress_observer'] = progress_observer
        ui_components['_observers'] = {
            'progress': {
                'start_handler': progress_observer.start_progress,
                'overall_progress': progress_observer.update_progress,
                'complete_handler': progress_observer.complete_progress,
                'error_handler': progress_observer.error_progress
            }
        }
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Observer setup error: {str(e)}")

def _create_fixed_progress_observer(ui_components: Dict[str, Any]):
    """Create observer object untuk progress events yang fixed."""
    
    class ProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.progress_system = ui_components.get('_progress_system', {})
        
        def update(self, event_type: str, sender, **kwargs):
            """Handle progress events dengan fixed error handling."""
            try:
                handlers = self.progress_system.get('handlers', {})
                
                if event_type in ['DOWNLOAD_START']:
                    message = kwargs.get('message', 'Memulai proses')
                    handlers.get('start', lambda x: None)(message)
                
                elif event_type in ['DOWNLOAD_PROGRESS']:
                    progress = kwargs.get('progress', 0)
                    message = kwargs.get('message', 'Processing...')
                    handlers.get('update', lambda x, y: None)(progress, message)
                
                elif event_type in ['DOWNLOAD_STEP_PROGRESS']:
                    step_name = kwargs.get('step_name', 'Step')
                    progress = kwargs.get('step_progress', 0)
                    message = kwargs.get('step_message', '')
                    handlers.get('step', lambda x, y, z: None)(step_name, progress, message)
                
                elif event_type in ['DOWNLOAD_COMPLETE']:
                    message = kwargs.get('message', 'Proses selesai')
                    duration = kwargs.get('duration', 0)
                    handlers.get('complete', lambda x, y: None)(message, duration)
                
                elif event_type in ['DOWNLOAD_ERROR']:
                    message = kwargs.get('message', 'Terjadi error')
                    handlers.get('error', lambda x: None)(message)
                    
            except Exception:
                pass  # Ignore observer errors
        
        # Helper methods untuk manual progress control
        def start_progress(self, message: str) -> None:
            """Start progress manually."""
            handlers = self.progress_system.get('handlers', {})
            handlers.get('start', lambda x: None)(message)
        
        def update_progress(self, value: int, message: str) -> None:
            """Update progress manually."""
            handlers = self.progress_system.get('handlers', {})
            handlers.get('update', lambda x, y: None)(value, message)
        
        def complete_progress(self, message: str, duration: float = 0) -> None:
            """Complete progress manually."""
            handlers = self.progress_system.get('handlers', {})
            handlers.get('complete', lambda x, y: None)(message, duration)
        
        def error_progress(self, message: str) -> None:
            """Error progress manually."""
            handlers = self.progress_system.get('handlers', {})
            handlers.get('error', lambda x: None)(message)
    
    return ProgressObserver(ui_components)

def _create_start_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk start progress dengan fixed UI updates."""
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
    
    return handler

def _create_update_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk progress update dengan fixed UI updates."""
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
    """Create handler untuk step progress dengan fixed UI updates.""" 
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
    """Create handler untuk complete progress dengan fixed UI updates."""
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
    
    return handler

def _create_error_handler(ui_components: Dict[str, Any]) -> Callable:
    """Create handler untuk error progress dengan fixed UI updates."""
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

# Helper functions untuk manual control dengan fixed availability
def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['start_handler'](message)

def update_progress(ui_components: Dict[str, Any], value: int, message: str = None) -> None:
    """Update progress secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['overall_progress'](value, message or f"Progress: {value}%")

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai", duration: float = 0) -> None:
    """Complete progress secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['complete_handler'](message, duration)

def error_progress(ui_components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Set error state secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['error_handler'](message)