"""
File: smartcash/ui/dataset/download/handlers/progress_handlers.py
Deskripsi: Fixed progress handlers dengan dual progress tracking yang benar dan konsisten
"""

from typing import Dict, Any, Callable

def setup_progress_handlers(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress tracking dengan dual progress system yang benar."""
    
    logger = ui_components.get('logger')
    
    try:
        # Create dual progress system
        ui_components['_progress_system'] = _create_dual_progress_system(ui_components)
        
        # Setup observers dengan fixed integration
        _setup_dual_progress_observers(ui_components)
        
        ui_components['progress_setup'] = True
        
        if logger:
            logger.info("📊 Dual progress system aktif")
        
    except Exception as e:
        if logger:
            logger.error(f"❌ Error setup progress: {str(e)}")
    
    return ui_components

def _create_dual_progress_system(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Create sistem dual progress tracking yang benar."""
    return {
        'overall_progress': 0,
        'step_progress': 0,
        'current_step': '',
        'overall_message': '',
        'step_message': '',
        'is_active': False,
        'handlers': {
            'start': _create_start_handler(ui_components),
            'overall_update': _create_overall_update_handler(ui_components),
            'step_update': _create_step_update_handler(ui_components),
            'complete': _create_complete_handler(ui_components),
            'error': _create_error_handler(ui_components)
        }
    }

def _setup_dual_progress_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observers untuk dual progress events."""
    try:
        from smartcash.components.observer import EventDispatcher
        
        progress_observer = _create_dual_progress_observer(ui_components)
        
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
                'complete_handler': progress_observer.complete_progress,
                'error_handler': progress_observer.error_progress
            }
        }
        
    except Exception as e:
        logger = ui_components.get('logger')
        if logger:
            logger.warning(f"⚠️ Observer setup error: {str(e)}")

def _create_dual_progress_observer(ui_components: Dict[str, Any]):
    """Create observer untuk dual progress events."""
    
    class DualProgressObserver:
        def __init__(self, ui_components):
            self.ui_components = ui_components
            self.progress_system = ui_components.get('_progress_system', {})
        
        def update(self, event_type: str, sender, **kwargs):
            """Handle dual progress events dengan separation yang jelas."""
            try:
                handlers = self.progress_system.get('handlers', {})
                
                if event_type == 'DOWNLOAD_START':
                    message = kwargs.get('message', 'Memulai proses')
                    handlers.get('start', lambda x: None)(message)
                
                elif event_type == 'DOWNLOAD_PROGRESS':
                    # Pisahkan overall vs step progress
                    if 'step_progress' in kwargs:
                        # Step progress update
                        step_progress = kwargs.get('step_progress', 0)
                        step_message = kwargs.get('message', '')
                        step_name = kwargs.get('step_name', 'Processing')
                        handlers.get('step_update', lambda x, y, z: None)(step_progress, step_message, step_name)
                    else:
                        # Overall progress update
                        overall_progress = kwargs.get('progress', 0)
                        overall_message = kwargs.get('message', 'Processing...')
                        handlers.get('overall_update', lambda x, y: None)(overall_progress, overall_message)
                
                elif event_type == 'DOWNLOAD_COMPLETE':
                    message = kwargs.get('message', 'Proses selesai')
                    handlers.get('complete', lambda x: None)(message)
                
                elif event_type == 'DOWNLOAD_ERROR':
                    message = kwargs.get('message', 'Terjadi error')
                    handlers.get('error', lambda x: None)(message)
                    
            except Exception:
                pass
        
        def start_progress(self, message: str) -> None:
            handlers = self.progress_system.get('handlers', {})
            handlers.get('start', lambda x: None)(message)
        
        def update_overall_progress(self, value: int, message: str) -> None:
            handlers = self.progress_system.get('handlers', {})
            handlers.get('overall_update', lambda x, y: None)(value, message)
        
        def update_step_progress(self, value: int, message: str, step_name: str = "Step") -> None:
            handlers = self.progress_system.get('handlers', {})
            handlers.get('step_update', lambda x, y, z: None)(value, message, step_name)
        
        def complete_progress(self, message: str) -> None:
            handlers = self.progress_system.get('handlers', {})
            handlers.get('complete', lambda x: None)(message)
        
        def error_progress(self, message: str) -> None:
            handlers = self.progress_system.get('handlers', {})
            handlers.get('error', lambda x: None)(message)
    
    return DualProgressObserver(ui_components)

def _create_start_handler(ui_components: Dict[str, Any]) -> Callable:
    """Handler untuk start progress dengan reset yang benar."""
    def handler(message: str = "Memulai proses...") -> None:
        # Show progress container
        if 'progress_container' in ui_components:
            ui_components['progress_container'].layout.display = 'block'
            ui_components['progress_container'].layout.visibility = 'visible'
        
        # Reset both progress bars ke 0
        _update_dual_progress_widgets(ui_components, overall=0, step=0, overall_msg=message, step_msg="Siap memulai")
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': True,
            'overall_progress': 0,
            'step_progress': 0,
            'overall_message': message,
            'step_message': "Siap memulai"
        })
    
    return handler

def _create_overall_update_handler(ui_components: Dict[str, Any]) -> Callable:
    """Handler untuk overall progress update."""
    def handler(progress: int, message: str = "Processing...") -> None:
        progress = max(0, min(100, progress))
        
        # Update hanya overall progress bar
        if 'progress_bar' in ui_components:
            ui_components['progress_bar'].value = progress
            ui_components['progress_bar'].description = f"Overall: {progress}%"
            ui_components['progress_bar'].layout.visibility = 'visible'
        
        if 'overall_label' in ui_components:
            ui_components['overall_label'].value = message
            ui_components['overall_label'].layout.visibility = 'visible'
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'overall_progress': progress,
            'overall_message': message
        })
    
    return handler

def _create_step_update_handler(ui_components: Dict[str, Any]) -> Callable:
    """Handler untuk step progress update yang terpisah."""
    def handler(progress: int, message: str, step_name: str = "Step") -> None:
        progress = max(0, min(100, progress))
        
        # Update hanya step progress bar
        if 'current_progress' in ui_components:
            ui_components['current_progress'].value = progress
            ui_components['current_progress'].description = f"Step: {progress}%"
            ui_components['current_progress'].layout.visibility = 'visible'
        
        if 'step_label' in ui_components:
            ui_components['step_label'].value = f"{step_name}: {message}"
            ui_components['step_label'].layout.visibility = 'visible'
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'step_progress': progress,
            'step_message': message,
            'current_step': step_name
        })
    
    return handler

def _create_complete_handler(ui_components: Dict[str, Any]) -> Callable:
    """Handler untuk complete progress."""
    def handler(message: str = "Proses selesai") -> None:
        # Set both progress bars ke 100%
        _update_dual_progress_widgets(ui_components, overall=100, step=100, overall_msg=message, step_msg="Selesai")
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': False,
            'overall_progress': 100,
            'step_progress': 100,
            'overall_message': message,
            'step_message': "Selesai"
        })
    
    return handler

def _create_error_handler(ui_components: Dict[str, Any]) -> Callable:
    """Handler untuk error progress."""
    def handler(message: str = "Terjadi error") -> None:
        # Reset both progress bars ke 0
        _update_dual_progress_widgets(ui_components, overall=0, step=0, overall_msg=f"❌ {message}", step_msg="Error", error=True)
        
        # Update system state
        progress_system = ui_components.get('_progress_system', {})
        progress_system.update({
            'is_active': False,
            'overall_progress': 0,
            'step_progress': 0,
            'overall_message': message,
            'step_message': "Error"
        })
    
    return handler

def _update_dual_progress_widgets(ui_components: Dict[str, Any], overall: int, step: int, 
                                 overall_msg: str, step_msg: str, error: bool = False) -> None:
    """Update both progress widgets secara terpisah."""
    # Update overall progress bar
    if 'progress_bar' in ui_components:
        ui_components['progress_bar'].value = overall
        ui_components['progress_bar'].description = "Error" if error else f"Overall: {overall}%"
        ui_components['progress_bar'].layout.visibility = 'visible'
    
    # Update step progress bar
    if 'current_progress' in ui_components:
        ui_components['current_progress'].value = step
        ui_components['current_progress'].description = "Error" if error else f"Step: {step}%"
        ui_components['current_progress'].layout.visibility = 'visible'
    
    # Update labels
    if 'overall_label' in ui_components:
        ui_components['overall_label'].value = overall_msg
        ui_components['overall_label'].layout.visibility = 'visible'
    
    if 'step_label' in ui_components:
        ui_components['step_label'].value = step_msg
        ui_components['step_label'].layout.visibility = 'visible'

# Helper functions untuk manual control
def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['start_handler'](message)

def update_overall_progress(ui_components: Dict[str, Any], value: int, message: str = None) -> None:
    """Update overall progress secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['overall_progress'](value, message or f"Overall: {value}%")

def update_step_progress(ui_components: Dict[str, Any], value: int, message: str = None, step_name: str = "Step") -> None:
    """Update step progress secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['step_progress'](value, message or f"{step_name}: {value}%", step_name)

def complete_progress(ui_components: Dict[str, Any], message: str = "Selesai") -> None:
    """Complete progress secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['complete_handler'](message)

def error_progress(ui_components: Dict[str, Any], message: str = "Terjadi error") -> None:
    """Set error state secara manual."""
    if '_observers' in ui_components and 'progress' in ui_components['_observers']:
        ui_components['_observers']['progress']['error_handler'](message)