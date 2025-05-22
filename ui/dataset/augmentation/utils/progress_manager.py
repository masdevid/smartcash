"""
File: smartcash/ui/dataset/augmentation/utils/progress_manager.py
Deskripsi: Manager progress bar untuk augmentasi dataset dengan logger bridge
"""

from typing import Dict, Any, Optional, Callable
from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge

def setup_ui_logger_if_needed(ui_components: Dict[str, Any]) -> None:
    """Setup UI logger bridge jika belum ada."""
    if 'logger' not in ui_components:
        ui_logger = create_ui_logger_bridge(ui_components, "progress_manager")
        ui_components['logger'] = ui_logger

def start_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Start progress tracking dengan pesan awal."""
    setup_ui_logger_if_needed(ui_components)
    
    # Reset progress bar terlebih dahulu
    reset_progress_bar(ui_components)
    
    # Tampilkan progress container
    show_progress(ui_components, message)
    
    # Set progress awal
    update_progress(ui_components, 0, message)
    
    ui_components['logger'].info(f"📊 Progress dimulai: {message}")

def complete_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Complete progress dengan pesan akhir."""
    setup_ui_logger_if_needed(ui_components)
    
    # Set progress ke 100%
    update_progress(ui_components, 100, message)
    
    # Log completion
    ui_components['logger'].success(f"✅ Progress selesai: {message}")
    
    # Sembunyikan progress setelah delay singkat
    import time
    time.sleep(1)
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'none'

def create_progress_callback(ui_components: Dict[str, Any]) -> Callable[[int, int, str], bool]:
    """
    Buat progress callback function.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Callback function yang mengembalikan True untuk continue, False untuk stop
    """
    setup_ui_logger_if_needed(ui_components)
    
    def progress_callback(current: int, total: int, message: str = "") -> bool:
        # Cek stop request
        if ui_components.get('stop_requested', False):
            ui_components['logger'].warning("⏹️ Stop request detected dalam progress callback")
            return False
        
        # Hitung persentase
        percentage = int((current / total) * 100) if total > 0 else 0
        
        # Update progress
        progress_message = f"{message} ({current}/{total})" if message else f"Progress: {current}/{total}"
        update_progress(ui_components, percentage, progress_message)
        
        return True
    
    return progress_callback

def reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """Reset progress bar ke kondisi awal."""
    setup_ui_logger_if_needed(ui_components)
    
    try:
        if 'progress_bar' in ui_components and ui_components['progress_bar'] is not None:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = "Progress: 0%"
            if hasattr(ui_components['progress_bar'], 'layout'):
                ui_components['progress_bar'].layout.visibility = 'hidden'
        
        for label_key in ['overall_label', 'step_label', 'progress_message']:
            if label_key in ui_components and ui_components[label_key] is not None:
                ui_components[label_key].value = ""
                if hasattr(ui_components[label_key], 'layout'):
                    ui_components[label_key].layout.visibility = 'hidden'
                
        if 'current_progress' in ui_components and ui_components['current_progress'] is not None:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = "Step 0/0"
            if hasattr(ui_components['current_progress'], 'layout'):
                ui_components['current_progress'].layout.visibility = 'hidden'
                
    except Exception as e:
        ui_components['logger'].warning(f"⚠️ Gagal mereset progress bar: {str(e)}")

def show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """Tampilkan progress container dan set progress awal."""
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = 0
        ui_components['progress_bar'].description = "Progress: 0%"
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    for label_key in ['progress_message', 'step_label', 'overall_label']:
        if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
            ui_components[label_key].value = message
            if hasattr(ui_components[label_key], 'layout'):
                ui_components[label_key].layout.visibility = 'visible'

def update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """Update progress bar dan pesan."""
    value = max(0, min(100, value))
    
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = value
        ui_components['progress_bar'].description = f"Progress: {value}%"
    
    if message:
        for label_key in ['progress_message', 'step_label', 'overall_label']:
            if label_key in ui_components and hasattr(ui_components[label_key], 'value'):
                ui_components[label_key].value = message
    
    # Notify observer jika ada
    observer_manager = ui_components.get('observer_manager')
    if observer_manager:
        try:
            observer_id = ui_components.get('observer_group', 'augmentation_progress')
            observer_manager.notify(observer_id, {'progress': value, 'message': message})
        except Exception:
            pass

def setup_multi_progress(ui_components: Dict[str, Any], tracking_keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Setup multi-progress tracking."""
    if tracking_keys is None:
        tracking_keys = {
            'overall_progress': 'progress_bar',
            'step_progress': 'progress_bar',
            'overall_label': 'overall_label',
            'step_label': 'step_label'
        }
    
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    # Add utility functions ke ui_components
    if not callable(ui_components.get('update_progress')):
        ui_components['update_progress'] = lambda value, message=None: update_progress(ui_components, value, message)
    
    if not callable(ui_components.get('reset_progress')):
        ui_components['reset_progress'] = lambda: reset_progress_bar(ui_components)
    
    if not callable(ui_components.get('show_progress')):
        ui_components['show_progress'] = lambda message: show_progress(ui_components, message)
    
    if not callable(ui_components.get('start_progress')):
        ui_components['start_progress'] = lambda message: start_progress(ui_components, message)
        
    if not callable(ui_components.get('complete_progress')):
        ui_components['complete_progress'] = lambda message: complete_progress(ui_components, message)
    
    return ui_components

def setup_progress_indicator(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """Setup progress indicator jika progress bar tidak tersedia."""
    if 'progress_bar' not in ui_components:
        from ipywidgets import FloatProgress, Label, VBox, HBox
        
        progress_bar = FloatProgress(value=0, min=0, max=100, description='Loading:')
        progress_message = Label(value='Mempersiapkan...')
        progress_container = VBox([HBox([progress_bar, progress_message])])
        
        ui_components['progress_bar'] = progress_bar
        ui_components['progress_message'] = progress_message
        ui_components['progress_container'] = progress_container
        
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                children = list(ui_components['ui'].children)
                children.append(progress_container)
                ui_components['ui'].children = tuple(children)
            except:
                pass
    
    return ui_components