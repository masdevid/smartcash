"""
File: smartcash/ui/dataset/download/utils/progress_manager.py
Deskripsi: Utilitas untuk mengelola progress bar dan feedback visual pada modul download dataset
"""

from typing import Dict, Any, Optional
from smartcash.ui.dataset.download.utils.logger_helper import log_message
from smartcash.ui.dataset.download.utils.notification_manager import notify_progress

def reset_progress_bar(ui_components: Dict[str, Any]) -> None:
    """
    Reset progress bar ke kondisi awal.
    
    Args:
        ui_components: Dictionary komponen UI
    """
    try:
        # Reset progress bar
        if 'progress_bar' in ui_components and ui_components['progress_bar'] is not None:
            ui_components['progress_bar'].value = 0
            ui_components['progress_bar'].description = "Progress: 0%"
            ui_components['progress_bar'].layout.visibility = 'hidden'
        
        # Reset labels
        if 'overall_label' in ui_components and ui_components['overall_label'] is not None:
            ui_components['overall_label'].value = ""
            ui_components['overall_label'].layout.visibility = 'hidden'
            
        if 'step_label' in ui_components and ui_components['step_label'] is not None:
            ui_components['step_label'].value = ""
            ui_components['step_label'].layout.visibility = 'hidden'
            
        if 'current_progress' in ui_components and ui_components['current_progress'] is not None:
            ui_components['current_progress'].value = 0
            ui_components['current_progress'].description = "Step 0/0"
            ui_components['current_progress'].layout.visibility = 'hidden'
    except Exception as e:
        log_message(ui_components, f"Gagal mereset progress bar: {str(e)}", "warning", "⚠️")

def show_progress(ui_components: Dict[str, Any], message: str) -> None:
    """
    Tampilkan progress container dan set progress awal.
    
    Args:
        ui_components: Dictionary komponen UI
        message: Pesan progress awal
    """
    # Tampilkan progress container jika ada
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    # Setup progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = 0
        
        # Pastikan progress bar terlihat
        if hasattr(ui_components['progress_bar'], 'layout'):
            ui_components['progress_bar'].layout.visibility = 'visible'
    
    # Update label progress jika ada
    if 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'value'):
        ui_components['progress_message'].value = message
        
        # Pastikan label terlihat
        if hasattr(ui_components['progress_message'], 'layout'):
            ui_components['progress_message'].layout.visibility = 'visible'
    
    # Update label progress step jika ada
    if 'step_label' in ui_components and hasattr(ui_components['step_label'], 'value'):
        ui_components['step_label'].value = message
        
        # Pastikan label terlihat
        if hasattr(ui_components['step_label'], 'layout'):
            ui_components['step_label'].layout.visibility = 'visible'
            
    # Update overall label jika ada
    if 'overall_label' in ui_components and hasattr(ui_components['overall_label'], 'value'):
        ui_components['overall_label'].value = "Proses download..."
        
        # Pastikan label terlihat
        if hasattr(ui_components['overall_label'], 'layout'):
            ui_components['overall_label'].layout.visibility = 'visible'

def update_progress(ui_components: Dict[str, Any], value: int, message: Optional[str] = None) -> None:
    """
    Update progress bar dan pesan.
    
    Args:
        ui_components: Dictionary komponen UI
        value: Nilai progress (0-100)
        message: Pesan progress opsional
    """
    # Pastikan nilai valid
    value = max(0, min(100, value))
    
    # Update progress bar
    if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
        ui_components['progress_bar'].value = value
    
    # Update pesan jika ada
    if message and 'progress_message' in ui_components and hasattr(ui_components['progress_message'], 'value'):
        ui_components['progress_message'].value = message
    
    # Update step label jika ada
    if message and 'step_label' in ui_components and hasattr(ui_components['step_label'], 'value'):
        ui_components['step_label'].value = message
    
    # Update observers progress jika tersedia
    observer_manager = ui_components.get('observer_manager')
    if observer_manager:
        # Update progress melalui observer jika tersedia
        try:
            # Format observer_id sesuai dengan kebutuhan (bisa diganti jika sistem observer berbeda)
            observer_id = ui_components.get('observer_group', 'download_progress')
            
            # Kirim progress update dengan observer manager
            observer_manager.notify(observer_id, {'progress': value, 'message': message})
        except Exception:
            # Ignore errors in observer notification
            pass

def setup_multi_progress(ui_components: Dict[str, Any], tracking_keys: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Setup multi-progress tracking untuk proses yang kompleks.
    
    Args:
        ui_components: Dictionary komponen UI
        tracking_keys: Dictionary dengan nama tracking key (opsional)
        
    Returns:
        Dictionary UI components yang telah disetup
    """
    # Setup key default jika tidak diberikan
    if tracking_keys is None:
        tracking_keys = {
            'overall_progress': 'progress_bar',
            'step_progress': 'progress_bar',
            'overall_label': 'overall_label',
            'step_label': 'step_label'
        }
    
    # Pasikan progress container terlihat
    if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
        ui_components['progress_container'].layout.display = 'block'
    
    # Tambahkan callback untuk update progress
    if not callable(ui_components.get('update_progress')):
        ui_components['update_progress'] = lambda value, message=None: update_progress(ui_components, value, message)
    
    # Tambahkan callback untuk reset progress
    if not callable(ui_components.get('reset_progress')):
        ui_components['reset_progress'] = lambda: reset_progress_bar(ui_components)
    
    # Tambahkan callback untuk show progress
    if not callable(ui_components.get('show_progress')):
        ui_components['show_progress'] = lambda message: show_progress(ui_components, message)
    
    return ui_components

def setup_progress_indicator(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup progress indicator sederhana jika progress bar tidak tersedia.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary UI components yang telah disetup
    """
    # Jika tidak ada progress bar, buat progress sederhana
    if 'progress_bar' not in ui_components:
        from ipywidgets import FloatProgress, Label, VBox, HBox
        
        # Buat progress bar
        progress_bar = FloatProgress(value=0, min=0, max=100, description='Loading:')
        
        # Buat label
        progress_message = Label(value='Mempersiapkan...')
        
        # Buat layout
        progress_container = VBox([
            HBox([progress_bar, progress_message])
        ])
        
        # Tambahkan ke UI components
        ui_components['progress_bar'] = progress_bar
        ui_components['progress_message'] = progress_message
        ui_components['progress_container'] = progress_container
        
        # Jika ui adalah VBox/HBox, tambahkan progress container
        if 'ui' in ui_components and hasattr(ui_components['ui'], 'children'):
            try:
                # Coba tambahkan ke UI
                children = list(ui_components['ui'].children)
                children.append(progress_container)
                ui_components['ui'].children = tuple(children)
            except:
                # Ignore error pada penambahan ke UI
                pass
    
    return ui_components 