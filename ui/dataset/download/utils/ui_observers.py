"""
File: smartcash/ui/dataset/download/utils/ui_observers.py
Deskripsi: Utilitas untuk mengelola observer UI pada proses download dataset
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import ObserverManager, EventTopics
from smartcash.ui.dataset.download.utils.notification_manager import DownloadUIEvents
from smartcash.ui.dataset.download.download_initializer import DOWNLOAD_LOGGER_NAMESPACE

def register_ui_observers(ui_components: Dict[str, Any]) -> ObserverManager:
    """
    Daftarkan observer untuk UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        ObserverManager: Manager untuk observer yang terdaftar
    """
    observer_manager = ObserverManager()
    
    # Observer untuk log output
    def log_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan log hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != DOWNLOAD_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.download'):
            return  # Skip log dari namespace lain
            
        if event_type in [DownloadUIEvents.LOG_INFO, DownloadUIEvents.LOG_WARNING, 
                         DownloadUIEvents.LOG_ERROR, DownloadUIEvents.LOG_SUCCESS]:
            if isinstance(ui_components, dict) and 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_stdout'):
                message = kwargs.get('message', '')
                level = kwargs.get('level', 'info')
                
                # Format pesan dengan emoji
                emoji_map = {
                    "info": "ℹ️",
                    "warning": "⚠️",
                    "error": "❌",
                    "success": "✅"
                }
                emoji = emoji_map.get(level.lower(), "ℹ️")
                formatted_message = f"{emoji} {message}"
                
                # Tampilkan pesan di log output
                if level.lower() == 'error':
                    ui_components['log_output'].append_stderr(formatted_message)
                else:
                    ui_components['log_output'].append_stdout(formatted_message)
                
                # Pastikan log accordion terbuka
                if 'log_accordion' in ui_components and hasattr(ui_components['log_accordion'], 'selected_index'):
                    ui_components['log_accordion'].selected_index = 0  # Buka accordion pertama
    
    # Observer untuk progress bar
    def progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan progress hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != DOWNLOAD_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.download'):
            return  # Skip progress dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [DownloadUIEvents.PROGRESS_START, DownloadUIEvents.PROGRESS_UPDATE,
                         DownloadUIEvents.PROGRESS_COMPLETE, DownloadUIEvents.PROGRESS_ERROR]:
            # Pastikan progress container terlihat
            if 'progress_container' in ui_components and hasattr(ui_components['progress_container'], 'layout'):
                ui_components['progress_container'].layout.display = 'block'
                ui_components['progress_container'].layout.visibility = 'visible'
                
            # Update progress bar
            if 'progress_bar' in ui_components and hasattr(ui_components['progress_bar'], 'value'):
                # Update progress bar
                progress = kwargs.get('progress', 0)
                total = kwargs.get('total', 100)
                message = kwargs.get('message', '')
                
                # Pastikan progress adalah integer
                try:
                    progress = int(float(progress))
                except (ValueError, TypeError):
                    progress = 0
                
                # Update nilai progress bar
                ui_components['progress_bar'].value = progress
                ui_components['progress_bar'].description = f"Progress: {progress}%"
                ui_components['progress_bar'].layout.visibility = 'visible'
                
                # Update pesan progress jika ada
                if message and 'overall_label' in ui_components:
                    ui_components['overall_label'].value = message
                    ui_components['overall_label'].layout.visibility = 'visible'
                
                # Jika complete, set progress ke 100%
                if event_type == DownloadUIEvents.PROGRESS_COMPLETE:
                    ui_components['progress_bar'].value = 100
                    ui_components['progress_bar'].description = "Progress: 100%"
                    
                # Jika error, tampilkan pesan error
                if event_type == DownloadUIEvents.PROGRESS_ERROR:
                    if 'log_output' in ui_components:
                        ui_components['log_output'].append_stderr(f"❌ {message}")
    
    # Observer untuk step progress
    def step_progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        # Periksa namespace untuk memastikan step progress hanya dari modul yang relevan
        namespace = kwargs.get('namespace', '')
        if namespace and namespace != DOWNLOAD_LOGGER_NAMESPACE and not namespace.startswith('smartcash.dataset.download'):
            return  # Skip step progress dari namespace lain
            
        if not isinstance(ui_components, dict):
            return
            
        if event_type in [DownloadUIEvents.STEP_PROGRESS_START, DownloadUIEvents.STEP_PROGRESS_UPDATE,
                         DownloadUIEvents.STEP_PROGRESS_COMPLETE, DownloadUIEvents.STEP_PROGRESS_ERROR]:
            # Update step progress bar
            if 'current_progress' in ui_components and hasattr(ui_components['current_progress'], 'value'):
                step_progress = kwargs.get('step_progress', 0)
                step_total = kwargs.get('step_total', 100)
                step_message = kwargs.get('step_message', '')
                current_step = kwargs.get('current_step', 1)
                total_steps = kwargs.get('total_steps', 5)
                
                # Pastikan progress adalah integer
                try:
                    step_progress = int(float(step_progress))
                except (ValueError, TypeError):
                    step_progress = 0
                
                # Update nilai step progress bar
                ui_components['current_progress'].value = step_progress
                ui_components['current_progress'].description = f"Step {current_step}/{total_steps}"
                ui_components['current_progress'].layout.visibility = 'visible'
                
                # Update pesan step progress jika ada
                if step_message and 'step_label' in ui_components:
                    ui_components['step_label'].value = step_message
                    ui_components['step_label'].layout.visibility = 'visible'
                
                # Jika complete, set progress ke 100%
                if event_type == DownloadUIEvents.STEP_PROGRESS_COMPLETE:
                    ui_components['current_progress'].value = 100
                    ui_components['current_progress'].description = f"Step {total_steps}/{total_steps}"
    
    # Daftarkan observer untuk semua event
    log_events = [
        DownloadUIEvents.LOG_INFO, 
        DownloadUIEvents.LOG_WARNING, 
        DownloadUIEvents.LOG_ERROR, 
        DownloadUIEvents.LOG_SUCCESS
    ]
    
    progress_events = [
        DownloadUIEvents.PROGRESS_START, 
        DownloadUIEvents.PROGRESS_UPDATE, 
        DownloadUIEvents.PROGRESS_COMPLETE, 
        DownloadUIEvents.PROGRESS_ERROR
    ]
    
    step_progress_events = [
        DownloadUIEvents.STEP_PROGRESS_START, 
        DownloadUIEvents.STEP_PROGRESS_UPDATE, 
        DownloadUIEvents.STEP_PROGRESS_COMPLETE, 
        DownloadUIEvents.STEP_PROGRESS_ERROR
    ]
    
    # Buat dan daftarkan observer
    observer_manager.create_simple_observer(log_events, log_observer, name="log_observer")
    observer_manager.create_simple_observer(progress_events, progress_observer, name="progress_observer")
    observer_manager.create_simple_observer(step_progress_events, step_progress_observer, name="step_progress_observer")
    
    # Simpan observer_manager ke ui_components
    ui_components['observer_manager'] = observer_manager
    
    return observer_manager
