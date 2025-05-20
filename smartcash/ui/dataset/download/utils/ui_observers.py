"""
File: smartcash/ui/dataset/download/utils/ui_observers.py
Deskripsi: Utilitas untuk mengelola observer UI pada proses download dataset
"""

from typing import Dict, Any, Optional
from smartcash.components.observer import ObserverManager, EventTopics
from smartcash.ui.dataset.download.utils.notification_manager import DownloadUIEvents

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
        if event_type in [DownloadUIEvents.LOG_INFO, DownloadUIEvents.LOG_WARNING, 
                         DownloadUIEvents.LOG_ERROR, DownloadUIEvents.LOG_SUCCESS]:
            if 'log_output' in ui_components and hasattr(ui_components['log_output'], 'append_stdout'):
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
                formatted_message = f"{emoji} {message}\n"
                
                # Tampilkan pesan di log output
                ui_components['log_output'].append_stdout(formatted_message)
    
    # Observer untuk progress bar
    def progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        if event_type in [DownloadUIEvents.PROGRESS_START, DownloadUIEvents.PROGRESS_UPDATE,
                         DownloadUIEvents.PROGRESS_COMPLETE, DownloadUIEvents.PROGRESS_ERROR]:
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
                
                # Update pesan progress jika ada
                if message and 'progress_message' in ui_components:
                    ui_components['progress_message'].value = message
    
    # Observer untuk step progress
    def step_progress_observer(event_type: str, sender: Any, **kwargs) -> None:
        if event_type in [DownloadUIEvents.STEP_PROGRESS_START, DownloadUIEvents.STEP_PROGRESS_UPDATE,
                         DownloadUIEvents.STEP_PROGRESS_COMPLETE, DownloadUIEvents.STEP_PROGRESS_ERROR]:
            # Update step progress bar
            if 'step_progress_bar' in ui_components and hasattr(ui_components['step_progress_bar'], 'value'):
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
                ui_components['step_progress_bar'].value = step_progress
                ui_components['step_progress_bar'].description = f"Step {current_step}/{total_steps}: {step_progress}%"
                
                # Update pesan step progress jika ada
                if step_message and 'step_progress_message' in ui_components:
                    ui_components['step_progress_message'].value = step_message
    
    # Daftarkan observer
    observer_manager.register(log_observer)
    observer_manager.register(progress_observer)
    observer_manager.register(step_progress_observer)
    
    return observer_manager
