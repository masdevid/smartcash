"""
File: smartcash/ui/dataset/download/observers/progress_observer.py
Deskripsi: Observer untuk progress tracking yang terintegrasi dengan UI progress bars
"""

from typing import Dict, Any
from smartcash.components.observer.base_observer import BaseObserver

class DownloadProgressObserver(BaseObserver):
    """Observer untuk tracking progress download ke UI progress bars."""
    
    def __init__(self, ui_components: Dict[str, Any], name: str = "download_progress"):
        super().__init__(name=name, priority=10)
        self.ui_components = ui_components
        self.logger = ui_components.get('logger')
    
    def update(self, event_type: str, sender: Any, **kwargs) -> None:
        """Update progress bars berdasarkan event dari service."""
        
        # Overall progress events
        if event_type in ['DOWNLOAD_PROGRESS', 'EXPORT_PROGRESS', 'BACKUP_PROGRESS']:
            self._update_overall_progress(kwargs)
        
        # Step progress events  
        elif event_type in ['DOWNLOAD_STEP_PROGRESS', 'EXPORT_STEP_PROGRESS']:
            self._update_step_progress(kwargs)
        
        # Start/Complete events
        elif event_type.endswith('_START'):
            self._handle_start_event(kwargs)
        elif event_type.endswith('_COMPLETE'):
            self._handle_complete_event(kwargs)
        elif event_type.endswith('_ERROR'):
            self._handle_error_event(kwargs)
    
    def _update_overall_progress(self, kwargs: Dict[str, Any]) -> None:
        """Update overall progress bar."""
        progress = kwargs.get('progress', 0)
        message = kwargs.get('message', 'Processing...')
        
        if 'progress_bar' in self.ui_components:
            self.ui_components['progress_bar'].value = min(100, max(0, progress))
            self.ui_components['progress_bar'].description = f"Progress: {progress}%"
            self.ui_components['progress_bar'].layout.visibility = 'visible'
        
        if 'overall_label' in self.ui_components:
            self.ui_components['overall_label'].value = message
            self.ui_components['overall_label'].layout.visibility = 'visible'
    
    def _update_step_progress(self, kwargs: Dict[str, Any]) -> None:
        """Update step progress bar."""
        step_progress = kwargs.get('step_progress', kwargs.get('progress', 0))
        current_step = kwargs.get('current_step', 1)
        total_steps = kwargs.get('total_steps', 5)
        step_message = kwargs.get('step_message', kwargs.get('message', ''))
        
        if 'current_progress' in self.ui_components:
            self.ui_components['current_progress'].value = min(100, max(0, step_progress))
            self.ui_components['current_progress'].description = f"Step {current_step}/{total_steps}"
            self.ui_components['current_progress'].layout.visibility = 'visible'
        
        if 'step_label' in self.ui_components:
            self.ui_components['step_label'].value = step_message
            self.ui_components['step_label'].layout.visibility = 'visible'
    
    def _handle_start_event(self, kwargs: Dict[str, Any]) -> None:
        """Handle start events."""
        message = kwargs.get('message', 'Memulai proses...')
        
        # Show progress container
        if 'progress_container' in self.ui_components:
            self.ui_components['progress_container'].layout.display = 'block'
            self.ui_components['progress_container'].layout.visibility = 'visible'
        
        self._update_overall_progress({'progress': 0, 'message': message})
        
        if self.logger:
            self.logger.info(f"🚀 {message}")
    
    def _handle_complete_event(self, kwargs: Dict[str, Any]) -> None:
        """Handle complete events."""
        message = kwargs.get('message', 'Proses selesai')
        duration = kwargs.get('duration', 0)
        
        self._update_overall_progress({'progress': 100, 'message': message})
        
        if self.logger:
            duration_str = f" ({duration:.1f}s)" if duration > 0 else ""
            self.logger.success(f"✅ {message}{duration_str}")
    
    def _handle_error_event(self, kwargs: Dict[str, Any]) -> None:
        """Handle error events."""
        message = kwargs.get('message', 'Terjadi error')
        
        if self.logger:
            self.logger.error(f"❌ {message}")