"""
File: smartcash/ui/handlers/download_progress_handler.py
Deskripsi: Handler untuk memantau dan memperbarui progres download dataset dengan integrasi UI utils
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable

from smartcash.ui.utils.constants import ICONS
from smartcash.ui.utils.alert_utils import create_status_indicator

class DownloadProgressHandler:
    """Handler untuk tracking progress download dataset dengan komponen utils standar."""
    
    def __init__(self, ui_components: Dict[str, Any], logger=None):
        """
        Inisialisasi DownloadProgressHandler.
        
        Args:
            ui_components: Komponen UI
            logger: Logger kustom (opsional)
        """
        self.ui_components = ui_components
        self.logger = logger or ui_components.get('logger')
        
        # Setup observer_manager dengan utils standar
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            self.observer_manager = ObserverManager()
        except (ImportError, AttributeError):
            self.observer_manager = None
        
        # Status handler
        self.is_downloading = False
        self.current_operation = None
        self.progress_by_step = {}
        
        # Akses pada progress bar dan label
        self.progress_bar = ui_components.get('progress_bar')
        self.progress_label = ui_components.get('progress_label')
        self.status_output = ui_components.get('status')
        
        # Register observer untuk tracking download dengan utils standar
        if self.observer_manager:
            self._register_download_observers()
            if self.logger: self.logger.info(f"{ICONS['success']} Progress handler berhasil diinisialisasi")
    
    def _register_download_observers(self):
        """Register observers untuk berbagai events download/import."""
        if not self.observer_manager:
            if self.logger: self.logger.warning(f"{ICONS['warning']} Observer manager tidak tersedia")
            return
            
        from smartcash.components.observer.event_topics_observer import EventTopics
        
        # Daftar event yang akan di-observe
        events = [
            # Download events
            EventTopics.DOWNLOAD_START, EventTopics.DOWNLOAD_PROGRESS, 
            EventTopics.DOWNLOAD_COMPLETE, EventTopics.DOWNLOAD_ERROR,
            
            # Export events
            EventTopics.EXPORT_START, EventTopics.EXPORT_PROGRESS, 
            EventTopics.EXPORT_COMPLETE, EventTopics.EXPORT_ERROR,
            
            # Backup events
            EventTopics.BACKUP_START, EventTopics.BACKUP_PROGRESS, 
            EventTopics.BACKUP_COMPLETE, EventTopics.BACKUP_ERROR,
            
            # ZIP processing events
            EventTopics.ZIP_PROCESSING_START, EventTopics.ZIP_PROCESSING_PROGRESS,
            EventTopics.ZIP_PROCESSING_COMPLETE, EventTopics.ZIP_PROCESSING_ERROR,
            
            # Other events
            EventTopics.UPLOAD_START, EventTopics.UPLOAD_PROGRESS,
            EventTopics.UPLOAD_COMPLETE, EventTopics.UPLOAD_ERROR,
            EventTopics.PULL_DATASET_START, EventTopics.PULL_DATASET_COMPLETE
        ]
        
        # Register untuk semua events dengan utils standar
        for event in events:
            self.observer_manager.create_simple_observer(
                event_type=event,
                callback=self._handle_download_event,
                name=f"UI_{event}_Observer",
                group="download_ui_observer"
            )
            
        if self.logger: self.logger.info(f"{ICONS['success']} Observer untuk {len(events)} events berhasil didaftarkan")
    
    def _handle_download_event(self, event_type: str, sender: Any, message: Optional[str] = None, **kwargs):
        """
        Handler untuk berbagai event download dengan utils standar.
        
        Args:
            event_type: Tipe event
            sender: Objek pengirim event
            message: Pesan event (opsional)
            **kwargs: Parameter tambahan
        """
        # Tentukan tipe operasi dengan utils standar
        operation_map = {
            'download.': 'download', 'export.': 'export', 
            'backup.': 'backup', 'zip_processing.': 'zip_processing',
            'zip_import.': 'zip_import', 'pull_dataset.': 'pull_dataset',
            'upload.': 'upload'
        }
        
        operation = 'unknown'
        for prefix, op_name in operation_map.items():
            if event_type.startswith(prefix):
                operation = op_name
                break
        
        # Set current operation jika belum di-set
        if self.current_operation is None and event_type.endswith(".start"):
            self.current_operation = operation
            self.is_downloading = True
            
        # Reset status jika operasi selesai atau error
        if event_type.endswith(".complete") or event_type.endswith(".error"):
            if self.current_operation == operation:
                self.current_operation = None
                self.is_downloading = False
                self.progress_by_step = {}
        
        # Update status UI dengan alert_utils standar
        status = kwargs.get('status', 'info')
        
        # Handle tipe event dengan utils standar
        if event_type.endswith(".start"): self._handle_start_event(operation, message, status)
        elif event_type.endswith(".progress"): self._handle_progress_event(operation, message, status, **kwargs)
        elif event_type.endswith(".complete"): self._handle_complete_event(operation, message, status, **kwargs)
        elif event_type.endswith(".error"): self._handle_error_event(operation, message)
            
    def _handle_start_event(self, operation: str, message: Optional[str], status: str):
        """Handler untuk event start dengan utils standar."""
        # Update status dengan fallback_utils standar
        from smartcash.ui.utils.fallback_utils import update_status_panel
        update_status_panel(
            self.ui_components, 
            status, 
            message or f"{ICONS['processing']} Memulai operasi {operation}"
        )
            
        # Reset progress bar dan label
        self.update_progress_bar(0, 100, message or f"Memulai {operation}...")
        if self.logger: self.logger.info(f"{ICONS['processing']} {message or f'Memulai operasi {operation}'}")
    
    def _handle_progress_event(self, operation: str, message: Optional[str], status: str, **kwargs):
        """Handler untuk event progress dengan utils standar."""
        # Extract progress dari kwargs
        progress = kwargs.get('progress', 0)
        total_steps = kwargs.get('total_steps', 100)
        current_step = kwargs.get('current_step', 0)
        percentage = kwargs.get('percentage')
        
        # Simpan progress untuk step ini jika ada
        step = kwargs.get('step')
        if step:
            self.progress_by_step[step] = {
                'progress': progress, 'total_steps': total_steps,
                'current_step': current_step, 'percentage': percentage
            }
            
        # Update progress bar dengan utils standar
        if percentage is not None:
            self.update_progress_bar(percentage, 100, message)
        elif total_steps and current_step:
            self.update_progress_bar(current_step, total_steps, message)
        elif progress and kwargs.get('total'):
            total = kwargs.get('total')
            percentage = int((progress / total) * 100) if total > 0 else 0
            self.update_progress_bar(percentage, 100, message)
                
        # Tampilkan pesan status untuk perubahan step
        if step:
            from smartcash.ui.utils.fallback_utils import update_status_panel
            update_status_panel(
                self.ui_components, status,
                message or f"{ICONS['processing']} Step {current_step}/{total_steps}: {step}"
            )
            
        # Log dengan interval yang teratur
        if step or (percentage is not None and percentage % 25 == 0):
            if self.logger:
                if percentage is not None:
                    self.logger.info(f"{ICONS['processing']} {message or f'Progress {operation}'}: {percentage}%")
                elif total_steps and current_step:
                    self.logger.info(f"{ICONS['processing']} {message or f'Progress {operation}'}: Step {current_step}/{total_steps}")
    
    def _handle_complete_event(self, operation: str, message: Optional[str], status: str, **kwargs):
        """Handler untuk event complete dengan utils standar."""
        duration = kwargs.get('duration')
        
        # Update progress bar ke 100%
        self.update_progress_bar(100, 100, message or f"{operation} selesai")
                
        # Update status dengan utils standar
        from smartcash.ui.utils.fallback_utils import update_status_panel
        
        # Format pesan sukses dengan detail tambahan
        complete_message = message or f"{ICONS['success']} {operation} selesai"
        if duration: complete_message += f" ({duration:.1f}s)"
        update_status_panel(self.ui_components, status, complete_message)
            
        # Reset progress tracking
        self.progress_by_step = {}
        
        # Log completion
        if self.logger: self.logger.success(f"{ICONS['success']} {message or f'Operasi {operation} selesai'}")
        
    def _handle_error_event(self, operation: str, message: Optional[str]):
        """Handler untuk event error dengan utils standar."""
        # Update progress label
        if self.progress_label:
            self.progress_label.value = message or f"Error pada {operation}"
            
        # Update status dengan utils standar
        from smartcash.ui.utils.fallback_utils import update_status_panel
        update_status_panel(
            self.ui_components, "error",
            message or f"{ICONS['error']} Error pada {operation}"
        )
        
        # Reset progress tracking
        self.progress_by_step = {}
        
        # Log error dengan utils standar
        if self.logger: self.logger.error(f"{ICONS['error']} {message or f'Error pada {operation}'}")
    
    def update_progress_bar(self, value: int, max_value: int, message: Optional[str] = None):
        """
        Update progress bar dan label dengan utils standar.
        
        Args:
            value: Nilai progres saat ini
            max_value: Nilai maksimum progres
            message: Pesan opsional untuk label
        """
        # Update progress bar jika ada
        if self.progress_bar:
            self.progress_bar.value = value
            self.progress_bar.max = max_value
            
            # Update deskripsi dengan persentase
            percentage = int((value / max_value) * 100) if max_value > 0 else 0
            self.progress_bar.description = f"Proses: {percentage}%"
            
        # Update label jika ada dan message diberikan
        if self.progress_label and message:
            self.progress_label.value = message
    
    def reset_progress_bar(self):
        """Reset progress bar dan label ke kondisi awal."""
        if self.progress_bar:
            self.progress_bar.value = 0
            self.progress_bar.description = 'Proses: 0%'
            
        if self.progress_label:
            self.progress_label.value = "Siap untuk download dataset"
            
        # Reset status dengan utils standar
        from smartcash.ui.utils.fallback_utils import update_status_panel
        update_status_panel(
            self.ui_components, "info",
            f"{ICONS['info']} Status download telah direset"
        )
                
        self.is_downloading = False
        self.current_operation = None
        self.progress_by_step = {}