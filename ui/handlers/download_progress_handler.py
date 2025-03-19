"""
File: smartcash/ui/handlers/download_progress_handler.py
Deskripsi: Handler khusus untuk memantau progres download dataset dan memperbarui UI dengan notifikasi
"""

import ipywidgets as widgets
from IPython.display import display, HTML
from typing import Dict, Any, Optional, Callable

from smartcash.common.logger import get_logger
from smartcash.components.observer import notify, EventTopics


class DownloadProgressHandler:
    """
    Handler untuk tracking progress download dataset menggunakan observer pattern.
    """
    
    def __init__(
        self, 
        ui_components: Dict[str, Any],
        logger=None
    ):
        """
        Inisialisasi DownloadProgressHandler.
        
        Args:
            ui_components: Komponen UI
            logger: Logger kustom (opsional)
        """
        self.ui_components = ui_components
        self.logger = logger or get_logger("download_progress_handler")
        
        # Setup observer_manager
        try:
            from smartcash.components.observer.manager_observer import ObserverManager
            self.observer_manager = ObserverManager()
        except (ImportError, AttributeError):
            self.observer_manager = None
        
        # Status handler
        self.is_downloading = False
        self.current_operation = None
        self.progress_by_step = {}
        
        # Akses pada progress bar jika tersedia
        self.progress_bar = ui_components.get('progress_bar')
        self.progress_label = ui_components.get('progress_label')
        self.status_output = ui_components.get('status_output')
        
        # Register observer untuk tracking download
        self._register_download_observers()
        
        if self.logger:
            self.logger.info(f"🖥️ DownloadProgressHandler diinisialisasi")
    
    def _register_download_observers(self):
        """Register observers untuk berbagai events download/import."""
        if not self.observer_manager:
            if self.logger:
                self.logger.warning("⚠️ Observer manager tidak tersedia, progress tracking tidak akan berfungsi")
            return
        
        # Register observer untuk download
        events = [
            # Download events
            EventTopics.DOWNLOAD_START,
            EventTopics.DOWNLOAD_PROGRESS,
            EventTopics.DOWNLOAD_COMPLETE,
            EventTopics.DOWNLOAD_ERROR,
            
            # Export events
            EventTopics.EXPORT_START,
            EventTopics.EXPORT_PROGRESS,
            EventTopics.EXPORT_COMPLETE,
            EventTopics.EXPORT_ERROR,
            
            # Backup events
            EventTopics.BACKUP_START,
            EventTopics.BACKUP_PROGRESS,
            EventTopics.BACKUP_COMPLETE,
            EventTopics.BACKUP_ERROR,
            
            # ZIP processing events
            EventTopics.ZIP_PROCESSING_START,
            EventTopics.ZIP_PROCESSING_PROGRESS,
            EventTopics.ZIP_PROCESSING_COMPLETE,
            EventTopics.ZIP_PROCESSING_ERROR,
            
            # ZIP import events
            EventTopics.ZIP_IMPORT_START,
            EventTopics.ZIP_IMPORT_COMPLETE,
            EventTopics.ZIP_IMPORT_ERROR,
            
            # One-step dataset pull events
            EventTopics.PULL_DATASET_START,
            EventTopics.PULL_DATASET_COMPLETE,
            EventTopics.PULL_DATASET_ERROR
        ]
        
        # Register untuk semua events dengan ObserverManager, bukan EventDispatcher
        for event in events:
            # Gunakan create_simple_observer dari ObserverManager
            self.observer_manager.create_simple_observer(
                event_type=event,
                callback=self._handle_download_event,
                name=f"UI_{event}_Observer",
                group="download_ui_observer"
            )
            
        if self.logger:
            self.logger.info(f"✅ Observer untuk {len(events)} events berhasil didaftarkan")
    
    def _handle_download_event(
        self, 
        event_type: str, 
        sender: Any, 
        message: Optional[str] = None,
        **kwargs
    ):
        """
        Handler untuk berbagai event download.
        
        Args:
            event_type: Tipe event
            sender: Objek pengirim event
            message: Pesan event (opsional)
            **kwargs: Parameter tambahan
        """
        # Tentukan tipe operasi
        if event_type.startswith("download."):
            operation = "download"
        elif event_type.startswith("export."):
            operation = "export"
        elif event_type.startswith("backup."):
            operation = "backup"
        elif event_type.startswith("zip_processing."):
            operation = "zip_processing"
        elif event_type.startswith("zip_import."):
            operation = "zip_import"
        elif event_type.startswith("pull_dataset."):
            operation = "pull_dataset"
        else:
            operation = "unknown"
        
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
        
        # Update status UI
        status = kwargs.get('status', 'info')
        
        # Handle tipe event
        if event_type.endswith(".start"):
            # Event mulai
            self._handle_start_event(operation, message, status)
        elif event_type.endswith(".progress"):
            # Event progress
            self._handle_progress_event(operation, message, status, **kwargs)
        elif event_type.endswith(".complete"):
            # Event selesai
            self._handle_complete_event(operation, message, status, **kwargs)
        elif event_type.endswith(".error"):
            # Event error
            self._handle_error_event(operation, message)
            
    def _handle_start_event(self, operation: str, message: Optional[str], status: str):
        """
        Handler untuk event start.
        
        Args:
            operation: Tipe operasi
            message: Pesan
            status: Status event
        """
        # Update status
        from smartcash.ui.dataset.download_initialization import update_status_panel
        update_status_panel(
            self.ui_components, 
            status, 
            message or f"Memulai operasi {operation}"
        )
            
        # Reset progress bar jika ada
        if self.progress_bar:
            self.progress_bar.value = 0
            self.progress_bar.max = 100
            
        if self.progress_label:
            self.progress_label.value = message or f"Memulai {operation}..."
            
        self.logger.info(f"🚀 {message or f'Memulai operasi {operation}'}")
    
    def _handle_progress_event(
        self, 
        operation: str, 
        message: Optional[str], 
        status: str,
        **kwargs
    ):
        """
        Handler untuk event progress.
        
        Args:
            operation: Tipe operasi
            message: Pesan
            status: Status event
            **kwargs: Parameter tambahan termasuk progress, total_steps, current_step
        """
        # Extract progress dari kwargs
        progress = kwargs.get('progress', 0)
        total_steps = kwargs.get('total_steps', 100)
        current_step = kwargs.get('current_step', 0)
        
        # Jika ada persentase langsung
        percentage = kwargs.get('percentage')
        
        # Simpan progress untuk step ini jika ada
        step = kwargs.get('step')
        if step:
            self.progress_by_step[step] = {
                'progress': progress,
                'total_steps': total_steps,
                'current_step': current_step,
                'percentage': percentage
            }
            
        # Update progress bar jika ada
        if self.progress_bar:
            # Jika persentase langsung tersedia
            if percentage is not None:
                self.progress_bar.value = percentage
            # Jika using step model
            elif total_steps and current_step:
                self.progress_bar.max = total_steps
                self.progress_bar.value = current_step
            # Fallback ke progress/total
            elif progress and kwargs.get('total'):
                total = kwargs.get('total')
                percentage = int((progress / total) * 100) if total > 0 else 0
                self.progress_bar.value = percentage
                
        # Update label progress jika ada
        if self.progress_label:
            if step and message:
                # Format pesan dengan progres
                if percentage is not None:
                    formatted_message = f"{message}: {percentage}%"
                elif total_steps and current_step:
                    formatted_message = f"{message} (Langkah {current_step}/{total_steps})"
                else:
                    formatted_message = message
                    
                self.progress_label.value = formatted_message
            else:
                self.progress_label.value = message or f"Sedang memproses {operation}..."
                
        # Tampilkan pesan status untuk perubahan step
        if step:
            from smartcash.ui.dataset.download_initialization import update_status_panel
            update_status_panel(
                self.ui_components,
                status,
                message or f"Step {current_step}/{total_steps}: {step}"
            )
            
        # Log setiap 25% perubahan atau perubahan step
        if step or (percentage is not None and percentage % 25 == 0):
            if percentage is not None:
                self.logger.info(f"🔄 {message or f'Progress {operation}'}: {percentage}%")
            elif total_steps and current_step:
                self.logger.info(f"🔄 {message or f'Progress {operation}'}: Step {current_step}/{total_steps}")
    
    def _handle_complete_event(
        self, 
        operation: str, 
        message: Optional[str], 
        status: str,
        **kwargs
    ):
        """
        Handler untuk event complete.
        
        Args:
            operation: Tipe operasi
            message: Pesan
            status: Status event
            **kwargs: Parameter tambahan
        """
        duration = kwargs.get('duration')
        
        # Update progress bar ke 100% jika ada
        if self.progress_bar:
            self.progress_bar.value = self.progress_bar.max
            
        # Update label progress
        if self.progress_label:
            # Format dengan durasi jika tersedia
            if duration:
                self.progress_label.value = message or f"{operation} selesai dalam {duration:.1f}s"
            else:
                self.progress_label.value = message or f"{operation} selesai"
                
        # Update status
        from smartcash.ui.dataset.download_initialization import update_status_panel
        # Format pesan sukses dengan detail tambahan
        complete_message = message or f"{operation} selesai"
        if duration:
            complete_message += f" ({duration:.1f}s)"
            
        update_status_panel(
            self.ui_components,
            status,
            complete_message
        )
            
        # Reset progress tracking
        self.progress_by_step = {}
        
        # Log completion
        self.logger.success(f"✅ {message or f'Operasi {operation} selesai'}")
        
    def _handle_error_event(self, operation: str, message: Optional[str]):
        """
        Handler untuk event error.
        
        Args:
            operation: Tipe operasi
            message: Pesan error
        """
        # Update progress label
        if self.progress_label:
            self.progress_label.value = message or f"Error pada {operation}"
            
        # Update status dengan error
        from smartcash.ui.dataset.download_initialization import update_status_panel
        update_status_panel(
            self.ui_components,
            "error",
            message or f"Error pada {operation}"
        )
        
        # Reset progress tracking
        self.progress_by_step = {}
        
        # Log error
        self.logger.error(f"❌ {message or f'Error pada {operation}'}")
    
    def reset_ui(self):
        """Reset komponen UI ke kondisi default."""
        if self.progress_bar:
            self.progress_bar.value = 0
            
        if self.progress_label:
            self.progress_label.value = "Siap untuk download dataset"
            
        from smartcash.ui.dataset.download_initialization import update_status_panel
        update_status_panel(
            self.ui_components,
            "info",
            "Status download telah direset"
        )
                
        self.is_downloading = False
        self.current_operation = None
        self.progress_by_step = {}