"""
File: smartcash/dataset/augmentor/communicator.py
Deskripsi: UI communication bridge untuk augmentasi dengan unified interface
"""

from typing import Dict, Any, Optional, Callable
from smartcash.common.logger import get_logger

class UICommunicator:
    """Unified interface untuk UI communication dengan fallback yang aman."""
    
    def __init__(self, ui_components: Dict[str, Any] = None):
        """
        Initialize UI communicator dengan components.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components or {}
        self.logger = self._setup_logger()
        self.progress_tracker = self._get_progress_tracker()
        
    def _setup_logger(self):
        """Setup logger dengan UI bridge jika tersedia."""
        if not self.ui_components:
            return get_logger("augmentation")
            
        try:
            from smartcash.ui.utils.logger_bridge import create_ui_logger_bridge
            return create_ui_logger_bridge(self.ui_components, "augmentation")
        except ImportError:
            return get_logger("augmentation")
    
    def _get_progress_tracker(self):
        """Dapatkan progress tracker dari UI components."""
        return self.ui_components.get('tracker')
    
    def progress(self, operation: str, current: int, total: int, message: str = ""):
        """
        Report progress ke UI tracker dengan validasi.
        
        Args:
            operation: Nama operasi ('overall', 'step', 'current')
            current: Progress saat ini (0-100)
            total: Total progress (biasanya 100)
            message: Pesan progress
        """
        if self.progress_tracker and hasattr(self.progress_tracker, 'update'):
            try:
                # Normalize progress ke 0-100 range
                percentage = min(100, max(0, int((current / max(1, total)) * 100)))
                self.progress_tracker.update(operation, percentage, message)
            except Exception:
                pass  # Silent fail untuk progress tracking
    
    def log_info(self, message: str):
        """Log info message dengan emoji kontekstual."""
        self.logger.info(f"ℹ️ {message}")
    
    def log_success(self, message: str):
        """Log success message dengan emoji kontekstual.""" 
        self.logger.success(f"✅ {message}")
    
    def log_warning(self, message: str):
        """Log warning message dengan emoji kontekstual."""
        self.logger.warning(f"⚠️ {message}")
    
    def log_error(self, message: str):
        """Log error message dengan emoji kontekstual."""
        self.logger.error(f"❌ {message}")
    
    def log_debug(self, message: str):
        """Log debug message dengan emoji kontekstual."""
        self.logger.debug(f"🔍 {message}")
    
    def start_operation(self, operation_name: str, total_steps: int = 100):
        """
        Mulai operasi dengan progress tracking.
        
        Args:
            operation_name: Nama operasi
            total_steps: Total steps untuk operasi
        """
        self.log_info(f"🚀 Memulai {operation_name}")
        if self.progress_tracker and hasattr(self.progress_tracker, 'show'):
            try:
                self.progress_tracker.show(operation_name.lower().replace(' ', '_'))
            except Exception:
                pass
    
    def complete_operation(self, operation_name: str, result_message: str = ""):
        """
        Selesaikan operasi dengan progress tracking.
        
        Args:
            operation_name: Nama operasi
            result_message: Pesan hasil operasi
        """
        final_message = result_message or f"{operation_name} selesai"
        self.log_success(final_message)
        
        if self.progress_tracker and hasattr(self.progress_tracker, 'complete'):
            try:
                self.progress_tracker.complete(final_message)
            except Exception:
                pass
    
    def error_operation(self, operation_name: str, error_message: str):
        """
        Handle error operasi dengan progress tracking.
        
        Args:
            operation_name: Nama operasi
            error_message: Pesan error
        """
        final_message = f"{operation_name} gagal: {error_message}"
        self.log_error(final_message)
        
        if self.progress_tracker and hasattr(self.progress_tracker, 'error'):
            try:
                self.progress_tracker.error(final_message)
            except Exception:
                pass
    
    def update_status(self, message: str, status_type: str = "info"):
        """
        Update status UI dengan pesan.
        
        Args:
            message: Pesan status
            status_type: Tipe status ('info', 'success', 'warning', 'error')
        """
        # Map status type ke log method
        log_method = getattr(self, f'log_{status_type}', self.log_info)
        log_method(message)
        
        # Update status panel jika tersedia
        if 'status_panel' in self.ui_components:
            try:
                from smartcash.ui.utils.alert_utils import update_status_panel
                update_status_panel(self.ui_components['status_panel'], message, status_type)
            except ImportError:
                pass
    
    def is_stop_requested(self) -> bool:
        """
        Check apakah user meminta stop operasi.
        
        Returns:
            True jika stop diminta
        """
        return self.ui_components.get('stop_requested', False)
    
    def report_progress_with_callback(self, progress_callback: Optional[Callable] = None, 
                                    step: str = "overall", current: int = 0, 
                                    total: int = 100, message: str = ""):
        """
        Report progress dengan callback dan tracker sekaligus.
        
        Args:
            progress_callback: Callback function eksternal
            step: Step operasi
            current: Progress saat ini
            total: Total progress
            message: Pesan progress
        """
        # Update internal tracker
        self.progress(step, current, total, message)
        
        # Call external callback jika ada
        if progress_callback and callable(progress_callback):
            try:
                progress_callback(step, current, total, message)
            except Exception:
                pass  # Silent fail untuk external callback

def create_communicator(ui_components: Dict[str, Any] = None) -> UICommunicator:
    """
    Factory function untuk membuat UI communicator.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance UICommunicator
    """
    return UICommunicator(ui_components)

# One-liner helper functions untuk backward compatibility
log_to_ui = lambda comm, msg, level="info": getattr(comm, f"log_{level}", comm.log_info)(msg)
progress_to_ui = lambda comm, op, curr, total, msg="": comm.progress(op, curr, total, msg)
start_ui_operation = lambda comm, name: comm.start_operation(name)
complete_ui_operation = lambda comm, name, msg="": comm.complete_operation(name, msg)
error_ui_operation = lambda comm, name, msg: comm.error_operation(name, msg)