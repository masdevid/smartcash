"""
File: smartcash/model/service/progress_tracker.py
Deskripsi: Implementasi progress tracker untuk training dan proses model
"""

import time
from typing import Dict, Any, Optional, List, Callable, Union
from smartcash.common.logger import get_logger
from smartcash.model.service.callback_interfaces import ProgressCallback, ProgressCallbackFn, StatusCallbackFn, ErrorCallbackFn, CompleteCallbackFn

class ProgressTracker:
    """Progress tracker untuk training dan proses model dengan dukungan callback UI"""
    
    def __init__(self, callback: Optional[Union[ProgressCallback, Dict[str, Callable]]] = None):
        self.logger = get_logger(__name__)
        self._callback = callback
        self._start_time = time.time()
        self._current_progress = 0
        self._total_progress = 1
        self._current_stage = "idle"
        self._current_substage = None
        self._current_message = ""
        self._is_complete = False
        self._success = False
        self._error = None
        
    def set_callback(self, callback: Union[ProgressCallback, Dict[str, Callable]]) -> None:
        """Set callback untuk progress tracking"""
        self._callback = callback
        self.logger.debug(f"🔄 Progress callback diatur: {type(callback).__name__}")
        
    def update(self, current: int, total: int, message: str, phase: str = "general") -> None:
        """Update progress dengan current/total dan pesan"""
        self._current_progress = current
        self._total_progress = max(total, 1)  # Hindari division by zero
        self._current_message = message
        
        # Log progress hanya pada perubahan signifikan (5% atau lebih)
        progress_pct = (current / max(total, 1)) * 100
        if current == 0 or current == total or current % max(1, total // 20) == 0:
            self.logger.info(f"📊 Progress [{phase}]: {progress_pct:.1f}% ({current}/{total}) - {message}")
            
        # Panggil callback jika tersedia
        self._call_progress_callback(current, total, message, phase)
        
    def update_status(self, status: str, phase: str = "general") -> None:
        """Update status message tanpa mengubah progress"""
        self._current_message = status
        self.logger.info(f"ℹ️ Status [{phase}]: {status}")
        
        # Panggil callback jika tersedia
        self._call_status_callback(status, phase)
        
    def update_stage(self, stage: str, substage: Optional[str] = None) -> None:
        """Update stage dan substage proses"""
        self._current_stage = stage
        self._current_substage = substage
        self.logger.info(f"🔄 Stage: {stage}" + (f" - Substage: {substage}" if substage else ""))
        
        # Panggil callback jika tersedia
        self._call_stage_callback(stage, substage)
        
    def complete(self, success: bool, message: str) -> None:
        """Tandai proses sebagai selesai dengan status success/failure"""
        self._is_complete = True
        self._success = success
        self._current_message = message
        
        # Update progress ke 100% jika sukses
        if success: self._current_progress = self._total_progress
        
        # Log completion
        elapsed = time.time() - self._start_time
        status_emoji = "✅" if success else "❌"
        self.logger.info(f"{status_emoji} Proses selesai ({elapsed:.2f}s): {message}")
        
        # Panggil callback jika tersedia
        self._call_complete_callback(success, message)
        
    def error(self, error_message: str, phase: str = "general") -> None:
        """Catat error dan update status"""
        self._error = error_message
        self._success = False
        self._current_message = f"Error: {error_message}"
        
        # Log error
        self.logger.error(f"❌ Error [{phase}]: {error_message}")
        
        # Panggil callback jika tersedia
        self._call_error_callback(error_message, phase)
        
    def reset(self) -> None:
        """Reset progress tracker untuk digunakan kembali"""
        self._start_time = time.time()
        self._current_progress = 0
        self._total_progress = 1
        self._current_stage = "idle"
        self._current_substage = None
        self._current_message = ""
        self._is_complete = False
        self._success = False
        self._error = None
        self.logger.debug("🔄 Progress tracker direset")
        
    def get_progress_percentage(self) -> float:
        """Dapatkan persentase progress saat ini"""
        return (self._current_progress / max(self._total_progress, 1)) * 100
        
    def get_elapsed_time(self) -> float:
        """Dapatkan waktu yang telah berlalu sejak mulai dalam detik"""
        return time.time() - self._start_time
        
    def get_estimated_time_remaining(self) -> Optional[float]:
        """Estimasi waktu yang tersisa berdasarkan progress saat ini"""
        if self._current_progress <= 0: return None
        
        elapsed = self.get_elapsed_time()
        progress_ratio = self._current_progress / max(self._total_progress, 1)
        
        if progress_ratio <= 0: return None
        
        total_estimated_time = elapsed / progress_ratio
        return max(0, total_estimated_time - elapsed)
    
    def get_status(self) -> Dict[str, Any]:
        """Dapatkan status lengkap progress saat ini"""
        return {
            "current": self._current_progress,
            "total": self._total_progress,
            "percentage": self.get_progress_percentage(),
            "message": self._current_message,
            "stage": self._current_stage,
            "substage": self._current_substage,
            "elapsed_time": self.get_elapsed_time(),
            "estimated_time_remaining": self.get_estimated_time_remaining(),
            "is_complete": self._is_complete,
            "success": self._success,
            "error": self._error
        }
    
    # Helper methods untuk memanggil callback dengan berbagai format
    def _call_progress_callback(self, current: int, total: int, message: str, phase: str) -> None:
        """Panggil callback progress dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah ProgressCallback interface
            if hasattr(self._callback, 'update_progress'):
                self._callback.update_progress(current, total, message, phase)
                return
                
            # Jika callback adalah dict dengan fungsi progress
            if isinstance(self._callback, dict) and 'progress' in self._callback:
                self._callback['progress'](current, total, message, phase)
                return
                
            # Jika callback adalah fungsi sederhana
            if callable(self._callback):
                self._callback(current, total, message, phase)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil progress callback: {str(e)}")
    
    def _call_status_callback(self, status: str, phase: str) -> None:
        """Panggil callback status dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah ProgressCallback interface
            if hasattr(self._callback, 'update_status'):
                self._callback.update_status(status, phase)
                return
                
            # Jika callback adalah dict dengan fungsi status
            if isinstance(self._callback, dict) and 'status' in self._callback:
                self._callback['status'](status, phase)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil status callback: {str(e)}")
    
    def _call_stage_callback(self, stage: str, substage: Optional[str]) -> None:
        """Panggil callback stage dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah ProgressCallback interface
            if hasattr(self._callback, 'update_stage'):
                self._callback.update_stage(stage, substage)
                return
                
            # Jika callback adalah dict dengan fungsi stage
            if isinstance(self._callback, dict) and 'stage' in self._callback:
                self._callback['stage'](stage, substage)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil stage callback: {str(e)}")
    
    def _call_complete_callback(self, success: bool, message: str) -> None:
        """Panggil callback complete dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah ProgressCallback interface
            if hasattr(self._callback, 'on_complete'):
                self._callback.on_complete(success, message)
                return
                
            # Jika callback adalah dict dengan fungsi complete
            if isinstance(self._callback, dict) and 'complete' in self._callback:
                self._callback['complete'](success, message)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil complete callback: {str(e)}")
    
    def _call_error_callback(self, error_message: str, phase: str) -> None:
        """Panggil callback error dengan berbagai format yang didukung"""
        if not self._callback: return
        
        try:
            # Jika callback adalah ProgressCallback interface
            if hasattr(self._callback, 'on_error'):
                self._callback.on_error(error_message, phase)
                return
                
            # Jika callback adalah dict dengan fungsi error
            if isinstance(self._callback, dict) and 'error' in self._callback:
                self._callback['error'](error_message, phase)
                return
        except Exception as e:
            self.logger.warning(f"⚠️ Error memanggil error callback: {str(e)}")
            
    # Properties untuk akses mudah
    @property
    def progress(self) -> float: return self.get_progress_percentage()
    @property
    def is_complete(self) -> bool: return self._is_complete
    @property
    def is_success(self) -> bool: return self._success
    @property
    def current_message(self) -> str: return self._current_message
    @property
    def current_stage(self) -> str: return self._current_stage
    @property
    def elapsed_time(self) -> float: return self.get_elapsed_time()
    @property
    def estimated_time_remaining(self) -> Optional[float]: return self.get_estimated_time_remaining()
