"""
File: smartcash/ui/dataset/augmentation/utils/stop_signal_manager.py
Deskripsi: Manager untuk propagasi stop signal dari UI sampai worker level
"""

import threading
from typing import Dict, Any, Optional, Callable
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class StopSignalManager:
    """Manager untuk mengelola stop signal propagation."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi stop signal manager.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self._stop_requested = False
        self._stop_callbacks = []
        self._lock = threading.Lock()  # Thread-safe access
    
    def request_stop(self, reason: str = "User requested") -> None:
        """
        Request stop untuk semua proses yang berjalan.
        
        Args:
            reason: Alasan stop request
        """
        with self._lock:
            if self._stop_requested:
                return  # Already requested
            
            self._stop_requested = True
            self.ui_components['stop_requested'] = True
            self.ui_components['augmentation_running'] = False
        
        log_message(self.ui_components, f"⏹️ Stop request: {reason}", "warning")
        
        # Panggil semua callbacks
        self._notify_stop_callbacks(reason)
    
    def is_stop_requested(self) -> bool:
        """
        Cek apakah ada stop request.
        
        Returns:
            True jika ada stop request
        """
        with self._lock:
            return self._stop_requested or self.ui_components.get('stop_requested', False)
    
    def reset_stop_signal(self) -> None:
        """Reset stop signal untuk proses baru."""
        with self._lock:
            self._stop_requested = False
            self.ui_components['stop_requested'] = False
            self._stop_callbacks.clear()
        
        log_message(self.ui_components, "🔄 Stop signal direset", "debug")
    
    def register_stop_callback(self, callback: Callable[[str], None]) -> None:
        """
        Register callback yang akan dipanggil saat stop request.
        
        Args:
            callback: Fungsi callback dengan parameter reason
        """
        with self._lock:
            if callback not in self._stop_callbacks:
                self._stop_callbacks.append(callback)
    
    def unregister_stop_callback(self, callback: Callable[[str], None]) -> None:
        """
        Unregister stop callback.
        
        Args:
            callback: Fungsi callback yang akan di-unregister
        """
        with self._lock:
            if callback in self._stop_callbacks:
                self._stop_callbacks.remove(callback)
    
    def _notify_stop_callbacks(self, reason: str) -> None:
        """
        Notify semua registered callbacks.
        
        Args:
            reason: Alasan stop request
        """
        callbacks_to_call = []
        with self._lock:
            callbacks_to_call = self._stop_callbacks.copy()
        
        for callback in callbacks_to_call:
            try:
                callback(reason)
            except Exception as e:
                log_message(self.ui_components, f"⚠️ Error dalam stop callback: {str(e)}", "warning")
    
    def create_progress_callback_with_stop_check(self, original_callback: Optional[Callable] = None) -> Callable:
        """
        Buat progress callback yang otomatis cek stop signal.
        
        Args:
            original_callback: Callback asli yang akan dibungkus
            
        Returns:
            Wrapped callback yang cek stop signal
        """
        def wrapped_callback(*args, **kwargs) -> bool:
            # Cek stop signal terlebih dahulu
            if self.is_stop_requested():
                log_message(self.ui_components, "⏹️ Stop signal detected dalam progress callback", "warning")
                return False  # Signal to stop processing
            
            # Panggil original callback jika ada
            if original_callback:
                try:
                    return original_callback(*args, **kwargs)
                except Exception as e:
                    log_message(self.ui_components, f"❌ Error dalam progress callback: {str(e)}", "error")
                    return False
            
            return True  # Continue processing
        
        return wrapped_callback
    
    def create_worker_stop_checker(self) -> Callable[[], bool]:
        """
        Buat function untuk cek stop signal di worker level.
        
        Returns:
            Function yang return True jika harus stop
        """
        def should_stop() -> bool:
            return self.is_stop_requested()
        
        return should_stop
    
    def wait_for_stop_or_completion(self, timeout: Optional[float] = None) -> bool:
        """
        Wait sampai stop request atau timeout.
        
        Args:
            timeout: Timeout dalam detik (None = no timeout)
            
        Returns:
            True jika stop requested, False jika timeout
        """
        import time
        
        start_time = time.time()
        
        while not self.is_stop_requested():
            if timeout and (time.time() - start_time) >= timeout:
                return False  # Timeout
            
            time.sleep(0.1)  # Sleep sebentar untuk tidak busy wait
        
        return True  # Stop requested
    
    def get_stop_status(self) -> Dict[str, Any]:
        """
        Dapatkan status lengkap stop signal.
        
        Returns:
            Dictionary status stop signal
        """
        with self._lock:
            return {
                'stop_requested': self._stop_requested,
                'ui_stop_requested': self.ui_components.get('stop_requested', False),
                'augmentation_running': self.ui_components.get('augmentation_running', False),
                'registered_callbacks': len(self._stop_callbacks)
            }

class WorkerStopSignal:
    """Stop signal yang bisa digunakan di worker level."""
    
    def __init__(self, stop_checker: Callable[[], bool]):
        """
        Inisialisasi worker stop signal.
        
        Args:
            stop_checker: Function untuk cek apakah harus stop
        """
        self.should_stop = stop_checker
        self.stopped = False
    
    def check_and_stop_if_needed(self) -> bool:
        """
        Cek stop signal dan set status jika perlu stop.
        
        Returns:
            True jika harus stop processing
        """
        if not self.stopped and self.should_stop():
            self.stopped = True
            return True
        
        return self.stopped
    
    def is_stopped(self) -> bool:
        """Cek apakah sudah dalam status stopped."""
        return self.stopped

def get_stop_signal_manager(ui_components: Dict[str, Any]) -> StopSignalManager:
    """
    Factory function untuk mendapatkan stop signal manager.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance StopSignalManager
    """
    if 'stop_signal_manager' not in ui_components:
        ui_components['stop_signal_manager'] = StopSignalManager(ui_components)
    
    return ui_components['stop_signal_manager']