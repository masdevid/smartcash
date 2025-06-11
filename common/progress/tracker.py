"""
File: smartcash/common/progress/tracker.py
Deskripsi: Implementasi terpadu untuk progress tracking dengan dukungan tqdm, callback dan clean API
"""

import time
from typing import Dict, Any, List, Optional, Callable, Union
from tqdm.auto import tqdm

def format_time(seconds: float) -> str:
    """Format waktu dalam detik ke string yang mudah dibaca."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

class ProgressTracker:
    """Tracker progres untuk operasi dengan dukungan tqdm, callback dan reporting."""
    
    def __init__(
        self,
        total: int = 0,
        desc: str = "",
        unit: str = "it",
        display: bool = True,
        logger = None,
        parent_tracker = None
    ):
        """
        Inisialisasi ProgressTracker.
        
        Args:
            total: Total unit yang akan diproses
            desc: Deskripsi progres
            unit: Unit untuk progress bar
            display: Apakah menampilkan progress bar
            logger: Logger kustom (opsional)
            parent_tracker: Tracker induk untuk operasi bersarang (opsional)
        """
        self.total = total
        self.desc = desc
        self.unit = unit
        self.display = display
        
        # Setup logger jika tersedia
        if logger is None:
            try:
                from smartcash.common.logger import get_logger
                self.logger = get_logger()
            except ImportError:
                import logging
                self.logger = logging.getLogger("progress_tracker")
        else:
            self.logger = logger
            
        self.parent_tracker = parent_tracker
        
        # Init state
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.is_completed = False
        self.pbar = None
        self.callbacks = []
        self.subtasks = []
        self.metrics = {}
        
        # Buat progress bar jika diminta
        if self.display and self.total > 0:
            self.pbar = tqdm(total=self.total, desc=self.desc, unit=self.unit)
    
    def update(self, n: int = 1, message: Optional[str] = None) -> None:
        """
        Update progres.
        
        Args:
            n: Jumlah unit yang diselesaikan
            message: Pesan status (opsional)
        """
        self.current += n
        self.last_update_time = time.time()
        
        if self.pbar:
            self.pbar.update(n)
            if message:
                self.pbar.set_description(f"{self.desc} - {message}")
                
        self._execute_callbacks()
        
        # Update parent tracker jika ada
        if self.parent_tracker and n > 0:
            portion = n / self.total if self.total > 0 else 0
            parent_update = max(1, int(portion * self.parent_tracker.total))
            self.parent_tracker.update(parent_update)
    
    def set_total(self, total: int) -> None:
        """
        Set total unit yang akan diproses.
        
        Args:
            total: Total unit
        """
        self.total = total
        if self.pbar:
            self.pbar.total = total
            self.pbar.refresh()
    
    def reset(self) -> None:
        """
        Reset progress tracker ke kondisi awal.
        
        Mengembalikan current progress ke 0 dan me-reset timer
        tanpa mengubah total atau deskripsi.
        """
        # Reset state
        self.current = 0
        self.start_time = time.time()
        self.last_update_time = self.start_time
        self.is_completed = False
        
        # Reset progress bar jika ada
        if self.pbar:
            self.pbar.reset()
            self.pbar.refresh()
        
        # Reset metrics tapi pertahankan konfigurasi
        self.metrics = {}
        
        # Notifikasi callbacks
        self._execute_callbacks()
        
        self.logger.debug(f" Progress tracker '{self.desc}' direset")
    
    def add_callback(self, callback: Callable[[Dict[str, Any]], None]) -> None:
        """
        Tambahkan callback function yang akan dipanggil pada setiap update.
        
        Args:
            callback: Fungsi callback yang menerima dictionary progres
        """
        self.callbacks.append(callback)
    
    def add_subtask(self, subtask: 'ProgressTracker') -> None:
        """
        Tambahkan subtask ke tracker induk.
        
        Args:
            subtask: ProgressTracker subtask
        """
        self.subtasks.append(subtask)
        subtask.parent_tracker = self
    
    def set_metrics(self, metrics: Dict[str, Any]) -> None:
        """
        Set metrik pelacakan tambahan.
        
        Args:
            metrics: Dictionary metrik
        """
        self.metrics.update(metrics)
        self._execute_callbacks()
    
    def get_progress(self) -> Dict[str, Any]:
        """
        Dapatkan status progres saat ini.
        
        Returns:
            Dictionary berisi status progres
        """
        elapsed = time.time() - self.start_time
        progress = min(1.0, self.current / max(1, self.total))
        remaining = (elapsed / max(0.001, progress)) * (1 - progress) if progress > 0 else 0
        
        result = {
            'desc': self.desc,
            'current': self.current,
            'total': self.total,
            'progress': progress,
            'progress_pct': progress * 100,
            'elapsed': elapsed,
            'elapsed_str': format_time(elapsed),
            'remaining': remaining,
            'remaining_str': format_time(remaining),
            'is_completed': self.is_completed,
            'metrics': self.metrics.copy()
        }
        
        # Tambahkan status subtask jika ada
        if self.subtasks:
            result['subtasks'] = [subtask.get_progress() for subtask in self.subtasks]
            
        return result
    
    def complete(self, message: Optional[str] = None) -> None:
        """
        Tandai progres sebagai selesai.
        
        Args:
            message: Pesan selesai (opsional)
        """
        if self.is_completed:
            return
            
        self.is_completed = True
        self.current = self.total
        
        if self.pbar:
            self.pbar.n = self.total
            if message:
                self.pbar.set_description(f"{self.desc} - {message}")
            self.pbar.close()
            self.pbar = None
            
        elapsed = time.time() - self.start_time
        self.logger.info(f" {self.desc} selesai dalam {format_time(elapsed)}")
        
        self._execute_callbacks()
    
    def _execute_callbacks(self) -> None:
        """Eksekusi semua callback yang terdaftar."""
        progress_info = self.get_progress()
        for callback in self.callbacks:
            try:
                callback(progress_info)
            except Exception as e:
                self.logger.warning(f" Error pada callback: {str(e)}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if not self.is_completed:
            if exc_type:
                self.metrics['error'] = str(exc_val)
                if self.pbar:
                    self.pbar.set_description(f"{self.desc} - Error: {str(exc_val)}")
            self.complete()

# Singleton instance
_progress_trackers = {}

def get_progress_tracker(
    name: str, 
    total: int = 0, 
    desc: str = "", 
    unit: str = "it",
    display: bool = True,
    logger = None
) -> ProgressTracker:
    """
    Dapatkan instance ProgressTracker (singleton per nama).
    
    Args:
        name: Nama unik untuk tracker
        total: Total unit yang akan diproses
        desc: Deskripsi progres
        unit: Unit untuk progress bar
        display: Apakah menampilkan progress bar
        logger: Logger kustom (opsional)
        
    Returns:
        Instance ProgressTracker
    """
    global _progress_trackers
    if name not in _progress_trackers:
        _progress_trackers[name] = ProgressTracker(total, desc, unit, display, logger)
    return _progress_trackers[name]