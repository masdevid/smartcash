"""
File: smartcash/dataset/preprocessor/utils/progress_bridge.py
Deskripsi: Bridge untuk menghubungkan progress tracker dengan proses preprocessing.
"""
from typing import Optional, Dict, Any, Callable


class ProgressBridge:
    """Bridge untuk menghubungkan progress tracker dengan proses preprocessing.
    
    Attributes:
        progress_tracker: Objek untuk melacak progress
        current_progress: Progress saat ini (0-100)
        status: Status terakhir
        messages: Daftar pesan progress
    """
    
    def __init__(self, progress_tracker=None):
        """Inisialisasi ProgressBridge.
        
        Args:
            progress_tracker: Objek untuk melacak progress (opsional)
        """
        self.progress_tracker = progress_tracker
        self.current_progress = 0
        self.status = "idle"
        self.messages = []
    
    def update(self, progress: float, status: str = None, message: str = None) -> None:
        """Update progress dan status.
        
        Args:
            progress: Nilai progress (0-100)
            status: Status saat ini (opsional)
            message: Pesan detail (opsional)
        """
        # Pastikan progress dalam rentang 0-100
        self.current_progress = max(0, min(100, progress))
        
        if status:
            self.status = status
            
        if message:
            self.messages.append({"status": status, "message": message})
        
        # Panggil progress tracker jika ada
        if self.progress_tracker and hasattr(self.progress_tracker, 'update'):
            self.progress_tracker.update(
                progress=self.current_progress,
                status=status,
                message=message
            )
    
    def get_progress(self) -> Dict[str, Any]:
        """Dapatkan progress saat ini.
        
        Returns:
            Dict berisi progress, status, dan pesan
        """
        return {
            "progress": self.current_progress,
            "status": self.status,
            "messages": self.messages
        }
    
    def reset(self) -> None:
        """Reset progress tracker ke kondisi awal."""
        self.current_progress = 0
        self.status = "idle"
        self.messages = []
