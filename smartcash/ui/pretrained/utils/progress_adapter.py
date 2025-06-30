"""
File: smartcash/ui/pretrained/utils/progress_adapter.py
Deskripsi: Adapter untuk mengintegrasikan ProgressTracker dengan modul pretrained
"""

from typing import Optional, Callable, Dict, Any
from smartcash.ui.components.progress_tracker.progress_tracker import ProgressTracker
from smartcash.ui.components.progress_tracker.progress_config import ProgressConfig, ProgressLevel

class PretrainedProgressAdapter:
    """Adapter untuk mengintegrasikan ProgressTracker dengan modul pretrained.
    
    Kelas ini menyediakan antarmuka yang kompatibel dengan callback progress
    yang digunakan di modul pretrained, sambil memanfaatkan fitur-fitur
    ProgressTracker yang lebih canggih.
    """
    
    def __init__(self, progress_tracker: Optional[ProgressTracker] = None):
        """Initialize adapter with optional ProgressTracker instance.
        
        Args:
            progress_tracker: Instance ProgressTracker yang akan digunakan.
                           Jika None, akan dibuat instance baru dengan konfigurasi default.
        """
        self._progress_tracker = progress_tracker or self._create_default_tracker()
        self._current_level = 'main'
    
    def _create_default_tracker(self) -> ProgressTracker:
        """Buat ProgressTracker dengan konfigurasi default untuk modul pretrained."""
        config = ProgressConfig()
        # Konfigurasi level progress default
        config.add_level(
            name='main',
            description='Main progress',
            total=100,
            level=ProgressLevel.PRIMARY,
            visible=True
        )
        return ProgressTracker(config)
    
    def get_tracker(self) -> ProgressTracker:
        """Dapatkan instance ProgressTracker yang digunakan."""
        return self._progress_tracker
    
    def update_progress(self, progress: int, message: str = "") -> None:
        """Update progress dengan pesan opsional.
        
        Args:
            progress: Nilai progress (0-100)
            message: Pesan status opsional
        """
        if not 0 <= progress <= 100:
            raise ValueError("Progress harus antara 0 dan 100")
            
        if message:
            self._progress_tracker.update_status(self._current_level, message)
        self._progress_tracker.update_progress(self._current_level, progress)
    
    def update_status(self, message: str) -> None:
        """Update status message tanpa mengubah progress.
        
        Args:
            message: Pesan status
        """
        self._progress_tracker.update_status(self._current_level, message)
    
    def set_level(self, level_name: str) -> None:
        """Set level progress yang aktif.
        
        Args:
            level_name: Nama level yang akan diaktifkan
        """
        self._current_level = level_name
    
    def reset(self) -> None:
        """Reset progress tracker ke keadaan awal."""
        self._progress_tracker.reset()
        self._current_level = 'main'
    
    def get_progress_callback(self) -> Callable[[int, str], None]:
        """Dapatkan callback function yang kompatibel dengan ProgressCallback."""
        return self.update_progress
    
    def get_status_callback(self) -> Callable[[str], None]:
        """Dapatkan callback function yang kompatibel dengan StatusCallback."""
        return self.update_status
    
    def __enter__(self):
        """Support for context manager protocol."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context manager exit."""
        self.reset()


def create_progress_adapter(ui_components: Optional[Dict[str, Any]] = None) -> PretrainedProgressAdapter:
    """Buat instance PretrainedProgressAdapter dengan konfigurasi yang sesuai.
    
    Args:
        ui_components: Komponen UI opsional untuk dihubungkan ke progress tracker
        
    Returns:
        Instance PretrainedProgressAdapter yang sudah dikonfigurasi
    """
    adapter = PretrainedProgressAdapter()
    
    # Jika ada komponen UI, hubungkan dengan progress tracker
    if ui_components and 'progress_bar' in ui_components:
        # Pastikan progress tracker menggunakan widget yang sesuai
        adapter.get_tracker().ui_manager.register_widget(
            'main', 
            ui_components['progress_bar']
        )
    
    return adapter
