"""
File: smartcash/ui/dataset/augmentation/utils/progress_coordinator.py
Deskripsi: Koordinator progress untuk menghindari bertabrakan antar level
"""

import time
from typing import Dict, Any, Optional, Callable
from smartcash.ui.dataset.augmentation.utils.logger_helper import log_message

class ProgressCoordinator:
    """Coordinator untuk mengelola progress reporting tanpa bertabrakan."""
    
    def __init__(self, ui_components: Dict[str, Any]):
        """
        Inisialisasi progress coordinator.
        
        Args:
            ui_components: Dictionary komponen UI
        """
        self.ui_components = ui_components
        self.last_update_time = 0
        self.update_interval = 0.5  # Minimum 0.5 detik antar update
        self.current_step = ""
        self.total_steps = 0
        self.completed_steps = 0
        self.is_active = False
    
    def start_progress(self, total_steps: int, initial_message: str = "Memulai proses...") -> None:
        """
        Mulai progress tracking.
        
        Args:
            total_steps: Total langkah yang akan dikerjakan
            initial_message: Pesan awal
        """
        self.total_steps = total_steps
        self.completed_steps = 0
        self.is_active = True
        self.last_update_time = time.time()
        
        self._update_ui_progress(0, initial_message)
        log_message(self.ui_components, f"🚀 {initial_message}", "info")
    
    def update_step(self, step_name: str, progress: int = None, message: str = None) -> None:
        """
        Update progress untuk step tertentu.
        
        Args:
            step_name: Nama step saat ini
            progress: Progress dalam persen (0-100)
            message: Pesan tambahan
        """
        if not self.is_active:
            return
        
        current_time = time.time()
        
        # Throttle update untuk menghindari spam
        if current_time - self.last_update_time < self.update_interval:
            return
        
        self.current_step = step_name
        
        if progress is not None:
            self.completed_steps = min(progress, 100)
        
        display_message = message or f"Memproses: {step_name}"
        
        self._update_ui_progress(self.completed_steps, display_message)
        self.last_update_time = current_time
    
    def complete_step(self, step_name: str, success: bool = True) -> None:
        """
        Tandai step sebagai selesai.
        
        Args:
            step_name: Nama step yang selesai
            success: Apakah step berhasil
        """
        if not self.is_active:
            return
        
        status = "success" if success else "error"
        icon = "✅" if success else "❌"
        
        log_message(self.ui_components, f"{icon} {step_name} selesai", status)
    
    def finish_progress(self, final_message: str = "Proses selesai", success: bool = True) -> None:
        """
        Selesaikan progress tracking.
        
        Args:
            final_message: Pesan akhir
            success: Apakah proses berhasil
        """
        if not self.is_active:
            return
        
        self.is_active = False
        final_progress = 100 if success else self.completed_steps
        
        self._update_ui_progress(final_progress, final_message)
        
        status = "success" if success else "error"
        icon = "🎉" if success else "❌"
        
        log_message(self.ui_components, f"{icon} {final_message}", status)
        
        # Hide progress setelah delay singkat jika berhasil
        if success and 'progress_container' in self.ui_components:
            import threading
            def hide_progress():
                time.sleep(2)
                if hasattr(self.ui_components['progress_container'], 'layout'):
                    self.ui_components['progress_container'].layout.display = 'none'
            
            # Tidak menggunakan threading di Colab, langsung hide
            if hasattr(self.ui_components.get('progress_container', {}), 'layout'):
                self.ui_components['progress_container'].layout.display = 'none'
    
    def _update_ui_progress(self, progress: int, message: str) -> None:
        """Update UI progress bar dan message."""
        # Update progress bar
        if 'progress_bar' in self.ui_components and hasattr(self.ui_components['progress_bar'], 'value'):
            self.ui_components['progress_bar'].value = progress
            self.ui_components['progress_bar'].description = f"Progress: {progress}%"
        
        # Update message labels
        for label_key in ['progress_message', 'step_label', 'overall_label']:
            if label_key in self.ui_components and hasattr(self.ui_components[label_key], 'value'):
                self.ui_components[label_key].value = message
        
        # Show progress container
        if 'progress_container' in self.ui_components and hasattr(self.ui_components['progress_container'], 'layout'):
            self.ui_components['progress_container'].layout.display = 'block'
    
    def create_service_callback(self) -> Callable:
        """
        Buat callback function untuk service level.
        
        Returns:
            Callback function yang kompatibel dengan service
        """
        def service_callback(current: int, total: int, message: str = "", **kwargs) -> bool:
            # Cek stop signal
            if self.ui_components.get('stop_requested', False):
                log_message(self.ui_components, "⏹️ Stop request detected", "warning")
                return False
            
            # Hitung progress percentage
            if total > 0:
                progress = int((current / total) * 100)
                self.update_step("Augmentasi", progress, f"{message} ({current}/{total})")
            
            return True  # Continue processing
        
        return service_callback
    
    def is_stop_requested(self) -> bool:
        """Cek apakah ada permintaan stop."""
        return self.ui_components.get('stop_requested', False)
    
    def reset_progress(self) -> None:
        """Reset progress ke kondisi awal."""
        self.is_active = False
        self.completed_steps = 0
        self.current_step = ""
        
        # Reset UI elements
        if 'progress_bar' in self.ui_components and hasattr(self.ui_components['progress_bar'], 'value'):
            self.ui_components['progress_bar'].value = 0
            self.ui_components['progress_bar'].description = "Progress: 0%"
        
        for label_key in ['progress_message', 'step_label', 'overall_label']:
            if label_key in self.ui_components and hasattr(self.ui_components[label_key], 'value'):
                self.ui_components[label_key].value = ""
        
        # Hide progress container
        if 'progress_container' in self.ui_components and hasattr(self.ui_components['progress_container'], 'layout'):
            self.ui_components['progress_container'].layout.display = 'none'

def get_progress_coordinator(ui_components: Dict[str, Any]) -> ProgressCoordinator:
    """
    Factory function untuk mendapatkan progress coordinator.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Instance ProgressCoordinator
    """
    if 'progress_coordinator' not in ui_components:
        ui_components['progress_coordinator'] = ProgressCoordinator(ui_components)
    
    return ui_components['progress_coordinator']