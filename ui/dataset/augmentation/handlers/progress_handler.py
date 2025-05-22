"""
File: smartcash/ui/dataset/augmentation/handlers/progress_handler.py
Deskripsi: Handler untuk progress tracking dan komunikasi dengan service (SRP)
"""

import time
from typing import Dict, Any, Callable

class ProgressHandler:
    """Handler untuk mengelola progress UI dan komunikasi dengan service."""
    
    def __init__(self, ui_components: Dict[str, Any], ui_logger):
        """
        Inisialisasi progress handler.
        
        Args:
            ui_components: Dictionary komponen UI
            ui_logger: UI Logger bridge
        """
        self.ui_components = ui_components
        self.ui_logger = ui_logger
        self.last_update_time = 0
        self.update_interval = 0.5  # Minimum interval antar update
        
    def start_progress(self, message: str) -> None:
        """Mulai progress tracking."""
        self._show_progress_container()
        self.update_progress(0, message)
        self.ui_logger.info(f"📊 {message}")
    
    def update_progress(self, value: int, message: str = "") -> None:
        """Update progress bar dan message."""
        current_time = time.time()
        
        # Throttle updates untuk menghindari spam
        if current_time - self.last_update_time < self.update_interval:
            return
            
        value = max(0, min(100, value))
        
        # Update progress bar
        if 'progress_bar' in self.ui_components:
            progress_bar = self.ui_components['progress_bar']
            if hasattr(progress_bar, 'value'):
                progress_bar.value = value
                progress_bar.description = f"Progress: {value}%"
        
        # Update message labels
        if message:
            for label_key in ['progress_message', 'step_label', 'overall_label']:
                if label_key in self.ui_components:
                    label_widget = self.ui_components[label_key]
                    if hasattr(label_widget, 'value'):
                        label_widget.value = message
        
        self.last_update_time = current_time
    
    def complete_progress(self, message: str, success: bool = True) -> None:
        """Selesaikan progress tracking."""
        final_progress = 100 if success else self.ui_components.get('progress_bar', {}).get('value', 0)
        self.update_progress(final_progress, message)
        
        if success:
            self.ui_logger.success(f"🎉 {message}")
        else:
            self.ui_logger.error(f"❌ {message}")
        
        # Hide progress setelah delay singkat jika berhasil
        if success:
            self._hide_progress_after_delay()
    
    def create_service_callback(self) -> Callable:
        """
        Buat callback function untuk komunikasi dengan service.
        
        Returns:
            Callback function yang kompatibel dengan service
        """
        def service_callback(current: int, total: int, message: str = "", **kwargs) -> bool:
            # Cek stop signal
            if self.ui_components.get('stop_requested', False):
                self.ui_logger.warning("⏹️ Stop request detected")
                return False
            
            # Hitung progress percentage (15-95% range untuk augmentasi)
            if total > 0:
                # Map service progress (0-100) ke UI progress range (15-95)
                service_progress = min(100, (current / total) * 100)
                ui_progress = 15 + int(service_progress * 0.8)  # 15 + (0-100 * 0.8) = 15-95
                
                # Format message
                if message:
                    display_message = f"{message} ({current}/{total})"
                else:
                    display_message = f"Memproses: {current}/{total}"
                
                self.update_progress(ui_progress, display_message)
            
            return True  # Continue processing
        
        return service_callback
    
    def _show_progress_container(self) -> None:
        """Tampilkan progress container."""
        if 'progress_container' in self.ui_components:
            container = self.ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.display = 'block'
        
        # Show individual progress elements
        for element_key in ['progress_bar', 'progress_message', 'step_label']:
            if element_key in self.ui_components:
                element = self.ui_components[element_key]
                if hasattr(element, 'layout'):
                    element.layout.visibility = 'visible'
    
    def _hide_progress_after_delay(self) -> None:
        """Sembunyikan progress setelah delay singkat."""
        # Untuk Colab, langsung hide tanpa threading
        if 'progress_container' in self.ui_components:
            container = self.ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.display = 'none'
    
    def reset_progress(self) -> None:
        """Reset progress ke kondisi awal."""
        self.update_progress(0, "")
        
        # Hide progress elements
        for element_key in ['progress_bar', 'progress_message', 'step_label', 'progress_container']:
            if element_key in self.ui_components:
                element = self.ui_components[element_key]
                if hasattr(element, 'layout'):
                    if element_key == 'progress_container':
                        element.layout.display = 'none'
                    else:
                        element.layout.visibility = 'hidden'