"""
File: smartcash/ui/dataset/augmentation/handlers/state_handler.py
Deskripsi: Handler untuk mengelola state UI dan stop signals (SRP)
"""

from typing import Dict, Any

class StateHandler:
    """Handler untuk mengelola state augmentasi dan UI controls."""
    
    def __init__(self, ui_components: Dict[str, Any], ui_logger):
        """
        Inisialisasi state handler.
        
        Args:
            ui_components: Dictionary komponen UI
            ui_logger: UI Logger bridge
        """
        self.ui_components = ui_components
        self.ui_logger = ui_logger
    
    def is_running(self) -> bool:
        """Cek apakah augmentasi sedang berjalan."""
        return self.ui_components.get('augmentation_running', False)
    
    def set_running(self, running: bool) -> None:
        """Set state augmentasi running dan update UI."""
        self.ui_components['augmentation_running'] = running
        
        if running:
            self._update_ui_before_augmentation()
            self.ui_logger.debug("🟢 State: Running")
        else:
            self._reset_ui_after_augmentation()
            self.ui_logger.debug("🔴 State: Stopped")
    
    def reset_signals(self) -> None:
        """Reset stop signal sebelum memulai."""
        self.ui_components['stop_requested'] = False
        self.ui_logger.debug("🔄 Stop signal direset")
    
    def request_stop(self, reason: str = "User requested") -> None:
        """Request stop untuk proses yang berjalan."""
        self.ui_components['stop_requested'] = True
        self.ui_components['augmentation_running'] = False
        self.ui_logger.warning(f"⏹️ Stop request: {reason}")
    
    def is_stop_requested(self) -> bool:
        """Cek apakah ada stop request."""
        return self.ui_components.get('stop_requested', False)
    
    def _update_ui_before_augmentation(self) -> None:
        """Update UI sebelum proses augmentasi dimulai."""
        # Nonaktifkan tombol kecuali stop
        self._disable_buttons(True)
        
        # Show stop button, hide augment button
        self._toggle_action_buttons(show_stop=True)
        
        # Show progress container
        if 'progress_container' in self.ui_components:
            container = self.ui_components['progress_container']
            if hasattr(container, 'layout'):
                container.layout.display = 'block'
    
    def _reset_ui_after_augmentation(self) -> None:
        """Reset UI setelah proses augmentasi selesai."""
        # Aktifkan kembali semua tombol
        self._disable_buttons(False)
        
        # Hide stop button, show augment button
        self._toggle_action_buttons(show_stop=False)
        
        # Clear confirmation area
        if 'confirmation_area' in self.ui_components:
            self.ui_components['confirmation_area'].clear_output()
    
    def _disable_buttons(self, disabled: bool) -> None:
        """Nonaktifkan/aktifkan tombol-tombol UI."""
        button_keys = ['augment_button', 'reset_button', 'cleanup_button', 'save_button']
        
        for key in button_keys:
            if key in self.ui_components:
                button = self.ui_components[key]
                if hasattr(button, 'disabled'):
                    button.disabled = disabled
    
    def _toggle_action_buttons(self, show_stop: bool) -> None:
        """Toggle visibility tombol aksi."""
        # Stop button
        if 'stop_button' in self.ui_components:
            stop_button = self.ui_components['stop_button']
            if hasattr(stop_button, 'layout'):
                stop_button.layout.display = 'block' if show_stop else 'none'
                if hasattr(stop_button, 'disabled'):
                    stop_button.disabled = not show_stop
        
        # Augment button
        if 'augment_button' in self.ui_components:
            augment_button = self.ui_components['augment_button']
            if hasattr(augment_button, 'layout'):
                augment_button.layout.display = 'none' if show_stop else 'block'
                if hasattr(augment_button, 'disabled'):
                    augment_button.disabled = show_stop