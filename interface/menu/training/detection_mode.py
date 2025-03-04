# File: smartcash/interface/menu/training/detection_mode.py
# Author: Alfrida Sabar
# Deskripsi: Menu pemilihan mode deteksi (single/multi layer)

from smartcash.interface.menu.base import BaseMenu, MenuItem

class DetectionModeMenu(BaseMenu):
    """Menu pemilihan mode deteksi mata uang."""
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        items = [
            MenuItem(
                title="Deteksi Lapis Tunggal",
                action=lambda: self._set_detection_mode('single'),
                description=(
                    "Layer: [banknote]"
                ),
                category="Mode Deteksi"
            ),
            MenuItem(
                title="Deteksi Lapis Banyak",
                action=lambda: self._set_detection_mode('multi'),
                description=(
                    "Layers: [banknote, nominal, security]"
                ),
                category="Mode Deteksi"
            ),
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        ]
        
        super().__init__("Pilih Mode Deteksi", items)
        
    def _set_detection_mode(self, mode: str) -> bool:
        """
        Set mode deteksi dan layer yang sesuai.
        
        Args:
            mode: Mode deteksi ('single' atau 'multi')
        
        Returns:
            bool: True jika berhasil
        """
        try:
            # Update detection mode
            self.config_manager.update('detection_mode', mode)
            
            # Update layers sesuai mode
            if mode == 'single':
                layers = ['banknote']
            else:
                layers = ['banknote', 'nominal', 'security']
                
            self.config_manager.update('layers', layers)
            
            # Save changes
            self.config_manager.save()
            
            # Show feedback
            self.display.show_success(
                f"Mode deteksi diatur ke: {mode} ({', '.join(layers)})"
            )
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mengatur mode deteksi: {str(e)}")
            return True