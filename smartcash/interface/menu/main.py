# File: smartcash/interface/menu/main.py
# Author: Alfrida Sabar
# Deskripsi: Implementasi menu utama SmartCash

from smartcash.interface.menu.base import BaseMenu, MenuItem

class MainMenu(BaseMenu):
    """Menu utama aplikasi."""
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        # Setup menu items
        items = [
            MenuItem(
                title="Pelatihan Model",
                action=self.app.show_training_menu,
                description="Konfigurasi dan mulai pelatihan model",
                category="Model"
            ),
            MenuItem(
                title="Evaluasi Model",
                action=self.app.show_evaluation_menu,
                description="Evaluasi performa model",
                category="Model"
            ),
            MenuItem(
                title="Keluar",
                action=lambda: False,
                description="Keluar dari aplikasi",
                category="Sistem"
            )
        ]
        
        super().__init__("SmartCash - Sistem Deteksi Uang Kertas", items)
