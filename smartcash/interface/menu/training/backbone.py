# File: smartcash/interface/menu/training/backbone.py
# Author: Alfrida Sabar
# Deskripsi: Menu pemilihan arsitektur backbone model dengan perbaikan error handling

from smartcash.interface.menu.base import BaseMenu, MenuItem
import curses
from typing import Dict, Any
from pathlib import Path

class BackboneMenu(BaseMenu):
    """Menu pemilihan arsitektur backbone."""
    
    BACKBONE_INFO = {
        'cspdarknet': {
            'name': 'CSPDarknet',
            'description': (
                "Backbone standar YOLOv5\n"
                "• Kecepatan inferensi tinggi\n"
                "• Ukuran model lebih kecil\n"
                "• Parameter lebih sedikit"
            ),
            'compatible_modes': ['single', 'multi']
        },
        'efficientnet': {
            'name': 'EfficientNet-B4',
            'description': (
                "Backbone EfficientNet-B4\n"
                "• Akurasi lebih tinggi\n"
                "• Compound scaling untuk feature extraction\n"
                "• Performa lebih baik pada detail kecil"
            ),
            'compatible_modes': ['single', 'multi']
        }
    }
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        # Get current detection mode to check compatibility
        current_mode = self.config_manager.current_config.get('detection_mode')
        
        items = []
        for backbone_id, info in self.BACKBONE_INFO.items():
            # Check if backbone compatible with current mode
            enabled = (
                current_mode is None or 
                current_mode in info['compatible_modes']
            )
            
            items.append(
                MenuItem(
                    title=info['name'],
                    action=lambda b=backbone_id: self._set_backbone(b),
                    description=info['description'],
                    category="Arsitektur",
                    enabled=enabled
                )
            )
            
        items.append(
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        )
        
        super().__init__("Pilih Arsitektur Model", items)
        
    def _set_backbone(self, backbone: str) -> bool:
        """
        Set arsitektur backbone model.
        
        Args:
            backbone: ID arsitektur yang dipilih
            
        Returns:
            bool: True jika berhasil
        """
        try:
            # Memastikan backbone valid sebelum melakukan update
            if backbone not in self.BACKBONE_INFO:
                self.display.show_error(f"Arsitektur tidak valid: {backbone}")
                return True
                
            # Cek kompatibilitas dengan mode deteksi
            current_mode = self.config_manager.current_config.get('detection_mode')
            if current_mode and current_mode not in self.BACKBONE_INFO[backbone]['compatible_modes']:
                self.display.show_error(
                    f"Arsitektur {backbone} tidak kompatibel dengan mode {current_mode}"
                )
                return True
                
            # Update backbone di config
            self.config_manager.update('backbone', backbone)
            
            # Update model config sesuai backbone dengan channels yang benar
            model_config = {
                'cspdarknet': {
                    'pretrained': True,
                    'feature_channels': [128, 256, 512]
                },
                'efficientnet': {
                    'pretrained': True,
                    'feature_channels': [56, 160, 448]
                }
            }
            
            # Update konfigurasi model dengan error handling yang lebih baik
            try:
                self.config_manager.update('model', model_config[backbone])
            except Exception as model_error:
                self.display.show_error(f"Gagal mengatur konfigurasi model: {str(model_error)}")
                return True
            
            # Save changes with error handling
            try:
                self.config_manager.save()
                # Show feedback
                self.display.show_success(
                    f"Arsitektur diatur ke: {self.BACKBONE_INFO[backbone]['name']}"
                )
            except Exception as save_error:
                self.display.show_error(f"Gagal menyimpan konfigurasi: {str(save_error)}")
                
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mengatur arsitektur: {str(e)}")
            return True
            
    def draw(self, stdscr, start_y: int) -> None:
        """Override draw untuk menambahkan info kompatibilitas."""
        super().draw(stdscr, start_y)
        
        # Show compatibility warning if needed
        current_mode = self.config_manager.current_config.get('detection_mode')
        if current_mode:
            height, width = stdscr.getmaxyx()
            warning_y = height - 5
            
            stdscr.attron(curses.color_pair(3))  # Yellow
            stdscr.addstr(
                warning_y, 
                2, 
                f"ℹ️ Mode deteksi saat ini: {current_mode}"
            )
            stdscr.attroff(curses.color_pair(3))