# File: smartcash/interface/menu/training/parameters.py
# Author: Alfrida Sabar
# Deskripsi: Menu konfigurasi parameter pelatihan model
import curses
from smartcash.interface.menu.base import BaseMenu, MenuItem

class TrainingParamsMenu(BaseMenu):
    """Menu konfigurasi parameter pelatihan."""
    
    PARAMETER_INFO = {
        'batch_size': {
            'name': 'Ukuran Batch',
            'description': (
                "Sampel data per iterasi\n"
            ),
            'validator': lambda x: 8 <= int(x) <= 128,
            'prompt': "Masukkan ukuran batch (8-128): ",
            'type': int
        },
        'learning_rate': {
            'name': 'Learning Rate',
            'description': (
                "Tingkat pembelajaran model"
            ),
            'validator': lambda x: 0.0001 <= float(x) <= 0.01,
            'prompt': "Masukkan learning rate (0.0001-0.01): ",
            'type': float
        },
        'epochs': {
            'name': 'Jumlah Epoch',
            'description': (
                "Iterasi pelatihan"
            ),
            'validator': lambda x: 10 <= int(x) <= 1000,
            'prompt': "Masukkan jumlah epoch (10-1000): ",
            'type': int
        },
        'early_stopping_patience': {
            'name': 'Early Stopping',
            'description': (
                "Iterasi tanpa perbaikan"
            ),
            'validator': lambda x: 5 <= int(x) <= 50,
            'prompt': "Masukkan nilai patience (5-50): ",
            'type': int
        }
    }
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        items = []
        for param_id, info in self.PARAMETER_INFO.items():
            items.append(
                MenuItem(
                    title=info['name'],
                    action=lambda p=param_id: self._set_parameter(p),
                    description=info['description'],
                    category="Parameter"
                )
            )
            
        # Tambah opsi reset ke default
        items.append(
            MenuItem(
                title="Reset ke Default",
                action=self._reset_parameters,
                description="Kembalikan semua parameter ke nilai default",
                category="Aksi"
            )
        )
        
        items.append(
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        )
        
        super().__init__("Konfigurasi Parameter Pelatihan", items)
        
    def _set_parameter(self, param_id: str) -> bool:
        """
        Set nilai parameter pelatihan.
        
        Args:
            param_id: ID parameter yang akan diubah
            
        Returns:
            bool: True jika berhasil
        """
        info = self.PARAMETER_INFO[param_id]
        
        try:
            # Get input value with validation
            while True:
                value = self.display.get_user_input(
                    info['prompt'],
                    validator=info['validator']
                )
                
                if value is None:  # User cancelled
                    return True
                    
                try:
                    # Convert dan validasi input
                    typed_value = info['type'](value)
                    if info['validator'](typed_value):
                        break
                except ValueError:
                    self.display.show_error("Input tidak valid")
                    continue
            
            # Update parameter in config
            self.config_manager.update(f'training.{param_id}', typed_value)
            self.config_manager.save()
            
            # Show feedback
            self.display.show_success(
                f"{info['name']} diatur ke: {typed_value}"
            )
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mengatur parameter: {str(e)}")
            return True
            
    def _reset_parameters(self) -> bool:
        """Reset semua parameter ke nilai default."""
        try:
            # Default values
            defaults = {
                'batch_size': 32,
                'learning_rate': 0.001,
                'epochs': 100,
                'early_stopping_patience': 10
            }
            
            # Update config with defaults
            self.config_manager.update('training', defaults)
            self.config_manager.save()
            
            # Show feedback
            self.display.show_success("Parameter direset ke nilai default")
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mereset parameter: {str(e)}")
            return True
            
    def draw(self, stdscr, start_y: int) -> None:
        """Override draw untuk menampilkan nilai parameter saat ini."""
        super().draw(stdscr, start_y)
        
        # Show current values
        height, width = stdscr.getmaxyx()
        current_y = start_y + 2
        
        training_config = self.config_manager.current_config.get('training', {})
        
        stdscr.attron(curses.color_pair(4))  # Cyan
        stdscr.addstr(current_y, width - 30, "Nilai Saat Ini:")
        stdscr.attroff(curses.color_pair(4))
        current_y += 1
        
        for param_id, info in self.PARAMETER_INFO.items():
            value = training_config.get(param_id, "Default")
            stdscr.addstr(
                current_y,
                width - 30,
                f"{info['name']}: {value}"
            )
            current_y += 1