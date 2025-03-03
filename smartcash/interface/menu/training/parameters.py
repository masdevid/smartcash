# File: smartcash/interface/menu/training/parameters.py
# Author: Alfrida Sabar
# Deskripsi: Menu konfigurasi parameter pelatihan model dengan perbaikan validasi

import curses
from smartcash.interface.menu.base import BaseMenu, MenuItem
from typing import Any, Callable, Dict

class TrainingParamsMenu(BaseMenu):
    """Menu konfigurasi parameter pelatihan."""
    
    # Parameter info tidak berubah kecuali penambahan default
    PARAMETER_INFO = {
        'batch_size': {
            'name': 'Ukuran Batch',
            'description': (
                "Jumlah sampel data per iterasi\n"
                "• Range: 8-128\n"
                "• Default: 32\n"
                "• ↑: Pelatihan cepat, memori tinggi\n"
                "• ↓: Pelatihan lambat, memori rendah"
            ),
            'validator': lambda x: 8 <= int(x) <= 128,
            'prompt': "Masukkan ukuran batch (8-128): ",
            'type': int,
            'default': 32
        },
        'learning_rate': {
            'name': 'Learning Rate',
            'description': (
                "Tingkat pembelajaran model\n"
                "• Range: 0.0001-0.01\n"
                "• Default: 0.001\n"
                "• ↑: Pembelajaran cepat, kurang stabil\n"
                "• ↓: Pembelajaran lambat, lebih stabil"
            ),
            'validator': lambda x: 0.0001 <= float(x) <= 0.01,
            'prompt': "Masukkan learning rate (0.0001-0.01): ",
            'type': float,
            'default': 0.001
        },
        'epochs': {
            'name': 'Jumlah Epoch',
            'description': (
                "Jumlah iterasi pelatihan\n"
                "• Range: 10-1000\n"
                "• Default: 100\n"
                "• ↑: Pelatihan lebih lama, akurasi potensial lebih tinggi\n"
                "• ↓: Pelatihan lebih singkat, akurasi potensial lebih rendah"
            ),
            'validator': lambda x: 10 <= int(x) <= 1000,
            'prompt': "Masukkan jumlah epoch (10-1000): ",
            'type': int,
            'default': 100
        },
        'early_stopping_patience': {
            'name': 'Early Stopping',
            'description': (
                "Jumlah epoch tanpa perbaikan sebelum berhenti\n"
                "• Range: 5-50\n"
                "• Default: 10\n"
                "• ↑: Lebih sabar, pelatihan bisa lebih lama\n"
                "• ↓: Kurang sabar, pelatihan bisa lebih singkat"
            ),
            'validator': lambda x: 5 <= int(x) <= 50,
            'prompt': "Masukkan nilai patience (5-50): ",
            'type': int,
            'default': 10
        }
    }
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        self.logger = app.logger if hasattr(app, 'logger') else None
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
        
    def _try_convert_value(self, value_str: str, param_id: str) -> Any:
        """
        Coba konversi string input ke tipe parameter yang diharapkan.
        
        Args:
            value_str: String input dari user
            param_id: ID parameter
            
        Returns:
            Nilai terkonversi atau None jika gagal
            
        Raises:
            ValueError: Jika konversi gagal
        """
        try:
            info = self.PARAMETER_INFO[param_id]
            return info['type'](value_str)
        except ValueError:
            raise ValueError(f"Tidak dapat mengkonversi '{value_str}' ke {info['type'].__name__}")
            
    def _set_parameter(self, param_id: str) -> bool:
        """
        Set nilai parameter pelatihan dengan perbaikan pada metode update.
        
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
                    validator=None,  # Handle validasi manual
                    default=str(info['default'])  # Tambahkan nilai default
                )
                
                if value is None:  # User cancelled
                    return True
                    
                if value.strip() == "":  # Gunakan nilai default
                    value = str(info['default'])
                    
                try:
                    # Coba konversi dan validasi input
                    typed_value = info['type'](value)
                    
                    # Validasi range
                    if not info['validator'](typed_value):
                        min_max = info['prompt'].split('(')[1].split(')')[0]
                        self.display.show_error(f"Nilai harus dalam range {min_max}")
                        continue
                        
                    break  # Validasi berhasil
                    
                except ValueError as ve:
                    self.display.show_error(f"Format nilai tidak valid: {str(ve)}")
                    continue
            
            # Update parameter in config
            try:
                # Kasus khusus untuk batch_size yang perlu diupdate di dua tempat
                if param_id == 'batch_size':
                    # Update training.batch_size
                    self.config_manager.update(f'training.{param_id}', typed_value)
                    
                    # Juga update model.batch_size jika ada
                    if 'model' in self.config_manager.current_config:
                        # Gunakan cara yang lebih aman - baca model config, update batch_size, lalu tulis kembali
                        model_config = self.config_manager.current_config.get('model', {}).copy()
                        model_config['batch_size'] = typed_value
                        self.config_manager.update('model', model_config)
                        
                        if self.logger:
                            self.logger.info(f"📝 Juga mengupdate model.batch_size ke {typed_value}")
                else:
                    # Parameter normal - hanya update di training
                    self.config_manager.update(f'training.{param_id}', typed_value)
                
                # Save config
                self.config_manager.save()
                self.display.show_success(f"{info['name']} diatur ke: {typed_value}")
                
            except Exception as e:
                self.display.show_error(f"Gagal mengupdate: {str(e)}")
                if self.logger:
                    self.logger.error(f"❌ Error update parameter: {str(e)}")
                
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal mengatur parameter: {str(e)}")
            return True 
    def _reset_parameters(self) -> bool:
        """Reset semua parameter ke nilai default."""
        try:
            # Default values from parameter info
            defaults = {
                param_id: info['default'] 
                for param_id, info in self.PARAMETER_INFO.items()
            }
            
            # Update config with defaults secara terpisah untuk mengisolasi error
            for param_id, default_value in defaults.items():
                try:
                    self.config_manager.update(f'training.{param_id}', default_value)
                except Exception as update_error:
                    self.display.show_error(
                        f"Gagal reset parameter {param_id}: {str(update_error)}"
                    )
                    return True
            
            # Save config dengan error handling
            try:
                self.config_manager.save()
                self.display.show_success("Parameter direset ke nilai default")
            except Exception as save_error:
                self.display.show_error(f"Gagal menyimpan konfigurasi: {str(save_error)}")
                
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
            # Get value with safe error handling
            try:
                value = training_config.get(param_id, "Default")
                stdscr.addstr(
                    current_y,
                    width - 30,
                    f"{info['name']}: {value}"
                )
            except Exception:
                # Fallback jika terjadi error saat menampilkan
                stdscr.addstr(
                    current_y,
                    width - 30,
                    f"{info['name']}: Error"
                )
            current_y += 1