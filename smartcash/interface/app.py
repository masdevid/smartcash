# File: smartcash/interface/app.py
# Author: Alfrida Sabar
# Deskripsi: Kelas aplikasi utama yang mengatur lifecycle dan koordinasi komponen interface

import curses
import sys
from pathlib import Path
from typing import Optional

from smartcash.cli.configuration_manager import ConfigurationManager
from smartcash.utils.logger import SmartCashLogger
from smartcash.interface.utils.display import DisplayManager
from smartcash.interface.menu.main import MainMenu
from smartcash.interface.menu.training import TrainingMenu
from smartcash.interface.menu.eval import EvaluationMenu

class SmartCashApp:
    """Kelas utama aplikasi SmartCash."""
    
    def __init__(self, config_path: Path):
        """
        Inisialisasi aplikasi.
        
        Args:
            config_path: Path ke file konfigurasi dasar
        """
        self.config_path = config_path
        self.logger = SmartCashLogger("smartcash-interface")
        
        # Inisialisasi ConfigurationManager
        try:
            self.config_manager = ConfigurationManager(str(config_path))
        except Exception as e:
            self.logger.error(f"Gagal inisialisasi config manager: {str(e)}")
            sys.exit(1)
            
        # Instance variables yang akan diset di setup()
        self.stdscr: Optional[curses.window] = None
        self.display: Optional[DisplayManager] = None
        self.current_menu = None
        self._is_setup = False
    
    def setup(self, stdscr: curses.window) -> None:
        """
        Setup aplikasi dengan curses window.
        
        Args:
            stdscr: Curses window utama
        """
        try:
            # Store window reference
            self.stdscr = stdscr
            
            # Essential curses setup
            curses.start_color()
            curses.use_default_colors()
            curses.curs_set(0)
            stdscr.keypad(True)
            
            # Setup display manager
            self.display = DisplayManager(stdscr)
            
            # Only create menu after display is ready
            self.current_menu = MainMenu(
                app=self,
                config_manager=self.config_manager,
                display=self.display
            )
            
            self._is_setup = True
            
        except Exception as e:
            self.logger.error(f"Gagal setup aplikasi: {str(e)}")
            raise
    
    def show_training_menu(self) -> bool:
        """
        Tampilkan menu pelatihan model.
        
        Returns:
            bool: True jika sukses, False jika kembali
        """
        try:
            self._ensure_setup()
            self.current_menu = TrainingMenu(
                app=self,
                config_manager=self.config_manager,
                display=self.display
            )
            return True
        except Exception as e:
            self.display.show_error(f"Gagal membuka menu training: {str(e)}")
            return True
    
    def show_evaluation_menu(self) -> bool:
        """
        Tampilkan menu evaluasi model.
        
        Returns:
            bool: True jika sukses, False jika kembali
        """
        try:
            self._ensure_setup()
            self.current_menu = EvaluationMenu(
                app=self,
                config_manager=self.config_manager,
                display=self.display
            )
            return True
        except Exception as e:
            self.display.show_error(f"Gagal membuka menu evaluasi: {str(e)}")
            return True
    
    def show_help(self) -> None:
        """Tampilkan bantuan penggunaan."""
        self._ensure_setup()
        help_content = {
            "Umum": (
                "• Gunakan ↑↓ untuk navigasi menu\n"
                "• Enter untuk memilih item\n"
                "• Q atau Ctrl+C untuk kembali/keluar\n"
                "• ESC untuk membatalkan input"
            ),
            "Pelatihan": (
                "• Lengkapi konfigurasi dengan urutan dari atas ke bawah\n"
                "• Mode lapis banyak membutuhkan GPU dengan memori lebih besar\n"
                "• Parameter dapat direset ke default jika diperlukan"
            ),
            "Evaluasi": (
                "• Evaluasi reguler menggunakan dataset testing standar\n"
                "• Skenario penelitian menguji kondisi pencahayaan dan posisi\n"
                "• Hasil evaluasi disimpan dalam format CSV"
            )
        }
        
        self.display.show_help(
            "Bantuan Penggunaan SmartCash",
            help_content
        )
    
    def show_confirm_exit(self) -> bool:
        """
        Tampilkan konfirmasi keluar.
        
        Returns:
            bool: True jika user konfirmasi keluar
        """
        self._ensure_setup()
        try:
            result = self.display.show_dialog(
                "Konfirmasi",
                "Apakah Anda yakin ingin keluar?",
                {"y": "Ya", "n": "Tidak"}
            )
            return result == 'y'
        except Exception as e:
            self.logger.error(f"Error saat konfirmasi keluar: {str(e)}")
            return True
    
    def _ensure_setup(self) -> None:
        """Memastikan aplikasi sudah di-setup."""
        if not self._is_setup:
            raise RuntimeError("Aplikasi belum di-setup. Panggil setup() terlebih dahulu.")
    
    def _handle_error(self, error: Exception) -> None:
        """
        Handle error dengan logging dan feedback.
        
        Args:
            error: Exception yang terjadi
        """
        error_msg = str(error)
        self.logger.error(f"Application error: {error_msg}")
        
        if self.display:
            self.display.show_error(f"Terjadi kesalahan: {error_msg}")
            if self.stdscr:
                self.stdscr.getch()
    
    def run(self, stdscr: Optional[curses.window] = None) -> None:
        """
        Run main application loop.
        
        Args:
            stdscr: Optional curses window dari wrapper
        """
        try:
            # Setup jika dipanggil melalui curses.wrapper
            if stdscr is not None:
                self.setup(stdscr)
            else:
                self._ensure_setup()
            
            # Main loop
            while True:
                try:
                    # Clear screen dan tampilkan menu saat ini
                    self.stdscr.clear()
                    self.current_menu.draw(self.stdscr, 2)
                    self.display.show_config_status(
                        self.config_manager.current_config
                    )
                    self.display.show_system_status()
                    self.stdscr.refresh()
                    
                    # Handle input
                    key = self.stdscr.getch()
                    if key == ord('q'):
                        if self.show_confirm_exit():
                            break
                        continue
                    elif key == ord('h'):
                        self.show_help()
                        continue
                    
                    # Proses input menu
                    result = self.current_menu.handle_input(key)
                    if result is False:  # Exit signal from menu
                        if isinstance(self.current_menu, MainMenu):
                            if self.show_confirm_exit():
                                break
                        else:
                            # Kembali ke menu utama
                            self.current_menu = MainMenu(
                                app=self,
                                config_manager=self.config_manager,
                                display=self.display
                            )
                        continue
                
                except KeyboardInterrupt:
                    if self.show_confirm_exit():
                        break
                except Exception as e:
                    self._handle_error(e)
            
            # Cleanup sebelum keluar
            self.config_manager.save()
            
        except Exception as e:
            self._handle_error(e)
        finally:
            self.logger.info("Aplikasi ditutup")