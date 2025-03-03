# File: smartcash/interface/app.py
# Author: Alfrida Sabar
# Deskripsi: Kelas aplikasi utama yang mengatur lifecycle dan koordinasi komponen interface

import curses
import sys
from pathlib import Path
from typing import Optional

from smartcash.cli.configuration_manager import ConfigurationManager
from smartcash.utils.logger import get_logger
from smartcash.interface.utils.display import DisplayManager
from smartcash.interface.menu.main import MainMenu
from smartcash.interface.menu.training import TrainingMenu
from smartcash.interface.menu.eval import EvaluationMenu
from smartcash.exceptions.handler import ErrorHandler
from smartcash.exceptions.base import (
    ConfigError, DataError, ValidationError, SmartCashError
)
from smartcash.utils.debug_helper import DebugHelper

class SmartCashApp:
    """Kelas utama aplikasi SmartCash."""
    
    def __init__(self, config_path: Path):
        """
        Inisialisasi aplikasi.
        
        Args:
            config_path: Path ke file konfigurasi dasar
        
        Raises:
            ConfigError: Jika gagal memuat konfigurasi
            ValidationError: Jika konfigurasi tidak valid
        """
        # Setup logger dan error handler
        self.logger = get_logger("smartcash")
        self.error_handler = ErrorHandler("smartcash.app")
        
        # Inisialisasi debug helper
        self.debug_helper = DebugHelper(self.logger)

        try:
            # Validasi path konfigurasi dan pastikan file dapat dibaca/ditulis
            config_check = self.debug_helper.check_config_file(str(config_path))
            if not config_check['exists']:
                raise ConfigError(f"File konfigurasi tidak ditemukan: {config_path}")
            if not config_check['readable']:
                raise ConfigError(f"File konfigurasi tidak dapat dibaca: {config_path}")
            if not config_check['writable']:
                self.logger.warning(f"⚠️ File konfigurasi tidak dapat ditulis: {config_path}")
            if not config_check['valid_yaml']:
                raise ConfigError(f"Format YAML tidak valid: {config_path}")
            
            self.config_path = config_path
            
            # Inisialisasi ConfigurationManager dengan error handling
            try:
                self.config_manager = ConfigurationManager(str(config_path))
            except Exception as e:
                # Log detail error untuk debugging
                self.debug_helper.log_error("config_manager_init", e)
                raise ConfigError(f"Gagal inisialisasi config manager: {str(e)}")
            
            # Instance variables untuk UI
            self.stdscr: Optional[curses.window] = None
            self.display: Optional[DisplayManager] = None
            self.current_menu = None
            self._is_setup = False
            
            # Test konfigurasi bisa disimpan
            test_result = self.debug_helper.test_config_save(
                self.config_manager,
                {"test_key": "test_value"}
            )
            if not test_result['success']:
                errors = "\n".join([e.get('error', 'Unknown') for e in test_result.get('errors', [])])
                self.logger.warning(
                    f"⚠️ Terdeteksi masalah penyimpanan konfigurasi:\n{errors}"
                )
            
            self.logger.info("✨ Aplikasi berhasil diinisialisasi")
            
        except SmartCashError as e:
            # Handle known errors
            self.error_handler.handle(e)
        except Exception as e:
            # Handle unexpected errors
            self.debug_helper.log_error("app_init", e)
            self.error_handler.handle(e)

    def setup(self, stdscr: curses.window) -> None:
        """
        Setup aplikasi dengan curses window dan perbaikan error handling.
        
        Args:
            stdscr: Curses window utama
            
        Raises:
            ConfigError: Jika setup gagal
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
            
            # Initialize main menu
            self.current_menu = MainMenu(
                app=self,
                config_manager=self.config_manager,
                display=self.display
            )
            
            # Test menu draw untuk verifikasi
            try:
                menu_test = self.debug_helper.test_menu_interactions(self.current_menu, stdscr)
                if not menu_test['success']:
                    errors = "\n".join([e.get('error', 'Unknown') for e in menu_test.get('errors', [])])
                    self.logger.warning(
                        f"⚠️ Terdeteksi masalah dengan menu:\n{errors}"
                    )
            except Exception as menu_test_error:
                self.debug_helper.log_error("menu_test", menu_test_error)
            
            self._is_setup = True
            self.logger.info("🎨 Interface berhasil disetup")
            
        except Exception as e:
            self.debug_helper.log_error("app_setup", e)
            raise ConfigError(f"Gagal setup aplikasi: {str(e)}")
    
    def show_training_menu(self) -> bool:
        """
        Tampilkan menu pelatihan model dengan error handling yang ditingkatkan.
        
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
            self.debug_helper.log_error("show_training_menu", e)
            self.display.show_error(f"Gagal membuka menu training: {str(e)}")
            return True
    def show_evaluation_menu(self) -> bool:
        """
        Tampilkan menu evaluasi model dengan error handling yang ditingkatkan.
        
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
            self.debug_helper.log_error("show_evaluation_menu", e)
            self.display.show_error(f"Gagal membuka menu evaluasi: {str(e)}")
            return True

    def show_debug_menu(self) -> bool:
        """
        Tampilkan menu debug untuk troubleshooting, termasuk opsi perbaikan mode vs model.
        
        Returns:
            bool: True
        """
        try:
            self._ensure_setup()
            
            # Buat menu dengan tambahan opsi perbaikan
            menu_items = [
                "Tampilkan Riwayat Error",
                "Test Penyimpanan Konfigurasi",
                "Perbaiki Issue Mode vs Model",  # Tambahkan opsi ini
                "Analisis Struktur Konfigurasi",  # Tambahkan opsi ini
                "Simpan Laporan Debug",
                "Tampilkan Info Debug",
                "Kembali"
            ]
            
            selected = 0
            while True:
                # Draw menu
                self.stdscr.clear()
                h, w = self.stdscr.getmaxyx()
                
                # Title
                title = "🐞 Menu Debug & Troubleshooting"
                self.stdscr.attron(curses.color_pair(1))
                self.stdscr.addstr(1, (w - len(title)) // 2, title)
                self.stdscr.attroff(curses.color_pair(1))
                
                # Items
                for i, item in enumerate(menu_items):
                    if i == selected:
                        self.stdscr.attron(curses.color_pair(3))
                        self.stdscr.addstr(3 + i, 2, f"> {item}")
                        self.stdscr.attroff(curses.color_pair(3))
                    else:
                        self.stdscr.addstr(3 + i, 2, f"  {item}")
                        
                # Info
                self.stdscr.attron(curses.color_pair(4))
                self.stdscr.addstr(3 + len(menu_items) + 1, 2, "↑↓: Navigasi | Enter: Pilih | Q: Kembali")
                self.stdscr.attroff(curses.color_pair(4))
                
                # Footer
                config_dir = self.config_manager.config_dir
                config_path = self.config_manager.base_config_path
                footer = f"Config Dir: {config_dir} | Base Config: {config_path}"
                
                if len(footer) > w - 4:
                    footer = footer[:w-7] + "..."
                    
                self.stdscr.addstr(h-1, 0, footer)
                
                self.stdscr.refresh()
                
                # Handle input
                key = self.stdscr.getch()
                
                if key == ord('q'):
                    break
                elif key == curses.KEY_UP and selected > 0:
                    selected -= 1
                elif key == curses.KEY_DOWN and selected < len(menu_items) - 1:
                    selected += 1
                elif key in [curses.KEY_ENTER, ord('\n'), ord('\r')]:
                    if selected == 0:  # Tampilkan Riwayat Error
                        self.display.show_error_history()
                    elif selected == 1:  # Test Penyimpanan
                        self._test_config_save()
                    elif selected == 2:  # Perbaiki Issue Mode vs Model
                        self._fix_mode_model_issue()
                    elif selected == 3:  # Analisis Struktur Konfigurasi
                        self._show_config_structure()
                    elif selected == 4:  # Simpan Laporan
                        path = self.debug_helper.save_debug_report()
                        if path:
                            self.display.show_success(f"Laporan debug disimpan ke {path}")
                        else:
                            self.display.show_error("Gagal menyimpan laporan debug")
                    elif selected == 5:  # Tampilkan Info
                        debug_info = self.config_manager.debug_config()
                        info_str = yaml.dump(debug_info, default_flow_style=False)
                        self.display.show_info(info_str, "Info Debug Konfigurasi")
                    elif selected == 6:  # Kembali
                        break
            
            return True
        except Exception as e:
            self.debug_helper.log_error("show_debug_menu", e)
            self.display.show_error(f"Gagal membuka menu debug: {str(e)}")
            return True
    def _test_config_save(self) -> None:
        """Test kemampuan menyimpan konfigurasi."""
        test_values = {
            'test_key': 'test_value',
            'training.test_param': 42,
            'model.test_param': 3.14
        }
        
        results = []
        
        for key, value in test_values.items():
            try:
                # Backup current value
                parts = key.split('.')
                old_value = None
                
                if len(parts) == 1:
                    old_value = self.config_manager.current_config.get(key)
                elif len(parts) == 2:
                    section = self.config_manager.current_config.get(parts[0], {})
                    old_value = section.get(parts[1])
                
                # Update value
                self.config_manager.update(key, value)
                
                # Try to save
                self.config_manager.save()
                results.append((key, True, ""))
                
                # Restore old value
                if old_value is not None:
                    self.config_manager.update(key, old_value)
                else:
                    # Remove test key
                    if len(parts) == 1:
                        if key in self.config_manager.current_config:
                            del self.config_manager.current_config[key]
                    elif len(parts) == 2:
                        section = self.config_manager.current_config.get(parts[0], {})
                        if parts[1] in section:
                            del section[parts[1]]
                
            except Exception as e:
                results.append((key, False, str(e)))
        
        # Format results for display
        report = "Hasil Test Penyimpanan Konfigurasi:\n\n"
        
        success_count = sum(1 for _, success, _ in results if success)
        report += f"Berhasil: {success_count}/{len(results)}\n\n"
        
        for key, success, error in results:
            if success:
                report += f"✅ {key}: Berhasil\n"
            else:
                report += f"❌ {key}: Gagal - {error}\n"
                
        # Show report
        self.display.show_info(report, "Hasil Test Penyimpanan")
    
    def show_help(self) -> None:
        """Tampilkan bantuan penggunaan."""
        self._ensure_setup()
        help_content = {
            "Umum": (
                "• Gunakan ↑↓ untuk navigasi menu\n"
                "• Enter untuk memilih item\n"
                "• Q atau Ctrl+C untuk kembali/keluar\n"
                "• ESC untuk membatalkan input\n"
                "• H untuk tampilan bantuan ini\n"
                "• D untuk tampilan menu debug (jika terjadi error)"
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
            ),
            "Troubleshooting": (
                "• Tekan D untuk akses menu debug jika terjadi error\n"
                "• Error terakhir akan ditampilkan di bagian bawah layar\n"
                "• Laporan debug disimpan di direktori utama\n"
                "• Jika penyimpanan gagal, coba periksa izin folder"
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
            self.debug_helper.log_error("show_confirm_exit", e)
            self.logger.error(f"Error saat konfirmasi keluar: {str(e)}")
            return True
    
    def _ensure_setup(self) -> None:
        """
        Memastikan aplikasi sudah di-setup.
        
        Raises:
            ValidationError: Jika aplikasi belum di-setup
        """
        if not self._is_setup:
            raise ValidationError(
                "Aplikasi belum di-setup. Panggil setup() terlebih dahulu."
            )
    
    def _cleanup(self) -> None:
        """Cleanup resources sebelum exit."""
        try:
            # Save config changes
            if hasattr(self, 'config_manager'):
                try:
                    self.config_manager.save()
                except Exception as save_error:
                    self.debug_helper.log_error("cleanup_save", save_error)
                    self.logger.warning(f"⚠️ Gagal menyimpan konfigurasi saat cleanup: {str(save_error)}")
            
            # Reset terminal state
            if hasattr(self, 'stdscr'):
                try:
                    curses.nocbreak()
                    self.stdscr.keypad(False)
                    curses.echo()
                    curses.endwin()
                except Exception as curses_error:
                    self.debug_helper.log_error("cleanup_curses", curses_error)
            
            self.logger.info("🧹 Cleanup selesai")
            
        except Exception as e:
            self.debug_helper.log_error("cleanup", e)
            self.logger.error(f"❌ Error saat cleanup: {str(e)}")
         
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
        Run main application loop with improved error handling.
        
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
                    elif key == ord('d'):
                        # Menu debug untuk troubleshooting
                        self.show_debug_menu()
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
                    # Log error tapi jangan exit
                    self.debug_helper.log_error("main_loop", e)
                    self.display.show_error(
                        f"Error: {str(e)}. Tekan 'D' untuk debug menu."
                    )
                    continue
            
            # Cleanup sebelum exit
            self._cleanup()
            
        except Exception as e:
            # Fatal error, cleanup dan exit
            self.debug_helper.log_error("fatal_error", e)
            self.error_handler.handle(
                e,
                cleanup_func=self._cleanup
            )
    def _fix_mode_model_issue(self) -> None:
        """Perbaiki masalah konfusi antara 'mode' dan 'model' di konfigurasi."""
        try:
            # Jalankan perbaikan
            result = self.debug_helper.fix_mode_model_confusion(self.config_manager)
            
            # Format laporan
            report = "Hasil Perbaikan Issue Mode vs Model:\n\n"
            
            if result['status'] == 'success':
                report += "✅ Status: Berhasil\n\n"
            else:
                report += f"❌ Status: Gagal - {result.get('error', 'Unknown error')}\n\n"
            
            # Masalah yang ditemukan
            if result['issues_found']:
                report += "🔍 Masalah yang ditemukan:\n"
                for issue in result['issues_found']:
                    report += f"  • {issue}\n"
                report += "\n"
            else:
                report += "🔍 Tidak ada masalah yang ditemukan\n\n"
            
            # Perbaikan yang diterapkan
            if result['fixes_applied']:
                report += "🔧 Perbaikan yang diterapkan:\n"
                for fix in result['fixes_applied']:
                    report += f"  • {fix}\n"
            else:
                report += "🔧 Tidak ada perbaikan yang diterapkan\n"
            
            # Tampilkan laporan
            self.display.show_info(report, "Hasil Perbaikan")
            
        except Exception as e:
            self.debug_helper.log_error("fix_mode_model", e)
            self.display.show_error(f"Gagal memperbaiki issue: {str(e)}")

    
    def _show_config_structure(self) -> None:
        """Tampilkan struktur konfigurasi untuk analisis."""
        try:
            # Dapatkan struktur konfigurasi
            structure = self.debug_helper.get_config_structure(self.config_manager.current_config)
            
            # Tampilkan struktur
            self.display.show_info(structure, "Struktur Konfigurasi")
        except Exception as e:
            self.debug_helper.log_error("show_config_structure", e)
            self.display.show_error(f"Gagal menganalisis konfigurasi: {str(e)}")