# File: smartcash/utils/debug_helper.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk debugging issues pada konfigurasi dan interface

import os
import sys
import json
import yaml
import traceback
from typing import Dict, Any, Optional, List
from pathlib import Path
import tempfile
from datetime import datetime

from smartcash.utils.logger import SmartCashLogger

class DebugHelper:
    """Utilitas untuk debugging issues pada konfigurasi dan antarmuka."""
    
    def __init__(self, logger: Optional[SmartCashLogger] = None):
        """
        Inisialisasi Debug Helper.
        
        Args:
            logger: Logger opsional
        """
        self.logger = logger or SmartCashLogger("debug")
        self.debug_logs = []
        self.info = self._collect_system_info()
        
    def _collect_system_info(self) -> Dict[str, Any]:
        """
        Kumpulkan informasi sistem untuk debugging.
        
        Returns:
            Dict berisi informasi sistem
        """
        info = {
            'python_version': sys.version,
            'platform': sys.platform,
            'cwd': os.getcwd(),
            'env_vars': {
                'PATH': os.getenv('PATH'),
                'PYTHONPATH': os.getenv('PYTHONPATH')
            },
            'file_permissions': {}
        }
        
        # Check permissions on common directories
        for dir_path in ['configs', 'data', 'runs']:
            if os.path.exists(dir_path):
                info['file_permissions'][dir_path] = {
                    'exists': True,
                    'is_dir': os.path.isdir(dir_path),
                    'readable': os.access(dir_path, os.R_OK),
                    'writable': os.access(dir_path, os.W_OK),
                    'executable': os.access(dir_path, os.X_OK)
                }
            else:
                info['file_permissions'][dir_path] = {
                    'exists': False
                }
        
        return info
    
    def log_error(self, context: str, error: Exception) -> None:
        """
        Log error untuk debugging.
        
        Args:
            context: Konteks di mana error terjadi
            error: Exception yang terjadi
        """
        error_info = {
            'timestamp': datetime.now().isoformat(),
            'context': context,
            'error_type': type(error).__name__,
            'error_message': str(error),
            'traceback': traceback.format_exc()
        }
        
        self.debug_logs.append(error_info)
        self.logger.error(
            f"🐞 Debug Error [{context}]: {error_info['error_type']}: {error_info['error_message']}"
        )
    
    def check_config_file(self, config_path: str) -> Dict[str, Any]:
        """
        Periksa file konfigurasi untuk masalah.
        
        Args:
            config_path: Path ke file konfigurasi
            
        Returns:
            Dict berisi hasil pengecekan
        """
        results = {
            'exists': False,
            'readable': False,
            'writable': False,
            'valid_yaml': False,
            'content': None,
            'errors': []
        }
        
        try:
            path = Path(config_path)
            results['exists'] = path.exists()
            
            if not results['exists']:
                results['errors'].append(f"File tidak ditemukan: {config_path}")
                return results
                
            results['readable'] = os.access(path, os.R_OK)
            results['writable'] = os.access(path, os.W_OK)
            
            if not results['readable']:
                results['errors'].append(f"File tidak dapat dibaca: {config_path}")
                return results
                
            # Try to read and parse
            with open(path, 'r') as f:
                content = f.read()
                results['content'] = content
                
                try:
                    config = yaml.safe_load(content)
                    results['valid_yaml'] = True
                    results['parsed_config'] = config
                except Exception as e:
                    results['errors'].append(f"YAML tidak valid: {str(e)}")
                    
            # Test if directory is writable by creating a temp file
            try:
                parent_dir = path.parent
                with tempfile.NamedTemporaryFile(dir=parent_dir, delete=True) as tmp:
                    tmp.write(b'test')
                results['dir_writable'] = True
            except Exception as e:
                results['dir_writable'] = False
                results['errors'].append(f"Direktori tidak dapat ditulis: {str(e)}")
                
            return results
            
        except Exception as e:
            results['errors'].append(f"Error saat memeriksa file: {str(e)}")
            return results
    
    def test_config_save(self, config_manager, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test menyimpan konfigurasi untuk debug.
        
        Args:
            config_manager: Configuration Manager
            test_config: Konfigurasi test
            
        Returns:
            Dict berisi hasil test
        """
        results = {
            'success': False,
            'errors': []
        }
        
        try:
            # Backup current config
            old_config = config_manager.current_config.copy()
            
            # Try to update and save with simple test value
            try:
                for key, value in test_config.items():
                    config_manager.update(key, value)
                    
                saved_path = config_manager.save()
                results['saved_path'] = str(saved_path)
                results['success'] = True
                
            except Exception as e:
                results['errors'].append({
                    'operation': 'save',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
            # Restore old config
            config_manager.current_config = old_config
            
            return results
            
        except Exception as e:
            results['errors'].append({
                'operation': 'test_save',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return results
    
    def test_menu_interactions(self, menu, stdscr) -> Dict[str, Any]:
        """
        Test interaksi menu untuk debug.
        
        Args:
            menu: Menu yang akan ditest
            stdscr: curses window
            
        Returns:
            Dict berisi hasil test
        """
        results = {
            'success': False,
            'errors': []
        }
        
        try:
            # Test drawing menu
            try:
                menu.draw(stdscr, 2)
                results['draw'] = True
            except Exception as e:
                results['errors'].append({
                    'operation': 'draw',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                results['draw'] = False
                
            # Test handling some basic keypress
            try:
                test_keys = [curses.KEY_UP, curses.KEY_DOWN, curses.KEY_ENTER]
                key_results = []
                
                for key in test_keys:
                    try:
                        menu.handle_input(key)
                        key_results.append({'key': key, 'success': True})
                    except Exception as e:
                        key_results.append({
                            'key': key, 
                            'success': False,
                            'error': str(e)
                        })
                
                results['key_tests'] = key_results
            except Exception as e:
                results['errors'].append({
                    'operation': 'handle_input',
                    'error': str(e),
                    'traceback': traceback.format_exc()
                })
                
            results['success'] = len(results['errors']) == 0
            return results
            
        except Exception as e:
            results['errors'].append({
                'operation': 'test_menu',
                'error': str(e),
                'traceback': traceback.format_exc()
            })
            return results
    
    def generate_debug_report(self) -> str:
        """
        Generate laporan debug lengkap.
        
        Returns:
            String berisi laporan debug
        """
        report = "=== SMARTCASH DEBUG REPORT ===\n\n"
        
        # System info
        report += "== SYSTEM INFO ==\n"
        report += f"Python: {self.info['python_version']}\n"
        report += f"Platform: {self.info['platform']}\n"
        report += f"Working Directory: {self.info['cwd']}\n\n"
        
        # File permissions
        report += "== FILE PERMISSIONS ==\n"
        for dir_path, perms in self.info['file_permissions'].items():
            report += f"{dir_path}: "
            if perms['exists']:
                if perms['is_dir']:
                    flags = []
                    if perms['readable']: flags.append('R')
                    if perms['writable']: flags.append('W')
                    if perms['executable']: flags.append('X')
                    report += f"DIR [{'/'.join(flags)}]\n"
                else:
                    report += "FILE\n"
            else:
                report += "NOT FOUND\n"
        report += "\n"
        
        # Error logs
        report += "== ERROR LOGS ==\n"
        if not self.debug_logs:
            report += "No errors logged\n"
        else:
            for i, log in enumerate(self.debug_logs):
                report += f"[{i+1}] {log['timestamp']} - {log['context']}\n"
                report += f"    {log['error_type']}: {log['error_message']}\n"
                tb_lines = log['traceback'].split('\n')
                for tb_line in tb_lines[-5:]:  # Show only last 5 lines of traceback
                    report += f"    {tb_line}\n"
                report += "\n"
        
        report += "=== END OF REPORT ===\n"
        return report
    
    def save_debug_report(self, path: str = "debug_report.txt") -> str:
        """
        Simpan laporan debug ke file.
        
        Args:
            path: Path file output
            
        Returns:
            Path laporan yang disimpan
        """
        report = self.generate_debug_report()
        
        try:
            with open(path, 'w') as f:
                f.write(report)
            
            self.logger.success(f"Laporan debug disimpan ke {path}")
            return path
            
        except Exception as e:
            self.logger.error(f"Gagal menyimpan laporan debug: {str(e)}")
            
            # Fallback to temp directory
            try:
                temp_path = tempfile.gettempdir() + "/smartcash_debug_report.txt"
                with open(temp_path, 'w') as f:
                    f.write(report)
                
                self.logger.success(f"Laporan debug disimpan ke {temp_path}")
                return temp_path
                
            except Exception as temp_error:
                self.logger.error(f"Gagal menyimpan laporan ke temp: {str(temp_error)}")
                return ""

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