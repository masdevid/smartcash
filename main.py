# File: src/main.py
# Author: Alfrida Sabar
# Deskripsi: Aplikasi utama SmartCash Detector dengan antarmuka berbasis context dan interface

from pathlib import Path
from termcolor import colored
import sys
from typing import Dict, List, Optional

from interfaces.model_interface import ModelInterface
from interfaces.evaluation_interface import EvaluationInterface
from interfaces.export_interface import ExportInterface
from interfaces.data_interface import DataInterface

from models.factory import ModelFactory
from utils.logging import ColoredLogger
from config.manager import ConfigManager

class SmartCashApp:
    """Aplikasi utama SmartCash Detector dengan dukungan context dan interface"""
    def __init__(self):
        self.logger = ColoredLogger('SmartCashApp')
        self.cfg = ConfigManager()
        
        # Initialize factory
        self.model_factory = ModelFactory()
        
        # Initialize interfaces
        self._init_interfaces()
        
        # Setup directories
        self.setup_directories()
        
    def _init_interfaces(self):
        """Initialize application interfaces"""
        data_path = self.cfg.data.data_dir
        self.interfaces = {
            'data': DataInterface(self.cfg),
            'model': ModelInterface(self.cfg, data_path),
            'evaluation': EvaluationInterface(self.cfg),
            'export': ExportInterface(self.cfg)
        }

    def setup_directories(self):
        """Siapkan direktori yang diperlukan"""
        dirs = ['weights', 'runs', 'exports', 'logs']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def tampilkan_menu(self) -> str:
        """Tampilkan menu utama dengan informasi status"""
        # Tampilkan header
        self._tampilkan_header()
        
        menu = """
🎯 SmartCash Detector - Menu Utama:

1. Manajemen Dataset
2. Operasi Model
3. Pengujian & Evaluasi
4. Ekspor & Deploy

0. Keluar
"""
        print(colored(menu, 'cyan'))
        return input(colored('Pilih menu: ', 'yellow'))

    def _tampilkan_header(self):
        """Tampilkan header aplikasi"""
        header = """
╔════════════════════════════════════════╗
║           SmartCash Detector           ║
║     Deteksi Nominal Uang Rupiah        ║
╚════════════════════════════════════════╝
"""
        print(colored(header, 'green'))

    def _konfirmasi_keluar(self) -> bool:
        """Konfirmasi keluar dari aplikasi"""
        return input(colored('\nKeluar dari aplikasi? (y/N): ', 'yellow')).lower() == 'y'

    def run(self):
        """Jalankan aplikasi"""
        print(colored('🎯 Selamat datang di SmartCash Detector!', 'cyan'))
        
        while True:
            try:
                choice = self.tampilkan_menu()
                
                if choice == '0':
                    self.logger.info('👋 Terima kasih telah menggunakan SmartCash Detector!')
                    break
                # Tangani menu utama
                if choice == '1':
                    self.interfaces['data'].handle_menu()
                elif choice == '2':
                    self.interfaces['model'].handle_menu()
                elif choice == '3':
                    self.interfaces['evaluation'].handle_menu()
                elif choice == '4':
                    self.interfaces['export'].handle_menu()
                else:
                    self.logger.error('❌ Pilihan menu tidak valid!')
                
            except KeyboardInterrupt:
                print("\n")
                self.logger.warning("⚠️ Operasi dibatalkan oleh pengguna")
                if self._konfirmasi_keluar():
                    break
                    
            except Exception as e:
                self.logger.error(f'❌ Terjadi kesalahan: {str(e)}')

if __name__ == '__main__':
    try:
        app = SmartCashApp()
        app.run()
    except KeyboardInterrupt:
        print("\n")
        sys.exit(0)