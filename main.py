# File: src/main.py
# Author: Alfrida Sabar
# Deskripsi: Aplikasi utama SmartCash Detector dengan antarmuka modular

from pathlib import Path
from termcolor import colored
from interfaces.model_interface import ModelInterface
from interfaces.data_interface import DataInterface
from interfaces.evaluation_interface import EvaluationInterface
from interfaces.export_interface import ExportInterface
from utils.logging import ColoredLogger
from config.manager import ConfigManager

class SmartCashApp:
    def __init__(self):
        self.logger = ColoredLogger('SmartCashApp')
        
        # Inisialisasi konfigurasi
        self.cfg = ConfigManager()
        self.setup_directories()
        
        # Inisialisasi interfaces
        self.interfaces = {
            'model': ModelInterface(self.cfg, self.cfg.data.data_dir),
            'data': DataInterface(self.cfg),
            'eval': EvaluationInterface(self.cfg),
            'export': ExportInterface(self.cfg)
        }

    def setup_directories(self):
        """Siapkan direktori yang diperlukan"""
        dirs = ['weights', 'runs', 'exports', 'logs']
        for dir_name in dirs:
            Path(dir_name).mkdir(exist_ok=True)

    def tampilkan_menu(self):
        menu = """
🔍 SmartCash Detector - Menu Utama:

1. Manajemen Data
2. Operasi Model
3. Pengujian & Evaluasi
4. Ekspor Model

0. Keluar
"""
        print(colored(menu, 'cyan'))
        return input(colored('Pilih menu (0-4): ', 'yellow'))

    def tampilkan_header(self):
        """Tampilkan header aplikasi"""
        header = """
╔════════════════════════════════════════╗
║           SmartCash Detector           ║
║     Deteksi Nominal Uang Rupiah        ║
╚════════════════════════════════════════╝
"""
        print(colored(header, 'green'))
        
    def tampilkan_info_sistem(self):
        """Tampilkan informasi sistem"""
        self.logger.info("ℹ️ Informasi Sistem:")
        self.logger.info(f"- Direktori Data: {self.cfg.data.data_dir}")
        self.logger.info(f"- Jumlah Kelas: {self.cfg.model.nc}")
        self.logger.info(f"- Ukuran Input: {self.cfg.model.img_size}x{self.cfg.model.img_size}")

    def run(self):
        """Jalankan aplikasi"""
        self.tampilkan_header()
        self.tampilkan_info_sistem()
        
        while True:
            pilihan = self.tampilkan_menu()
            
            try:
                if pilihan == '0':
                    self.logger.info('👋 Terima kasih telah menggunakan SmartCash Detector!')
                    break
                elif pilihan == '1':
                    self.interfaces['data'].handle_menu()
                elif pilihan == '2':
                    self.interfaces['model'].handle_menu()
                elif pilihan == '3':
                    self.interfaces['eval'].handle_menu()
                elif pilihan == '4':
                    self.interfaces['export'].handle_menu()
                else:
                    self.logger.error('❌ Pilihan menu tidak valid!')
            except KeyboardInterrupt:
                self.logger.warning("\n⚠️ Operasi dibatalkan oleh pengguna")
            except Exception as e:
                self.logger.error(f'❌ Terjadi kesalahan: {str(e)}')

if __name__ == '__main__':
    app = SmartCashApp()
    app.run()