# File: src/interfaces/evaluation_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka untuk evaluasi dan pengujian model

from .base_interface import BaseInterface
from evaluation.evaluator import DetectorEvaluator
from utils.metrics import MeanAveragePrecision
from termcolor import colored

class EvaluationInterface(BaseInterface):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.evaluator = None

    def tampilkan_menu(self):
        menu = """
🔍 Menu Evaluasi & Pengujian:

1. Jalankan Skenario Pengujian
2. Evaluasi Performa Model
3. Analisis Error
4. Bandingkan Model

0. Kembali ke Menu Utama
"""
        return self.prompt(menu, color='cyan')

    def handle_menu(self):
        """Tangani pilihan menu evaluasi"""
        while True:
            pilihan = self.tampilkan_menu()
            
            try:
                if pilihan == '0':
                    break
                elif pilihan == '1':
                    self.jalankan_skenario()
                elif pilihan == '2':
                    self.evaluasi_performa()
                elif pilihan == '3':
                    self.analisis_error()
                elif pilihan == '4':
                    self.bandingkan_model()
                else:
                    self.show_error("❌ Pilihan menu tidak valid!")
                    continue
                
                # Konfirmasi kembali ke menu
                if not self.confirm("\nKembali ke menu Evaluasi & Pengujian?"):
                    break
                    
            except KeyboardInterrupt:
                self.logger.warning("\n⚠️ Operasi dibatalkan oleh pengguna")
                continue
            except Exception as e:
                self.show_error(f"❌ Terjadi kesalahan: {str(e)}")

    def jalankan_skenario(self):
        """Jalankan skenario pengujian"""
        self.logger.info("\n🧪 Pilih Skenario Pengujian:")
        print(colored("1. Pencahayaan Normal", 'yellow'))
        print(colored("2. Pencahayaan Rendah", 'yellow'))
        print(colored("3. Objek Kecil", 'yellow'))
        print(colored("4. Oklusi", 'yellow'))
        
        skenario = self.prompt("Pilih skenario", default="1")
        threshold = float(self.prompt("Confidence threshold", default="0.25"))
        
        try:
            # TODO: Implement scenario evaluation
            self.logger.info("Menjalankan skenario pengujian...")
        except Exception as e:
            self.show_error(f"❌ Gagal menjalankan skenario: {str(e)}")

    def evaluasi_performa(self):
        """Evaluasi performa model"""
        self.logger.info("📊 Memulai evaluasi performa...")
        
        try:
            # TODO: Implement performance evaluation
            self.logger.info("Mengevaluasi performa model...")
        except Exception as e:
            self.show_error(f"❌ Gagal mengevaluasi performa: {str(e)}")

    def analisis_error(self):
        """Analisis kesalahan deteksi"""
        self.logger.info("🔍 Menganalisis kesalahan deteksi...")
        try:
            # TODO: Implement error analysis
            self.logger.info("Menganalisis kesalahan deteksi...")
        except Exception as e:
            self.show_error(f"❌ Gagal menganalisis error: {str(e)}")

    def bandingkan_model(self):
        """Bandingkan performa antar model"""
        self.logger.info("🔄 Perbandingan Model...")
        try:
            # TODO: Implement model comparison
            self.logger.info("Membandingkan model...")
        except Exception as e:
            self.show_error(f"❌ Gagal membandingkan model: {str(e)}")