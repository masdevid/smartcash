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
            hasil = self.evaluator.evaluate_scenario(
                scenario=skenario,
                conf_thresh=threshold
            )
            self.tampilkan_hasil_evaluasi(hasil)
        except Exception as e:
            self.show_error(f"❌ Gagal menjalankan skenario: {str(e)}")

    def evaluasi_performa(self):
        """Evaluasi performa model"""
        self.logger.info("📊 Memulai evaluasi performa...")
        
        try:
            metrics = self.evaluator.evaluate_all()
            
            self.logger.info("\n📈 Hasil Evaluasi:")
            for metrik, nilai in metrics.items():
                warna = 'green' if nilai > 0.8 else 'yellow' if nilai > 0.6 else 'red'
                print(colored(f"- {metrik}: {nilai:.4f}", warna))
        except Exception as e:
            self.show_error(f"❌ Gagal mengevaluasi performa: {str(e)}")

    def analisis_error(self):
        """Analisis kesalahan deteksi"""
        self.logger.info("🔍 Menganalisis kesalahan deteksi...")
        try:
            errors = self.evaluator.analyze_errors()
            
            self.logger.info("\n❌ Analisis Error:")
            self.logger.info("1. False Positives:")
            for kelas, jumlah in errors['fp'].items():
                self.logger.info(f"  • Rp{kelas}: {jumlah}")
                
            self.logger.info("\n2. False Negatives:")
            for kelas, jumlah in errors['fn'].items():
                self.logger.info(f"  • Rp{kelas}: {jumlah}")
        except Exception as e:
            self.show_error(f"❌ Gagal menganalisis error: {str(e)}")

    def bandingkan_model(self):
        """Bandingkan performa antar model"""
        self.logger.info("🔄 Perbandingan Model...")
        # Implementasi perbandingan model