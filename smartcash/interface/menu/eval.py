# File: smartcash/interface/menu/eval.py
# Author: Alfrida Sabar
# Deskripsi: Menu evaluasi model dengan validasi konfigurasi

from smartcash.handlers.evaluation_handler import EvaluationHandler
from smartcash.interface.menu.base import BaseMenu, MenuItem
from smartcash.cli.configuration_manager import ConfigurationManager
   
class EvaluationMenu(BaseMenu):
    """Menu evaluasi model dengan validasi konfigurasi."""
    
    def __init__(self, app, config_manager, display):
        self.app = app
        self.config_manager = config_manager
        self.display = display
        
        # Siapkan items menu
        items = [
            MenuItem(
                title="Evaluasi Model Reguler",
                action=self.evaluate_regular,
                description="Evaluasi model pada dataset testing standar",
                category="Evaluasi"
            ),
            MenuItem(
                title="Evaluasi Skenario Penelitian",
                action=self.evaluate_research,
                description="Evaluasi model pada skenario penelitian",
                category="Evaluasi"
            ),
            MenuItem(
                title="Kembali",
                action=lambda: False,
                category="Navigasi"
            )
        ]
        
        super().__init__("Menu Evaluasi Model", items)
        
        # Validasi dan atur status item menu
        self._validate_menu_configuration()
    
    def _validate_menu_configuration(self) -> None:
        """
        Validasi konfigurasi dan atur status item menu.
        Pastikan item evaluasi hanya dapat diakses jika konfigurasi lengkap.
        """
        config = self.config_manager.current_config
        
        # Cek kelengkapan konfigurasi yang diperlukan
        required_configs = [
            'data_source',    # Sumber data
            'backbone',       # Arsitektur backbone
            'detection_mode'  # Mode deteksi
        ]
        
        # Periksa apakah semua konfigurasi yang diperlukan sudah diset
        configs_complete = all(config.get(req) for req in required_configs)
        
        # Nonaktifkan item evaluasi jika konfigurasi belum lengkap
        if not configs_complete:
            for item in self.items:
                if item.title in ["Evaluasi Model Reguler", "Evaluasi Skenario Penelitian"]:
                    item.enabled = False
        else:
            for item in self.items:
                if item.title in ["Evaluasi Model Reguler", "Evaluasi Skenario Penelitian"]:
                    item.enabled = True
    
    def _check_configuration_and_checkpoints(self) -> bool:
        """
        Periksa kelengkapan konfigurasi dan ketersediaan checkpoint.
        
        Returns:
            bool: True jika konfigurasi dan checkpoint tersedia
        """
        config = self.config_manager.current_config
        
        # Validasi konfigurasi dasar
        required_configs = [
            'data_source',    # Sumber data
            'backbone',       # Arsitektur backbone
            'detection_mode'  # Mode deteksi
        ]
        
        # Periksa apakah semua konfigurasi yang diperlukan sudah diset
        if not all(config.get(req) for req in required_configs):
            self.display.show_error(
                "Konfigurasi belum lengkap. Silakan lengkapi konfigurasi terlebih dahulu."
            )
            return False
        
        # Cek ketersediaan checkpoint
        try:
            evaluator = EvaluationHandler(config=config)
            checkpoints = evaluator.list_checkpoints()
            
            if not checkpoints:
                self.display.show_error(
                    "Tidak ada model yang telah dilatih. Silakan latih model terlebih dahulu."
                )
                return False
            
            return True
        
        except Exception as e:
            self.display.show_error(f"Gagal memeriksa model: {str(e)}")
            return False
    
    def evaluate_regular(self) -> bool:
        """Evaluasi model pada dataset testing standar."""
        # Validasi konfigurasi dan checkpoint
        if not self._check_configuration_and_checkpoints():
            return True
        
        try:
            # Konfirmasi sebelum evaluasi
            konfirmasi = self.display.show_dialog(
                "Konfirmasi Evaluasi",
                "Apakah Anda yakin ingin melakukan evaluasi model?",
                {"y": "Ya", "n": "Tidak"}
            )
            
            if konfirmasi != 'y':
                return True
            
            # Tampilkan loading/progress
            self.display.show_success("🔍 Memulai evaluasi model...")
            
            # Lakukan evaluasi
            evaluator = EvaluationHandler(
                config=self.config_manager.current_config
            )
            results = evaluator.evaluate(eval_type='regular')
            
            # Tampilkan hasil
            self._tampilkan_hasil_evaluasi(results)
            
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal melakukan evaluasi: {str(e)}")
            return True
    
    def evaluate_research(self) -> bool:
        """Evaluasi model pada skenario penelitian."""
        # Validasi konfigurasi dan checkpoint
        if not self._check_configuration_and_checkpoints():
            return True
        
        try:
            # Konfirmasi sebelum evaluasi
            konfirmasi = self.display.show_dialog(
                "Konfirmasi Evaluasi Penelitian",
                "Apakah Anda yakin ingin menjalankan skenario penelitian?",
                {"y": "Ya", "n": "Tidak"}
            )
            
            if konfirmasi != 'y':
                return True
            
            # Tampilkan loading/progress
            self.display.show_success("🔬 Memulai skenario penelitian...")
            
            # Lakukan evaluasi
            evaluator = EvaluationHandler(
                config=self.config_manager.current_config
            )
            results = evaluator.evaluate(eval_type='research')
            
            # Tampilkan hasil
            self._tampilkan_hasil_penelitian(results)
            
            return True
            
        except Exception as e:
            self.display.show_error(f"Gagal menjalankan skenario: {str(e)}")
            return True
    
    def _tampilkan_hasil_evaluasi(self, results: dict) -> None:
        """
        Tampilkan hasil evaluasi dalam dialog.
        
        Args:
            results: Dictionary hasil evaluasi
        """
        # Format pesan hasil
        pesan_hasil = (
            f"📊 Hasil Evaluasi Model:\n\n"
            f"Akurasi: {results.get('accuracy', 'N/A'):.4f}\n"
            f"Precision: {results.get('precision', 'N/A'):.4f}\n"
            f"Recall: {results.get('recall', 'N/A'):.4f}\n"
            f"F1-Score: {results.get('f1', 'N/A'):.4f}\n"
            f"mAP: {results.get('mAP', 'N/A'):.4f}\n"
            f"Waktu Inferensi: {results.get('inference_time', 'N/A')*1000:.2f} ms"
        )
        
        # Tampilkan dialog
        self.display.show_dialog(
            "Hasil Evaluasi Model",
            pesan_hasil,
            {"o": "OK"}
        )
    
    def _tampilkan_hasil_penelitian(self, results: dict) -> None:
        """
        Tampilkan hasil evaluasi penelitian dalam dialog.
        
        Args:
            results: Dictionary hasil evaluasi penelitian
        """
        # Konversi DataFrame ke string
        df = results.get('research_results', None)
        if df is None:
            self.display.show_error("Tidak ada data hasil penelitian")
            return
        
        # Format pesan hasil
        pesan_hasil = "🔬 Hasil Skenario Penelitian:\n\n"
        for _, row in df.iterrows():
            pesan_hasil += (
                f"Skenario: {row['Skenario']}\n"
                f"Akurasi: {row['Akurasi']:.4f}\n"
                f"F1-Score: {row['F1-Score']:.4f}\n"
                f"Waktu Inferensi: {row['Waktu Inferensi']*1000:.2f} ms\n\n"
            )
        
        # Tampilkan dialog
        self.display.show_dialog(
            "Hasil Skenario Penelitian",
            pesan_hasil,
            {"o": "OK"}
        )