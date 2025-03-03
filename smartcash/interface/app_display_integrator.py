# File: smartcash/interface/app_display_integrator.py
# Author: Alfrida Sabar
# Deskripsi: Integrator antara DisplayManager dan aplikasi SmartCash

import curses
from typing import Dict, Optional, Any

from smartcash.interface.display.display_manager import DisplayManager
from smartcash.cli.configuration_manager import ConfigurationManager
from smartcash.utils.logger import SmartCashLogger

class AppDisplayIntegrator:
    """
    Kelas yang menangani integrasi antara DisplayManager dan komponen aplikasi lainnya.
    Menyediakan helper function untuk skenario umum dalam konteks aplikasi.
    """
    
    def __init__(
        self, 
        stdscr: curses.window, 
        config_manager: ConfigurationManager,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi display integrator.
        
        Args:
            stdscr: Curses window utama
            config_manager: Manager konfigurasi aplikasi
            logger: Logger opsional
        """
        self.stdscr = stdscr
        self.config_manager = config_manager
        self.logger = logger or SmartCashLogger(__name__)
        
        # Inisialisasi display manager
        self.display_manager = DisplayManager(stdscr)
        
        # Siapkan handler untuk menangkap event
        self._setup_event_handlers()
        
    def _setup_event_handlers(self) -> None:
        """Setup callback dan handler untuk event."""
        # Contoh event handler: update log saat ada perubahan konfigurasi
        # Akan diimplementasikan jika sistem event sudah ada
        pass
    
    def refresh_display(self) -> None:
        """Refresh semua komponen display dengan konfigurasi terbaru."""
        try:
            # Dapatkan konfigurasi terbaru
            current_config = self.config_manager.current_config
            
            # Tampilkan
            self.display_manager.draw(current_config)
            
        except Exception as e:
            self.logger.error(f"❌ Error saat refresh display: {str(e)}")
            # Jangan raise exception, cukup log
    
    def show_config_validation_results(
        self, 
        validation_results: Dict[str, bool]
    ) -> None:
        """
        Tampilkan hasil validasi konfigurasi dengan format yang baik.
        
        Args:
            validation_results: Dict hasil validasi konfigurasi
        """
        # Format hasil validasi
        message_lines = ["Hasil validasi konfigurasi:"]
        
        for key, is_valid in validation_results.items():
            status = "✅ Valid" if is_valid else "❌ Tidak valid"
            message_lines.append(f"{key}: {status}")
            
        if not all(validation_results.values()):
            message_lines.append("\nBeberapa konfigurasi tidak valid. Silakan perbaiki.")
            
        # Tampilkan sebagai dialog
        self.display_manager.show_dialog(
            "Validasi Konfigurasi",
            "\n".join(message_lines)
        )
    
    def show_training_progress(
        self, 
        epoch: int, 
        epochs: int, 
        metrics: Dict[str, float]
    ) -> None:
        """
        Tampilkan kemajuan training dengan metrik.
        
        Args:
            epoch: Epoch saat ini
            epochs: Total epoch
            metrics: Metrik training
        """
        # Format metrik
        formatted_metrics = []
        for key, value in metrics.items():
            if isinstance(value, float):
                formatted_metrics.append(f"{key}: {value:.4f}")
            else:
                formatted_metrics.append(f"{key}: {value}")
                
        metrics_str = ", ".join(formatted_metrics)
        
        # Tampilkan progress
        self.display_manager.show_progress(
            f"Epoch {epoch}/{epochs} - {metrics_str}",
            epoch,
            epochs
        )
    
    def prompt_for_training_params(self) -> Dict[str, Any]:
        """
        Minta parameter training dari user.
        
        Returns:
            Dict parameter training
        """
        # Tentukan field untuk form
        fields = [
            {
                'name': 'batch_size',
                'label': 'Ukuran Batch',
                'type': 'number',
                'default': '32',
                'validator': lambda v: v.isdigit() and int(v) > 0
            },
            {
                'name': 'learning_rate',
                'label': 'Learning Rate',
                'type': 'number',
                'default': '0.001',
                'validator': lambda v: float(v) > 0
            },
            {
                'name': 'epochs',
                'label': 'Jumlah Epoch',
                'type': 'number',
                'default': '100',
                'validator': lambda v: v.isdigit() and int(v) > 0
            },
            {
                'name': 'backbone',
                'label': 'Backbone',
                'type': 'select',
                'default': 'efficientnet',
                'options': ['cspdarknet', 'efficientnet']
            }
        ]
        
        # Tampilkan form
        return self.display_manager.show_form("Parameter Training", fields)
    
    def update_config_from_user(self, section: str) -> bool:
        """
        Update konfigurasi berdasarkan input user.
        
        Args:
            section: Bagian konfigurasi yang akan diupdate
            
        Returns:
            True jika berhasil, False jika dibatalkan
        """
        if section == 'training':
            # Dapatkan parameter training
            params = self.prompt_for_training_params()
            
            # Jika user membatalkan
            if not params:
                return False
                
            # Update konfigurasi
            for key, value in params.items():
                # Konversi ke tipe yang sesuai
                if key in ['batch_size', 'epochs']:
                    value = int(value)
                elif key == 'learning_rate':
                    value = float(value)
                    
                # Update section training atau root berdasarkan key
                if key == 'backbone':
                    self.config_manager.update(key, value)
                else:
                    self.config_manager.update(f'training.{key}', value)
                    
            # Simpan perubahan
            try:
                self.config_manager.save()
                self.display_manager.show_success("Parameter training berhasil diupdate")
                return True
            except Exception as e:
                self.display_manager.show_error(f"Gagal menyimpan konfigurasi: {str(e)}")
                return False
                
        # Section lain bisa ditambahkan di sini
        return False
    
    def show_help_for_section(self, section: str) -> None:
        """
        Tampilkan bantuan untuk section tertentu.
        
        Args:
            section: Bagian yang perlu bantuan
        """
        # Buat konten bantuan berdasarkan section
        help_content = {}
        
        if section == 'main':
            help_content = {
                "Umum": (
                    "• Gunakan ↑↓ untuk navigasi menu\n"
                    "• Enter untuk memilih item\n"
                    "• Q atau Ctrl+C untuk kembali/keluar\n"
                    "• ESC untuk membatalkan input\n"
                    "• H untuk tampilan bantuan ini\n"
                    "• D untuk tampilan menu debug (jika terjadi error)"
                ),
                "Navigasi": (
                    "• Gunakan Tab untuk berpindah antara panel\n"
                    "• Home/End untuk scroll ke awal/akhir\n"
                    "• Page Up/Down untuk scroll per halaman"
                )
            }
        elif section == 'training':
            help_content = {
                "Parameter": (
                    "• batch_size: Jumlah sampel per iterasi (8-128)\n"
                    "• learning_rate: Tingkat pembelajaran (0.0001-0.01)\n"
                    "• epochs: Jumlah iterasi pelatihan (10-1000)\n"
                    "• early_stopping: Jumlah epoch tanpa perbaikan (5-50)"
                ),
                "Backbone": (
                    "• CSPDarknet: Backbone standar YOLOv5, kecepatan tinggi\n"
                    "• EfficientNet: Peningkatan akurasi, lebih baik untuk detail"
                )
            }
        
        # Tampilkan bantuan
        self.display_manager.show_help(f"Bantuan: {section.capitalize()}", help_content)