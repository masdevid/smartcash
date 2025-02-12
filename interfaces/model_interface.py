# File: src/interfaces/model_interface.py
# Author: Alfrida Sabar
# Deskripsi: Antarmuka modular untuk operasi model SmartCash Detector dengan konteks yang lebih baik

from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass
from termcolor import colored
from .base_interface import BaseInterface
from models.factory import ModelFactory, ModelType, ModelConfig
from utils.logging import ColoredLogger
from config.manager import ConfigManager

@dataclass
class ModelContext:
    """Konteks status dan konfigurasi model"""
    model_type: Optional[ModelType] = None
    weights_path: Optional[str] = None
    training_status: Dict = None
    last_evaluation: Dict = None
    is_modified: bool = False
class ModelOperationHandler:
    """Handler untuk operasi-operasi model"""
    def __init__(self, config: ConfigManager, data_path: Path):
        self.logger = ColoredLogger('ModelHandler')
        self.cfg = config
        self.data_path = data_path
        self.factory = ModelFactory()
        
    def create_model(self, context: ModelContext) -> Dict:
        """Buat model baru dengan konteks yang diberikan"""
        try:
            config = ModelConfig(
                type=context.model_type,
                weights_path=context.weights_path,
                img_size=self.cfg.model.img_size,
                nc=self.cfg.model.nc
            )
            
            model = self.factory.create_model(config)
            return {'success': True, 'model': model}
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def validate_weights(self, weights_path: str) -> Dict:
        """Validasi file weights"""
        path = Path(weights_path)
        if not path.exists():
            return {'valid': False, 'error': '❌ File weights tidak ditemukan'}
        if not path.suffix == '.pt':
            return {'valid': False, 'error': '❌ Format file tidak valid'}
        return {'valid': True}

class ModelInterface(BaseInterface):
    """Antarmuka utama untuk manajemen model"""
    def __init__(self, config: ConfigManager, data_path: Path):
        super().__init__()
        self.cfg = config
        self.data_path = data_path
        self.handler = ModelOperationHandler(config, data_path)
        self.context = ModelContext()
        self.model = None

    def tampilkan_menu(self) -> str:
        """Tampilkan menu utama dengan status model"""
        menu = """
🤖 Menu Operasi Model:

1. Training Model
   └─ Latih model baru atau lanjutkan training
2. Hyperparameter Tuning
   └─ Optimasi parameter model
3. Status Model
   └─ Informasi dan metrik model
4. Ekspor Model
   └─ Ekspor ke format yang didukung

0. Kembali ke Menu Utama
"""
        return self.prompt(menu, color='cyan')

    def handle_menu(self):
        """Handle pilihan menu utama"""
        while True:
            pilihan = self.tampilkan_menu()
            
            try:
                if pilihan == '0':
                    break
                elif pilihan == '1':
                    self._handle_training()
                elif pilihan == '2':
                    self._handle_tuning()
                elif pilihan == '3':
                    self._display_detailed_status()
                elif pilihan == '4':
                    self._handle_export()
                else:
                    self.show_error("❌ Pilihan menu tidak valid!")
                    continue
                
                # Konfirmasi kembali ke menu
                if not self.confirm("\nKembali ke menu Model Operations?"):
                    break
                    
            except KeyboardInterrupt:
                self.logger.warning("\n⚠️ Operasi dibatalkan oleh pengguna")
                continue
            except Exception as e:
                self.show_error(f"❌ Terjadi kesalahan: {str(e)}")

    def _handle_training(self):
        """Handle proses training model"""
        self.logger.info("\n🎯 Konfigurasi Training Model")
        
        # Pilih tipe model jika belum ada
        if not self.context.model_type:
            model_type = self._select_model_type()
            if not model_type:
                return
            self.context.model_type = model_type
        
        # Konfigurasi weights
        weights_config = self._configure_weights()
        if weights_config.get('cancelled'):
            return
        self.context.weights_path = weights_config.get('path')
        
        # Buat model
        result = self.handler.create_model(self.context)
        if not result['success']:
            self.show_error(f"❌ Gagal membuat model: {result['error']}")
            return
        
        self.model = result['model']
        
        # Konfigurasi parameter training
        training_config = self._get_training_config()
        if training_config.get('cancelled'):
            return
            
        # Konfirmasi dan mulai training
        self._display_training_summary(training_config)
        if self.confirm("\nMulai training?"):
            try:
                self.model.train(
                    data_yaml=self.data_path / 'rupiah.yaml',
                    **training_config
                )
                self.context.training_status = {
                    'epochs_completed': training_config['epochs'],
                    'last_batch_size': training_config['batch_size']
                }
                self.context.is_modified = True
                self.show_success("✨ Training selesai dengan sukses!")
            except Exception as e:
                self.show_error(f"❌ Training gagal: {str(e)}")

    def _handle_tuning(self):
        """Handle proses tuning hyperparameter"""
        if not self.model:
            self.show_error("❌ Harap train model terlebih dahulu!")
            return
            
        self.logger.info("\n⚙️ Hyperparameter Tuning")
        # TODO: Implement hyperparameter tuning

    def _handle_export(self):
        """Handle proses ekspor model"""
        if not self.model:
            self.show_error("❌ Tidak ada model untuk diekspor!")
            return
            
        self.logger.info("\n📦 Ekspor Model")
        # TODO: Implement model export

    def _select_model_type(self) -> Optional[ModelType]:
        """Pilih tipe model dengan informasi kontekstual"""
        options = {
            "1": ("YOLOv5 with CSPDarknet", ModelType.YOLO_DARKNET, 
                  "Model dasar, performa standar"),
            "2": ("YOLOv5 with EfficientNet-B4", ModelType.YOLO_EFFICIENT,
                  "Rekomendasi: performa optimal untuk deteksi nominal"),
            "3": ("YOLOv5 with EfficientNet-B0", ModelType.YOLO_EFFICIENT_SMALL,
                  "Model ringan, cocok untuk perangkat terbatas"),
            "4": ("YOLOv5 with EfficientNet-B7", ModelType.YOLO_EFFICIENT_LARGE,
                  "Model berat, akurasi maksimal")
        }

        self.logger.info("\n📊 Model yang Tersedia:")
        for key, (name, _, desc) in options.items():
            print(colored(f"{key}. {name}", 'yellow'))
            print(f"   └─ {desc}")

        choice = self.prompt("\nPilih model", default="2")
        return options.get(choice, (None, None, None))[1]

    def _configure_weights(self) -> Dict:
        """Konfigurasi weights dengan validasi"""
        self.logger.info("\n💾 Konfigurasi Weights:")
        print(colored("1. Mulai dari awal (tanpa pre-trained weights)", 'yellow'))
        print(colored("2. Gunakan weights yang direkomendasikan", 'yellow'))
        print(colored("3. Gunakan weights kustom", 'yellow'))
        print(colored("0. Batal", 'yellow'))

        choice = self.prompt("\nPilih opsi", default="2")
        
        if choice == "0":
            return {'cancelled': True}
        elif choice == "1":
            return {'path': None}
        elif choice == "2":
            return {'path': "pretrained"}
        elif choice == "3":
            path = self.prompt("Masukkan path ke file weights")
            result = self.handler.validate_weights(path)
            if not result['valid']:
                self.show_error(result['error'])
                return {'cancelled': True}
            return {'path': path}
        else:
            self.show_error("❌ Pilihan tidak valid!")
            return {'cancelled': True}

    def _get_training_config(self) -> Dict:
        """Dapatkan konfigurasi training dengan validasi"""
        config = {}
        
        self.logger.info("\n⚙️ Konfigurasi Training:")
        
        # Epochs
        epochs = self.prompt("Jumlah epochs", default="100")
        if not epochs.isdigit() or int(epochs) < 1:
            self.show_error("❌ Jumlah epochs tidak valid!")
            return {'cancelled': True}
        config['epochs'] = int(epochs)
        
        # Batch size
        batch_size = self.prompt("Batch size", default="16")
        if not batch_size.isdigit() or int(batch_size) < 1:
            self.show_error("❌ Batch size tidak valid!")
            return {'cancelled': True}
        config['batch_size'] = int(batch_size)
        
        # Learning rate
        lr = self.prompt("Learning rate", default="0.01")
        try:
            config['learning_rate'] = float(lr)
        except ValueError:
            self.show_error("❌ Learning rate tidak valid!")
            return {'cancelled': True}
            
        return config

    def _display_model_status(self):
        """Tampilkan status model saat ini"""
        status = "\n📊 Status Model Saat Ini:"
        status += f"\n  • Tipe: {self.context.model_type.value}"
        status += f"\n  • Weights: {'Menggunakan' if self.context.weights_path else 'Tidak menggunakan'}"
        
        if self.context.training_status:
            status += f"\n  • Epochs: {self.context.training_status['epochs_completed']}"
            status += f"\n  • Batch Size: {self.context.training_status['last_batch_size']}"
        
        print(colored(status, 'cyan'))

    def _display_training_summary(self, config: Dict):
        """Tampilkan ringkasan konfigurasi training"""
        summary = "\n📋 Ringkasan Konfigurasi Training:"
        summary += f"\n  • Model: {self.context.model_type.value}"
        summary += f"\n  • Weights: {'Pre-trained' if self.context.weights_path else 'Fresh'}"
        summary += f"\n  • Epochs: {config['epochs']}"
        summary += f"\n  • Batch Size: {config['batch_size']}"
        summary += f"\n  • Learning Rate: {config['learning_rate']}"
        
        print(colored(summary, 'cyan'))

    def _display_detailed_status(self):
        """Tampilkan status detail model"""
        if not self.model:
            self.show_error("❌ Tidak ada model yang aktif!")
            return
            
        self.logger.info("\n📊 Status Detail Model:")
        
        try:
            # Informasi Model Umum
            print(colored("🤖 Informasi Model", 'cyan'))
            print(f"  • Tipe Model: {self.context.model_type.value}")
            print(f"  • Ukuran Input: {self.cfg.model.img_size}x{self.cfg.model.img_size}")
            print(f"  • Jumlah Kelas: {self.cfg.model.nc}")
            
            # Informasi Weights
            print(colored("\n💾 Informasi Weights", 'cyan'))
            print(f"  • Path Weights: {self.context.weights_path or 'Tidak menggunakan weights'}")
            
            # Status Training
            if self.context.training_status:
                print(colored("\n🏋️ Status Training", 'cyan'))
                print(f"  • Epochs Terselesaikan: {self.context.training_status.get('epochs_completed', 'N/A')}")
                print(f"  • Batch Size Terakhir: {self.context.training_status.get('last_batch_size', 'N/A')}")
            
            # Informasi Backbone (jika tersedia)
            if hasattr(self.model, 'backbone'):
                print(colored("\n🧠 Informasi Backbone", 'cyan'))
                backbone = self.model.backbone
                if hasattr(backbone, 'width') and hasattr(backbone, 'depth'):
                    print(f"  • Lebar: {backbone.width}")
                    print(f"  • Kedalaman: {backbone.depth}")
            
            # Informasi Parameter Model
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            
            print(colored("\n📊 Statistik Parameter", 'cyan'))
            print(f"  • Total Parameter: {total_params:,}")
            print(f"  • Parameter Trainable: {trainable_params:,}")
            print(f"  • Persentase Parameter Trainable: {trainable_params/total_params*100:.2f}%")
            
            # Status Modifikasi
            print(colored("\n🔄 Status Modifikasi", 'cyan'))
            print(f"  • Model Dimodifikasi: {'Ya' if self.context.is_modified else 'Tidak'}")
            
        except Exception as e:
            self.show_error(f"❌ Gagal menampilkan status detail: {str(e)}")