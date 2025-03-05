# File: smartcash/utils/model_exporter.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk mengekspor model dalam berbagai format yang dapat digunakan di produksi

from pathlib import Path
import torch
from typing import Dict, Optional, Callable
from smartcash.handlers.checkpoint_handler import CheckpointHandler
from smartcash.utils.logger import SmartCashLogger

class ModelExporter:
    """Kelas untuk mengekspor model dalam berbagai format"""
    
    def __init__(self, model_manager, checkpoint_handler: CheckpointHandler, logger=None):
        self.model_manager = model_manager
        self.checkpoint_handler = checkpoint_handler
        self.logger = logger or SmartCashLogger("model_exporter")
        
        # Setup direktori output
        self.export_dir = Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_torchscript(self, checkpoint_path=None, optimize=True):
        """Ekspor model ke format TorchScript"""
        self.logger.info("🔄 Memulai ekspor ke TorchScript...")
        
        try:
            # Muat model
            model = self.model_manager.load_model(self.checkpoint_handler.get_checkpoint_path(checkpoint_path))
            model.eval()
            
            # Ekstrak info backbone dan mode dari model
            backbone_type = model.backbone.__class__.__name__
            detection_layers = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export model ke TorchScript
            scripted_model = torch.jit.script(model)
            
            # Simpan model
            export_path = self.export_dir / f"{backbone_type}_{timestamp}.pt"
            scripted_model.save(str(export_path))
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model: {str(e)}")
            raise

    def export_to_onnx(self, checkpoint_path=None, opset_version=12):
        """Ekspor model ke format ONNX"""
        self.logger.info("🔄 Memulai ekspor ke ONNX...")
        
        try:
            # Muat model
            model = self.model_manager.load_model(self.checkpoint_handler.get_checkpoint_path(checkpoint_path))
            model.eval()
            
            # Ekstrak info backbone dan mode dari model
            backbone_type = model.backbone.__class__.__name__
            detection_layers = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export model ke ONNX
            export_path = self.export_dir / f"{backbone_type}_{timestamp}.onnx"
            
            # Simpan model
            torch.onnx.export(
                model,
                torch.randn(1, 3, 224, 224),
                str(export_path),
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model: {str(e)}")
            raise

    def copy_to_drive(self, export_path, drive_dir="/content/drive/MyDrive/SmartCash/exports"):
        """Salin model yang diekspor ke Google Drive"""
        self.logger.info(f"📤 Menyalin {export_path} ke Google Drive...")
        
        try:
            # Pastikan direktori tujuan ada
            drive_dir = Path(drive_dir)
            drive_dir.mkdir(parents=True, exist_ok=True)
            
            # Salin file
            import shutil
            shutil.copy(export_path, drive_dir)
            
            self.logger.success(f"✅ Berhasil menyalin ke {drive_dir}")
            return str(drive_dir / Path(export_path).name)
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin ke Google Drive: {str(e)}")
            raise