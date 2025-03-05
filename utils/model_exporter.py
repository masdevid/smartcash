# File: smartcash/utils/model_exporter.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk mengekspor model dalam berbagai format yang dapat digunakan di produksi

import os
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Callable
from smartcash.utils.logger import SmartCashLogger

class ModelExporter:
    """Kelas untuk mengekspor model dalam berbagai format"""
    
    def __init__(self, model_manager, logger=None):
        self.model_manager = model_manager
        self.logger = logger or SmartCashLogger("model_exporter")
        
        # Setup direktori output
        self.export_dir = Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
    def export_to_torchscript(self, checkpoint_path=None, optimize=True):
        """Ekspor model ke format TorchScript"""
        self.logger.info("🔄 Memulai ekspor ke TorchScript...")
        
        try:
            # Muat model
            model = self.model_manager.load_model(checkpoint_path)
            model.eval()
            
            # Ekstrak info backbone dan mode dari model
            backbone_type = model.backbone.__class__.__name__
            detection_layers = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Buat nama file export
            model_name = f"smartcash_{backbone_type.lower()}_{'-'.join(detection_layers)}_{timestamp}"
            export_path = self.export_dir / f"{model_name}.pt"
            
            # Buat dummy input
            dummy_input = torch.randn(1, 3, 640, 640, device=next(model.parameters()).device)
            
            # Export model ke TorchScript
            with torch.no_grad():
                # Coba export dengan tracing (lebih cepat)
                try:
                    traced_model = torch.jit.trace(model, dummy_input)
                    if optimize:
                        traced_model = torch.jit.optimize_for_inference(traced_model)
                    traced_model.save(str(export_path))
                    self.logger.success(f"✅ Model berhasil diekspor ke {export_path} (traced)")
                except Exception as e:
                    self.logger.warning(f"⚠️ Ekspor dengan tracing gagal: {str(e)}")
                    self.logger.info("🔄 Mencoba ekspor dengan scripting...")
                    
                    # Backup dengan scripting (lebih serbaguna)
                    script_model = torch.jit.script(model)
                    script_model.save(str(export_path))
                    self.logger.success(f"✅ Model berhasil diekspor ke {export_path} (scripted)")
            
            # Ukuran file
            file_size = export_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"📊 Ukuran file: {file_size:.2f} MB")
            
            return {
                'path': str(export_path),
                'size': f"{file_size:.2f} MB",
                'format': 'TorchScript',
                'timestamp': timestamp
            }
            
        except Exception as e:
            self.logger.error(f"❌ Ekspor model gagal: {str(e)}")
            return None
            
    def export_to_onnx(self, checkpoint_path=None, opset_version=12):
        """Ekspor model ke format ONNX"""
        self.logger.info("🔄 Memulai ekspor ke ONNX...")
        
        try:
            # Muat model
            model = self.model_manager.load_model(checkpoint_path)
            model.eval()
            
            # Ekstrak info backbone dan mode dari model
            backbone_type = model.backbone.__class__.__name__
            detection_layers = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Buat nama file export
            model_name = f"smartcash_{backbone_type.lower()}_{'-'.join(detection_layers)}_{timestamp}"
            export_path = self.export_dir / f"{model_name}.onnx"
            
            # Buat dummy input
            dummy_input = torch.randn(1, 3, 640, 640, device=next(model.parameters()).device)
            
            # Export model ke ONNX
            try:
                torch.onnx.export(
                    model,
                    dummy_input,
                    export_path,
                    verbose=False,
                    opset_version=opset_version,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
                self.logger.success(f"✅ Model berhasil diekspor ke {export_path}")
                
                # Ukuran file
                file_size = export_path.stat().st_size / (1024 * 1024)  # MB
                self.logger.info(f"📊 Ukuran file: {file_size:.2f} MB")
                
                return {
                    'path': str(export_path),
                    'size': f"{file_size:.2f} MB",
                    'format': 'ONNX',
                    'timestamp': timestamp
                }
            except Exception as e:
                self.logger.error(f"❌ Ekspor ke ONNX gagal: {str(e)}")
                return None
            
        except Exception as e:
            self.logger.error(f"❌ Ekspor model gagal: {str(e)}")
            return None

    def copy_to_drive(self, export_path, drive_dir="/content/drive/MyDrive/SmartCash/exports"):
        """Salin model yang diekspor ke Google Drive"""
        try:
            # Cek apakah Google Drive terpasang
            if not os.path.exists("/content/drive"):
                self.logger.warning("⚠️ Google Drive tidak terpasang")
                return None
                
            # Cek apakah drive_dir ada
            if not os.path.exists(drive_dir):
                os.makedirs(drive_dir, exist_ok=True)
                
            # Salin file
            import shutil
            dest_path = os.path.join(drive_dir, os.path.basename(export_path))
            shutil.copy2(export_path, dest_path)
            
            self.logger.success(f"✅ Model berhasil disalin ke {dest_path}")
            return dest_path
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin model ke Drive: {str(e)}")
            return None