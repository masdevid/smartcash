# File: smartcash/handlers/model/integration/exporter_adapter.py
# Author: Alfrida Sabar
# Deskripsi: Adapter untuk integrasi dengan ModelExporter

from typing import Dict, Optional, Any, Union
from pathlib import Path
from datetime import datetime
import torch

from smartcash.utils.logger import get_logger, SmartCashLogger
from smartcash.exceptions.base import ModelError

class ExporterAdapter:
    """
    Adapter untuk integrasi dengan ModelExporter.
    Menyediakan antarmuka yang konsisten untuk ekspor model ke berbagai format.
    """
    
    def __init__(
        self,
        config: Dict,
        logger: Optional[SmartCashLogger] = None
    ):
        """
        Inisialisasi exporter adapter.
        
        Args:
            config: Konfigurasi aplikasi
            logger: Custom logger (opsional)
        """
        self.config = config
        self.logger = logger or get_logger("exporter_adapter")
        
        # Setup direktori output
        export_dir = config.get('model', {}).get('export_dir', "exports")
        self.export_dir = Path(export_dir)
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"📦 ExporterAdapter diinisialisasi (output: {self.export_dir})")
    
    def export_to_torchscript(
        self,
        model: torch.nn.Module,
        input_shape: Optional[list] = None,
        filename: Optional[str] = None,
        optimize: bool = True
    ) -> str:
        """
        Export model ke format TorchScript.
        
        Args:
            model: Model yang akan diexport
            input_shape: Bentuk input tensor [B, C, H, W] (opsional)
            filename: Nama file output (opsional)
            optimize: Flag untuk optimasi model (opsional)
            
        Returns:
            Path ke file TorchScript
        """
        self.logger.info("🔄 Memulai ekspor ke TorchScript...")
        
        try:
            # Set model ke mode evaluasi
            model.eval()
            
            # Tentukan input shape jika tidak ada
            if input_shape is None:
                input_shape = [1, 3, 640, 640]
            
            # Buat example input
            example_input = torch.rand(*input_shape)
            
            # Trace model
            with torch.no_grad():
                traced_model = torch.jit.trace(model, example_input)
                
                if optimize:
                    traced_model = torch.jit.optimize_for_inference(traced_model)
            
            # Generate nama file jika tidak ada
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backbone_type = getattr(model, 'backbone_type', 'model')
                filename = f"{backbone_type}_{timestamp}.pt"
            
            # Simpan model
            export_path = self.export_dir / filename
            traced_model.save(str(export_path))
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model ke TorchScript: {str(e)}")
            raise ModelError(f"Gagal mengekspor model ke TorchScript: {str(e)}")
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        input_shape: Optional[list] = None,
        filename: Optional[str] = None,
        opset_version: int = 12
    ) -> str:
        """
        Export model ke format ONNX.
        
        Args:
            model: Model yang akan diexport
            input_shape: Bentuk input tensor [B, C, H, W] (opsional)
            filename: Nama file output (opsional)
            opset_version: Versi ONNX opset (opsional)
            
        Returns:
            Path ke file ONNX
        """
        self.logger.info("🔄 Memulai ekspor ke ONNX...")
        
        try:
            # Pastikan package onnx tersedia
            try:
                import onnx
                import onnxruntime
            except ImportError as e:
                self.logger.error(f"❌ Package ONNX tidak tersedia: {str(e)}")
                raise ModelError("Package ONNX tidak tersedia. Install dengan 'pip install onnx onnxruntime'")
            
            # Set model ke mode evaluasi
            model.eval()
            
            # Tentukan input shape jika tidak ada
            if input_shape is None:
                input_shape = [1, 3, 640, 640]
            
            # Buat example input
            example_input = torch.rand(*input_shape)
            
            # Generate nama file jika tidak ada
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backbone_type = getattr(model, 'backbone_type', 'model')
                filename = f"{backbone_type}_{timestamp}.onnx"
            
            # Simpan model
            export_path = self.export_dir / filename
            
            # Export ke ONNX
            torch.onnx.export(
                model,
                example_input,
                str(export_path),
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            # Validasi model ONNX
            onnx_model = onnx.load(str(export_path))
            onnx.checker.check_model(onnx_model)
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model ke ONNX: {str(e)}")
            raise ModelError(f"Gagal mengekspor model ke ONNX: {str(e)}")
    
    def copy_to_drive(
        self,
        export_path: Union[str, Path],
        drive_dir: Optional[str] = None
    ) -> str:
        """
        Salin model yang diekspor ke Google Drive.
        
        Args:
            export_path: Path ke file model
            drive_dir: Direktori tujuan di Drive (opsional)
            
        Returns:
            Path tujuan di Drive
        """
        self.logger.info(f"📤 Menyalin {export_path} ke Google Drive...")
        
        try:
            # Pastikan berjalan di Colab
            try:
                import google.colab
            except ImportError:
                self.logger.warning("⚠️ Bukan di Google Colab, tidak dapat menyalin ke Drive")
                return str(export_path)
            
            # Tentukan direktori tujuan jika tidak ada
            if drive_dir is None:
                drive_dir = "/content/drive/MyDrive/SmartCash/exports"
            
            # Pastikan direktori tujuan ada
            drive_dir = Path(drive_dir)
            drive_dir.mkdir(parents=True, exist_ok=True)
            
            # Salin file
            import shutil
            source_path = Path(export_path)
            target_path = drive_dir / source_path.name
            shutil.copy(str(source_path), str(target_path))
            
            self.logger.success(f"✅ Berhasil menyalin ke {target_path}")
            return str(target_path)
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin ke Google Drive: {str(e)}")
            raise ModelError(f"Gagal menyalin ke Google Drive: {str(e)}")