# File: smartcash/utils/model_exporter.py
# Author: Alfrida Sabar
# Deskripsi: Utilitas untuk mengekspor model dalam berbagai format produksi dengan manajemen checkpoint yang komprehensif

import os
import torch
import yaml
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, Union

from smartcash.utils.logger import SmartCashLogger
from smartcash.handlers.checkpoint import CheckpointManager
from smartcash.utils.environment_manager import EnvironmentManager

class ModelExporter:
    """
    Kelas untuk mengekspor model dalam berbagai format produksi dengan dukungan checkpoint komprehensif.
    
    Fitur:
    - Ekspor model ke format TorchScript, ONNX
    - Manajemen checkpoint dengan metadata lengkap
    - Integrasi dengan Google Drive
    - Pelacakan riwayat ekspor
    """
    
    def __init__(
        self, 
        model_manager, 
        checkpoint_handler: CheckpointManager, 
        logger: Optional[SmartCashLogger] = None,
        env_manager: Optional[EnvironmentManager] = None
    ):
        """
        Inisialisasi ModelExporter.
        
        Args:
            model_manager: Instance ModelManager
            checkpoint_handler: Instance CheckpointManager
            logger: Logger kustom (opsional)
            env_manager: Instance EnvironmentManager (opsional)
        """
        self.model_manager = model_manager
        self.checkpoint_handler = checkpoint_handler
        self.logger = logger or SmartCashLogger("model_exporter")
        self.env_manager = env_manager or EnvironmentManager()
        
        # Setup direktori output
        self.export_dir = Path("exports")
        self.export_dir.mkdir(parents=True, exist_ok=True)
        
        # Direktori riwayat ekspor
        self.history_file = self.export_dir / "export_history.yaml"
        
    def export_to_torchscript(
        self, 
        checkpoint_path: Optional[str] = None, 
        optimize: bool = True, 
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ekspor model ke format TorchScript dengan metadata komprehensif.
        
        Args:
            checkpoint_path: Path checkpoint (opsional)
            optimize: Aktifkan optimasi model (default: True)
            metadata: Metadata tambahan untuk ekspor
            
        Returns:
            Path file TorchScript
        """
        self.logger.info("🔄 Memulai ekspor ke TorchScript...")
        
        try:
            # Muat model dari checkpoint
            model, checkpoint = self.checkpoint_handler.load_checkpoint(checkpoint_path)
            model.eval()
            
            # Ekstrak informasi model
            backbone_type = model.backbone.__class__.__name__
            layer_names = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Nama file dengan konvensi baru
            export_filename = (
                f"smartcash_{backbone_type.lower()}_"
                f"torchscript_{timestamp}.pt"
            )
            export_path = self.export_dir / export_filename
            
            # Optimasi jika diperlukan
            if optimize:
                try:
                    model = torch.jit.script(model)
                except Exception as e:
                    self.logger.warning(f"⚠️ Optimasi TorchScript gagal: {str(e)}")
            
            # Simpan model
            torch.jit.save(model, str(export_path))
            
            # Siapkan metadata untuk riwayat
            export_metadata = {
                'format': 'torchscript',
                'backbone': backbone_type,
                'layers': layer_names,
                'timestamp': timestamp,
                'checkpoint': checkpoint_path,
                'metadata': metadata or {}
            }
            
            # Update riwayat ekspor
            self._update_export_history(export_metadata, export_path)
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model: {str(e)}")
            raise
    
    def export_to_onnx(
        self, 
        checkpoint_path: Optional[str] = None, 
        opset_version: int = 12, 
        metadata: Optional[Dict] = None
    ) -> str:
        """
        Ekspor model ke format ONNX dengan metadata komprehensif.
        
        Args:
            checkpoint_path: Path checkpoint (opsional)
            opset_version: Versi ONNX opset
            metadata: Metadata tambahan untuk ekspor
            
        Returns:
            Path file ONNX
        """
        self.logger.info("🔄 Memulai ekspor ke ONNX...")
        
        try:
            # Muat model dari checkpoint
            model, checkpoint = self.checkpoint_handler.load_checkpoint(checkpoint_path)
            model.eval()
            
            # Ekstrak informasi model
            backbone_type = model.backbone.__class__.__name__
            layer_names = getattr(model, 'detection_layers', ['banknote'])
            
            # Buat timestamp untuk penamaan file
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Nama file dengan konvensi baru
            export_filename = (
                f"smartcash_{backbone_type.lower()}_"
                f"onnx_{timestamp}.onnx"
            )
            export_path = self.export_dir / export_filename
            
            # Persiapkan input dummy
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Simpan model ke ONNX
            torch.onnx.export(
                model,
                dummy_input,
                str(export_path),
                opset_version=opset_version,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes={
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            )
            
            # Siapkan metadata untuk riwayat
            export_metadata = {
                'format': 'onnx',
                'backbone': backbone_type,
                'layers': layer_names,
                'timestamp': timestamp,
                'checkpoint': checkpoint_path,
                'opset_version': opset_version,
                'metadata': metadata or {}
            }
            
            # Update riwayat ekspor
            self._update_export_history(export_metadata, export_path)
            
            self.logger.success(f"✅ Berhasil mengekspor model ke {export_path}")
            return str(export_path)
        
        except Exception as e:
            self.logger.error(f"❌ Gagal mengekspor model: {str(e)}")
            raise
    
    def _update_export_history(
        self, 
        metadata: Dict[str, Any], 
        export_path: Path
    ) -> None:
        """
        Update riwayat ekspor dalam file YAML.
        
        Args:
            metadata: Metadata ekspor
            export_path: Path file yang diekspor
        """
        try:
            # Baca riwayat yang ada atau buat baru
            if self.history_file.exists():
                with open(self.history_file, 'r') as f:
                    export_history = yaml.safe_load(f) or {'exports': []}
            else:
                export_history = {'exports': []}
            
            # Tambahkan metadata baru
            export_history['exports'].append({
                'file': str(export_path),
                **metadata
            })
            
            # Simpan riwayat
            with open(self.history_file, 'w') as f:
                yaml.dump(export_history, f, default_flow_style=False)
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal memperbarui riwayat ekspor: {str(e)}")
    
    def copy_to_drive(
        self, 
        export_path: Union[str, Path], 
        drive_dir: Optional[str] = None
    ) -> Optional[str]:
        """
        Salin model yang diekspor ke Google Drive.
        
        Args:
            export_path: Path model yang diekspor
            drive_dir: Direktori Drive (opsional)
            
        Returns:
            Path model di Drive atau None jika gagal
        """
        # Gunakan default drive jika tidak ditentukan
        if drive_dir is None:
            drive_dir = "/content/drive/MyDrive/SmartCash/exports"
        
        # Pastikan Colab terdeteksi dan Drive di-mount
        if not self.env_manager.is_colab or not self.env_manager.is_drive_mounted:
            self.logger.warning("⚠️ Google Colab atau Drive tidak terdeteksi")
            return None
        
        try:
            # Pastikan direktori tujuan ada
            drive_path = Path(drive_dir)
            drive_path.mkdir(parents=True, exist_ok=True)
            
            # Salin file
            import shutil
            destination = drive_path / Path(export_path).name
            shutil.copy(export_path, destination)
            
            self.logger.success(f"✅ Model berhasil disalin ke {destination}")
            return str(destination)
        
        except Exception as e:
            self.logger.error(f"❌ Gagal menyalin ke Drive: {str(e)}")
            return None
    
    def list_exports(self) -> Dict[str, Any]:
        """
        Dapatkan daftar model yang sudah diekspor.
        
        Returns:
            Dictionary riwayat ekspor
        """
        try:
            if not self.history_file.exists():
                return {'total_exports': 0, 'exports': []}
            
            with open(self.history_file, 'r') as f:
                return yaml.safe_load(f) or {'total_exports': 0, 'exports': []}
        
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal membaca riwayat ekspor: {str(e)}")
            return {'total_exports': 0, 'exports': []}