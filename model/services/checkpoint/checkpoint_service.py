"""
File: smartcash/model/services/checkpoint/checkpoint_service.py
Deskripsi: Layanan untuk mengelola checkpoint model, memungkinkan penyimpanan dan pemulihan state model
"""

import os
import torch
import shutil
import tempfile
import json
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import time

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError


class CheckpointService(ICheckpointService):
    """
    Layanan untuk mengelola checkpoint model dengan fitur penyimpanan, 
    pemulihan, dan manajemen versi. Mendukung format PyTorch dan ONNX.
    """
    
    def __init__(
        self,
        checkpoint_dir: str = "runs/train/checkpoints",
        max_checkpoints: int = 5,
        logger = None
    ):
        """
        Inisialisasi Checkpoint Service.
        
        Args:
            checkpoint_dir: Direktori untuk menyimpan checkpoint
            max_checkpoints: Jumlah maksimum checkpoint yang disimpan
            logger: Logger untuk mencatat aktivitas
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logger or get_logger("checkpoint_service")
        
        # Buat direktori jika belum ada
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"✨ CheckpointService diinisialisasi (dir: {checkpoint_dir})")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        path: str,
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """
        Simpan checkpoint model.
        
        Args:
            model: Model yang akan disimpan
            path: Path atau nama file untuk menyimpan checkpoint
            optimizer: Optimizer untuk disimpan (opsional)
            epoch: Nomor epoch saat ini
            metadata: Metadata tambahan untuk disimpan
            is_best: Flag untuk menandai checkpoint terbaik
            
        Returns:
            Path lengkap ke file checkpoint yang disimpan
            
        Raises:
            ModelCheckpointError: Jika gagal menyimpan checkpoint
        """
        try:
            checkpoint_path = Path(path)
            
            # Pastikan direktori ada
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.checkpoint_dir / checkpoint_path
                
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Siapkan data checkpoint
            checkpoint_data = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            
            # Tambahkan optimizer state jika disediakan
            if optimizer is not None:
                checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
            
            # Gunakan temporary file untuk atomic write
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
                torch.save(checkpoint_data, tmp_file.name)
                # Pindahkan file sementara ke tujuan akhir
                shutil.move(tmp_file.name, checkpoint_path)
            
            self.logger.info(f"💾 Checkpoint disimpan ke {checkpoint_path}")
            
            # Simpan juga sebagai best model jika diperlukan
            if is_best:
                best_path = checkpoint_path.parent / 'best.pt'
                shutil.copy2(checkpoint_path, best_path)
                self.logger.info(f"🏆 Checkpoint ditandai sebagai terbaik: {best_path}")
            
            # Kelola jumlah checkpoint
            self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Gagal menyimpan checkpoint: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Union[Dict[str, Any], torch.nn.Module]:
        """
        Load checkpoint model.
        
        Args:
            path: Path ke file checkpoint
            model: Model untuk load state (opsional)
            optimizer: Optimizer untuk load state (opsional)
            map_location: Device untuk load checkpoint
            
        Returns:
            Model yang diload atau Dictionary checkpoint data jika model tidak disediakan
            
        Raises:
            ModelCheckpointError: Jika gagal load checkpoint
        """
        try:
            checkpoint_path = Path(path)
            
            # Cek path relatif terhadap checkpoint_dir
            if not checkpoint_path.is_absolute():
                checkpoint_path = self.checkpoint_dir / checkpoint_path
                
            # Cek jika file ada
            if not checkpoint_path.exists():
                # Coba dengan ekstensi .pt
                if not str(checkpoint_path).endswith('.pt'):
                    checkpoint_path = Path(f"{checkpoint_path}.pt")
                
                # Cek lagi dengan ekstensi
                if not checkpoint_path.exists():
                    raise FileNotFoundError(f"File checkpoint tidak ditemukan: {path}")
            
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Load state ke model jika disediakan
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                metadata = checkpoint.get('metadata', {})
                epoch = checkpoint.get('epoch', 0)
                
                self.logger.info(
                    f"📂 Checkpoint diload dari {checkpoint_path}\n"
                    f"   • Epoch: {epoch}\n"
                    f"   • Metadata: {', '.join(f'{k}={v}' for k, v in metadata.items()) if metadata else 'None'}"
                )
                
                return model
            else:
                # Return dictionary jika model tidak disediakan
                self.logger.info(f"📂 Checkpoint data diload dari {checkpoint_path}")
                return checkpoint
                
        except Exception as e:
            error_msg = f"❌ Gagal load checkpoint: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """
        Dapatkan path ke checkpoint terbaru.
        
        Returns:
            Path ke checkpoint terbaru atau None jika tidak ada
        """
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        
        if not checkpoints:
            return None
            
        # Urutkan berdasarkan waktu modifikasi (terbaru di awal)
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        return str(latest)
    
    def get_best_checkpoint(self) -> Optional[str]:
        """
        Dapatkan path ke checkpoint terbaik.
        
        Returns:
            Path ke checkpoint terbaik atau None jika tidak ada
        """
        best_path = self.checkpoint_dir / 'best.pt'
        
        if best_path.exists():
            return str(best_path)
        return None
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """
        Daftar semua checkpoint yang tersedia.
        
        Args:
            sort_by: Kriteria pengurutan ('time', 'name', atau 'epoch')
            
        Returns:
            List checkpoint info dengan metadata
        """
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        result = []
        
        for ckpt in checkpoints:
            try:
                # Load hanya metadata
                checkpoint = torch.load(ckpt, map_location='cpu')
                
                info = {
                    'path': str(ckpt),
                    'name': ckpt.name,
                    'timestamp': checkpoint.get('timestamp', ckpt.stat().st_mtime),
                    'epoch': checkpoint.get('epoch', 0),
                    'metadata': checkpoint.get('metadata', {})
                }
                
                # Cek apakah ini best checkpoint
                info['is_best'] = (ckpt.name == 'best.pt')
                
                result.append(info)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal membaca checkpoint {ckpt}: {str(e)}")
        
        # Urutkan hasil
        if sort_by == 'time':
            result.sort(key=lambda x: x['timestamp'], reverse=True)
        elif sort_by == 'name':
            result.sort(key=lambda x: x['name'])
        elif sort_by == 'epoch':
            result.sort(key=lambda x: x['epoch'], reverse=True)
            
        return result
    
    def export_to_onnx(
        self,
        model: torch.nn.Module,
        output_path: str,
        input_shape: List[int] = [1, 3, 640, 640],
        opset_version: int = 12,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """
        Export model PyTorch ke ONNX.
        
        Args:
            model: Model PyTorch yang akan diekspor
            output_path: Path untuk menyimpan model ONNX
            input_shape: Bentuk input tensor
            opset_version: Versi ONNX opset
            dynamic_axes: Dictionary axes dinamis untuk input/output
            
        Returns:
            Path lengkap ke file ONNX yang disimpan
            
        Raises:
            ModelCheckpointError: Jika gagal melakukan ekspor
        """
        try:
            onnx_path = Path(output_path)
            
            # Pastikan direktori ada
            if not onnx_path.is_absolute():
                onnx_path = self.checkpoint_dir / onnx_path
                
            onnx_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Tambahkan ekstensi .onnx jika belum ada
            if not str(onnx_path).endswith('.onnx'):
                onnx_path = Path(f"{onnx_path}.onnx")
            
            # Set model ke mode evaluasi
            model.eval()
            
            # Buat dummy input
            dummy_input = torch.randn(input_shape, requires_grad=True)
            
            # Set dynamic axes jika tidak ada
            if dynamic_axes is None:
                dynamic_axes = {
                    'input': {0: 'batch_size'},
                    'output': {0: 'batch_size'}
                }
            
            # Export ke ONNX
            torch.onnx.export(
                model,
                dummy_input,
                onnx_path,
                export_params=True,
                opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'],
                output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            self.logger.info(f"📤 Model berhasil diekspor ke ONNX: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            error_msg = f"❌ Gagal mengekspor model ke ONNX: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    def _cleanup_old_checkpoints(self) -> None:
        """
        Hapus checkpoint lama jika melebihi jumlah maksimum yang ditentukan.
        Checkpoint 'best.pt' tidak akan dihapus.
        """
        # Dapatkan semua checkpoint kecuali best.pt
        checkpoints = [
            p for p in self.checkpoint_dir.glob('*.pt') 
            if p.name != 'best.pt'
        ]
        
        # Jika jumlah checkpoint tidak melebihi maksimum, tidak perlu tindakan
        if len(checkpoints) <= self.max_checkpoints:
            return
            
        # Urutkan berdasarkan waktu modifikasi (terlama di awal)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        # Hapus checkpoint terlama hingga jumlah maksimum
        for i in range(len(checkpoints) - self.max_checkpoints):
            try:
                os.remove(checkpoints[i])
                self.logger.debug(f"🧹 Checkpoint lama dihapus: {checkpoints[i]}")
            except Exception as e:
                self.logger.warning(f"⚠️ Gagal menghapus checkpoint lama: {str(e)}")
                
    def add_metadata(self, checkpoint_path: str, metadata: Dict[str, Any]) -> bool:
        """
        Tambahkan atau update metadata pada checkpoint yang ada.
        
        Args:
            checkpoint_path: Path ke file checkpoint
            metadata: Metadata yang akan ditambahkan/diupdate
            
        Returns:
            Boolean yang menunjukkan keberhasilan operasi
        """
        try:
            ckpt_path = Path(checkpoint_path)
            
            # Cek path relatif terhadap checkpoint_dir
            if not ckpt_path.is_absolute():
                ckpt_path = self.checkpoint_dir / ckpt_path
                
            # Load checkpoint
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            
            # Update metadata
            if 'metadata' not in checkpoint:
                checkpoint['metadata'] = {}
                
            checkpoint['metadata'].update(metadata)
            
            # Simpan kembali
            torch.save(checkpoint, ckpt_path)
            
            self.logger.info(f"📝 Metadata ditambahkan ke checkpoint: {ckpt_path}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Gagal menambahkan metadata: {str(e)}")
            return False