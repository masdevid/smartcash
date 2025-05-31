"""
File: smartcash/model/service/checkpoint_service.py
Deskripsi: Implementasi checkpoint service untuk training model
"""

import os
import torch
import json
import time
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from functools import partial

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError
from smartcash.model.service.progress_tracker import ProgressTracker

class CheckpointService(ICheckpointService):
    """Checkpoint service untuk training model dengan progress tracking dan UI integration"""
    
    def __init__(
        self,
        checkpoint_dir: str = "runs/train/checkpoints",
        max_checkpoints: int = 5,
        save_best: bool = True,
        save_last: bool = True,
        save_interval: int = 0,
        metric_name: str = "val_loss",
        mode: str = "min",
        logger = None,
        progress_callback: Optional[Callable] = None
    ):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.save_best = save_best
        self.save_last = save_last
        self.save_interval = save_interval
        self.metric_name = metric_name
        self.mode = mode
        self.logger = logger or get_logger(__name__)
        self.progress_tracker = ProgressTracker(progress_callback)
        
        # Buat direktori checkpoint jika belum ada
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Inisialisasi best metric
        self.best_metric = float('inf') if mode == 'min' else float('-inf')
        self.best_epoch = -1
        self.last_checkpoint_path = None
        self.best_checkpoint_path = None
        
        self.logger.info(f"✨ CheckpointService initialized (dir: {checkpoint_dir}, metric: {metric_name}, mode: {mode})")
    
    def save_checkpoint(
        self,
        model: torch.nn.Module,
        path: Optional[str] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        epoch: int = 0,
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Simpan checkpoint dengan progress tracking"""
        try:
            # Update progress
            self.progress_tracker.update(0, 4, "🔄 Memulai penyimpanan checkpoint...")
            
            # Prepare checkpoint path
            if path is None:
                path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
            else:
                path = Path(path)
                if not path.is_absolute():
                    path = self.checkpoint_dir / path
            
            # Update progress
            self.progress_tracker.update(1, 4, f"📁 Menyiapkan path: {path.name}")
            
            # Prepare metadata
            full_metadata = {
                "epoch": epoch,
                "timestamp": time.time(),
                "metrics": metrics or {},
                "custom": metadata or {}
            }
            
            # Prepare checkpoint data
            checkpoint_data = {
                "model": model.state_dict(),
                "metadata": full_metadata
            }
            
            # Tambahkan optimizer dan scheduler jika tersedia
            if optimizer is not None:
                checkpoint_data["optimizer"] = optimizer.state_dict()
            if scheduler is not None:
                checkpoint_data["scheduler"] = scheduler.state_dict()
                
            # Update progress
            self.progress_tracker.update(2, 4, "📦 Menyiapkan data checkpoint...")
            
            # Simpan checkpoint dengan atomic save
            os.makedirs(path.parent, exist_ok=True)
            torch.save(checkpoint_data, str(path) + ".tmp")
            if os.path.exists(str(path)):
                os.remove(str(path))
            os.rename(str(path) + ".tmp", str(path))
            
            # Update progress
            self.progress_tracker.update(3, 4, "💾 Menyimpan checkpoint...")
            
            # Update last checkpoint path
            self.last_checkpoint_path = str(path)
            
            # Check apakah ini best checkpoint
            if metrics and self.metric_name in metrics:
                current_metric = metrics[self.metric_name]
                is_best = (self.mode == 'min' and current_metric < self.best_metric) or \
                          (self.mode == 'max' and current_metric > self.best_metric)
                
                if is_best:
                    self.best_metric = current_metric
                    self.best_epoch = epoch
                    
                    # Simpan sebagai best checkpoint jika diperlukan
                    if self.save_best:
                        best_path = self.checkpoint_dir / "best.pt"
                        shutil.copy2(path, best_path)
                        self.best_checkpoint_path = str(best_path)
                        self.logger.info(f"🏆 Best checkpoint: {best_path} ({self.metric_name}: {current_metric:.4f})")
            
            # Simpan sebagai last checkpoint jika diperlukan
            if self.save_last:
                last_path = self.checkpoint_dir / "last.pt"
                shutil.copy2(path, last_path)
            
            # Cleanup old checkpoints jika melebihi max_checkpoints
            if self.max_checkpoints > 0:
                self._cleanup_old_checkpoints()
            
            # Update progress
            self.progress_tracker.update(4, 4, f"✅ Checkpoint tersimpan: {path.name}")
            
            # Log info
            self.logger.info(f"✅ Checkpoint tersimpan: {path}")
            
            return str(path)
            
        except Exception as e:
            error_msg = f"❌ Error menyimpan checkpoint: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        map_location: Optional[str] = None
    ) -> Tuple[torch.nn.Module, Optional[Dict[str, Any]]]:
        """Load checkpoint dengan progress tracking"""
        try:
            # Update progress
            self.progress_tracker.update(0, 3, "🔄 Memulai loading checkpoint...")
            
            # Resolve path
            if not os.path.isabs(path):
                path = self.checkpoint_dir / path
            path = Path(path)
            
            # Check if file exists
            if not path.exists():
                error_msg = f"❌ Checkpoint tidak ditemukan: {path}"
                self.logger.error(error_msg)
                self.progress_tracker.error(error_msg, "checkpoint")
                raise ModelCheckpointError(error_msg)
            
            # Update progress
            self.progress_tracker.update(1, 3, f"📂 Loading checkpoint: {path.name}")
            
            # Load checkpoint
            checkpoint = torch.load(path, map_location=map_location)
            
            # Load model state dict jika model diberikan
            if model is not None and "model" in checkpoint:
                model.load_state_dict(checkpoint["model"])
            
            # Load optimizer state dict jika optimizer diberikan
            if optimizer is not None and "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
            
            # Load scheduler state dict jika scheduler diberikan
            if scheduler is not None and "scheduler" in checkpoint:
                scheduler.load_state_dict(checkpoint["scheduler"])
            
            # Extract metadata
            metadata = checkpoint.get("metadata", {})
            
            # Update best metric jika tersedia di metadata
            if "metrics" in metadata and self.metric_name in metadata["metrics"]:
                metric_value = metadata["metrics"][self.metric_name]
                if (self.mode == 'min' and metric_value < self.best_metric) or \
                   (self.mode == 'max' and metric_value > self.best_metric):
                    self.best_metric = metric_value
                    self.best_epoch = metadata.get("epoch", -1)
            
            # Update progress
            self.progress_tracker.update(3, 3, f"✅ Checkpoint berhasil diload: {path.name}")
            
            # Log info
            self.logger.info(f"✅ Checkpoint berhasil diload: {path}")
            
            return model, metadata
            
        except Exception as e:
            error_msg = f"❌ Error loading checkpoint: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """List semua checkpoint dengan metadata"""
        try:
            # Cari semua file checkpoint
            checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
            
            # Early return jika tidak ada checkpoint
            if not checkpoint_files:
                return []
            
            # Update progress
            self.progress_tracker.update(0, len(checkpoint_files), f"🔍 Mencari {len(checkpoint_files)} checkpoint...")
            
            # Extract metadata dari setiap checkpoint
            checkpoints = []
            for i, checkpoint_path in enumerate(checkpoint_files):
                try:
                    # Update progress
                    self.progress_tracker.update(i, len(checkpoint_files), f"📊 Menganalisis: {checkpoint_path.name}")
                    
                    # Get metadata
                    metadata = self._get_checkpoint_metadata(checkpoint_path)
                    checkpoints.append(metadata)
                except Exception as e:
                    self.logger.warning(f"⚠️ Error mengekstrak metadata dari {checkpoint_path.name}: {str(e)}")
            
            # Sort checkpoints
            if sort_by == 'time':
                checkpoints.sort(key=lambda x: x.get('timestamp', 0), reverse=True)
            elif sort_by == 'epoch':
                checkpoints.sort(key=lambda x: x.get('epoch', 0), reverse=True)
            elif sort_by == 'metric':
                if self.mode == 'min':
                    checkpoints.sort(key=lambda x: x.get('metrics', {}).get(self.metric_name, float('inf')))
                else:
                    checkpoints.sort(key=lambda x: x.get('metrics', {}).get(self.metric_name, float('-inf')), reverse=True)
            
            # Update progress
            self.progress_tracker.update(len(checkpoint_files), len(checkpoint_files), f"✅ {len(checkpoints)} checkpoint ditemukan")
            
            return checkpoints
            
        except Exception as e:
            error_msg = f"❌ Error listing checkpoints: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            return []
    
    def get_checkpoint_info(self, path: str) -> Dict[str, Any]:
        """Dapatkan informasi detail tentang checkpoint"""
        try:
            # Resolve path
            if not os.path.isabs(path):
                path = self.checkpoint_dir / path
            checkpoint_path = Path(path)
            
            # Check if file exists
            if not checkpoint_path.exists():
                error_msg = f"❌ Checkpoint tidak ditemukan: {checkpoint_path}"
                self.logger.error(error_msg)
                raise ModelCheckpointError(error_msg)
            
            # Get metadata
            metadata = self._get_checkpoint_metadata(checkpoint_path)
            
            return metadata
            
        except Exception as e:
            error_msg = f"❌ Error mendapatkan info checkpoint: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    def _get_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract metadata dari checkpoint file"""
        try:
            # Load checkpoint
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract metadata
            metadata = checkpoint.get("metadata", {})
            
            # Add file info
            stat = checkpoint_path.stat()
            metadata.update({
                'path': str(checkpoint_path),
                'filename': checkpoint_path.name,
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'size_formatted': self._format_file_size(stat.st_size)
            })
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error mengekstrak metadata dari {checkpoint_path.name}: {str(e)}")
            
            # Return basic info jika gagal extract metadata
            stat = checkpoint_path.stat()
            return {
                'path': str(checkpoint_path),
                'filename': checkpoint_path.name,
                'size': stat.st_size,
                'mtime': stat.st_mtime,
                'size_formatted': self._format_file_size(stat.st_size),
                'error': str(e)
            }
    
    def _cleanup_old_checkpoints(self) -> None:
        """Cleanup old checkpoints, keeping only max_checkpoints"""
        try:
            # Get all checkpoint files except best.pt and last.pt
            checkpoint_files = [p for p in self.checkpoint_dir.glob("*.pt") 
                               if p.name != "best.pt" and p.name != "last.pt"]
            
            # Early return jika tidak perlu cleanup
            if len(checkpoint_files) <= self.max_checkpoints:
                return
            
            # Sort berdasarkan waktu modifikasi (terbaru dulu)
            checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            
            # Hapus checkpoint lama
            for checkpoint_path in checkpoint_files[self.max_checkpoints:]:
                try:
                    os.remove(checkpoint_path)
                    self.logger.debug(f"🗑️ Removed old checkpoint: {checkpoint_path.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Error removing {checkpoint_path.name}: {str(e)}")
                    
        except Exception as e:
            self.logger.warning(f"⚠️ Error during checkpoint cleanup: {str(e)}")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size dari bytes ke human-readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} B"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.1f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.1f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.1f} GB"
    
    def set_progress_callback(self, callback: Callable) -> None:
        """Set progress callback"""
        self.progress_tracker = ProgressTracker(callback)
    
    # One-liner utilities
    get_best_checkpoint_path = lambda self: self.best_checkpoint_path or (str(self.checkpoint_dir / "best.pt") if (self.checkpoint_dir / "best.pt").exists() else None)
    get_last_checkpoint_path = lambda self: self.last_checkpoint_path or (str(self.checkpoint_dir / "last.pt") if (self.checkpoint_dir / "last.pt").exists() else None)
    get_checkpoint_dir = lambda self: str(self.checkpoint_dir)
    get_best_metric = lambda self: self.best_metric
    get_best_epoch = lambda self: self.best_epoch
    should_save_checkpoint = lambda self, epoch: self.save_interval > 0 and epoch % self.save_interval == 0
    is_better_metric = lambda self, current, best: (self.mode == 'min' and current < best) or (self.mode == 'max' and current > best)
