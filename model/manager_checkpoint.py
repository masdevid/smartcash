"""
File: smartcash/model/manager_checkpoint.py
Deskripsi: Unified checkpoint manager dengan progress tracking dan UI integration
"""

import os
import torch
import json
import time
import shutil
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path
from functools import partial

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError
from smartcash.model.utils import (
    prepare_checkpoint_data, atomic_save, resolve_checkpoint_path, 
    prepare_onnx_path, cleanup_old_checkpoints, get_checkpoint_metadata,
    ProgressTracker, export_model_to_onnx, format_file_size
)


class ModelCheckpointManager(ICheckpointService):
    """Unified checkpoint manager dengan progress tracking dan UI integration"""
    
    def __init__(
        self,
        model_manager = None,
        checkpoint_dir: str = "runs/train/checkpoints",
        max_checkpoints: int = 5,
        logger = None,
        progress_callback: Optional[Callable] = None
    ):
        self.model_manager = model_manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logger or get_logger()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self._progress_tracker = ProgressTracker(progress_callback)
        self.logger.info(f"✨ Unified CheckpointManager initialized (dir: {checkpoint_dir})")
    
    # Properties untuk compatibility
    @property
    def progress_callback(self): return self._progress_tracker._callback
    
    @progress_callback.setter
    def progress_callback(self, callback): self._progress_tracker.set_callback(callback)
    
    @property
    def operation_progress(self): return self._progress_tracker.progress
    
    def save_checkpoint(
        self,
        model: Optional[torch.nn.Module] = None,
        path: str = "checkpoint.pt",
        optimizer: Optional[torch.optim.Optimizer] = None,
        epoch: int = 0,
        metadata: Optional[Dict[str, Any]] = None,
        is_best: bool = False
    ) -> str:
        """Save checkpoint dengan progress tracking"""
        try:
            self._update_progress(0, 4, "🔄 Memulai penyimpanan checkpoint...")
            
            # Use model from manager if not provided
            model = model or (self.model_manager.model if self.model_manager else None)
            if not model: raise ModelCheckpointError("❌ Model tidak tersedia untuk disimpan")
            
            # Prepare checkpoint path dan update progress
            checkpoint_path = self._prepare_checkpoint_path(path)
            self._update_progress(1, 4, f"📁 Menyiapkan path: {checkpoint_path.name}")
            
            # Prepare data, save checkpoint, dan update progress
            checkpoint_data = self._prepare_checkpoint_data(model, optimizer, epoch, metadata)
            self._update_progress(2, 4, "📦 Menyiapkan data checkpoint...")
            self._atomic_save(checkpoint_data, checkpoint_path)
            self._update_progress(3, 4, "💾 Menyimpan checkpoint...")
            
            # Handle best model jika diperlukan
            if is_best:
                best_path = checkpoint_path.parent / 'best.pt'
                shutil.copy2(checkpoint_path, best_path)
                self.logger.info(f"🏆 Best checkpoint: {best_path}")
            
            # Cleanup dan finalisasi
            self._cleanup_old_checkpoints()
            self._update_progress(4, 4, f"✅ Checkpoint tersimpan: {checkpoint_path.name}")
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Save checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress(4, 4, error_msg)
            raise ModelCheckpointError(error_msg)
    
    def load_checkpoint(
        self,
        path: str,
        model: Optional[torch.nn.Module] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        map_location: Optional[str] = None
    ) -> Union[Dict[str, Any], torch.nn.Module]:
        """Load checkpoint dengan progress tracking"""
        try:
            # Update progress dan resolve path
            self._update_progress(0, 3, f"🔍 Mencari checkpoint: {path}")
            checkpoint_path = self._resolve_checkpoint_path(path)
            self._update_progress(1, 3, f"📂 Loading checkpoint: {checkpoint_path.name}")
            
            # Load checkpoint data
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Apply states dan return
            if model is not None and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self._update_progress(2, 3, f"✅ Model berhasil dimuat dari {checkpoint_path.name}")
                self.logger.info(f"✅ Model berhasil dimuat dari: {checkpoint_path}")
                return model
            
            # Return data jika tidak ada model
            self._update_progress(2, 3, f"✅ Checkpoint berhasil dimuat: {checkpoint_path.name}")
            self.logger.info(f"✅ Checkpoint berhasil dimuat: {checkpoint_path}")
            return checkpoint
            
        except Exception as e:
            error_msg = f"❌ Load checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress(3, 3, error_msg)
            raise ModelCheckpointError(error_msg)
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """List semua checkpoint dengan metadata"""
        checkpoints = []
        
        # Filter dan proses checkpoint files
        valid_files = [f for f in self.checkpoint_dir.glob('*.pt') 
                      if not (f.name.startswith('.') or f.name.startswith('tmp'))]
        
        for checkpoint_file in valid_files:
            try:
                # Load metadata dan tambahkan file info
                metadata = get_checkpoint_metadata(checkpoint_file, self.logger)
                stat = checkpoint_file.stat()
                
                metadata.update({
                    'path': str(checkpoint_file), 'filename': checkpoint_file.name,
                    'size': stat.st_size, 'mtime': stat.st_mtime,
                    'size_formatted': format_file_size(stat.st_size)
                })
                checkpoints.append(metadata)
            except Exception as e:
                self.logger.warning(f"Couldn't read checkpoint {checkpoint_file}: {e}")
        
        # Sort berdasarkan attribute yang diminta dengan dict lookup
        sort_keys = {
            'time': lambda x: x.get('mtime', 0),
            'epoch': lambda x: x.get('epoch', 0),
            'size': lambda x: x.get('size', 0),
            'name': lambda x: x.get('filename', '').lower()
        }
        
        reverse = sort_by != 'name'  # Reverse untuk semua kecuali name
        checkpoints.sort(key=sort_keys.get(sort_by, sort_keys['time']), reverse=reverse)
        return checkpoints
    
    def export_to_onnx(
        self,
        model: Optional[torch.nn.Module] = None,
        output_path: Optional[str] = None,
        input_shape: List[int] = [1, 3, 640, 640],
        opset_version: int = 12,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """Export model ke format ONNX dengan progress tracking"""
        try:
            # Get model dan validasi
            model = model or (self.model_manager.model if self.model_manager else None)
            if not model: raise ModelCheckpointError("❌ Model tidak tersedia untuk diexport")
            
            # Prepare output path dengan handling untuk berbagai kasus
            if output_path is None:
                output_path = self._prepare_onnx_path()
            else:
                output_path = Path(output_path)
                if not output_path.is_absolute(): output_path = self.checkpoint_dir / output_path
                if not str(output_path).endswith('.onnx'): output_path = Path(f"{output_path}.onnx")
                output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export model ke ONNX dengan utility function
            return export_model_to_onnx(
                model=model, output_path=output_path, input_shape=input_shape,
                opset_version=opset_version, dynamic_axes=dynamic_axes,
                progress_tracker=self._progress_tracker, logger=self.logger
            )
            
        except Exception as e:
            error_msg = f"❌ Export ONNX error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress(3, 3, error_msg)
            raise ModelCheckpointError(error_msg)
    
    def get_checkpoint_info(self, path: str) -> Dict[str, Any]:
        """Get detailed info tentang checkpoint"""
        try:
            checkpoint_path = self._resolve_checkpoint_path(path)
            metadata = get_checkpoint_metadata(checkpoint_path, self.logger)
            stat = checkpoint_path.stat()
            
            # Add file info dengan one-liner update
            metadata.update({
                'path': str(checkpoint_path), 'filename': checkpoint_path.name,
                'size': stat.st_size, 'mtime': stat.st_mtime,
                'size_formatted': self._format_file_size(stat.st_size)
            })
            return metadata
        except Exception as e:
            error_msg = f"Couldn't get checkpoint info: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    # One-liner getters untuk checkpoint paths
    get_latest_checkpoint = lambda self: self.list_checkpoints(sort_by='time')[0]['path'] if self.list_checkpoints(sort_by='time') else None
    get_best_checkpoint = lambda self: str(self.checkpoint_dir / 'best.pt') if (self.checkpoint_dir / 'best.pt').exists() else None
    
    def cleanup_checkpoints(self, keep_best: bool = True, keep_latest: int = 3) -> Dict[str, Any]:
        """Cleanup old checkpoints dengan progress tracking"""
        # Filter checkpoints dan early return jika tidak perlu cleanup
        checkpoints = [p for p in self.checkpoint_dir.glob('*.pt') if p.name != 'best.pt' or not keep_best]
        if len(checkpoints) <= keep_latest:
            return {'removed': 0, 'kept': len(checkpoints), 'errors': []}
        
        # Sort dan split untuk keep/remove
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_keep, to_remove = checkpoints[:keep_latest], checkpoints[keep_latest:]
        result = {'removed': 0, 'kept': len(to_keep), 'errors': []}
        
        # Hapus checkpoint lama dengan progress tracking
        for i, ckpt in enumerate(to_remove):
            try:
                self._update_progress(i, len(to_remove), f"🗑️ Removing: {ckpt.name}")
                os.remove(ckpt)
                result['removed'] += 1
            except Exception as e:
                error = f"Error removing {ckpt.name}: {str(e)}"
                result['errors'].append(error)
                self.logger.warning(f"⚠️ {error}")
        
        # Update progress final
        self._update_progress(len(to_remove), len(to_remove), 
                      f"🧹 Cleanup complete: kept {result['kept']}, removed {result['removed']}")
        return result
    
    # Helper methods - one-liners menggunakan utilitas
    _prepare_checkpoint_path = lambda self, path: resolve_checkpoint_path(path, self.checkpoint_dir)
    _prepare_checkpoint_data = lambda self, model, optimizer, epoch, metadata: prepare_checkpoint_data(model, optimizer, epoch, metadata, self.model_manager)
    _atomic_save = lambda self, checkpoint_data, checkpoint_path: atomic_save(checkpoint_data, checkpoint_path)
    _resolve_checkpoint_path = lambda self, path: resolve_checkpoint_path(path, self.checkpoint_dir)
    _prepare_onnx_path = lambda self: prepare_onnx_path(self.checkpoint_dir, getattr(self.model_manager, 'model_type', 'model') if self.model_manager else 'model')
    _cleanup_old_checkpoints = lambda self: cleanup_old_checkpoints(self.checkpoint_dir, self.max_checkpoints, self.logger)
    _update_progress = lambda self, current, total, message: self._progress_tracker.update(current, total, message)
    _format_file_size = lambda self, size_bytes: format_file_size(size_bytes)
    
    # One-liner utilities dan helpers
    set_progress_callback = lambda self, callback: setattr(self, 'progress_callback', callback)
    get_checkpoint_count = lambda self: len(list(self.checkpoint_dir.glob('*.pt')))
    get_total_size = lambda self: sum(p.stat().st_size for p in self.checkpoint_dir.glob('*.pt'))
    has_checkpoints = lambda self: self.get_checkpoint_count() > 0
    get_checkpoint_dir = lambda self: str(self.checkpoint_dir)
    get_average_size = lambda self: self.get_total_size() / self.get_checkpoint_count() if self.get_checkpoint_count() > 0 else 0
    get_formatted_total_size = lambda self: self._format_file_size(self.get_total_size())