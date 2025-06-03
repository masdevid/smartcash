"""
File: smartcash/model/manager_checkpoint.py
Deskripsi: Unified checkpoint manager dengan UI integration dan one-liner style
"""

import os
import torch
import json
import time
import shutil
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError
from smartcash.model.service.progress_tracker import ProgressTracker

class ModelCheckpointManager(ICheckpointService):
    """Unified checkpoint manager dengan UI integration dan one-liner style"""
    
    def __init__(self, model_manager=None, checkpoint_dir: str = "runs/train/checkpoints", 
                 max_checkpoints: int = 5, logger=None, progress_callback: Optional[Callable] = None):
        self.model_manager = model_manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logger or get_logger()
        self.progress_tracker = ProgressTracker(progress_callback)
        
        # Setup checkpoint directory dan initialize properties
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.last_checkpoint_path = None
        
        self.logger.info(f"✨ Unified CheckpointManager initialized (dir: {checkpoint_dir})")
    
    def save_checkpoint(self, model: Optional[torch.nn.Module] = None, path: str = "checkpoint.pt",
                       optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0,
                       metadata: Optional[Dict[str, Any]] = None, is_best: bool = False) -> str:
        """Save checkpoint dengan progress tracking dan atomic operations"""
        try:
            self.progress_tracker.update(0, 4, "🔄 Memulai penyimpanan checkpoint...")
            
            # Use model from manager jika tidak disediakan
            model = model or (self.model_manager.model if self.model_manager else None)
            model or self._raise_error("❌ Model tidak tersedia untuk disimpan")
            
            # Prepare paths dan data
            checkpoint_path = self._prepare_checkpoint_path(path)
            self.progress_tracker.update(1, 4, f"📁 Menyiapkan path: {checkpoint_path.name}")
            
            checkpoint_data = self._prepare_checkpoint_data(model, optimizer, epoch, metadata)
            self.progress_tracker.update(2, 4, "📦 Menyiapkan data checkpoint...")
            
            # Atomic save operation
            self._atomic_save(checkpoint_data, checkpoint_path)
            self.progress_tracker.update(3, 4, "💾 Menyimpan checkpoint...")
            
            # Handle best model dan cleanup
            is_best and self._handle_best_checkpoint(checkpoint_path)
            self._cleanup_old_checkpoints()
            self.last_checkpoint_path = str(checkpoint_path)
            
            self.progress_tracker.update(4, 4, f"✅ Checkpoint tersimpan: {checkpoint_path.name}")
            self.logger.info(f"✅ Checkpoint tersimpan: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Save checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def load_checkpoint(self, path: str, model: Optional[torch.nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       map_location: Optional[str] = None) -> Union[Dict[str, Any], torch.nn.Module]:
        """Load checkpoint dengan progress tracking"""
        try:
            self.progress_tracker.update(0, 3, f"🔍 Mencari checkpoint: {path}")
            checkpoint_path = self._resolve_checkpoint_path(path)
            self.progress_tracker.update(1, 3, f"📂 Loading: {checkpoint_path.name}")
            
            # Load checkpoint data
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Apply states jika model/optimizer disediakan
            if model and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer and 'optimizer_state_dict' in checkpoint and optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.progress_tracker.update(3, 3, f"✅ Model berhasil dimuat dari {checkpoint_path.name}")
                self.logger.info(f"✅ Model berhasil dimuat: {checkpoint_path}")
                return model
            
            # Return raw data jika tidak ada model
            self.progress_tracker.update(3, 3, f"✅ Checkpoint data dimuat: {checkpoint_path.name}")
            return checkpoint
            
        except Exception as e:
            error_msg = f"❌ Load checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """List checkpoints dengan metadata extraction"""
        checkpoints = []
        
        # Process valid checkpoint files
        valid_files = [f for f in self.checkpoint_dir.glob('*.pt') if not (f.name.startswith('.') or f.name.startswith('tmp'))]
        
        for checkpoint_file in valid_files:
            try:
                metadata = self._get_checkpoint_metadata(checkpoint_file)
                stat = checkpoint_file.stat()
                
                metadata.update({
                    'path': str(checkpoint_file), 'filename': checkpoint_file.name,
                    'size': stat.st_size, 'mtime': stat.st_mtime,
                    'size_formatted': self._format_file_size(stat.st_size)
                })
                checkpoints.append(metadata)
            except Exception as e:
                self.logger.warning(f"⚠️ Couldn't read checkpoint {checkpoint_file}: {e}")
        
        # Sort berdasarkan kriteria dengan one-liner
        sort_keys = {'time': lambda x: x.get('mtime', 0), 'epoch': lambda x: x.get('epoch', 0),
                    'size': lambda x: x.get('size', 0), 'name': lambda x: x.get('filename', '').lower()}
        
        checkpoints.sort(key=sort_keys.get(sort_by, sort_keys['time']), reverse=(sort_by != 'name'))
        return checkpoints
    
    def export_to_onnx(self, model: Optional[torch.nn.Module] = None, output_path: Optional[str] = None,
                      input_shape: List[int] = [1, 3, 640, 640], opset_version: int = 12,
                      dynamic_axes: Optional[Dict] = None) -> str:
        """Export model ke ONNX format dengan progress tracking"""
        try:
            # Get model dan prepare path
            model = model or (self.model_manager.model if self.model_manager else None)
            model or self._raise_error("❌ Model tidak tersedia untuk export")
            
            output_path = self._prepare_onnx_path(output_path)
            
            # Export dengan progress tracking
            self.progress_tracker.update(0, 3, f"🔄 Memulai export ONNX: {output_path.name}")
            
            # Create dummy input dan export
            dummy_input = torch.zeros(*input_shape)
            model.eval()
            
            self.progress_tracker.update(1, 3, "📦 Mengeksport model ke ONNX...")
            torch.onnx.export(
                model, dummy_input, str(output_path),
                export_params=True, opset_version=opset_version,
                do_constant_folding=True, input_names=['input'],
                output_names=['output'], dynamic_axes=dynamic_axes or {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            )
            
            self.progress_tracker.update(3, 3, f"✅ ONNX export selesai: {output_path.name}")
            self.logger.info(f"✅ Model berhasil dieksport ke ONNX: {output_path}")
            return str(output_path)
            
        except Exception as e:
            error_msg = f"❌ Export ONNX error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "export")
            raise ModelCheckpointError(error_msg)
    
    def cleanup_checkpoints(self, keep_best: bool = True, keep_latest: int = 3) -> Dict[str, Any]:
        """Cleanup old checkpoints dengan progress tracking"""
        checkpoints = [p for p in self.checkpoint_dir.glob('*.pt') if p.name != 'best.pt' or not keep_best]
        
        if len(checkpoints) <= keep_latest:
            return {'removed': 0, 'kept': len(checkpoints), 'errors': []}
        
        # Sort dan determine checkpoints to remove
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_keep, to_remove = checkpoints[:keep_latest], checkpoints[keep_latest:]
        result = {'removed': 0, 'kept': len(to_keep), 'errors': []}
        
        # Remove old checkpoints dengan progress tracking
        for i, ckpt in enumerate(to_remove):
            try:
                self.progress_tracker.update(i, len(to_remove), f"🗑️ Removing: {ckpt.name}")
                os.remove(ckpt)
                result['removed'] += 1
            except Exception as e:
                error = f"Error removing {ckpt.name}: {str(e)}"
                result['errors'].append(error)
                self.logger.warning(f"⚠️ {error}")
        
        self.progress_tracker.update(len(to_remove), len(to_remove), f"🧹 Cleanup complete: kept {result['kept']}, removed {result['removed']}")
        return result
    
    def get_checkpoint_info(self, path: str) -> Dict[str, Any]:
        """Get detailed checkpoint information"""
        try:
            checkpoint_path = self._resolve_checkpoint_path(path)
            metadata = self._get_checkpoint_metadata(checkpoint_path)
            stat = checkpoint_path.stat()
            
            metadata.update({
                'path': str(checkpoint_path), 'filename': checkpoint_path.name,
                'size': stat.st_size, 'mtime': stat.st_mtime,
                'size_formatted': self._format_file_size(stat.st_size)
            })
            return metadata
        except Exception as e:
            error_msg = f"❌ Couldn't get checkpoint info: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    # Helper methods dengan one-liner implementations
    def _prepare_checkpoint_path(self, path: str) -> Path:
        """Prepare dan resolve checkpoint path"""
        checkpoint_path = Path(path) if Path(path).is_absolute() else self.checkpoint_dir / path
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
    
    def _prepare_checkpoint_data(self, model, optimizer, epoch, metadata) -> Dict[str, Any]:
        """Prepare checkpoint data dictionary"""
        checkpoint_data = {'model_state_dict': model.state_dict(), 'epoch': epoch, 'timestamp': time.time(), 'metadata': metadata or {}}
        optimizer and checkpoint_data.update({'optimizer_state_dict': optimizer.state_dict()})
        self.model_manager and hasattr(self.model_manager, 'get_config') and checkpoint_data.update({'model_config': self.model_manager.get_config()})
        return checkpoint_data
    
    def _atomic_save(self, checkpoint_data: Dict[str, Any], checkpoint_path: Path):
        """Atomic save operation untuk prevent corruption"""
        temp_path = str(checkpoint_path) + ".tmp"
        torch.save(checkpoint_data, temp_path)
        checkpoint_path.exists() and os.remove(checkpoint_path)
        os.rename(temp_path, checkpoint_path)
    
    def _resolve_checkpoint_path(self, path: str) -> Path:
        """Resolve checkpoint path dengan validation"""
        checkpoint_path = Path(path) if Path(path).is_absolute() else self.checkpoint_dir / path
        checkpoint_path.exists() or self._raise_error(f"❌ Checkpoint tidak ditemukan: {checkpoint_path}")
        return checkpoint_path
    
    def _prepare_onnx_path(self, output_path: Optional[str]) -> Path:
        """Prepare ONNX output path"""
        if output_path is None:
            model_name = getattr(self.model_manager, 'model_type', 'model') if self.model_manager else 'model'
            output_path = self.checkpoint_dir / f"{model_name}_{int(time.time())}.onnx"
        else:
            output_path = Path(output_path)
            output_path.is_absolute() or (output_path := self.checkpoint_dir / output_path)
            str(output_path).endswith('.onnx') or (output_path := Path(f"{output_path}.onnx"))
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _handle_best_checkpoint(self, checkpoint_path: Path):
        """Handle best checkpoint saving"""
        best_path = self.checkpoint_dir / 'best.pt'
        shutil.copy2(checkpoint_path, best_path)
        self.best_checkpoint_path = str(best_path)
        self.logger.info(f"🏆 Best checkpoint: {best_path}")
    
    def _cleanup_old_checkpoints(self):
        """Cleanup old checkpoints berdasarkan max_checkpoints"""
        if self.max_checkpoints <= 0: return
        
        checkpoint_files = [p for p in self.checkpoint_dir.glob('*.pt') if p.name not in ['best.pt', 'last.pt']]
        
        if len(checkpoint_files) <= self.max_checkpoints: return
        
        # Remove oldest checkpoints
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        [self._safe_remove(ckpt) for ckpt in checkpoint_files[self.max_checkpoints:]]
    
    def _get_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract metadata dari checkpoint file"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            return checkpoint.get("metadata", {}) if isinstance(checkpoint, dict) else {}
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting metadata from {checkpoint_path.name}: {str(e)}")
            return {'error': str(e)}
    
    def _safe_remove(self, path: Path):
        """Safely remove file dengan error handling"""
        try:
            os.remove(path)
            self.logger.debug(f"🗑️ Removed old checkpoint: {path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error removing {path.name}: {str(e)}")
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size to human readable"""
        units = ['B', 'KB', 'MB', 'GB']
        size = float(size_bytes)
        for unit in units:
            if size < 1024.0: return f"{size:.1f} {unit}"
            size /= 1024.0
        return f"{size:.1f} TB"
    
    def _raise_error(self, message: str): raise ModelCheckpointError(message)
    
    # One-liner utilities dan properties
    get_latest_checkpoint = lambda self: (self.list_checkpoints(sort_by='time')[0]['path'] if self.list_checkpoints(sort_by='time') else None)
    get_best_checkpoint = lambda self: self.best_checkpoint_path or (str(self.checkpoint_dir / 'best.pt') if (self.checkpoint_dir / 'best.pt').exists() else None)
    get_last_checkpoint = lambda self: self.last_checkpoint_path or (str(self.checkpoint_dir / 'last.pt') if (self.checkpoint_dir / 'last.pt').exists() else None)
    get_checkpoint_dir = lambda self: str(self.checkpoint_dir)
    get_checkpoint_count = lambda self: len(list(self.checkpoint_dir.glob('*.pt')))
    get_total_size = lambda self: sum(p.stat().st_size for p in self.checkpoint_dir.glob('*.pt'))
    has_checkpoints = lambda self: self.get_checkpoint_count() > 0
    get_formatted_total_size = lambda self: self._format_file_size(self.get_total_size())
    set_progress_callback = lambda self, callback: setattr(self.progress_tracker, '_callback', callback)
    
    # Properties untuk compatibility
    @property
    def progress_callback(self): return self.progress_tracker._callback
    
    @progress_callback.setter
    def progress_callback(self, callback): self.progress_tracker._callback = callback
    
    @property
    def operation_progress(self): return self.progress_tracker.progress