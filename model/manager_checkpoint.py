"""
File: smartcash/model/manager_checkpoint.py
Deskripsi: Unified checkpoint manager dengan progress tracking dan UI integration
"""

import os
import torch
import shutil
import tempfile
import json
import time
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError


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
        self.progress_callback = progress_callback
        
        # Create directories
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Progress tracking state
        self._operation_progress = {'current': 0, 'total': 0, 'message': ''}
        
        self.logger.info(f"✨ Unified CheckpointManager initialized (dir: {checkpoint_dir})")
    
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
            if model is None and self.model_manager:
                model = self.model_manager.model
            
            if model is None:
                raise ModelCheckpointError("❌ Model tidak tersedia untuk disimpan")
            
            checkpoint_path = self._prepare_checkpoint_path(path)
            self._update_progress(1, 4, f"📁 Menyiapkan path: {checkpoint_path.name}")
            
            # Prepare checkpoint data
            checkpoint_data = self._prepare_checkpoint_data(model, optimizer, epoch, metadata)
            self._update_progress(2, 4, "📦 Menyiapkan data checkpoint...")
            
            # Atomic save using temporary file
            self._atomic_save(checkpoint_data, checkpoint_path)
            self._update_progress(3, 4, "💾 Menyimpan checkpoint...")
            
            # Handle best model
            if is_best:
                best_path = checkpoint_path.parent / 'best.pt'
                shutil.copy2(checkpoint_path, best_path)
                self.logger.info(f"🏆 Best checkpoint: {best_path}")
            
            # Cleanup old checkpoints
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
            self._update_progress(0, 3, f"🔍 Mencari checkpoint: {path}")
            
            checkpoint_path = self._resolve_checkpoint_path(path)
            self._update_progress(1, 3, f"📂 Loading checkpoint: {checkpoint_path.name}")
            
            # Load checkpoint data
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            
            # Use model from manager if not provided
            if model is None and self.model_manager:
                model = self.model_manager.model
            
            if model is not None:
                model.load_state_dict(checkpoint['model_state_dict'])
                
                if optimizer is not None and 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self._update_progress(2, 3, "🔧 Loading state ke model...")
                
                metadata = checkpoint.get('metadata', {})
                epoch = checkpoint.get('epoch', 0)
                
                self.logger.info(f"📂 Checkpoint loaded: epoch {epoch}, metadata: {metadata}")
                self._update_progress(3, 3, f"✅ Checkpoint loaded: epoch {epoch}")
                
                return model
            else:
                self._update_progress(3, 3, "✅ Checkpoint data loaded")
                return checkpoint
                
        except Exception as e:
            error_msg = f"❌ Load checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress(3, 3, error_msg)
            raise ModelCheckpointError(error_msg)
    
    def list_checkpoints(self, sort_by: str = 'time') -> List[Dict[str, Any]]:
        """List checkpoints dengan metadata"""
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        result = []
        
        for i, ckpt in enumerate(checkpoints):
            self._update_progress(i, len(checkpoints), f"📋 Scanning: {ckpt.name}")
            
            try:
                checkpoint = torch.load(ckpt, map_location='cpu')
                
                info = {
                    'path': str(ckpt),
                    'name': ckpt.name,
                    'timestamp': checkpoint.get('timestamp', ckpt.stat().st_mtime),
                    'epoch': checkpoint.get('epoch', 0),
                    'metadata': checkpoint.get('metadata', {}),
                    'is_best': (ckpt.name == 'best.pt'),
                    'size': self._format_file_size(ckpt.stat().st_size)
                }
                
                result.append(info)
                
            except Exception as e:
                self.logger.warning(f"⚠️ Error reading {ckpt}: {str(e)}")
        
        # Sort results
        sort_keys = {'time': 'timestamp', 'name': 'name', 'epoch': 'epoch'}
        if sort_by in sort_keys:
            result.sort(key=lambda x: x[sort_keys[sort_by]], reverse=(sort_by != 'name'))
        
        self._update_progress(len(checkpoints), len(checkpoints), f"✅ Found {len(result)} checkpoints")
        return result
    
    def export_to_onnx(
        self,
        model: Optional[torch.nn.Module] = None,
        output_path: str = "model.onnx",
        input_shape: List[int] = [1, 3, 640, 640],
        opset_version: int = 12,
        dynamic_axes: Optional[Dict] = None
    ) -> str:
        """Export model ke ONNX dengan progress tracking"""
        try:
            self._update_progress(0, 4, "🔄 Memulai export ONNX...")
            
            # Use model from manager if not provided
            if model is None and self.model_manager:
                model = self.model_manager.model
            
            if model is None:
                raise ModelCheckpointError("❌ Model tidak tersedia untuk export")
            
            onnx_path = self._prepare_onnx_path(output_path)
            self._update_progress(1, 4, f"📁 Preparing ONNX path: {onnx_path.name}")
            
            # Prepare model
            model.eval()
            dummy_input = torch.randn(input_shape, requires_grad=True)
            self._update_progress(2, 4, "🔧 Preparing model untuk export...")
            
            # Set dynamic axes if not provided
            dynamic_axes = dynamic_axes or {'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
            
            # Export to ONNX
            torch.onnx.export(
                model, dummy_input, onnx_path,
                export_params=True, opset_version=opset_version,
                do_constant_folding=True,
                input_names=['input'], output_names=['output'],
                dynamic_axes=dynamic_axes
            )
            
            self._update_progress(3, 4, "🚀 Exporting ke ONNX...")
            self._update_progress(4, 4, f"✅ ONNX exported: {onnx_path.name}")
            
            self.logger.info(f"📤 ONNX export success: {onnx_path}")
            return str(onnx_path)
            
        except Exception as e:
            error_msg = f"❌ ONNX export error: {str(e)}"
            self.logger.error(error_msg)
            self._update_progress(4, 4, error_msg)
            raise ModelCheckpointError(error_msg)
    
    def get_latest_checkpoint(self) -> Optional[str]:
        """Get latest checkpoint path"""
        checkpoints = list(self.checkpoint_dir.glob('*.pt'))
        return str(max(checkpoints, key=lambda p: p.stat().st_mtime)) if checkpoints else None
    
    def get_best_checkpoint(self) -> Optional[str]:
        """Get best checkpoint path"""
        best_path = self.checkpoint_dir / 'best.pt'
        return str(best_path) if best_path.exists() else None
    
    def cleanup_checkpoints(self, keep_best: bool = True, keep_latest: int = 3) -> Dict[str, Any]:
        """Cleanup old checkpoints dengan progress tracking"""
        checkpoints = [p for p in self.checkpoint_dir.glob('*.pt') if p.name != 'best.pt' or not keep_best]
        
        if len(checkpoints) <= keep_latest:
            return {'removed': 0, 'kept': len(checkpoints), 'errors': []}
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        to_remove = checkpoints[:-keep_latest]
        
        result = {'removed': 0, 'kept': len(checkpoints) - len(to_remove), 'errors': []}
        
        for i, ckpt in enumerate(to_remove):
            self._update_progress(i, len(to_remove), f"🗑️ Removing: {ckpt.name}")
            
            try:
                os.remove(ckpt)
                result['removed'] += 1
                self.logger.debug(f"🧹 Removed old checkpoint: {ckpt}")
            except Exception as e:
                result['errors'].append(f"Error removing {ckpt}: {str(e)}")
        
        self._update_progress(len(to_remove), len(to_remove), f"✅ Cleanup complete: {result['removed']} removed")
        return result
    
    # Helper methods
    def _prepare_checkpoint_path(self, path: str) -> Path:
        """Prepare checkpoint path"""
        checkpoint_path = Path(path)
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        return checkpoint_path
    
    def _prepare_checkpoint_data(self, model, optimizer, epoch, metadata) -> Dict[str, Any]:
        """Prepare checkpoint data"""
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        # Add model manager info
        if self.model_manager:
            checkpoint_data['model_type'] = getattr(self.model_manager, 'model_type', 'unknown')
            checkpoint_data['model_config'] = getattr(self.model_manager, 'config', {})
        
        # Add optimizer state if provided
        if optimizer is not None:
            checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
        
        return checkpoint_data
    
    def _atomic_save(self, checkpoint_data: Dict, checkpoint_path: Path):
        """Atomic save using temporary file"""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
            torch.save(checkpoint_data, tmp_file.name)
            shutil.move(tmp_file.name, checkpoint_path)
    
    def _resolve_checkpoint_path(self, path: str) -> Path:
        """Resolve checkpoint path dengan fallback"""
        checkpoint_path = Path(path)
        
        if not checkpoint_path.is_absolute():
            checkpoint_path = self.checkpoint_dir / checkpoint_path
        
        if not checkpoint_path.exists():
            # Try with .pt extension
            if not str(checkpoint_path).endswith('.pt'):
                checkpoint_path = Path(f"{checkpoint_path}.pt")
            
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"Checkpoint not found: {path}")
        
        return checkpoint_path
    
    def _prepare_onnx_path(self, output_path: str) -> Path:
        """Prepare ONNX export path"""
        onnx_path = Path(output_path)
        
        if not onnx_path.is_absolute():
            onnx_path = self.checkpoint_dir / onnx_path
        
        onnx_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not str(onnx_path).endswith('.onnx'):
            onnx_path = Path(f"{onnx_path}.onnx")
        
        return onnx_path
    
    def _cleanup_old_checkpoints(self):
        """Cleanup old checkpoints based on max_checkpoints"""
        checkpoints = [p for p in self.checkpoint_dir.glob('*.pt') if p.name != 'best.pt']
        
        if len(checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by modification time (oldest first)
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        
        for ckpt in checkpoints[:-self.max_checkpoints]:
            try:
                os.remove(ckpt)
                self.logger.debug(f"🧹 Auto-removed old checkpoint: {ckpt}")
            except Exception as e:
                self.logger.warning(f"⚠️ Error removing old checkpoint: {str(e)}")
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress tracking"""
        self._operation_progress = {'current': current, 'total': total, 'message': message}
        
        if self.progress_callback:
            try:
                self.progress_callback(current, total, message)
            except Exception:
                pass  # Silent fail untuk callback
    
    def _format_file_size(self, size_bytes: int) -> str:
        """Format file size"""
        for unit in ['B', 'KB', 'MB', 'GB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} TB"
    
    # Properties untuk compatibility
    @property
    def operation_progress(self) -> Dict[str, Any]:
        return self._operation_progress
    
    # One-liner utilities
    set_progress_callback = lambda self, callback: setattr(self, 'progress_callback', callback)
    get_checkpoint_count = lambda self: len(list(self.checkpoint_dir.glob('*.pt')))
    get_total_size = lambda self: sum(p.stat().st_size for p in self.checkpoint_dir.glob('*.pt'))