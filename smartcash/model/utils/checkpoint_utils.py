"""
File: smartcash/model/utils/checkpoint_utils.py
Deskripsi: Utilitas untuk operasi checkpoint model
"""

import os
import torch
import shutil
import tempfile
import time
from pathlib import Path
from typing import Dict, Any, Optional, Union

from smartcash.common.logger import get_logger
from smartcash.common.exceptions import ModelCheckpointError

# Format file size dengan one-liner
format_file_size = lambda size_bytes: next((f"{size_bytes/1024**i:.1f} {unit}" for i, unit in enumerate(['B', 'KB', 'MB', 'GB', 'TB']) if size_bytes < 1024**(i+1)), f"{size_bytes/1024**4:.1f} TB")

def prepare_checkpoint_data(model, optimizer=None, epoch=0, metadata=None, model_manager=None) -> Dict[str, Any]:
    """Prepare checkpoint data untuk disimpan"""
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'timestamp': time.time(),
        'metadata': metadata or {}
    }
    
    # Add model manager info
    if model_manager:
        checkpoint_data['model_type'] = getattr(model_manager, 'model_type', 'unknown')
        checkpoint_data['model_config'] = getattr(model_manager, 'config', {})
    
    # Add optimizer state if provided
    if optimizer is not None:
        checkpoint_data['optimizer_state_dict'] = optimizer.state_dict()
    
    return checkpoint_data

def atomic_save(checkpoint_data: Dict, checkpoint_path: Path) -> None:
    """Atomic save using temporary file untuk mencegah corrupt checkpoint"""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pt') as tmp_file:
        torch.save(checkpoint_data, tmp_file.name)
        shutil.move(tmp_file.name, checkpoint_path)

def resolve_checkpoint_path(path: str, checkpoint_dir: Path) -> Path:
    """Resolve checkpoint path dengan fallback"""
    # Konversi ke Path dan handle relative path
    checkpoint_path = Path(path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_dir / checkpoint_path
    
    # Coba dengan .pt extension jika file tidak ditemukan
    if not checkpoint_path.exists() and not str(checkpoint_path).endswith('.pt'):
        checkpoint_path = Path(f"{checkpoint_path}.pt")
    
    # Raise error jika masih tidak ditemukan
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    return checkpoint_path

def prepare_onnx_path(output_path: str, checkpoint_dir: Path) -> Path:
    """Prepare ONNX export path"""
    # Konversi ke Path dan handle relative path
    onnx_path = Path(output_path) if output_path else checkpoint_dir / 'model.onnx'
    if not onnx_path.is_absolute():
        onnx_path = checkpoint_dir / onnx_path
    
    # Pastikan direktori parent ada
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Tambahkan ekstensi .onnx jika belum ada
    if not str(onnx_path).endswith('.onnx'):
        onnx_path = Path(f"{onnx_path}.onnx")
    
    return onnx_path

def cleanup_old_checkpoints(checkpoint_dir: Path, max_checkpoints: int, logger=None) -> None:
    """Cleanup old checkpoints based on max_checkpoints"""
    logger = logger or get_logger()
    
    # Filter checkpoints (exclude best.pt)
    checkpoints = [p for p in checkpoint_dir.glob('*.pt') if p.name != 'best.pt']
    if len(checkpoints) <= max_checkpoints: return
    
    # Sort by modification time (oldest first) dan hapus yang melebihi max_checkpoints
    checkpoints.sort(key=lambda p: p.stat().st_mtime)
    for ckpt in checkpoints[:-max_checkpoints]:
        try:
            os.remove(ckpt)
            logger.debug(f"🧹 Auto-removed old checkpoint: {ckpt}")
        except Exception as e:
            logger.warning(f"⚠️ Error removing old checkpoint: {str(e)}")

def get_checkpoint_metadata(checkpoint_path: Path) -> Dict[str, Any]:
    """Extract metadata dari checkpoint file"""
    try:
        # Load checkpoint dan extract metadata
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        metadata = checkpoint.get('metadata', {})
        
        # Tambahkan informasi file dan checkpoint
        metadata.update({
            'file_size': format_file_size(checkpoint_path.stat().st_size),
            'timestamp': checkpoint.get('timestamp', 0),
            'epoch': checkpoint.get('epoch', 0),
            'path': str(checkpoint_path)
        })
        return metadata
    except Exception as e:
        raise ModelCheckpointError(f"Error reading checkpoint metadata: {str(e)}")
