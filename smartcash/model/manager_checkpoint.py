"""
File: smartcash/model/manager_checkpoint.py
Deskripsi: Checkpoint manager dengan format penamaan {backbone}_{model_type}_{layer_mode} dan one-liner style
"""

import os
import torch
import time
import shutil
from typing import Dict, List, Optional, Union, Any, Callable
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.common.interfaces.checkpoint_interface import ICheckpointService
from smartcash.common.exceptions import ModelCheckpointError
from smartcash.model.service.progress_tracker import ProgressTracker

class ModelCheckpointManager(ICheckpointService):
    """Checkpoint manager dengan format penamaan {backbone}_{model_type}_{layer_mode} dan one-liner style"""
    
    def __init__(self, model_manager=None, checkpoint_dir: str = "runs/train/checkpoints", 
                 max_checkpoints: int = 5, logger=None, progress_callback: Optional[Callable] = None):
        self.model_manager = model_manager
        self.checkpoint_dir = Path(checkpoint_dir)
        self.max_checkpoints = max_checkpoints
        self.logger = logger or get_logger()
        self.progress_tracker = ProgressTracker(progress_callback)
        
        # Setup checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_checkpoint_path = self.last_checkpoint_path = None
        
        self.logger.info(f"✨ CheckpointManager initialized dengan format {'{backbone}_{model_type}_{layer_mode}'}")
    
    def save_checkpoint(self, model: Optional[torch.nn.Module] = None, path: str = "checkpoint.pt",
                       optimizer: Optional[torch.optim.Optimizer] = None, epoch: int = 0,
                       metadata: Optional[Dict[str, Any]] = None, is_best: bool = False) -> str:
        """Save checkpoint dengan naming format baru"""
        try:
            self.progress_tracker.update(0, 4, "🔄 Memulai penyimpanan checkpoint...")
            
            model = model or (self.model_manager.model if self.model_manager else None)
            model or self._raise_error("❌ Model tidak tersedia untuk disimpan")
            
            checkpoint_path = self._generate_checkpoint_path(path, epoch, is_best)
            self.progress_tracker.update(1, 4, f"📁 Generated path: {checkpoint_path.name}")
            
            checkpoint_data = self._prepare_checkpoint_data(model, optimizer, epoch, metadata)
            self.progress_tracker.update(2, 4, "📦 Menyiapkan checkpoint data...")
            
            self._atomic_save(checkpoint_data, checkpoint_path)
            self.progress_tracker.update(3, 4, "💾 Menyimpan checkpoint...")
            
            is_best and self._handle_best_checkpoint(checkpoint_path)
            self._cleanup_old_checkpoints()
            self.last_checkpoint_path = str(checkpoint_path)
            
            self.progress_tracker.update(4, 4, f"✅ Checkpoint tersimpan: {checkpoint_path.name}")
            self.logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Save checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def load_checkpoint(self, path: str, model: Optional[torch.nn.Module] = None,
                       optimizer: Optional[torch.optim.Optimizer] = None,
                       map_location: Optional[str] = None) -> Union[Dict[str, Any], torch.nn.Module]:
        """Load checkpoint dengan path resolution"""
        try:
            self.progress_tracker.update(0, 3, f"🔍 Resolving checkpoint: {path}")
            
            checkpoint_path = self._resolve_checkpoint_path(path)
            self.progress_tracker.update(1, 3, f"📂 Loading: {checkpoint_path.name}")
            
            checkpoint = torch.load(checkpoint_path, map_location=map_location)
            self._log_metadata(checkpoint)
            
            if model and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer and 'optimizer_state_dict' in checkpoint and optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                
                self.progress_tracker.update(3, 3, f"✅ Model loaded: {checkpoint_path.name}")
                self.logger.info(f"✅ Model berhasil dimuat: {checkpoint_path}")
                return model
            
            self.progress_tracker.update(3, 3, f"✅ Checkpoint data loaded: {checkpoint_path.name}")
            return checkpoint
            
        except Exception as e:
            error_msg = f"❌ Load checkpoint error: {str(e)}"
            self.logger.error(error_msg)
            self.progress_tracker.error(error_msg, "checkpoint")
            raise ModelCheckpointError(error_msg)
    
    def list_checkpoints(self, sort_by: str = 'time', filter_by_current_model: bool = True) -> List[Dict[str, Any]]:
        """List checkpoints dengan filtering dan metadata"""
        current_signature = self._get_model_signature() if filter_by_current_model else None
        
        valid_files = [
            f for f in self.checkpoint_dir.glob('*.pt') 
            if not (f.name.startswith('.') or f.name.startswith('tmp')) and
               (not current_signature or f.name.startswith(current_signature))
        ]
        
        checkpoints = []
        for checkpoint_file in valid_files:
            try:
                metadata = self._get_checkpoint_metadata(checkpoint_file)
                stat = checkpoint_file.stat()
                filename_info = self._parse_filename(checkpoint_file.name)
                
                checkpoints.append({
                    'path': str(checkpoint_file), 'filename': checkpoint_file.name,
                    'size': stat.st_size, 'mtime': stat.st_mtime,
                    'size_formatted': self._format_file_size(stat.st_size),
                    **metadata, **filename_info
                })
            except Exception as e:
                self.logger.warning(f"⚠️ Couldn't read checkpoint {checkpoint_file}: {e}")
        
        sort_keys = {
            'time': lambda x: x.get('mtime', 0), 'epoch': lambda x: x.get('epoch', 0),
            'size': lambda x: x.get('size', 0), 'name': lambda x: x.get('filename', '').lower(),
            'backbone': lambda x: x.get('backbone', ''), 'model_type': lambda x: x.get('model_type', '')
        }
        
        checkpoints.sort(key=sort_keys.get(sort_by, sort_keys['time']), reverse=(sort_by != 'name'))
        return checkpoints
    
    def export_to_onnx(self, model: Optional[torch.nn.Module] = None, output_path: Optional[str] = None,
                      input_shape: List[int] = [1, 3, 640, 640], opset_version: int = 12,
                      dynamic_axes: Optional[Dict] = None) -> str:
        """Export model ke ONNX dengan naming format"""
        try:
            model = model or (self.model_manager.model if self.model_manager else None)
            model or self._raise_error("❌ Model tidak tersedia untuk export")
            
            output_path = self._prepare_onnx_path(output_path)
            self.progress_tracker.update(0, 3, f"🔄 Memulai ONNX export: {output_path.name}")
            
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
        """Cleanup dengan model signature awareness"""
        model_signature = self._get_model_signature()
        checkpoints = [
            p for p in self.checkpoint_dir.glob('*.pt') 
            if p.name.startswith(model_signature) and 
               (not keep_best or not p.name.endswith('_best.pt'))
        ]
        
        if len(checkpoints) <= keep_latest:
            return {'removed': 0, 'kept': len(checkpoints), 'errors': [], 'model_signature': model_signature}
        
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        to_keep, to_remove = checkpoints[:keep_latest], checkpoints[keep_latest:]
        result = {'removed': 0, 'kept': len(to_keep), 'errors': [], 'model_signature': model_signature}
        
        for i, ckpt in enumerate(to_remove):
            try:
                self.progress_tracker.update(i, len(to_remove), f"🗑️ Removing: {ckpt.name}")
                os.remove(ckpt)
                result['removed'] += 1
            except Exception as e:
                error = f"Error removing {ckpt.name}: {str(e)}"
                result['errors'].append(error)
                self.logger.warning(f"⚠️ {error}")
        
        self.progress_tracker.update(len(to_remove), len(to_remove), 
                                   f"🧹 Cleanup complete: kept {result['kept']}, removed {result['removed']} for {model_signature}")
        return result
    
    def get_checkpoint_info(self, path: str) -> Dict[str, Any]:
        """Get detailed checkpoint information"""
        try:
            checkpoint_path = self._resolve_checkpoint_path(path)
            metadata = self._get_checkpoint_metadata(checkpoint_path)
            stat = checkpoint_path.stat()
            filename_info = self._parse_filename(checkpoint_path.name)
            
            return {
                'path': str(checkpoint_path), 'filename': checkpoint_path.name,
                'size': stat.st_size, 'mtime': stat.st_mtime,
                'size_formatted': self._format_file_size(stat.st_size),
                **metadata, **filename_info
            }
        except Exception as e:
            error_msg = f"❌ Couldn't get checkpoint info: {str(e)}"
            self.logger.error(error_msg)
            raise ModelCheckpointError(error_msg)
    
    # Private helper methods
    def _generate_checkpoint_path(self, path: str, epoch: int, is_best: bool) -> Path:
        """Generate checkpoint path dengan format {backbone}_{model_type}_{layer_mode}"""
        backbone = self._get_model_info('backbone', 'efficientnet_b4')
        model_type = self._get_model_info('model_type', 'efficient_basic')
        layer_mode = self._get_model_info('layer_mode', 'single')
        
        backbone_clean = backbone.replace('efficientnet_', 'effnet_').replace('cspdarknet_', 'csp_')
        model_type_clean = model_type.replace('efficient_', 'eff_')
        base_name = f"{backbone_clean}_{model_type_clean}_{layer_mode}"
        
        if is_best:
            filename = f"{base_name}_best.pt"
        elif path == "checkpoint.pt" or path.endswith('.pt'):
            filename = f"{base_name}_epoch_{epoch:03d}.pt"
        else:
            path_stem = Path(path).stem
            filename = f"{base_name}_{path_stem}_epoch_{epoch:03d}.pt" if not path_stem.startswith(base_name) else f"{path_stem}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        self.logger.debug(f"🏷️ Generated checkpoint name: {filename}")
        return checkpoint_path
    
    def _get_model_info(self, key: str, default: str) -> str:
        """Get model info dengan fallback chain"""
        return (
            getattr(self.model_manager, f'get_{key}', lambda: None)() if self.model_manager and hasattr(self.model_manager, f'get_{key}') else
            getattr(self.model_manager, key, None) if self.model_manager and hasattr(self.model_manager, key) else
            self.model_manager.config.get(key, default) if self.model_manager and hasattr(self.model_manager, 'config') else
            default
        )
    
    def _get_model_signature(self) -> str:
        """Get current model signature untuk filtering checkpoints"""
        backbone = self._get_model_info('backbone', 'efficientnet_b4')
        model_type = self._get_model_info('model_type', 'efficient_basic')
        layer_mode = self._get_model_info('layer_mode', 'single')
        
        backbone_clean = backbone.replace('efficientnet_', 'effnet_').replace('cspdarknet_', 'csp_')
        model_type_clean = model_type.replace('efficient_', 'eff_')
        
        return f"{backbone_clean}_{model_type_clean}_{layer_mode}"
    
    def _prepare_checkpoint_data(self, model, optimizer, epoch, metadata) -> Dict[str, Any]:
        """Prepare checkpoint data dengan metadata"""
        checkpoint_data = {
            'model_state_dict': model.state_dict(),
            'epoch': epoch,
            'timestamp': time.time(),
            'metadata': metadata or {}
        }
        
        optimizer and checkpoint_data.update({'optimizer_state_dict': optimizer.state_dict()})
        
        if self.model_manager:
            model_metadata = {
                'backbone': self._get_model_info('backbone', 'unknown'),
                'model_type': self._get_model_info('model_type', 'unknown'),
                'layer_mode': self._get_model_info('layer_mode', 'unknown'),
                'detection_layers': self._get_model_info('detection_layers', []),
                'num_classes': self._get_model_info('num_classes', 0),
                'model_config': getattr(self.model_manager, 'get_config', lambda: {})()
            }
            checkpoint_data['model_metadata'] = model_metadata
        
        return checkpoint_data
    
    def _atomic_save(self, checkpoint_data: Dict[str, Any], checkpoint_path: Path):
        """Atomic save operation untuk prevent corruption"""
        temp_path = str(checkpoint_path) + ".tmp"
        torch.save(checkpoint_data, temp_path)
        checkpoint_path.exists() and os.remove(checkpoint_path)
        os.rename(temp_path, checkpoint_path)
    
    def _handle_best_checkpoint(self, checkpoint_path: Path):
        """Handle best checkpoint dengan naming format"""
        best_filename = f"{self._get_model_signature()}_best.pt"
        best_path = self.checkpoint_dir / best_filename
        
        shutil.copy2(checkpoint_path, best_path)
        self.best_checkpoint_path = str(best_path)
        self.logger.info(f"🏆 Best checkpoint: {best_path}")
    
    def _cleanup_old_checkpoints(self):
        """Cleanup old checkpoints berdasarkan max_checkpoints"""
        if self.max_checkpoints <= 0: return
        
        model_signature = self._get_model_signature()
        checkpoint_files = [
            p for p in self.checkpoint_dir.glob('*.pt') 
            if p.name.startswith(model_signature) and 
               not any(suffix in p.name for suffix in ['_best.pt', '_last.pt'])
        ]
        
        if len(checkpoint_files) <= self.max_checkpoints: return
        
        checkpoint_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        [self._safe_remove(ckpt) for ckpt in checkpoint_files[self.max_checkpoints:]]
    
    def _resolve_checkpoint_path(self, path: str) -> Path:
        """Path resolution dengan format-aware search"""
        if Path(path).is_absolute() and Path(path).exists():
            return Path(path)
        
        checkpoint_path = self.checkpoint_dir / path
        if checkpoint_path.exists():
            return checkpoint_path
        
        model_signature = self._get_model_signature()
        search_patterns = [f"{model_signature}_best.pt", f"{model_signature}_epoch_*.pt", f"{model_signature}*.pt"]
        
        for pattern in search_patterns:
            matches = list(self.checkpoint_dir.glob(pattern))
            if matches:
                latest_match = max(matches, key=lambda p: p.stat().st_mtime)
                self.logger.debug(f"🔍 Found checkpoint: {latest_match.name}")
                return latest_match
        
        if not checkpoint_path.exists():
            self._raise_error(f"❌ Checkpoint tidak ditemukan: {path}")
        
        return checkpoint_path
    
    def _log_metadata(self, checkpoint: Dict[str, Any]) -> None:
        """Log metadata dengan format"""
        model_meta = checkpoint.get('model_metadata', {})
        model_meta and self.logger.info(
            f"📋 Checkpoint info: "
            f"{model_meta.get('backbone', 'unknown')} | "
            f"{model_meta.get('model_type', 'unknown')} | "
            f"{model_meta.get('layer_mode', 'unknown')} | "
            f"{len(model_meta.get('detection_layers', []))} layers"
        )
    
    def _parse_filename(self, filename: str) -> Dict[str, Any]:
        """Parse filename untuk extract info"""
        try:
            name_parts = filename.replace('.pt', '').split('_')
            info = {}
            
            if len(name_parts) >= 1:
                backbone_part = name_parts[0]
                if backbone_part.startswith('effnet'):
                    info['backbone'] = backbone_part.replace('effnet', 'efficientnet')
                elif backbone_part.startswith('csp'):
                    info['backbone'] = backbone_part.replace('csp', 'cspdarknet')
                else:
                    info['backbone'] = backbone_part
            
            if len(name_parts) >= 2:
                model_type_part = name_parts[1]
                if model_type_part.startswith('eff'):
                    info['model_type'] = model_type_part.replace('eff', 'efficient')
                else:
                    info['model_type'] = model_type_part
            
            if len(name_parts) >= 3 and name_parts[2] in ['single', 'multilayer']:
                info['layer_mode'] = name_parts[2]
            
            if 'best' in filename:
                info['checkpoint_type'] = 'best'
            elif 'epoch' in filename:
                info['checkpoint_type'] = 'epoch'
                epoch_parts = [part for part in name_parts if part.startswith('epoch') or part.isdigit()]
                epoch_parts and info.update({'epoch': int(epoch_parts[-1]) if epoch_parts[-1].isdigit() else 0})
            else:
                info['checkpoint_type'] = 'custom'
            
            return info
            
        except Exception:
            return {'backbone': 'unknown', 'model_type': 'unknown', 'layer_mode': 'unknown', 'checkpoint_type': 'unknown'}
    
    def _get_checkpoint_metadata(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Extract metadata dari checkpoint file"""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            if isinstance(checkpoint, dict):
                standard_meta = checkpoint.get("metadata", {})
                model_meta = checkpoint.get("model_metadata", {})
                return {**standard_meta, **model_meta}
            return {}
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting metadata from {checkpoint_path.name}: {str(e)}")
            return {'error': str(e)}
    
    def _prepare_onnx_path(self, output_path: Optional[str]) -> Path:
        """Prepare ONNX output path dengan format naming"""
        if output_path is None:
            model_signature = self._get_model_signature()
            timestamp = int(time.time())
            output_path = self.checkpoint_dir / f"{model_signature}_export_{timestamp}.onnx"
        else:
            output_path = Path(output_path)
            if not output_path.is_absolute():
                output_path = self.checkpoint_dir / output_path
            if not str(output_path).endswith('.onnx'):
                model_signature = self._get_model_signature()
                output_path = Path(f"{output_path}_{model_signature}.onnx")
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        return output_path
    
    def _safe_remove(self, path: Path):
        """Safely remove file dengan error handling"""
        try:
            os.remove(path)
            self.logger.debug(f"🗑️ Removed checkpoint: {path.name}")
        except Exception as e:
            self.logger.warning(f"⚠️ Error removing checkpoint {path.name}: {str(e)}")
    
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
    get_best_checkpoint = lambda self: (next((ckpt['path'] for ckpt in self.list_checkpoints() if ckpt.get('checkpoint_type') == 'best'), None))
    get_model_signature = lambda self: self._get_model_signature()
    get_checkpoint_count = lambda self: len(self.list_checkpoints())
    has_checkpoints = lambda self: self.get_checkpoint_count() > 0
    get_checkpoints_by_signature = lambda self, signature: [ckpt for ckpt in self.list_checkpoints() if ckpt.get('filename', '').startswith(signature)]
    
    # Properties untuk compatibility
    @property
    def current_model_signature(self): return self._get_model_signature()
    
    @property
    def checkpoint_count(self): return self.get_checkpoint_count()