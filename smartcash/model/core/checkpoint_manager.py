"""
File: smartcash/model/core/checkpoint_manager.py
Deskripsi: Manager untuk checkpoint operations dengan progress tracking
"""

import os
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.common.logger import get_logger
from smartcash.model.utils.progress_bridge import ModelProgressBridge

class CheckpointManager:
    """💾 Manager untuk checkpoint operations dengan automatic naming"""
    
    def __init__(self, config: Dict[str, Any], progress_bridge: ModelProgressBridge):
        self.config = config
        self.progress_bridge = progress_bridge
        self.logger = get_logger("model.checkpoint")
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', '/data/checkpoints'))
        self.format_template = checkpoint_config.get('format', 'best_{model_name}_{backbone}_{date:%Y%m%d}.pt')
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
        self.auto_cleanup = checkpoint_config.get('auto_cleanup', True)
        
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"💾 CheckpointManager initialized | Dir: {self.save_dir}")
    
    def save_checkpoint(self, model: torch.nn.Module, metrics: Optional[Dict] = None, 
                       checkpoint_name: Optional[str] = None, **kwargs) -> str:
        """💾 Save model checkpoint dengan automatic naming"""
        
        try:
            self.progress_bridge.start_operation("Saving Checkpoint", 4)
            
            # Generate checkpoint filename
            self.progress_bridge.update(1, "📝 Generating checkpoint name...")
            if checkpoint_name is None:
                checkpoint_name = self._generate_checkpoint_name(metrics, **kwargs)
            
            checkpoint_path = self.save_dir / checkpoint_name
            
            # Prepare checkpoint data
            self.progress_bridge.update(2, "📦 Preparing checkpoint data...")
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'model_config': getattr(model, 'config', {}),
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat(),
                'torch_version': torch.__version__,
                'model_info': self._get_model_info(model)
            }
            
            # Add optimizer dan scheduler jika ada
            if 'optimizer' in kwargs:
                checkpoint_data['optimizer_state_dict'] = kwargs['optimizer'].state_dict()
            if 'scheduler' in kwargs:
                checkpoint_data['scheduler_state_dict'] = kwargs['scheduler'].state_dict()
            if 'epoch' in kwargs:
                checkpoint_data['epoch'] = kwargs['epoch']
            
            # Save checkpoint
            self.progress_bridge.update(3, f"💾 Saving to {checkpoint_name}...")
            torch.save(checkpoint_data, checkpoint_path)
            
            # Cleanup old checkpoints
            if self.auto_cleanup:
                self.progress_bridge.update(4, "🧹 Cleaning up old checkpoints...")
                self._cleanup_old_checkpoints()
            
            self.progress_bridge.complete(4, f"✅ Checkpoint saved: {checkpoint_name}")
            
            # Log save info
            file_size = checkpoint_path.stat().st_size / (1024 * 1024)  # MB
            self.logger.info(f"💾 Checkpoint saved: {checkpoint_name} ({file_size:.1f}MB)")
            
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Checkpoint save failed: {str(e)}"
            self.logger.error(error_msg)
            self.progress_bridge.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: Optional[str] = None, 
                       strict: bool = True, **kwargs) -> Dict[str, Any]:
        """📂 Load model dari checkpoint"""
        
        try:
            self.progress_bridge.start_operation("Loading Checkpoint", 3)
            
            # Find checkpoint jika tidak specified
            if checkpoint_path is None:
                self.progress_bridge.update(1, "🔍 Finding best checkpoint...")
                checkpoint_path = self._find_best_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("❌ No checkpoint found")
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint data
            self.progress_bridge.update(2, f"📂 Loading {checkpoint_path.name}...")
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu')
            
            # Load model state
            if 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
            else:
                # Fallback untuk checkpoint format lama
                model.load_state_dict(checkpoint_data, strict=strict)
            
            # Load optimizer dan scheduler jika ada
            if 'optimizer' in kwargs and 'optimizer_state_dict' in checkpoint_data:
                kwargs['optimizer'].load_state_dict(checkpoint_data['optimizer_state_dict'])
            
            if 'scheduler' in kwargs and 'scheduler_state_dict' in checkpoint_data:
                kwargs['scheduler'].load_state_dict(checkpoint_data['scheduler_state_dict'])
            
            self.progress_bridge.complete(3, f"✅ Checkpoint loaded: {checkpoint_path.name}")
            
            # Return checkpoint info
            return {
                'checkpoint_path': str(checkpoint_path),
                'metrics': checkpoint_data.get('metrics', {}),
                'epoch': checkpoint_data.get('epoch', 0),
                'timestamp': checkpoint_data.get('timestamp', ''),
                'model_info': checkpoint_data.get('model_info', {}),
                'torch_version': checkpoint_data.get('torch_version', ''),
                'loaded_components': {
                    'model': True,
                    'optimizer': 'optimizer' in kwargs and 'optimizer_state_dict' in checkpoint_data,
                    'scheduler': 'scheduler' in kwargs and 'scheduler_state_dict' in checkpoint_data
                }
            }
            
        except Exception as e:
            error_msg = f"❌ Checkpoint load failed: {str(e)}"
            self.logger.error(error_msg)
            self.progress_bridge.error(error_msg)
            raise RuntimeError(error_msg)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """📋 List available checkpoints dengan metadata"""
        
        try:
            checkpoints = []
            
            for checkpoint_file in self.save_dir.glob("*.pt"):
                try:
                    # Load metadata dari checkpoint
                    checkpoint_data = torch.load(checkpoint_file, map_location='cpu')
                    
                    # Extract info
                    file_stat = checkpoint_file.stat()
                    checkpoint_info = {
                        'filename': checkpoint_file.name,
                        'path': str(checkpoint_file),
                        'size_mb': file_stat.st_size / (1024 * 1024),
                        'modified': datetime.fromtimestamp(file_stat.st_mtime).isoformat(),
                        'metrics': checkpoint_data.get('metrics', {}),
                        'epoch': checkpoint_data.get('epoch', 0),
                        'timestamp': checkpoint_data.get('timestamp', ''),
                        'model_info': checkpoint_data.get('model_info', {}),
                        'torch_version': checkpoint_data.get('torch_version', ''),
                    }
                    
                    checkpoints.append(checkpoint_info)
                    
                except Exception as e:
                    self.logger.warning(f"⚠️ Error reading checkpoint {checkpoint_file.name}: {str(e)}")
                    continue
            
            # Sort by modification time (newest first)
            checkpoints.sort(key=lambda x: x['modified'], reverse=True)
            
            self.logger.info(f"📋 Found {len(checkpoints)} checkpoints")
            return checkpoints
            
        except Exception as e:
            self.logger.error(f"❌ Error listing checkpoints: {str(e)}")
            return []
    
    def _generate_checkpoint_name(self, metrics: Optional[Dict] = None, **kwargs) -> str:
        """📝 Generate checkpoint filename berdasarkan template"""
        
        # Get model info
        model_config = self.config.get('model', {})
        
        # Template variables
        current_date = datetime.now()
        variables = {
            'model_name': model_config.get('model_name', 'smartcash'),
            'backbone': model_config.get('backbone', 'unknown'),
            'layer_mode': model_config.get('layer_mode', 'single'),
            'date': current_date,  # Pass datetime object for strftime formatting
            'time': current_date.strftime('%H%M'),
            'epoch': kwargs.get('epoch', 0)
        }
        
        # Add metric info jika ada
        if metrics:
            best_metric = min(metrics.values()) if metrics else 0
            variables['metric'] = f"{best_metric:.4f}".replace('.', '')
        
        # Format filename
        try:
            filename = self.format_template.format(**variables)
        except KeyError as e:
            self.logger.warning(f"⚠️ Template variable missing: {e}, using default")
            date_str = current_date.strftime('%Y%m%d')
            filename = f"best_{variables['model_name']}_{variables['backbone']}_{date_str}.pt"
        
        return filename
    
    def _find_best_checkpoint(self) -> Optional[str]:
        """🔍 Find best checkpoint berdasarkan naming pattern"""
        
        checkpoints = list(self.save_dir.glob("*.pt"))
        if not checkpoints:
            return None
        
        # Prioritas: best_*.pt > terbaru berdasarkan waktu
        best_checkpoints = [cp for cp in checkpoints if cp.name.startswith('best_')]
        if best_checkpoints:
            # Sort by modification time
            best_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(best_checkpoints[0])
        
        # Fallback: checkpoint terbaru
        checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoints[0])
    
    def _cleanup_old_checkpoints(self) -> None:
        """🧹 Cleanup old checkpoints, keep max_checkpoints"""
        
        if self.max_checkpoints <= 0:
            return
        
        checkpoints = list(self.save_dir.glob("*.pt"))
        
        # Keep best_*.pt files
        regular_checkpoints = [cp for cp in checkpoints if not cp.name.startswith('best_')]
        
        if len(regular_checkpoints) > self.max_checkpoints:
            # Sort by modification time
            regular_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Remove old checkpoints
            for old_checkpoint in regular_checkpoints[self.max_checkpoints:]:
                try:
                    old_checkpoint.unlink()
                    self.logger.debug(f"🗑️ Removed old checkpoint: {old_checkpoint.name}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to remove {old_checkpoint.name}: {str(e)}")
    
    def _get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """ℹ️ Extract model information untuk checkpoint metadata"""
        
        try:
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            model_info = {
                'total_parameters': total_params,
                'trainable_parameters': trainable_params,
                'model_size_mb': total_params * 4 / (1024 * 1024),  # Estimate
                'architecture': model.__class__.__name__
            }
            
            # Add model config jika tersedia
            if hasattr(model, 'config'):
                model_info['model_config'] = model.config
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error extracting model info: {str(e)}")
            return {'error': str(e)}
    
    def delete_checkpoint(self, checkpoint_path: str) -> bool:
        """🗑️ Delete specific checkpoint"""
        try:
            path = Path(checkpoint_path)
            if path.exists():
                path.unlink()
                self.logger.info(f"🗑️ Deleted checkpoint: {path.name}")
                return True
            return False
        except Exception as e:
            self.logger.error(f"❌ Failed to delete checkpoint: {str(e)}")
            return False


# Factory function
def create_checkpoint_manager(config: Dict[str, Any], progress_bridge: ModelProgressBridge) -> CheckpointManager:
    """🏭 Factory untuk membuat CheckpointManager"""
    return CheckpointManager(config, progress_bridge)