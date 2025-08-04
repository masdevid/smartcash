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
from smartcash.model.training.utils.progress_tracker import TrainingProgressTracker

class CheckpointManager:
    """💾 Manager untuk checkpoint operations dengan automatic naming"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger("model.checkpoint")
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', 'data/checkpoints'))
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
        self.auto_cleanup = checkpoint_config.get('auto_cleanup', True)
        
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"💾 CheckpointManager initialized | Dir: {self.save_dir}")
    
    def save_checkpoint(self, model: torch.nn.Module, metrics: Optional[Dict] = None, 
                       checkpoint_name: Optional[str] = None, **kwargs) -> str:
        """💾 Save model checkpoint dengan automatic naming"""
        
        try:
            # Generate checkpoint filename
            if checkpoint_name is None:
                checkpoint_name = self._generate_checkpoint_name(metrics, **kwargs)
            
            checkpoint_path = self.save_dir / checkpoint_name
            
            # CRITICAL FIX: Get model configuration from proper source
            model_config = self._get_model_config_for_checkpoint(model, **kwargs)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
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
            if 'phase' in kwargs:
                checkpoint_data['phase'] = kwargs['phase']
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Log save info
            is_best = kwargs.get('is_best', False)
            log_message = f"✅ {'Best' if is_best else 'Last'} checkpoint saved successfully at {checkpoint_path}"
            self.logger.info(log_message)
            
            # Create phase backup immediately if this is a best model
            if is_best:
                phase_num = kwargs.get('phase', 1)  # Default to phase 1 if not specified
                backup_path = self._create_phase_backup_immediate(str(checkpoint_path), phase_num)
                if backup_path:
                    self.logger.info(f"📦 Phase {phase_num} backup created: {Path(backup_path).name}")
            
            # Cleanup old checkpoints
            if self.auto_cleanup:
                self._cleanup_old_checkpoints()
            
            return str(checkpoint_path)
            
        except Exception as e:
            error_msg = f"❌ Checkpoint save failed: {str(e)}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def load_checkpoint(self, model: torch.nn.Module, checkpoint_path: Optional[str] = None, 
                       strict: bool = True, **kwargs) -> Dict[str, Any]:
        """📂 Load model dari checkpoint"""
        
        try:
            # Find checkpoint jika tidak specified
            if checkpoint_path is None:
                checkpoint_path = self._find_best_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("❌ No checkpoint found")
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint data
            from smartcash.common.checkpoint_utils import safe_load_checkpoint
            checkpoint_data = safe_load_checkpoint(str(checkpoint_path))
            
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
            raise RuntimeError(error_msg)
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """📋 List available checkpoints dengan metadata"""
        
        try:
            checkpoints = []
            
            for checkpoint_file in self.save_dir.glob("*.pt"):
                try:
                    # Load metadata dari checkpoint with safe loading
                    from smartcash.common.checkpoint_utils import safe_load_checkpoint
                    checkpoint_data = safe_load_checkpoint(str(checkpoint_file))
                    
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
        """📝 Generate checkpoint filename using naming convention
        
        - For best checkpoints: best_{backbone}_{two_phase|single_phase}_{single|multi}_{frozen|unfrozen}_{pretrained:if true}_{yyyymmdd}.pt
        - For regular checkpoints: last_{backbone}_{two_phase|single_phase}_{single|multi}_{frozen|unfrozen}_{pretrained:if true}.pt
        """
        
        try:
            # Check if this is a best checkpoint
            is_best = kwargs.get('is_best', False)
            
            # Get model configuration for naming (both best and regular checkpoints)
            model_config = self.config.get('model', {})
            training_config = self.config.get('training', {})
            
            # Extract naming components
            backbone = model_config.get('backbone', 'unknown')
            
            # Determine training mode (phase mode)
            training_mode = training_config.get('training_mode', 'two_phase')
            if training_mode not in ['single_phase', 'two_phase']:
                # Fallback: determine from phases configuration
                total_phases = len([k for k in self.config.get('training_phases', {}) if 'phase_' in k])
                training_mode = 'single_phase' if total_phases == 1 else 'two_phase'
            
            # Determine layer mode
            layer_mode = model_config.get('layer_mode', 'multi')
            if 'detection_layers' in model_config:
                num_layers = len(model_config['detection_layers'])
                layer_mode = 'single' if num_layers == 1 else 'multi'
            
            # Determine freeze status
            freeze_backbone = model_config.get('freeze_backbone', False)
            freeze_status = 'frozen' if freeze_backbone else 'unfrozen'
            
            # Check if pretrained is enabled
            pretrained = model_config.get('pretrained', False)
            pretrained_suffix = 'pretrained' if pretrained else ''
            
            # For regular checkpoints, build name with backbone info (no date)
            if not is_best:
                name_parts = [
                    'last',
                    backbone,
                    training_mode,
                    layer_mode,
                    freeze_status
                ]
                
                # Add pretrained suffix only if True
                if pretrained_suffix:
                    name_parts.append(pretrained_suffix)
                
                filename = '_'.join(name_parts) + '.pt'
                self.logger.debug(f"📝 Generated regular checkpoint name: {filename}")
                return filename
            
            # For best checkpoints, add date
            # Get current date
            current_date = datetime.now()
            date_str = current_date.strftime('%Y%m%d')
            
            # Build checkpoint name parts for best checkpoints
            name_parts = [
                'best',
                backbone,
                training_mode,
                layer_mode,
                freeze_status
            ]
            
            # Add pretrained suffix only if True
            if pretrained_suffix:
                name_parts.append(pretrained_suffix)
            
            name_parts.append(date_str)
            filename = '_'.join(name_parts) + '.pt'
            
            self.logger.debug(f"📝 Generated checkpoint name: {filename}")
            return filename
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating checkpoint name: {e}, using fallback")
            # Fallback to simple naming
            model_config = self.config.get('model', {})
            backbone = model_config.get('backbone', 'unknown')
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"best_{backbone}_{date_str}.pt"
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
        
        # Keep best_*.pt files and last_*.pt files (updated pattern)
        regular_checkpoints = [cp for cp in checkpoints 
                              if not cp.name.startswith('best_') and not cp.name.startswith('last_')]
        
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
    
    def _get_model_config_for_checkpoint(self, model: torch.nn.Module, **kwargs) -> Dict[str, Any]:
        """🔧 Get model configuration for checkpoint saving"""
        try:
            # Method 1: Try to get from model_api if passed in kwargs (MOST ACCURATE)
            if 'model_api' in kwargs:
                model_api = kwargs['model_api']
                if hasattr(model_api, 'config'):
                    api_config = getattr(model_api, 'config', {})
                    api_model_config = api_config.get('model', {})
                    if api_model_config:
                        self.logger.debug("✅ Got model config from model_api")
                        return api_model_config
            
            # Method 2: Check if model has a get_model_config method
            if hasattr(model, 'get_model_config'):
                try:
                    model_config = model.get_model_config()
                    if model_config:
                        self.logger.debug("✅ Got model config from model.get_model_config()")
                        return model_config
                except Exception as e:
                    self.logger.debug(f"Failed to get config from model method: {e}")
            
            # Method 3: Check if model has a config attribute
            if hasattr(model, 'config') and model.config:
                self.logger.debug("✅ Got model config from model.config attribute")
                return model.config
            
            # Method 4: Use the CheckpointManager's own config as fallback
            model_config = self.config.get('model', {})
            if model_config:
                self.logger.debug("✅ Got model config from CheckpointManager config")
                return model_config
            
            # Method 5: Try to extract basic config from model structure
            if hasattr(model, 'multi_layer_config') and model.multi_layer_config:
                self.logger.debug("✅ Got model config from model.multi_layer_config")
                return model.multi_layer_config
            
            self.logger.warning("⚠️ No model configuration found, saving empty config")
            return {}
            
        except Exception as e:
            self.logger.error(f"❌ Error getting model config for checkpoint: {e}")
            # Return configuration from CheckpointManager as fallback
            return self.config.get('model', {})

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
    
    def _create_phase_backup_immediate(self, checkpoint_path: str, phase_num: int) -> Optional[str]:
        """📦 Create phase-specific backup immediately when best model is saved"""
        if not checkpoint_path:
            return None
            
        try:
            import shutil
            
            best_checkpoint_path = Path(checkpoint_path)
            
            # Generate phase-specific backup filename
            # Convert: best_cspdarknet_two_phase_multi_unfrozen_pretrained_20250804.pt
            # To: best_cspdarknet_two_phase_multi_unfrozen_pretrained_20250804_phase1.pt
            backup_checkpoint_path = best_checkpoint_path.parent / (
                best_checkpoint_path.stem + f'_phase{phase_num}' + best_checkpoint_path.suffix
            )
            
            # Remove existing backup if it exists (avoid accumulation)
            if backup_checkpoint_path.exists():
                backup_checkpoint_path.unlink()
                self.logger.debug(f"🗑️ Removed existing phase {phase_num} backup: {backup_checkpoint_path.name}")
            
            # Copy the best checkpoint to phase backup
            shutil.copy2(best_checkpoint_path, backup_checkpoint_path)
            
            self.logger.debug(f"✅ Created Phase {phase_num} backup: {backup_checkpoint_path.name}")
            return str(backup_checkpoint_path)
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to create Phase {phase_num} backup: {e}")
            return None


# Factory function
def create_checkpoint_manager(config: Dict[str, Any]) -> CheckpointManager:
    """🏭 Factory untuk membuat CheckpointManager"""
    return CheckpointManager(config)