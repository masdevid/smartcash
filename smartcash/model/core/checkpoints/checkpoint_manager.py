"""
File: smartcash/model/core/checkpoint_manager.py
Deskripsi: Manager untuk checkpoint operations dengan progress tracking
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

from smartcash.common.logger import get_logger
from .best_metrics_manager import BestMetricsManager

class CheckpointManager:
    """💾 Manager untuk checkpoint operations dengan automatic naming"""
    
    def __init__(self, config: Dict[str, Any], is_resuming: bool = False):
        self.config = config
        self.logger = get_logger("model.checkpoint")
        
        # Checkpoint configuration
        checkpoint_config = config.get('checkpoint', {})
        self.save_dir = Path(checkpoint_config.get('save_dir', 'data/checkpoints'))
        self.max_checkpoints = checkpoint_config.get('max_checkpoints', 5)
        self.auto_cleanup = checkpoint_config.get('auto_cleanup', True)
        
        # Ensure directory exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize best metrics manager
        self.best_metrics_manager = BestMetricsManager(self.save_dir, is_resuming=is_resuming)
        
        self.logger.info(f"💾 CheckpointManager initialized | Dir: {self.save_dir}")
    
    def save_checkpoint(self, model: torch.nn.Module, metrics: Optional[Dict] = None, 
                       checkpoint_name: Optional[str] = None, **kwargs) -> str:
        """💾 Save model checkpoint with best metrics preservation"""
        
        try:
            # Set current phase and load previous best metrics if needed
            phase_num = kwargs.get('phase', 1)
            self.best_metrics_manager.set_current_phase(phase_num)
            
            # Load previous best metrics for this phase if not already loaded
            # BUT skip loading if this is a fresh phase start
            should_load_previous = (
                phase_num not in self.best_metrics_manager.phase_best_metrics and
                getattr(self.best_metrics_manager, '_fresh_phase_start', None) != phase_num
            )
            
            if should_load_previous:
                self.best_metrics_manager.load_previous_best_metrics(phase_num)
            else:
                self.logger.info(f"🆕 Skipping previous metrics load for Phase {phase_num} (fresh start)")
            
            # Determine if this should be saved as best
            is_best = kwargs.get('is_best', False)
            if not is_best and metrics:
                # Check if current metrics are better than previous best
                is_best = self.best_metrics_manager.should_save_as_best(
                    metrics, 
                    comparison_metric='val_accuracy',
                    comparison_mode='max'
                )
                kwargs['is_best'] = is_best
            
            # Generate checkpoint filename
            if checkpoint_name is None:
                checkpoint_name = self._generate_checkpoint_name(metrics, **kwargs)
            
            checkpoint_path = self.save_dir / checkpoint_name
            
            # CRITICAL FIX: Get model configuration from proper source
            model_config = self._get_model_config_for_checkpoint(model, **kwargs)
            
            # Get best metrics manager metadata
            epoch = kwargs.get('epoch', 0)
            phase = kwargs.get('phase', 1)
            manager_metadata = self.best_metrics_manager.get_checkpoint_save_metadata(epoch, phase)
            
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'metrics': metrics or {},
                'timestamp': datetime.now().isoformat(),
                'torch_version': torch.__version__,
                'model_info': self._get_model_info(model)
            }
            
            # Add best metrics manager metadata
            checkpoint_data.update(manager_metadata)
            
            # Add optimizer and scheduler if available
            if 'optimizer' in kwargs:
                checkpoint_data['optimizer_state_dict'] = kwargs['optimizer'].state_dict()
                self.logger.debug("💾 Saved optimizer state")
            if 'scheduler' in kwargs:
                checkpoint_data['scheduler_state_dict'] = kwargs['scheduler'].state_dict()
                self.logger.debug("💾 Saved scheduler state")
            if 'epoch' in kwargs:
                checkpoint_data['epoch'] = kwargs['epoch']
            if 'phase' in kwargs:
                checkpoint_data['phase'] = kwargs['phase']
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Update best metrics manager if this is a best checkpoint
            if is_best and metrics:
                self.best_metrics_manager.update_best_metrics(metrics, epoch, phase)
            
            # Log save info
            log_message = f"✅ {'Best' if is_best else 'Regular'} checkpoint saved successfully at {checkpoint_path}"
            if is_best and metrics:
                val_acc = metrics.get('val_accuracy', 0)
                val_loss = metrics.get('val_loss', 0)
                log_message += f" | Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}"
            self.logger.info(log_message)
            
            # Create phase backup immediately if this is a best model
            if is_best:
                phase_num = kwargs.get('phase', 1)
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
        """📂 Load model from checkpoint with best metrics restoration"""
        
        try:
            # Find checkpoint if not specified
            if checkpoint_path is None:
                checkpoint_path = self._find_best_checkpoint()
                if checkpoint_path is None:
                    raise FileNotFoundError("❌ No checkpoint found")
            
            # If resuming and loading into Phase 2, ensure we load the best checkpoint from Phase 1
            if self.best_metrics_manager.is_resuming and kwargs.get('resume_phase', 1) == 2:
                best_phase1_checkpoint = self.best_metrics_manager._find_best_checkpoint_for_phase(1)
                if best_phase1_checkpoint:
                    self.logger.info(f"🔄 Resuming into Phase 2: Loading best Phase 1 checkpoint: {best_phase1_checkpoint.name}")
                    checkpoint_path = best_phase1_checkpoint
                else:
                    self.logger.warning("⚠️ Resuming into Phase 2 but no best Phase 1 checkpoint found. Loading last checkpoint.")
            
            checkpoint_path = Path(checkpoint_path)
            if not checkpoint_path.exists():
                raise FileNotFoundError(f"❌ Checkpoint not found: {checkpoint_path}")
            
            # Load checkpoint data
            from smartcash.common.checkpoint_utils import safe_load_checkpoint
            checkpoint_data = safe_load_checkpoint(str(checkpoint_path))
            
            # Restore best metrics manager state from checkpoint
            self.best_metrics_manager.restore_from_checkpoint_metadata(checkpoint_data)
            # Set the is_resuming flag in best_metrics_manager
            self.best_metrics_manager.set_is_resuming(True)
            
            # Load model state
            if 'model_state_dict' in checkpoint_data:
                model.load_state_dict(checkpoint_data['model_state_dict'], strict=strict)
            else:
                # Fallback for old checkpoint format
                model.load_state_dict(checkpoint_data, strict=strict)
            
            # Load optimizer and scheduler if available
            optimizer_loaded = False
            scheduler_loaded = False
            
            if 'optimizer' in kwargs and 'optimizer_state_dict' in checkpoint_data:
                kwargs['optimizer'].load_state_dict(checkpoint_data['optimizer_state_dict'])
                optimizer_loaded = True
                self.logger.info("✅ Restored optimizer state from checkpoint")
            
            if 'scheduler' in kwargs and 'scheduler_state_dict' in checkpoint_data:
                kwargs['scheduler'].load_state_dict(checkpoint_data['scheduler_state_dict'])
                scheduler_loaded = True
                self.logger.info("✅ Restored scheduler state from checkpoint")
            
            # Extract checkpoint info
            loaded_metrics = checkpoint_data.get('metrics', {})
            loaded_epoch = checkpoint_data.get('epoch', 0)
            loaded_phase = checkpoint_data.get('phase', 1)
            
            self.logger.info(f"✅ Checkpoint loaded successfully from epoch {loaded_epoch}, phase {loaded_phase}")
            if loaded_metrics:
                val_acc = loaded_metrics.get('val_accuracy', 0)
                val_loss = loaded_metrics.get('val_loss', 0)
                self.logger.info(f"📊 Loaded metrics - Accuracy: {val_acc:.4f}, Loss: {val_loss:.4f}")
            
            # Return checkpoint info
            return {
                'checkpoint_path': str(checkpoint_path),
                'metrics': loaded_metrics,
                'epoch': loaded_epoch,
                'phase': loaded_phase,
                'timestamp': checkpoint_data.get('timestamp', ''),
                'model_info': checkpoint_data.get('model_info', {}),
                'torch_version': checkpoint_data.get('torch_version', ''),
                'loaded_components': {
                    'model': True,
                    'optimizer': optimizer_loaded,
                    'scheduler': scheduler_loaded,
                    'best_metrics_manager': True
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
    
    def get_last_checkpoint_path(self) -> Optional[str]:
        """🔍 Find the most recent 'last_*.pt' checkpoint."""
        try:
            last_checkpoints = list(self.save_dir.glob("last_*.pt"))
            if not last_checkpoints:
                return None
            
            # Sort by modification time (newest first)
            last_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return str(last_checkpoints[0])
        except Exception as e:
            self.logger.error(f"❌ Failed to find last checkpoint: {str(e)}")
            return None

    def check_for_resumable_checkpoint(self, backbone: str) -> Optional[Dict[str, Any]]:
        """🔄 Check for resumable checkpoint for given backbone."""
        try:
            # Find all checkpoint files for this backbone
            pattern = f"{backbone}_phase*_epoch*_best_*.pt"
            checkpoint_files = list(self.save_dir.glob(pattern))
            
            if not checkpoint_files:
                self.logger.info(f"📋 No resumable checkpoints found for {backbone}")
                return None
            
            # Sort by modification time (most recent first)
            checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Try to load the most recent checkpoint
            for checkpoint_file in checkpoint_files:
                try:
                    resume_info = self.load_checkpoint_for_resume(str(checkpoint_file))
                    if resume_info:
                        return resume_info
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to load checkpoint {checkpoint_file.name}: {e}")
                    continue
            
            self.logger.info(f"❌ No valid resumable checkpoints found for {backbone}")
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Error checking for resumable checkpoint: {e}")
            return None

    def load_checkpoint_for_resume(self, checkpoint_path: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
        """📂 Load checkpoint data specifically for training resume."""
        try:
            from smartcash.common.checkpoint_utils import safe_load_checkpoint
            checkpoint_data = safe_load_checkpoint(checkpoint_path)
            
            if not checkpoint_data:
                return None
            
            # Extract resume information
            resume_info = {
                'checkpoint_path': checkpoint_path,
                'epoch': checkpoint_data.get('epoch', 0),
                'phase': checkpoint_data.get('phase', 1),
                'metrics': checkpoint_data.get('metrics', {}),
                'session_id': checkpoint_data.get('session_id'),
                'model_config': checkpoint_data.get('model_config', {}),
                'optimizer_state': checkpoint_data.get('optimizer_state_dict'),
                'scheduler_state': checkpoint_data.get('scheduler_state_dict')
            }
            
            if verbose:
                self.logger.info(f"📂 Loaded resume info from {Path(checkpoint_path).name}")
                self.logger.info(f"   Phase: {resume_info['phase']}, Epoch: {resume_info['epoch']}")
            
            return resume_info
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load checkpoint for resume: {e}")
            return None

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
    
    def get_best_metrics_manager(self) -> BestMetricsManager:
        """Get the best metrics manager instance for external configuration."""
        return self.best_metrics_manager
    
    def configure_early_stopping_with_best_metrics(self, early_stopping, phase_num: int = None):
        """
        Configure early stopping instance to preserve previous best metrics.
        
        Args:
            early_stopping: Early stopping instance to configure
            phase_num: Phase number for phase-specific configuration
        """
        return self.best_metrics_manager.configure_early_stopping_with_previous_best(
            early_stopping, phase_num
        )


# Factory function
def create_checkpoint_manager(config: Dict[str, Any], is_resuming: bool = False) -> CheckpointManager:
    """🏭 Factory untuk membuat CheckpointManager"""
    return CheckpointManager(config)