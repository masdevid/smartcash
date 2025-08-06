#!/usr/bin/env python3
"""
Checkpoint Utilities

This module handles checkpoint management operations including backup creation,
model discovery, and checkpoint naming conventions.
"""

import shutil
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class CheckpointUtils:
    """Utility class for checkpoint management operations."""
    
    @staticmethod
    def create_phase_backup(checkpoint_path: str, phase_num: int) -> Optional[str]:
        """
        Create backup copy of phase checkpoint with phase suffix.
        
        Args:
            checkpoint_path: Path to the checkpoint to backup
            phase_num: Phase number for backup naming
            
        Returns:
            Path to the backup checkpoint or None if failed
        """
        if not checkpoint_path:
            return None
            
        try:
            best_checkpoint_path = Path(checkpoint_path)
            backup_checkpoint_path = best_checkpoint_path.parent / (
                best_checkpoint_path.stem + f'_phase{phase_num}' + best_checkpoint_path.suffix
            )
            shutil.copy2(best_checkpoint_path, backup_checkpoint_path)
            logger.info(f"✅ Created Phase {phase_num} backup model: {backup_checkpoint_path}")
            return str(backup_checkpoint_path)
        except Exception as e:
            logger.warning(f"⚠️ Failed to create Phase {phase_num} backup model: {e}")
            return None
    
    @staticmethod
    def find_phase_backup_model(phase_num: int, config: Dict[str, Any]) -> Optional[str]:
        """
        Find existing phase backup model.
        
        Args:
            phase_num: Phase number to find backup for
            config: Configuration containing checkpoint directory
            
        Returns:
            Path to the phase backup model or None if not found
        """
        try:
            checkpoint_dir = Path(config.get('checkpoint_dir', 'data/checkpoints'))
            
            # Look for backup models with phase suffix
            pattern = str(checkpoint_dir / f"*_phase{phase_num}.pt")
            backup_files = glob.glob(pattern)
            
            if backup_files:
                # Sort by modification time, get the most recent
                backup_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                backup_path = backup_files[0]
                logger.info(f"📦 Found Phase {phase_num} backup model: {backup_path}")
                return backup_path
            else:
                logger.warning(f"⚠️ No Phase {phase_num} backup model found in {checkpoint_dir}")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to find Phase {phase_num} backup model: {e}")
            return None
    
    @staticmethod
    def find_standard_best_model(config: Dict[str, Any]) -> Optional[str]:
        """
        Find standard best model using naming convention (excludes backup models with _phase suffixes).
        
        Args:
            config: Configuration containing checkpoint directory
            
        Returns:
            Path to the standard best model or None if not found
        """
        try:
            checkpoint_dir = Path(config.get('checkpoint_dir', 'data/checkpoints'))
            
            # Look for best models with standard naming convention: best_*.pt
            pattern = str(checkpoint_dir / "best_*.pt")
            all_best_files = glob.glob(pattern)
            
            # Filter out backup models with _phase1 or _phase2 suffixes
            standard_best_files = []
            for file_path in all_best_files:
                file_name = Path(file_path).stem  # Get filename without extension
                # Exclude files that end with _phase1 or _phase2
                if not (file_name.endswith('_phase1') or file_name.endswith('_phase2')):
                    standard_best_files.append(file_path)
            
            if standard_best_files:
                # Sort by modification time, get the most recent
                standard_best_files.sort(key=lambda x: Path(x).stat().st_mtime, reverse=True)
                best_path = standard_best_files[0]
                logger.debug(f"📦 Found standard best model: {best_path}")
                return best_path
            else:
                logger.debug(f"⚠️ No standard best model found in {checkpoint_dir} (excluding backup models)")
                return None
        except Exception as e:
            logger.error(f"❌ Failed to find standard best model: {e}")
            return None
    
    @staticmethod
    def override_standard_best_with_backup(backup_path: str, config: Dict[str, Any]):
        """
        Override standard best model with backup model content (file copy only).
        
        Args:
            backup_path: Path to the backup checkpoint
            config: Configuration containing checkpoint directory
        """
        try:
            backup_checkpoint_path = Path(backup_path)
            
            # Find the current best model using the naming convention
            standard_best_path = CheckpointUtils.find_standard_best_model(config)
            
            if standard_best_path:
                # Copy backup to standard best model location (override)
                shutil.copy2(backup_checkpoint_path, standard_best_path)
                logger.info(f"✅ Standard best model overridden with Phase 1 backup: {standard_best_path}")
                logger.info("🔄 Training will load from standard best model (now contains Phase 1 backup)")
            else:
                # No existing best model, create new one with full naming convention
                new_best_path = CheckpointUtils.generate_new_best_model_name(config)
                
                shutil.copy2(backup_checkpoint_path, new_best_path)
                logger.info(f"✅ Created new standard best model from Phase 1 backup: {new_best_path}")
                
        except Exception as e:
            logger.error(f"❌ Failed to override standard best model with backup: {e}")
    
    @staticmethod
    def generate_new_best_model_name(config: Dict[str, Any]) -> Path:
        """
        Generate new best model filename using the full naming convention.
        
        Args:
            config: Configuration containing model and training parameters
            
        Returns:
            Path to the new best model
        """
        try:
            checkpoint_dir = Path(config.get('checkpoint_dir', 'data/checkpoints'))
            
            # Get model configuration for naming
            model_config = config.get('model', {})
            training_config = config.get('training', {})
            
            # Extract naming components (same logic as checkpoint manager)
            backbone = model_config.get('backbone', config.get('backbone', 'unknown'))
            
            # Determine training mode
            training_mode = training_config.get('training_mode', config.get('training_mode', 'two_phase'))
            if training_mode not in ['single_phase', 'two_phase']:
                training_mode = 'two_phase'  # Default fallback
            
            # Determine layer mode
            layer_mode = model_config.get('layer_mode', 'multi')
            if 'detection_layers' in model_config:
                num_layers = len(model_config['detection_layers'])
                layer_mode = 'single' if num_layers == 1 else 'multi'
            
            # Determine freeze status
            freeze_backbone = model_config.get('freeze_backbone', False)
            freeze_status = 'frozen' if freeze_backbone else 'unfrozen'
            
            # Check if pretrained is enabled
            pretrained = model_config.get('pretrained', config.get('pretrained', False))
            pretrained_suffix = 'pretrained' if pretrained else ''
            
            # Get current date
            date_str = datetime.now().strftime('%Y%m%d')
            
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
            
            return checkpoint_dir / filename
            
        except Exception as e:
            # Fallback to simple naming if anything goes wrong
            logger.warning(f"⚠️ Error generating full checkpoint name: {e}, using fallback")
            
            checkpoint_dir = Path(config.get('checkpoint_dir', 'data/checkpoints'))
            backbone = config.get('backbone', 'unknown')
            date_str = datetime.now().strftime('%Y%m%d')
            filename = f"best_{backbone}_{date_str}.pt"
            return checkpoint_dir / filename
    
    @staticmethod
    def load_standard_best_model(model, config: Dict[str, Any]):
        """
        Load standard best model into current model for Phase 2 training.
        
        Args:
            model: Model instance to load weights into
            config: Configuration containing checkpoint directory
        """
        try:
            import torch
            
            # Find the current best model using the naming convention
            best_model_path = CheckpointUtils.find_standard_best_model(config)
            
            if best_model_path:
                logger.info(f"🔄 Loading standard best model for Phase 2 training: {best_model_path}")
                # Use weights_only=False for compatibility with PyTorch 2.6+
                checkpoint = torch.load(best_model_path, map_location='cpu', weights_only=False)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                    logger.info("✅ Standard best model loaded into current model")
                else:
                    logger.warning("⚠️ No model_state_dict found in standard best model")
            else:
                logger.warning("⚠️ No standard best model found")
                
        except Exception as e:
            logger.error(f"❌ Failed to load standard best model: {e}")