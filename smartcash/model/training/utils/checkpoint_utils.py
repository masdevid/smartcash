#!/usr/bin/env python3
"""
Checkpoint management utilities for the unified training pipeline.

This module provides functions for checkpoint saving, loading, and management
to keep the main pipeline code clean and focused.
"""

import time
import torch
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


def check_for_resumable_checkpoint(checkpoint_dir: str, backbone: str) -> Optional[Dict[str, Any]]:
    """
    Check for the latest resumable checkpoint for the given backbone.
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        backbone: Model backbone name
        
    Returns:
        Resume information dict or None if no checkpoint found
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return None
        
        # Find all checkpoint files for this backbone
        pattern = f"{backbone}_phase*_epoch*_best_*.pt"
        checkpoint_files = list(checkpoint_path.glob(pattern))
        
        if not checkpoint_files:
            return None
        
        # Sort by modification time (most recent first)
        checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        # Try to load the most recent checkpoint
        for checkpoint_file in checkpoint_files:
            try:
                checkpoint = torch.load(checkpoint_file, map_location='cpu')
                
                # Extract information from checkpoint
                resume_info = {
                    'checkpoint_path': str(checkpoint_file),
                    'phase': checkpoint.get('phase', 1),
                    'epoch': checkpoint.get('epoch', 0),
                    'metrics': checkpoint.get('metrics', {}),
                    'config': checkpoint.get('config', {}),
                    'session_id': checkpoint.get('session_id'),
                    'model_state_dict': checkpoint.get('model_state_dict'),
                    'checkpoint_name': checkpoint_file.name
                }
                
                logger.info(f"✅ Found valid checkpoint: {checkpoint_file.name}")
                return resume_info
                
            except Exception as e:
                logger.warning(f"⚠️ Could not load checkpoint {checkpoint_file.name}: {e}")
                continue
        
        return None
        
    except Exception as e:
        logger.error(f"Error checking for resumable checkpoint: {e}")
        return None


def generate_checkpoint_name(
    backbone: str,
    model: torch.nn.Module,
    config: Dict[str, Any],
    is_best: bool = True
) -> str:
    """
    Generate checkpoint filename according to naming conventions.
    
    Args:
        backbone: Model backbone name
        model: PyTorch model for device detection
        config: Training configuration
        is_best: Whether this is a best model checkpoint
        
    Returns:
        Checkpoint filename
    """
    try:
        # Determine phase mode
        total_phases = len([k for k in config.get('training_phases', {}) if 'phase_' in k])
        phase_mode = 'single_phase' if total_phases == 1 else 'two_phase'
        
        # Determine layer mode from config
        layer_mode = config['model'].get('layer_mode', 'multi')
        if 'detection_layers' in config['model']:
            num_layers = len(config['model']['detection_layers'])
            layer_mode = 'single' if num_layers == 1 else 'multi'
        
        # Determine device type
        device = next(model.parameters()).device
        device_type = 'cpu' if device.type == 'cpu' else ('mps' if device.type == 'mps' else 'gpu')
        
        # Get batch size
        batch_size = config.get('training', {}).get('data', {}).get('batch_size', 'auto')
        
        # Get current date
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Generate name based on type
        if is_best:
            checkpoint_name = f"best_{backbone}_{phase_mode}_{layer_mode}_{device_type}_{batch_size}_{date_str}.pt"
        else:
            timestamp = int(time.time())
            checkpoint_name = f"{backbone}_{phase_mode}_{layer_mode}_{device_type}_{batch_size}_{timestamp}.pt"
        
        return checkpoint_name
        
    except Exception as e:
        logger.error(f"Error generating checkpoint name: {e}")
        # Fallback naming
        timestamp = int(time.time())
        return f"{backbone}_checkpoint_{timestamp}.pt"


def save_checkpoint_to_disk(
    checkpoint_path: Path,
    model_state_dict: Dict[str, Any],
    epoch: int,
    phase: int,
    metrics: Dict[str, float],
    config: Dict[str, Any],
    session_id: Optional[str] = None
) -> bool:
    """
    Save checkpoint data to disk.
    
    Args:
        checkpoint_path: Path where to save the checkpoint
        model_state_dict: Model state dictionary
        epoch: Current epoch number
        phase: Current phase number
        metrics: Training metrics
        config: Training configuration
        session_id: Training session ID
        
    Returns:
        True if saved successfully, False otherwise
    """
    try:
        checkpoint_data = {
            'model_state_dict': model_state_dict,
            'epoch': epoch,
            'phase': phase,
            'metrics': metrics,
            'config': config,
            'session_id': session_id,
            'timestamp': time.time()
        }
        
        # Create directory if it doesn't exist
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        logger.info(f"💾 Checkpoint saved: {checkpoint_path.name}")
        return True
        
    except Exception as e:
        logger.error(f"Error saving checkpoint to disk: {e}")
        return False


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    backbone: str,
    keep_best: int = 3,
    keep_recent: int = 5
) -> int:
    """
    Clean up old checkpoints, keeping only the most recent and best ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        backbone: Model backbone name to filter checkpoints
        keep_best: Number of best checkpoints to keep
        keep_recent: Number of recent checkpoints to keep
        
    Returns:
        Number of checkpoints removed
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            return 0
        
        # Find all checkpoint files for this backbone
        all_checkpoints = list(checkpoint_path.glob(f"{backbone}*.pt"))
        best_checkpoints = [f for f in all_checkpoints if 'best_' in f.name]
        regular_checkpoints = [f for f in all_checkpoints if 'best_' not in f.name]
        
        removed_count = 0
        
        # Clean up best checkpoints (keep most recent by modification time)
        if len(best_checkpoints) > keep_best:
            best_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for checkpoint in best_checkpoints[keep_best:]:
                try:
                    checkpoint.unlink()
                    removed_count += 1
                    logger.info(f"🗑️ Removed old best checkpoint: {checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint.name}: {e}")
        
        # Clean up regular checkpoints
        if len(regular_checkpoints) > keep_recent:
            regular_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for checkpoint in regular_checkpoints[keep_recent:]:
                try:
                    checkpoint.unlink()
                    removed_count += 1
                    logger.info(f"🗑️ Removed old checkpoint: {checkpoint.name}")
                except Exception as e:
                    logger.warning(f"Could not remove checkpoint {checkpoint.name}: {e}")
        
        if removed_count > 0:
            logger.info(f"🧹 Cleaned up {removed_count} old checkpoints")
        
        return removed_count
        
    except Exception as e:
        logger.error(f"Error during checkpoint cleanup: {e}")
        return 0


def load_checkpoint_for_resume(checkpoint_path: str) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data for resuming training.
    
    Args:
        checkpoint_path: Path to checkpoint file
        
    Returns:
        Checkpoint data dictionary or None if failed
    """
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Validate checkpoint structure
        required_keys = ['model_state_dict', 'epoch', 'phase']
        for key in required_keys:
            if key not in checkpoint:
                logger.warning(f"Checkpoint missing required key: {key}")
                return None
        
        logger.info(f"✅ Loaded checkpoint: {Path(checkpoint_path).name}")
        return checkpoint
        
    except Exception as e:
        logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None