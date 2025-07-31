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


# Import shared safe loading function
from smartcash.common.checkpoint_utils import safe_load_checkpoint


def _load_checkpoint_raw(checkpoint_path: str) -> Dict[str, Any]:
    """Load raw checkpoint data with proper safety measures and version compatibility."""
    return safe_load_checkpoint(checkpoint_path)


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
                # Use version-compatible checkpoint loading
                checkpoint = _load_checkpoint_raw(str(checkpoint_file))
                
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
    Format: best_{backbone}_{two_phase|single_phase}_{single|multi}_{frozen|unfrozen}_{pretrained:if true}_{yyyymmdd}.pt
    
    Args:
        backbone: Model backbone name
        model: PyTorch model for device detection
        config: Training configuration
        is_best: Whether this is a best model checkpoint
        
    Returns:
        Checkpoint filename
    """
    try:
        # Determine training mode (phase mode)
        training_config = config.get('training', {})
        training_mode = training_config.get('training_mode', 'two_phase')
        if training_mode not in ['single_phase', 'two_phase']:
            # Fallback: determine from phases configuration
            total_phases = len([k for k in config.get('training_phases', {}) if 'phase_' in k])
            training_mode = 'single_phase' if total_phases == 1 else 'two_phase'
        
        # Determine layer mode from config
        model_config = config.get('model', {})
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
        
        # Get current date
        date_str = datetime.now().strftime('%Y%m%d')
        
        # Generate name based on type
        if is_best:
            # Build checkpoint name parts
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
            checkpoint_name = '_'.join(name_parts) + '.pt'
        else:
            timestamp = int(time.time())
            checkpoint_name = f"{backbone}_{training_mode}_{layer_mode}_{freeze_status}_{timestamp}.pt"
        
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


def find_latest_checkpoint(checkpoint_dir: str, backbone: str = None) -> Optional[str]:
    """
    Auto-detect the latest 'last_*.pt' checkpoint for resuming training.
    
    This function follows the sequence described in debug.md:
    - 🔁 Resume Phase (last.pt) -> Keeps optimizer, LR schedule, and epoch in sync
    
    Args:
        checkpoint_dir: Directory to search for checkpoints
        backbone: Optional backbone name to filter checkpoints
        
    Returns:
        Path to the latest checkpoint file or None if not found
    """
    try:
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            logger.warning(f"📁 Checkpoint directory does not exist: {checkpoint_dir}")
            return None
        
        # Search patterns in order of preference (following debug.md)
        search_patterns = []
        
        if backbone:
            # Look for backbone-specific last checkpoints first
            search_patterns.extend([
                f"last_{backbone}_*.pt",
                f"last_{backbone}.pt"
            ])
        
        # General last checkpoint patterns
        search_patterns.extend([
            "last_*.pt",
            "last.pt"
        ])
        
        all_candidates = []
        
        # Collect all matching checkpoints
        for pattern in search_patterns:
            candidates = list(checkpoint_path.glob(pattern))
            all_candidates.extend(candidates)
        
        if not all_candidates:
            logger.info(f"📋 No 'last_*.pt' checkpoints found in {checkpoint_dir}")
            return None
        
        # Remove duplicates and sort by modification time (most recent first)
        unique_candidates = list(set(all_candidates))
        unique_candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        
        latest_checkpoint = unique_candidates[0]
        
        # Verify the checkpoint is loadable
        try:
            _load_checkpoint_raw(str(latest_checkpoint))
            logger.info(f"✅ Auto-detected latest checkpoint: {latest_checkpoint.name}")
            return str(latest_checkpoint)
        except Exception as e:
            logger.warning(f"⚠️ Latest checkpoint {latest_checkpoint.name} is corrupted: {e}")
            
            # Try the next candidates
            for candidate in unique_candidates[1:]:
                try:
                    _load_checkpoint_raw(str(candidate))
                    logger.info(f"✅ Using fallback checkpoint: {candidate.name}")
                    return str(candidate)
                except Exception:
                    continue
            
            logger.error("❌ No valid checkpoints found")
            return None
        
    except Exception as e:
        logger.error(f"Error auto-detecting checkpoint: {e}")
        return None


def load_checkpoint_for_resume(checkpoint_path: str, verbose: bool = True) -> Optional[Dict[str, Any]]:
    """
    Load checkpoint data for resuming training.
    
    This function provides intelligent checkpoint loading with:
    - Automatic last_*.pt detection for more recent training state  
    - Phase detection from checkpoint structure and naming
    - Proper epoch calculation for resume
    - Comprehensive validation and user feedback
    
    Args:
        checkpoint_path: Path to checkpoint file
        verbose: Whether to print detailed analysis information
        
    Returns:
        Resume information dictionary or None if failed
    """
    try:
        # First, try to find a more recent 'last_*.pt' checkpoint in the same directory
        checkpoint_dir = Path(checkpoint_path).parent
        
        # Find all last_*.pt files in the directory
        last_checkpoints = list(checkpoint_dir.glob('last_*.pt'))
        
        # Also check for legacy last.pt
        legacy_last = checkpoint_dir / 'last.pt'
        if legacy_last.exists():
            last_checkpoints.append(legacy_last)
        
        # If we have last checkpoints, find the most recent one
        if last_checkpoints:
            try:
                # Sort by modification time to find the most recent
                last_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                most_recent_last = last_checkpoints[0]
                
                last_ckpt = _load_checkpoint_raw(most_recent_last)
                last_epoch = last_ckpt.get('epoch', 0)
                current_ckpt = _load_checkpoint_raw(checkpoint_path)
                current_epoch = current_ckpt.get('epoch', 0)
                
                if last_epoch > current_epoch:
                    if verbose:
                        logger.info(f"🔍 Found more recent last checkpoint (epoch {last_epoch} vs {current_epoch})")
                        logger.info(f"   Using: {most_recent_last}")
                    checkpoint = last_ckpt
                    checkpoint_path = str(most_recent_last)
                else:
                    checkpoint = current_ckpt
            except Exception as e:
                if verbose:
                    logger.warning(f"⚠️ Could not load last checkpoint, using specified checkpoint: {e}")
                checkpoint = _load_checkpoint_raw(checkpoint_path)
        else:
            checkpoint = _load_checkpoint_raw(checkpoint_path)
        
        # Analyze checkpoint and create resume info
        resume_info = _create_resume_info(checkpoint, checkpoint_path, verbose)
        
        return resume_info
        
    except Exception as e:
        logger.error(f"Error loading checkpoint for resume {checkpoint_path}: {e}")
        return None


def _create_resume_info(checkpoint: Dict[str, Any], checkpoint_path: str, verbose: bool = True) -> Dict[str, Any]:
    """Create comprehensive resume information from checkpoint."""
    import time
    
    # Extract epoch and calculate resume epoch
    saved_epoch = checkpoint.get('epoch', 0)
    resume_epoch = saved_epoch + 1  # Correct epoch calculation
    
    # Determine phase based on checkpoint analysis
    checkpoint_name = Path(checkpoint_path).name.lower()
    metrics = checkpoint.get('metrics', {})
    
    # Try to determine phase from checkpoint structure
    if 'phase' in checkpoint:
        # Direct phase information in checkpoint
        resume_phase = checkpoint.get('phase', 1)
    elif 'phase_1' in checkpoint_name:
        resume_phase = 1
    elif 'phase_2' in checkpoint_name:
        resume_phase = 2
    else:
        # Analyze checkpoint to determine phase
        # Phase 1: Early training, basic metrics
        # Phase 2: Advanced training, full validation metrics
        
        # Check for phase 2 indicators
        # Phase 2 is characterized by:
        # - Detection metrics: val_map50, val_detection_map50
        # - Multi-layer metrics: val_layer_2_*, val_layer_3_*
        # - Research metrics: val_research_primary_metric, val_hierarchical_accuracy
        has_detection_metrics = any(key in ['val_map50', 'val_detection_map50'] for key in metrics.keys())
        has_multilayer_metrics = any(key.startswith('val_layer_2_') or key.startswith('val_layer_3_') for key in metrics.keys())
        has_research_metrics = any(key in ['val_research_primary_metric', 'val_hierarchical_accuracy'] for key in metrics.keys())
        has_layer_contribution = any('contribution' in key for key in metrics.keys())
        
        # More comprehensive phase 2 detection
        phase_2_indicators = has_detection_metrics or has_multilayer_metrics or has_research_metrics or has_layer_contribution
        
        if saved_epoch == 0 and not phase_2_indicators:
            # Very early checkpoint, likely phase 1
            resume_phase = 1
        elif phase_2_indicators:
            # Advanced metrics suggest phase 2
            resume_phase = 2
        else:
            # Default to phase 1 for ambiguous cases
            resume_phase = 1
    
    # Create resume info dictionary
    resume_info = {
        'checkpoint_path': checkpoint_path,
        'checkpoint_name': Path(checkpoint_path).name,
        'epoch': resume_epoch,
        'phase': resume_phase,
        'model_state_dict': checkpoint.get('model_state_dict'),
        'metrics': metrics,
        'config': checkpoint.get('model_config', {}),
        'session_id': f"resume_{int(time.time())}",
        'timestamp': checkpoint.get('timestamp', time.time()),
        'model_info': checkpoint.get('model_info', {}),
        'saved_epoch': saved_epoch  # Keep original for debugging
    }
    
    # Provide detailed analysis if verbose
    if verbose:
        _print_resume_analysis(resume_info, metrics, saved_epoch, resume_epoch, resume_phase)
    
    return resume_info


def _print_resume_analysis(resume_info: Dict[str, Any], metrics: Dict, saved_epoch: int, 
                          resume_epoch: int, resume_phase: int):
    """Print essential resume information for user feedback."""
    logger.info(f"📊 Loaded checkpoint from phase {resume_phase} and epoch {resume_epoch}")
    logger.info(f"   Checkpoint file: {resume_info['checkpoint_name']}")
    
    # Show key metrics if available
    if metrics:
        key_metrics = {k: v for k, v in metrics.items() if k in ['val_map50', 'val_loss', 'train_loss']}
        if key_metrics:
            logger.info(f"   Key metrics: {key_metrics}")