#!/usr/bin/env python3
"""
Best Metrics Manager

This module handles preservation and restoration of best metrics during training resume,
ensuring that resumed training doesn't incorrectly overwrite better checkpoints.
"""

import torch
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class BestMetricsManager:
    """Manages phase-specific best metrics during training operations."""
    
    def __init__(self, checkpoint_dir: Union[str, Path]):
        """
        Initialize best metrics manager.
        
        Args:
            checkpoint_dir: Directory containing checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.best_metrics_history = {}
        
        # Phase-specific state - each phase maintains independent tracking
        self.phase_best_metrics = {}  # {phase_num: {metrics}}
        self.phase_current_best = {}  # {phase_num: best_value}
        self.current_phase = None
        
    def load_previous_best_metrics(self, phase_num: int) -> Dict[str, Any]:
        """
        Load previous best metrics for a specific phase from existing checkpoints.
        
        Args:
            phase_num: Phase number to load metrics for
            
        Returns:
            Dictionary containing previous best metrics for this phase
        """
        try:
            self.current_phase = phase_num
            
            # Find phase-specific backup checkpoint first
            phase_backup = self._find_phase_backup(phase_num)
            
            if phase_backup:
                logger.info(f"🔍 Loading Phase {phase_num} backup checkpoint: {phase_backup.name}")
                checkpoint_data = self._safe_load_checkpoint(phase_backup)
            else:
                # If no phase backup, look for best checkpoint from this phase
                best_checkpoint = self._find_best_checkpoint_for_phase(phase_num)
                
                if best_checkpoint:
                    logger.info(f"🔍 Loading Phase {phase_num} best checkpoint: {best_checkpoint.name}")
                    checkpoint_data = self._safe_load_checkpoint(best_checkpoint)
                else:
                    logger.info(f"🔍 No previous Phase {phase_num} checkpoint found - starting fresh")
                    return {}
            
            if not checkpoint_data:
                logger.warning(f"⚠️ Failed to load Phase {phase_num} checkpoint data")
                return {}
                
            # Extract metrics
            checkpoint_metrics = checkpoint_data.get('metrics', {})
            epoch = checkpoint_data.get('epoch', 0)
            phase = checkpoint_data.get('phase', phase_num)
            timestamp = checkpoint_data.get('timestamp', '')
            
            # Store in phase-specific state
            self.phase_best_metrics[phase_num] = checkpoint_metrics.copy()
            
            # Store current best value for comparison
            val_accuracy = checkpoint_metrics.get('val_accuracy', 0)
            self.phase_current_best[phase_num] = val_accuracy
            
            best_metrics_info = {
                'metrics': checkpoint_metrics,
                'epoch': epoch,
                'phase': phase,
                'timestamp': timestamp,
                'checkpoint_path': str(phase_backup or best_checkpoint)
            }
            
            logger.info(f"✅ Loaded Phase {phase_num} best metrics from epoch {epoch}")
            logger.info(f"📊 Phase {phase_num} best val_accuracy: {val_accuracy:.4f}")
            logger.info(f"📊 Phase {phase_num} best val_loss: {checkpoint_metrics.get('val_loss', 0):.4f}")
            
            return best_metrics_info
            
        except Exception as e:
            logger.error(f"❌ Failed to load Phase {phase_num} best metrics: {e}")
            return {}
    
    def should_save_as_best(self, current_metrics: Dict[str, Any], 
                           comparison_metric: str = 'val_accuracy',
                           comparison_mode: str = 'max') -> bool:
        """
        Determine if current metrics warrant saving as new best checkpoint for current phase.
        
        Args:
            current_metrics: Current epoch metrics
            comparison_metric: Metric to use for comparison
            comparison_mode: 'max' for higher-is-better, 'min' for lower-is-better
            
        Returns:
            True if current metrics are better than previous best for this phase
        """
        try:
            if self.current_phase is None:
                logger.warning("⚠️ No current phase set - cannot compare metrics")
                return False
                
            current_value = current_metrics.get(comparison_metric)
            if current_value is None:
                logger.warning(f"⚠️ Comparison metric '{comparison_metric}' not found in current metrics")
                return False
            
            # Get previous best value for current phase
            previous_best = self.phase_current_best.get(self.current_phase)
            
            if previous_best is None:
                logger.info(f"🆕 No previous Phase {self.current_phase} best {comparison_metric} found - saving as first best")
                return True
            
            # Compare based on mode
            if comparison_mode == 'max':
                is_better = current_value > previous_best
                direction = "↗️" if is_better else "↘️"
            else:  # 'min'
                is_better = current_value < previous_best
                direction = "↘️" if is_better else "↗️"
            
            logger.info(f"📊 Phase {self.current_phase} {comparison_metric} comparison: {current_value:.6f} vs previous best {previous_best:.6f}")
            
            if is_better:
                logger.info(f"✨ New Phase {self.current_phase} best {comparison_metric}! {direction} {current_value:.6f} > {previous_best:.6f}")
                return True
            else:
                logger.info(f"📉 Phase {self.current_phase} {comparison_metric} {direction} {current_value:.6f} <= {previous_best:.6f} (not better)")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error comparing Phase {self.current_phase} metrics for best checkpoint decision: {e}")
            return False
    
    def update_best_metrics(self, new_metrics: Dict[str, Any], epoch: int, phase: int):
        """
        Update the manager's phase-specific best metrics tracking.
        
        Args:
            new_metrics: New best metrics to store
            epoch: Epoch where these metrics were achieved
            phase: Phase where these metrics were achieved
        """
        # Update phase-specific tracking
        if phase not in self.phase_best_metrics:
            self.phase_best_metrics[phase] = {}
        
        self.phase_best_metrics[phase].update(new_metrics)
        
        # Update current best value for this phase
        val_accuracy = new_metrics.get('val_accuracy', 0)
        self.phase_current_best[phase] = val_accuracy
        
        # Update history
        self.best_metrics_history[f"epoch_{epoch}_phase_{phase}"] = {
            'metrics': new_metrics.copy(),
            'epoch': epoch,
            'phase': phase,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"📝 Updated Phase {phase} best metrics tracking for epoch {epoch}")
        logger.info(f"📊 Phase {phase} new best val_accuracy: {val_accuracy:.4f}")
    
    def set_current_phase(self, phase_num: int):
        """Set the current active phase for metrics comparison."""
        self.current_phase = phase_num
        logger.info(f"🔄 Set current phase to {phase_num}")
    
    def reset_phase_state(self, phase_num: int):
        """Reset all tracking state for a specific phase (for fresh phase start)."""
        if phase_num in self.phase_best_metrics:
            del self.phase_best_metrics[phase_num]
        
        if phase_num in self.phase_current_best:
            del self.phase_current_best[phase_num]
        
        # Remove phase history
        keys_to_remove = [k for k in self.best_metrics_history.keys() if f"_phase_{phase_num}" in k]
        for key in keys_to_remove:
            del self.best_metrics_history[key]
        
        logger.info(f"🧹 Reset all state for Phase {phase_num} (fresh start)")
    
    def transition_to_phase(self, new_phase_num: int, reset_state: bool = True):
        """
        Transition to a new phase, optionally resetting state.
        
        Args:
            new_phase_num: The new phase number
            reset_state: Whether to reset state for the new phase (default: True)
        """
        logger.info(f"🔄 Phase transition: {self.current_phase} -> {new_phase_num}")
        
        if reset_state:
            self.reset_phase_state(new_phase_num)
            logger.info(f"🧹 Phase {new_phase_num} state reset - starting fresh")
        
        self.current_phase = new_phase_num
        logger.info(f"✅ Transitioned to Phase {new_phase_num}")
    
    def configure_early_stopping_with_previous_best(self, early_stopping, phase_num: int):
        """
        Configure phase-specific early stopping with previous best metrics for this phase.
        
        Args:
            early_stopping: Early stopping instance to configure
            phase_num: Phase number for phase-specific configuration
        """
        try:
            # Set current phase
            self.set_current_phase(phase_num)
            
            # Load previous best metrics for this phase if not already loaded
            if phase_num not in self.phase_best_metrics:
                self.load_previous_best_metrics(phase_num)
            
            # Configure early stopping with phase-specific previous best
            phase_metrics = self.phase_best_metrics.get(phase_num, {})
            
            if hasattr(early_stopping, 'best_score') and phase_metrics:
                metric_name = getattr(early_stopping, 'metric', 'val_accuracy')
                previous_best_value = phase_metrics.get(metric_name)
                
                if previous_best_value is not None:
                    early_stopping.best_score = previous_best_value
                    
                    # Also set the epoch info if available
                    if hasattr(early_stopping, 'best_epoch'):
                        # Find the epoch info from phase history
                        for info in self.best_metrics_history.values():
                            if (info['phase'] == phase_num and 
                                info['metrics'].get(metric_name) == previous_best_value):
                                early_stopping.best_epoch = info['epoch']
                                break
                    
                    logger.info(f"🔧 Configured Phase {phase_num} early stopping with previous best {metric_name}: {previous_best_value:.6f}")
                    return True
            
            logger.info(f"ℹ️ Phase {phase_num} early stopping configured without previous best metrics (fresh start)")
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to configure Phase {phase_num} early stopping with previous best: {e}")
            return False
    
    def get_checkpoint_save_metadata(self, epoch: int, phase: int) -> Dict[str, Any]:
        """
        Get metadata to include when saving checkpoints.
        
        Args:
            epoch: Current epoch
            phase: Current phase
            
        Returns:
            Metadata dictionary for checkpoint saving
        """
        return {
            'best_metrics_manager': {
                'phase_best_metrics': self.phase_best_metrics.copy(),
                'phase_current_best': self.phase_current_best.copy(),
                'best_metrics_history': self.best_metrics_history.copy(),
                'current_epoch': epoch,
                'current_phase': phase,
                'manager_timestamp': datetime.now().isoformat()
            }
        }
    
    def restore_from_checkpoint_metadata(self, checkpoint_data: Dict[str, Any]):
        """
        Restore manager state from checkpoint metadata.
        
        Args:
            checkpoint_data: Checkpoint data containing manager state
        """
        try:
            manager_data = checkpoint_data.get('best_metrics_manager', {})
            
            if manager_data:
                self.phase_best_metrics = manager_data.get('phase_best_metrics', {})
                self.phase_current_best = manager_data.get('phase_current_best', {})
                self.best_metrics_history = manager_data.get('best_metrics_history', {})
                
                logger.info("✅ Restored BestMetricsManager phase-specific state from checkpoint")
                logger.info(f"📊 Restored phase best metrics: {list(self.phase_best_metrics.keys())}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Failed to restore BestMetricsManager from checkpoint: {e}")
            return False
    
    def _find_best_checkpoint(self) -> Optional[Path]:
        """Find the best checkpoint in the directory."""
        if not self.checkpoint_dir.exists():
            return None
        
        # Look for best_*.pt files first
        best_checkpoints = list(self.checkpoint_dir.glob("best_*.pt"))
        
        if best_checkpoints:
            # Sort by modification time and return newest
            best_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return best_checkpoints[0]
        
        # Fallback: look for any .pt files
        all_checkpoints = list(self.checkpoint_dir.glob("*.pt"))
        if all_checkpoints:
            all_checkpoints.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return all_checkpoints[0]
        
        return None
    
    def _safe_load_checkpoint(self, checkpoint_path: Path) -> Optional[Dict[str, Any]]:
        """Safely load checkpoint data."""
        try:
            # Use weights_only=False for compatibility with PyTorch 2.6+
            checkpoint_data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            return checkpoint_data
        except Exception as e:
            logger.error(f"❌ Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def _find_phase_backup(self, phase_num: int) -> Optional[Path]:
        """Find phase-specific backup checkpoint."""
        if not self.checkpoint_dir.exists():
            return None
        
        # Look for phase backup files (e.g., best_*_phase1.pt)
        phase_backup_pattern = f"*_phase{phase_num}.pt"
        phase_backups = list(self.checkpoint_dir.glob(phase_backup_pattern))
        
        if phase_backups:
            # Sort by modification time and return newest
            phase_backups.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return phase_backups[0]
        
        return None
    
    def _find_best_checkpoint_for_phase(self, phase_num: int) -> Optional[Path]:
        """Find best checkpoint for a specific phase."""
        if not self.checkpoint_dir.exists():
            return None
        
        # Look for best checkpoints that might belong to this phase
        best_checkpoints = list(self.checkpoint_dir.glob("best_*.pt"))
        
        # Filter out phase backups (those already have phase suffix)
        non_backup_bests = [cp for cp in best_checkpoints if not any(f'_phase{i}' in cp.name for i in range(1, 10))]
        
        if non_backup_bests:
            # Sort by modification time and return newest
            non_backup_bests.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            return non_backup_bests[0]
        
        return None


def create_best_metrics_manager(checkpoint_dir: Union[str, Path]) -> BestMetricsManager:
    """
    Factory function to create a BestMetricsManager instance.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        BestMetricsManager instance
    """
    return BestMetricsManager(checkpoint_dir)