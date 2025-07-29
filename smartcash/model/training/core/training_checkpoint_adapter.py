#!/usr/bin/env python3
"""
Training checkpoint adapter for the unified training pipeline.

This module provides an adapter between the training pipeline and the
main checkpoint manager, maintaining interface compatibility while
leveraging the comprehensive checkpoint management features.
"""

from typing import Dict, Any, Optional
from pathlib import Path

from smartcash.common.logger import get_logger
from smartcash.model.core.checkpoint_manager import CheckpointManager

# Try to import ModelProgressBridge, create mock if not available
try:
    from smartcash.model.utils.progress_bridge import ModelProgressBridge
except ImportError:
    # Create a simple mock for ModelProgressBridge
    class ModelProgressBridge:
        def start_operation(self, name, steps): 
            pass
        def update(self, step, message): 
            pass
        def complete(self, step, message): 
            pass
        def error(self, message): 
            pass

logger = get_logger(__name__)


class TrainingCheckpointAdapter:
    """Adapter for training pipeline to use the main CheckpointManager."""
    
    def __init__(self, model, model_api, config):
        """
        Initialize training checkpoint adapter.
        
        Args:
            model: PyTorch model
            model_api: Model API instance  
            config: Training configuration
        """
        self.model = model
        self.model_api = model_api
        self.config = config
        
        # Create progress bridge for the checkpoint manager
        self.progress_bridge = ModelProgressBridge()
        
        # Initialize the main checkpoint manager
        self.checkpoint_manager = CheckpointManager(config, self.progress_bridge)
        
        # Best model tracking (for backward compatibility)
        self.best_metrics = {}
        self.best_checkpoint_path = None
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       phase_num: int, is_best: bool = False) -> Optional[str]:
        """
        Save checkpoint using the main checkpoint manager.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics dictionary
            phase_num: Current phase number
            is_best: Whether this is the best model so far
            
        Returns:
            Path to saved checkpoint or None if failed
        """
        try:
            # Prepare checkpoint name if this is the best model
            checkpoint_name = None
            if is_best:
                checkpoint_name = self._generate_best_checkpoint_name()
            
            # Use the main checkpoint manager
            saved_path = self.checkpoint_manager.save_checkpoint(
                model=self.model,
                metrics=metrics,
                checkpoint_name=checkpoint_name,
                epoch=epoch,
                phase=phase_num,
                is_best=is_best
            )
            
            # Update tracking for backward compatibility
            if is_best and saved_path:
                self.best_checkpoint_path = saved_path
                self.best_metrics = metrics.copy()
            
            # Log checkpoint info
            if saved_path:
                self._log_checkpoint_info(saved_path, epoch, phase_num, is_best)
            
            return saved_path
            
        except Exception as e:
            logger.error(f"Error saving checkpoint via adapter: {str(e)}")
            
            # Fallback: try model API if available
            if self.model_api:
                try:
                    checkpoint_info = {
                        'epoch': epoch,
                        'phase': phase_num,
                        'metrics': metrics,
                        'is_best': is_best,
                        'config': self.config
                    }
                    saved_path = self.model_api.save_checkpoint(**checkpoint_info)
                    if saved_path and is_best:
                        self.best_checkpoint_path = saved_path
                        self.best_metrics = metrics.copy()
                    return saved_path
                except Exception as api_error:
                    logger.error(f"Model API fallback also failed: {str(api_error)}")
            
            return None
    
    def _generate_best_checkpoint_name(self) -> str:
        """Generate a checkpoint name for best models."""
        # Use the main checkpoint manager's naming logic with is_best=True
        return self.checkpoint_manager._generate_checkpoint_name(is_best=True)
    
    def _log_checkpoint_info(self, saved_path: str, epoch: int, phase_num: int, is_best: bool):
        """Log checkpoint information."""
        checkpoint_type = "Best" if is_best else "Regular"
        logger.info(f"💾 {checkpoint_type} checkpoint saved: {Path(saved_path).name}")
    
    def get_best_checkpoint_info(self) -> Dict[str, Any]:
        """
        Get information about the best checkpoint.
        
        Returns:
            Dictionary containing best checkpoint information
        """
        return {
            'path': self.best_checkpoint_path,
            'metrics': self.best_metrics.copy() if self.best_metrics else {},
            'exists': self.best_checkpoint_path is not None
        }
    
    def ensure_best_checkpoint(self, epoch: int, metrics: Dict[str, float], phase_num: int) -> str:
        """
        Ensure a best checkpoint exists, creating one if necessary.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            phase_num: Current phase number
            
        Returns:
            Path to best checkpoint
        """
        if not self.best_checkpoint_path:
            self.best_checkpoint_path = self.save_checkpoint(epoch, metrics, phase_num, is_best=True)
            if not self.best_checkpoint_path:
                logger.warning("Failed to create best checkpoint")
                return ""
        
        return self.best_checkpoint_path
    
    def update_best_if_better(self, epoch: int, metrics: Dict[str, float], 
                             phase_num: int, monitor_metric: str = 'val_map50') -> bool:
        """
        Update best checkpoint if current metrics are better.
        
        Args:
            epoch: Current epoch number
            metrics: Current metrics
            phase_num: Current phase number
            monitor_metric: Metric to monitor for best model
            
        Returns:
            True if this is a new best model, False otherwise
        """
        current_value = metrics.get(monitor_metric, 0.0)
        
        if not self.best_metrics or current_value > self.best_metrics.get(monitor_metric, 0.0):
            self.save_checkpoint(epoch, metrics, phase_num, is_best=True)
            return True
        
        return False
    
    def list_checkpoints(self):
        """List available checkpoints using the main checkpoint manager."""
        return self.checkpoint_manager.list_checkpoints()
    
    def load_checkpoint(self, checkpoint_path: Optional[str] = None, strict: bool = True, **kwargs):
        """Load checkpoint using the main checkpoint manager."""
        return self.checkpoint_manager.load_checkpoint(
            self.model, checkpoint_path, strict, **kwargs
        )