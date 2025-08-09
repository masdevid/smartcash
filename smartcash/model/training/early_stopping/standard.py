"""
Standard early stopping implementation.

Traditional early stopping that monitors a single metric with configurable patience.
"""

import torch
from typing import Optional, Dict, Any
from .base import BaseEarlyStopping


class StandardEarlyStopping(BaseEarlyStopping):
    """Standard early stopping implementation with single metric monitoring."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001, 
                 metric: str = 'val_accuracy', mode: str = 'max',
                 restore_best_weights: bool = True, save_best_path: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize standard early stopping.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change considered as improvement
            metric: Metric to monitor for early stopping
            mode: 'max' for metrics where higher is better, 'min' for lower is better
            restore_best_weights: Restore model to best weights when stopping
            save_best_path: Path to save best model weights
            verbose: Print early stopping messages
        """
        super().__init__(patience, min_delta, save_best_path, verbose)
        
        self.metric = metric
        self.mode = mode.lower()
        self.restore_best_weights = restore_best_weights
        
        # State tracking
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        
        # Validation
        if self.mode not in ['max', 'min']:
            raise ValueError(f"❌ Mode must be 'max' or 'min', got: {self.mode}")
        
        if self.verbose:
            print(f"   Monitoring: {metric} ({'↗️ higher' if mode == 'max' else '↘️ lower'} is better)")
    
    def __call__(self, metrics, model: Optional[torch.nn.Module] = None, 
                 epoch: int = None) -> bool:
        """
        Check if training should stop based on monitored metric.
        
        Args:
            metrics: Dictionary containing training metrics or direct score
            model: Model for weight saving/restoration
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        # Get current score with robust handling
        if isinstance(metrics, dict):
            current_score = metrics.get(self.metric)
            if current_score is None:
                # Try alternative metric names
                alt_names = {
                    'val_loss': ['loss', 'val_loss', 'validation_loss'],
                    'val_accuracy': ['accuracy', 'val_accuracy', 'validation_accuracy'],
                    'loss': ['loss', 'val_loss', 'total_loss']
                }
                
                for alt_name in alt_names.get(self.metric, [self.metric]):
                    current_score = metrics.get(alt_name)
                    if current_score is not None:
                        if self.verbose and alt_name != self.metric:
                            print(f"📋 Using {alt_name} instead of {self.metric}")
                        break
                
                if current_score is None:
                    if self.verbose:
                        available_metrics = [k for k, v in metrics.items() if isinstance(v, (int, float))]
                        print(f"⚠️ Metric '{self.metric}' not found. Available: {available_metrics}")
                    return False
        else:
            # Support direct score input
            current_score = float(metrics)
        
        # Ensure we have a valid numeric score
        if not isinstance(current_score, (int, float)) or current_score != current_score:  # NaN check
            if self.verbose:
                print(f"⚠️ Invalid score for {self.metric}: {current_score}")
            return False
            
        epoch = epoch or len(self.history['scores'])
        
        # Record history
        self.history['scores'].append(current_score)
        
        # Check for improvement
        is_improvement = self._is_improvement(current_score)
        self.history['improvements'].append(is_improvement)
        
        if is_improvement:
            self.best_score = current_score
            self.best_epoch = epoch
            self.wait = 0
            
            # Save best model
            if model is not None:
                self.save_best_model(model, epoch, metrics if isinstance(metrics, dict) else {self.metric: current_score})
            
            if self.verbose:
                direction = "↗️" if self.mode == 'max' else "↘️"
                print(f"✨ Early stopping: {self.metric} improved {direction} {current_score:.6f} (epoch {epoch + 1})")
        
        else:
            self.wait += 1
            if self.verbose and self.wait > 0:
                print(f"⏳ Early stopping: {self.metric} no improvement ({self.wait}/{self.patience})")
        
        # Record wait count
        self.history['wait_counts'].append(self.wait)
        
        # Check for stopping
        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.should_stop = True
            
            if self.verbose:
                print(f"🛑 Early stopping triggered! Best {self.metric}: {self.best_score:.6f} (epoch {self.best_epoch + 1})")
            
            # Restore best weights
            if self.restore_best_weights and model is not None:
                restored = self.restore_best_weights(model)
                if self.verbose and restored:
                    print(f"🔄 Restored model to best weights from epoch {self.best_epoch + 1}")
        
        return self.should_stop
    
    def _is_improvement(self, score: float) -> bool:
        """Check if current score is improvement with robust comparison."""
        if self.best_score is None:
            return True
        
        # Ensure we have valid scores
        if not isinstance(score, (int, float)) or score != score:  # NaN check
            return False
            
        if not isinstance(self.best_score, (int, float)) or self.best_score != self.best_score:
            return True  # Reset if best_score is invalid
        
        if self.mode == 'max':
            improvement = score > (self.best_score + self.min_delta)
        else:  # mode == 'min'
            improvement = score < (self.best_score - self.min_delta)
            
        return improvement
    
    def reset(self) -> None:
        """Reset early stopping state for new training session."""
        self.best_score = None
        self.best_epoch = 0
        self.wait = 0
        self.stopped_epoch = 0
        self.should_stop = False
        self.best_weights = None
        
        # Clear history
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        if self.verbose:
            print("🔄 Standard early stopping state reset")
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model."""
        return {
            'best_score': self.best_score,
            'best_epoch': self.best_epoch,
            'metric': self.metric,
            'mode': self.mode,
            'current_wait': self.wait,
            'patience': self.patience,
            'stopped': self.should_stop,
            'stopped_epoch': self.stopped_epoch,
            'has_best_weights': self.best_weights is not None
        }
    
    def get_plateau_duration(self) -> int:
        """Get current plateau duration (epochs without improvement)."""
        return self.wait