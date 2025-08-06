"""
Adaptive early stopping implementation.

Early stopping with adaptive patience that adjusts based on training progress.
"""

from typing import Optional, Dict, Any
import torch
from .standard import StandardEarlyStopping


class AdaptiveEarlyStopping(StandardEarlyStopping):
    """Early stopping with adaptive patience based on training progress."""
    
    def __init__(self, initial_patience: int = 10, patience_factor: float = 1.5,
                 max_patience: int = 50, improvement_threshold: float = 0.01, **kwargs):
        """
        Initialize adaptive early stopping.
        
        Args:
            initial_patience: Initial patience value
            patience_factor: Factor to increase patience when significant improvement occurs
            max_patience: Maximum patience value
            improvement_threshold: Threshold for significant improvement
            **kwargs: Additional arguments for base StandardEarlyStopping
        """
        super().__init__(patience=initial_patience, **kwargs)
        
        self.initial_patience = initial_patience
        self.patience_factor = patience_factor
        self.max_patience = max_patience
        self.improvement_threshold = improvement_threshold
        
        # Track significant improvements
        self.significant_improvements = 0
        self.last_significant_score = None
        
        if self.verbose:
            print(f"   Adaptive: factor={patience_factor}, max_patience={max_patience}")
    
    def __call__(self, metrics, model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """Check with adaptive patience logic."""
        
        # Get current score for adaptive logic
        if isinstance(metrics, dict):
            current_score = metrics.get(self.metric)
            if current_score is None:
                return super().__call__(metrics, model, epoch)
        else:
            current_score = float(metrics)
        
        # Check for significant improvement
        if self._is_significant_improvement(current_score):
            self.significant_improvements += 1
            self.last_significant_score = current_score
            
            # Increase patience after significant improvement
            new_patience = min(
                int(self.patience * self.patience_factor),
                self.max_patience
            )
            
            if new_patience != self.patience:
                old_patience = self.patience
                self.patience = new_patience
                if self.verbose:
                    print(f"📈 Significant improvement detected! Patience increased: {old_patience} → {new_patience}")
        
        # Call parent implementation
        return super().__call__(metrics, model, epoch)
    
    def _is_significant_improvement(self, score: float) -> bool:
        """Check if improvement is significant enough to adjust patience."""
        if self.last_significant_score is None:
            self.last_significant_score = score
            return True
        
        if self.mode == 'max':
            improvement = score - self.last_significant_score
            return improvement > self.improvement_threshold
        else:
            improvement = self.last_significant_score - score
            return improvement > self.improvement_threshold
    
    def reset(self) -> None:
        """Reset adaptive early stopping state."""
        super().reset()
        
        # Reset adaptive-specific state
        self.patience = self.initial_patience
        self.significant_improvements = 0
        self.last_significant_score = None
        
        if self.verbose:
            print("🔄 Adaptive early stopping state reset")
    
    def get_adaptive_info(self) -> Dict[str, Any]:
        """Get adaptive early stopping information."""
        base_info = self.get_best_info()
        adaptive_info = {
            'initial_patience': self.initial_patience,
            'current_patience': self.patience,
            'max_patience': self.max_patience,
            'significant_improvements': self.significant_improvements,
            'patience_factor': self.patience_factor,
            'improvement_threshold': self.improvement_threshold,
            'last_significant_score': self.last_significant_score
        }
        
        return {**base_info, **adaptive_info}
    
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model including adaptive details."""
        base_info = super().get_best_info()
        base_info['adaptive_info'] = {
            'significant_improvements': self.significant_improvements,
            'current_patience': self.patience,
            'initial_patience': self.initial_patience
        }
        return base_info