"""
Base early stopping class for SmartCash training.

Provides common functionality and interface for all early stopping implementations.
"""

import torch
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any


class BaseEarlyStopping(ABC):
    """Abstract base class for all early stopping implementations."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.001,
                 save_best_path: Optional[str] = None, verbose: bool = True):
        """
        Initialize base early stopping functionality.
        
        Args:
            patience: Number of epochs with no improvement before stopping
            min_delta: Minimum change considered as improvement
            save_best_path: Path to save best model weights
            verbose: Print early stopping messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.save_best_path = save_best_path
        self.verbose = verbose
        
        # Common state tracking
        self.should_stop = False
        self.stopped_epoch = 0
        self.best_weights = None
        
        # History tracking for analysis
        self.history = {
            'scores': [],
            'improvements': [],
            'wait_counts': []
        }
        
        if self.verbose:
            print(f"🔍 {self.__class__.__name__} initialized with patience={patience}")
    
    @abstractmethod
    def __call__(self, metrics: Dict[str, float], model: Optional[torch.nn.Module] = None,
                 epoch: int = None) -> bool:
        """
        Check if training should stop based on metrics.
        
        Args:
            metrics: Dictionary containing training metrics
            model: Model for weight saving/restoration
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        pass
    
    @abstractmethod
    def reset(self) -> None:
        """Reset early stopping state for new training session."""
        pass
    
    @abstractmethod
    def get_best_info(self) -> Dict[str, Any]:
        """Get information about best model state."""
        pass
    
    def save_best_model(self, model: torch.nn.Module, epoch: int, 
                       metrics: Dict[str, float], additional_data: Dict[str, Any] = None) -> None:
        """
        Save best model with comprehensive checkpoint data.
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Current metrics
            additional_data: Additional data to save in checkpoint
        """
        if not self.save_best_path or model is None:
            return
            
        try:
            save_path = Path(self.save_best_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save model weights in memory for restoration
            self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
            # Prepare checkpoint data
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'metrics': metrics,
                'early_stopping_config': {
                    'class': self.__class__.__name__,
                    'patience': self.patience,
                    'min_delta': self.min_delta,
                },
                'timestamp': time.time()
            }
            
            # Add additional data if provided
            if additional_data:
                checkpoint.update(additional_data)
            
            torch.save(checkpoint, save_path)
            
            if self.verbose:
                print(f"💾 Best model saved: {save_path}")
                
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Error saving best model: {str(e)}")
    
    def restore_best_weights(self, model: torch.nn.Module) -> bool:
        """
        Restore model to best weights if available.
        
        Args:
            model: Model to restore weights to
            
        Returns:
            True if weights were restored, False otherwise
        """
        if self.best_weights is None or model is None:
            return False
        
        try:
            model.load_state_dict(self.best_weights)
            if self.verbose:
                print(f"🔄 Restored model to best weights")
            return True
        except Exception as e:
            if self.verbose:
                print(f"⚠️ Failed to restore best weights: {str(e)}")
            return False
    
    def get_improvement_ratio(self) -> float:
        """Get ratio of epochs with improvement."""
        if not self.history['improvements']:
            return 0.0
        
        improvements = sum(self.history['improvements'])
        total_epochs = len(self.history['improvements'])
        
        return improvements / total_epochs
    
    def get_summary(self) -> str:
        """Get summary string for logging."""
        status = "🛑 STOPPED" if self.should_stop else f"⏳ Active"
        return f"Early stopping: {status} | {self.__class__.__name__}"