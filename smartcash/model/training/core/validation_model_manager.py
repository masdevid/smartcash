#!/usr/bin/env python3
"""
Validation model mode management for the unified training pipeline.

This module handles switching the model between training and evaluation modes
with optimizations for performance.
"""

import torch
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ValidationModelManager:
    """Handles model mode management for validation."""
    
    def __init__(self, model):
        """
        Initialize validation model manager.
        
        Args:
            model: PyTorch model
        """
        self.model = model
    
    def switch_to_eval_mode(self):
        """
        Optimized model switching to evaluation mode.
        
        This method implements optimizations to reduce the time required
        to switch the model from training to evaluation mode.
        """
        try:
            # Check if model is already in eval mode to avoid unnecessary work
            if not self.model.training:
                logger.debug("Model already in eval mode, skipping switch")
                return
            
            # Switch to eval mode
            self.model.eval()
            
            # For CUDA models, ensure synchronization happens now to avoid later sync overhead
            if torch.cuda.is_available() and next(self.model.parameters()).is_cuda:
                torch.cuda.synchronize()
            
            logger.debug("🔄 Model switched to eval mode (optimized)")
            
        except Exception as e:
            # Fallback to simple eval() if optimization fails
            logger.debug(f"⚠️ Optimized eval switch failed, using fallback: {e}")
            self.model.eval()
    
    def switch_to_train_mode(self):
        """Switch model back to training mode."""
        try:
            self.model.train()
            logger.debug("🔄 Model switched to train mode")
        except Exception as e:
            logger.warning(f"Error switching to train mode: {e}")