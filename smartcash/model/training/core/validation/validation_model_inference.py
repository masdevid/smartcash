#!/usr/bin/env python3
"""
Model inference for validation batch processing.

This module handles model inference operations during validation,
including optimized inference and prediction formatting.
"""

import torch
from typing import Optional, Any

from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class ValidationModelInference:
    """Handles model inference operations for validation."""
    
    def __init__(self, model: torch.nn.Module, device: torch.device, use_amp: bool = False):
        """
        Initialize model inference handler.
        
        Args:
            model: The model to run inference with
            device: Device to run inference on
            use_amp: Whether to use automatic mixed precision
        """
        self.model = model
        self.device = device
        self.use_amp = use_amp
        self.prediction_processor = getattr(model, 'prediction_processor', None)
    
    def run_inference_optimized(self, images: torch.Tensor) -> torch.Tensor:
        """
        Optimized model inference with better memory management.
        
        Args:
            images: Input tensor of shape [B, C, H, W]
            
        Returns:
            Model predictions in expected format
            
        Raises:
            ValueError: If input is invalid
            RuntimeError: If inference fails
        """
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(f"Input must be 4D tensor [B, C, H, W], got {type(images)}")
            
        batch_size = images.size(0)
        if batch_size == 0:
            return torch.zeros((0, 0, 6), device=self.device)
        
        try:
            # Optimized inference with memory management
            with torch.no_grad(), torch.amp.autocast('cuda', enabled=self.use_amp and self.device.type == 'cuda'):
                if self.prediction_processor is not None:
                    outputs = self.prediction_processor(images)
                else:
                    outputs = self.model(images)
            
            # Fast output processing - keep multi-scale predictions for validation
            if isinstance(outputs, (list, tuple)):
                if not outputs:
                    raise ValueError("Model returned empty output")
                # For validation, we may need all scales - return the full list
                # The downstream components will handle multi-scale processing
                return outputs
            
            if not isinstance(outputs, torch.Tensor):
                raise ValueError(f"Expected tensor output, got {type(outputs)}")
                
            # Handle different output formats
            if outputs.dim() == 3 and outputs.size(-1) >= 6:
                predictions = outputs[..., :6]  # [B, N, 6]
            elif outputs.dim() == 3:
                # Pad if needed
                needed_features = 6 - outputs.size(-1)
                if needed_features > 0:
                    padding = torch.zeros(outputs.size(0), outputs.size(1), needed_features, device=outputs.device)
                    predictions = torch.cat([outputs, padding], dim=-1)
                else:
                    predictions = outputs
            else:
                # Try to reshape to expected format
                predictions = outputs.view(batch_size, -1, outputs.size(-1))[..., :6]
            
            return predictions
            
        except Exception as e:
            logger.error(f"Optimized inference error: {e}")
            raise RuntimeError(f"Model inference failed: {e}")
    
    def run_inference(self, images: torch.Tensor) -> torch.Tensor:
        """
        Standard model inference with validation checks.
        
        Args:
            images: Input tensor of shape [B, C, H, W]
            
        Returns:
            Tensor of shape [B, N, 6] where each detection has format:
            [x1, y1, x2, y2, confidence, class_id]
            
        Raises:
            ValueError: If input is invalid or model output is in unexpected format
            RuntimeError: If inference fails or produces invalid output
        """
        if not isinstance(images, torch.Tensor) or images.dim() != 4:
            raise ValueError(
                f"Input must be a 4D tensor [B, C, H, W], got {type(images)} with shape {getattr(images, 'shape', 'N/A')}"
            )
            
        batch_size = images.size(0)
        if batch_size == 0:
            logger.warning("Received empty batch for inference")
            return torch.zeros((0, 0, 6), device=self.device)
        
        try:
            with torch.no_grad():
                if self.use_amp:
                    with torch.amp.autocast('cuda', enabled=True):
                        if self.prediction_processor is not None:
                            outputs = self.prediction_processor(images)
                        else:
                            outputs = self.model(images)
                else:
                    if self.prediction_processor is not None:
                        outputs = self.prediction_processor(images)
                    else:
                        outputs = self.model(images)
            
            # Handle different output formats
            if isinstance(outputs, (list, tuple)):
                if not outputs:
                    raise ValueError("Model returned empty output sequence")
                outputs = outputs[0]  # Take first output if multiple returned
            
            # Ensure output is a tensor with expected shape
            if not isinstance(outputs, torch.Tensor):
                raise ValueError(
                    f"Expected model output to be a tensor, got {type(outputs).__name__}"
                )
                
            # Reshape output if needed (handles different model output formats)
            if outputs.dim() == 3 and outputs.size(-1) >= 6:
                # Already in [B, N, 6+] format
                predictions = outputs[..., :6]  # Take first 6 columns
            else:
                raise ValueError(
                    f"Unexpected output shape: {outputs.shape}. "
                    "Expected [B, N, 6+] where last dim is [x1, y1, x2, y2, conf, cls, ...]"
                )
            
            # Validate confidence and class scores
            if predictions.numel() > 0:
                conf_scores = predictions[..., 4]
                cls_scores = predictions[..., 5]
                
                if (conf_scores < 0).any() or (conf_scores > 1).any():
                    logger.warning("Model output contains confidence scores outside [0, 1] range")
                
                if (cls_scores < 0).any() or (cls_scores >= 17).any():  # 17 SmartCash classes
                    logger.warning("Model output contains invalid class indices")
            
            return predictions
            
        except RuntimeError as e:
            error_msg = f"Runtime error during model inference: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
            
        except Exception as e:
            error_msg = f"Unexpected error during model inference: {e}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from e
    
    def is_smartcash_model(self) -> bool:
        """Check if the model is a SmartCashYOLO model."""
        return hasattr(self.model, 'model') and hasattr(self.model, 'training')