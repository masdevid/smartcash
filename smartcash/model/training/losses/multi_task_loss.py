"""
File: smartcash/model/training/losses/multi_task_loss.py
Description: Multi-task loss implementation for YOLO object detection
Responsibility: Handle multi-task learning with uncertainty-based weighting
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any, Optional
import math

from smartcash.common.logger import get_logger


class MultiTaskLoss:
    """
    Multi-task loss with uncertainty-based weighting.
    Implements the approach from "Multi-Task Learning Using Uncertainty to Weigh Losses"
    """
    
    def __init__(self, num_tasks: int, init_log_vars: Optional[List[float]] = None, 
                 min_variance: float = 1e-3, max_variance: float = 10.0):
        """
        Initialize multi-task loss.
        
        Args:
            num_tasks: Number of tasks
            init_log_vars: Initial log variances (one per task)
            min_variance: Minimum allowed variance
            max_variance: Maximum allowed variance
        """
        self.logger = get_logger(__name__)
        self.min_variance = min_variance
        self.max_variance = max_variance
        
        # Initialize log variances as learnable parameters
        if init_log_vars is None:
            init_log_vars = [0.0] * num_tasks
        self.log_vars = torch.nn.Parameter(torch.FloatTensor(init_log_vars))
        
    def __call__(self, losses: List[torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute weighted multi-task loss.
        
        Args:
            losses: List of loss tensors, one per task
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        if len(losses) != len(self.log_vars):
            raise ValueError(f"Expected {len(self.log_vars)} losses, got {len(losses)}")
        
        # Compute precision (inverse of variance) for each task
        precisions = torch.exp(-self.log_vars)
        
        # Clip precisions to avoid numerical instability
        precisions = torch.clamp(precisions, 1.0/self.max_variance, 1.0/self.min_variance)
        
        # Compute weighted losses
        weighted_losses = []
        metrics = {}
        
        for i, (loss, precision) in enumerate(zip(losses, precisions)):
            # Compute weighted loss for this task
            weighted_loss = precision * loss + 0.5 * self.log_vars[i]
            weighted_losses.append(weighted_loss)
            
            # Log metrics
            metrics[f'task_{i}_loss'] = loss.item()
            metrics[f'task_{i}_weight'] = precision.item()
            metrics[f'task_{i}_log_var'] = self.log_vars[i].item()
        
        # Sum all weighted losses
        total_loss = sum(weighted_losses)
        
        # Add metrics
        metrics['total_loss'] = total_loss.item()
        metrics['avg_precision'] = precisions.mean().item()
        metrics['avg_log_var'] = self.log_vars.mean().item()
        
        return total_loss, metrics
    
    def get_weights(self) -> Dict[str, float]:
        """Get current task weights and variances"""
        weights = {}
        for i, (w, v) in enumerate(zip(torch.exp(-self.log_vars), self.log_vars)):
            weights[f'task_{i}_weight'] = w.item()
            weights[f'task_{i}_log_var'] = v.item()
        return weights
