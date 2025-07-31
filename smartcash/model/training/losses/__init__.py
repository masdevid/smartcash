"""
SmartCash Model Training Losses Module

This module provides modular loss functions following Single Responsibility Principle:

- base_loss.py: Core YOLO loss implementation with IoU, classification, and objectness losses
- target_builder.py: Target preparation, anchor matching, and grid coordinate computation
- loss_coordinator.py: High-level loss coordination and multi-task loss management

Usage:
    from smartcash.model.training.losses import YOLOLoss, LossCoordinator
    from smartcash.model.training.losses import create_loss_coordinator
"""

from .base_loss import YOLOLoss
from .loss_coordinator import LossCoordinator, create_loss_coordinator, compute_coordinated_loss
from .target_builder import build_targets_for_yolo, filter_targets_for_layer

__all__ = [
    'YOLOLoss',
    'LossCoordinator', 
    'create_loss_coordinator',
    'compute_coordinated_loss',
    'build_targets_for_yolo',
    'filter_targets_for_layer'
]