"""
File: smartcash/model/training/loss_manager.py
Description: Backwards-compatible loss manager that delegates to modular loss system
Responsibility: Maintain API compatibility while using new modular loss architecture
"""

from typing import Dict, Tuple, Any, Optional
import torch

from smartcash.common.logger import get_logger
from smartcash.model.training.losses import YOLOLoss, LossCoordinator, create_loss_coordinator


class LossManager:
    """
    Backwards-compatible loss manager that delegates to the new modular loss system.
    
    This class maintains the existing API while internally using the new SRP-based
    loss architecture for better maintainability and testability.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize loss manager with backwards compatibility
        
        Args:
            config: Training configuration dictionary
        """
        self.config = config
        self.logger = get_logger(__name__)
        
        # Initialize the new modular loss coordinator
        self._coordinator = create_loss_coordinator(config)
        
        # Expose coordinator properties for backwards compatibility
        self.current_phase = self._coordinator.current_phase
        self.box_weight = self._coordinator.box_weight
        self.obj_weight = self._coordinator.obj_weight
        self.cls_weight = self._coordinator.cls_weight
        self.focal_loss = self._coordinator.focal_loss
        self.label_smoothing = self._coordinator.label_smoothing
        self.use_dynamic_weighting = self._coordinator.use_dynamic_weighting
        self.min_variance = self._coordinator.min_variance
        self.max_variance = self._coordinator.max_variance
        
        # Backwards compatibility: expose loss functions
        self.loss_functions = self._coordinator.loss_functions
        self.use_multi_task_loss = self._coordinator.use_multi_task_loss
        
        # Multi-task loss reference for backwards compatibility
        if hasattr(self._coordinator, 'multi_task_loss'):
            self.multi_task_loss = self._coordinator.multi_task_loss
        
        self.logger.info("🔄 LossManager initialized with new modular architecture")
    
    def set_current_phase(self, phase: int) -> None:
        """
        Set the current training phase (delegates to coordinator)
        
        Args:
            phase: Training phase (1 or 2)
        """
        self._coordinator.set_current_phase(phase)
        # Update local reference for backwards compatibility
        self.current_phase = self._coordinator.current_phase
        
        # Update exposed properties
        self.loss_functions = self._coordinator.loss_functions
        self.use_multi_task_loss = self._coordinator.use_multi_task_loss
        if hasattr(self._coordinator, 'multi_task_loss'):
            self.multi_task_loss = self._coordinator.multi_task_loss
    
    def compute_loss(self, predictions: Dict[str, list], 
                    targets: torch.Tensor, img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Compute total loss for single or multilayer detection (delegates to coordinator)
        
        Args:
            predictions: Dict with format {layer_name: [pred_p3, pred_p4, pred_p5]}
            targets: Batch targets [batch_idx, class, x, y, w, h]
            img_size: Input image size
            
        Returns:
            total_loss: Combined loss
            loss_breakdown: Detailed loss components
        """
        return self._coordinator.compute_loss(predictions, targets, img_size)
    
    def get_loss_weights(self) -> Dict[str, float]:
        """Get current loss weights (delegates to coordinator)"""
        return self._coordinator.get_loss_weights()
    
    def update_loss_weights(self, box_weight: Optional[float] = None,
                           obj_weight: Optional[float] = None,
                           cls_weight: Optional[float] = None) -> None:
        """Update loss weights dynamically during training (delegates to coordinator)"""
        self._coordinator.update_loss_weights(box_weight, obj_weight, cls_weight)
        
        # Update local references for backwards compatibility
        self.box_weight = self._coordinator.box_weight
        self.obj_weight = self._coordinator.obj_weight
        self.cls_weight = self._coordinator.cls_weight
    
    def get_loss_breakdown_summary(self, loss_dict: Dict[str, Any]) -> str:
        """Get formatted summary of loss breakdown (delegates to coordinator)"""
        return self._coordinator.get_loss_breakdown_summary(loss_dict)
    
    # Backwards compatibility methods that were in the original LossManager
    def _is_multilayer_mode(self) -> bool:
        """Check if model uses multilayer detection (backwards compatibility)"""
        return self._coordinator._is_multilayer_mode()
    
    def _get_loss_params(self) -> Dict[str, Any]:
        """Get standard loss parameters (backwards compatibility)"""
        return self._coordinator._get_loss_params()


# Convenience functions for backwards compatibility
def create_loss_manager(config: Dict[str, Any]) -> LossManager:
    """Factory function for creating loss manager (backwards compatibility)"""
    return LossManager(config)


def compute_yolo_loss(predictions: Dict[str, list], targets: torch.Tensor,
                     config: Dict[str, Any], img_size: int = 640) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """One-liner for computing YOLO loss (backwards compatibility)"""
    loss_manager = LossManager(config)
    return loss_manager.compute_loss(predictions, targets, img_size)


# Export the old YOLOLoss class for backwards compatibility
# This allows existing code to import YOLOLoss from loss_manager
from smartcash.model.training.losses import YOLOLoss

__all__ = [
    'LossManager',
    'YOLOLoss',  # Backwards compatibility
    'create_loss_manager',
    'compute_yolo_loss'
]