"""
File: smartcash/model/training/losses/target_builder.py
Description: Target building and filtering logic for YOLO loss computation
Responsibility: Handle target preparation, anchor matching, and grid coordinate computation
"""

import math
import torch
from typing import List, Dict, Tuple, Union, Optional


def build_targets_for_yolo(predictions: Union[List[torch.Tensor], Tuple[torch.Tensor, ...]], 
                          targets: torch.Tensor, anchors: torch.Tensor, img_size: int,
                          logger=None) -> Tuple[List, List, List, List]:
    """
    Build targets for each prediction scale
    
    Args:
        predictions: List or tuple of predictions from different scales
        targets: [num_targets, 6] format: [batch_idx, class, x, y, w, h]
        anchors: Anchor tensors for each scale
        img_size: Input image size
        logger: Logger instance
        
    Returns:
        tcls: List of class targets for each scale
        tbox: List of box targets for each scale
        indices: List of indices for each scale
        anchors: List of anchors for each scale
    """
    # Initialize empty lists for targets
    tcls, tbox, indices, anchors_out = [], [], [], []
    
    # Convert predictions to list if it's a tuple
    if isinstance(predictions, tuple):
        predictions = list(predictions)
    
    # Ensure predictions is a list of tensors
    if not isinstance(predictions, (list, tuple)):
        predictions = [predictions]
    
    # Ensure all predictions are tensors
    tensor_predictions = []
    for p in predictions:
        if torch.is_tensor(p):
            tensor_predictions.append(p)
        else:
            try:
                # Try to convert to tensor
                if isinstance(p, (list, tuple)) and all(torch.is_tensor(x) for x in p):
                    for tensor_item in p:
                        if torch.is_tensor(tensor_item):
                            tensor_predictions.append(tensor_item)
                else:
                    tensor_predictions.append(torch.tensor(p, device=targets.device) if hasattr(targets, 'device') 
                                           else torch.tensor(p))
            except (ValueError, RuntimeError, TypeError) as e:
                if logger:
                    logger.warning(f"Failed to convert prediction to tensor: {e}")
                continue
    
    predictions = tensor_predictions
        
    # Number of anchors per scale
    na = anchors.shape[1] if hasattr(anchors, 'shape') and anchors is not None else 3
    
    # Number of positive targets
    nt = targets.shape[0] if hasattr(targets, 'shape') and len(targets.shape) > 1 else 0
    
    # Define the grid offset
    g = 0.5  # grid cell offset
    
    # Offsets for grid cells
    off = torch.tensor(
        [
            [0, 0],  # no offset
            [1, 0],  # x offset
            [0, 1],  # y offset
            [-1, 0], # negative x offset
            [0, -1]  # negative y offset
        ], 
        device=targets.device if torch.is_tensor(targets) and hasattr(targets, 'device') 
              else torch.device('cpu')
    ).float() * g  # offset
    
    # Limit processing to available anchor scales
    max_scales = len(anchors) if hasattr(anchors, 'shape') and anchors is not None else 3
    
    # Reshape predictions if necessary
    if predictions:
        reshaped_predictions = _reshape_predictions_if_necessary(predictions[0], logger)
    else:
        reshaped_predictions = []

    # Handle both single tensor and list of tensors for predictions
    if not isinstance(reshaped_predictions, list):
        reshaped_predictions = [reshaped_predictions]
        
    for i, pred in enumerate(reshaped_predictions[:max_scales]):
        # Ensure pred is a tensor
        if not torch.is_tensor(pred):
            pred = torch.tensor(pred, device=off.device)
            predictions[i] = pred
            
        # Get the shape of the current prediction
        pred_shape = pred.shape if hasattr(pred, 'shape') else torch.tensor(pred).shape

        # Reshape predictions if they are in a flattened format
        pred = _reshape_predictions_if_necessary(pred, logger)
        pred_shape = pred.shape
        
        # Skip if prediction is empty or has invalid shape
        if len(pred_shape) < 2:
            device = targets.device if hasattr(targets, 'device') else torch.device('cpu')
            tcls.append(torch.zeros(0, device=device))
            tbox.append(torch.zeros((0, 4), device=device))
            indices.append((torch.zeros(0, dtype=torch.long, device=device), 
                          torch.zeros(0, dtype=torch.long, device=device)))
            anchors_out.append(torch.zeros((0, 2), device=device))
            continue
            
        # Get the current anchors for this scale with bounds checking
        anchors_i = _get_anchors_for_scale(anchors, i, targets.device, logger)
            
        # Ensure anchors_i has the correct shape [num_anchors, 2]
        if len(anchors_i.shape) == 1:
            anchors_i = anchors_i.view(-1, 2)
        elif len(anchors_i.shape) == 3 and anchors_i.shape[0] == 1:
            anchors_i = anchors_i.squeeze(0)
        
        # Initialize gain tensor with the correct shape
        gain = torch.ones(7, device=targets.device)
        
        # Handle different prediction shapes and set gain
        gain = _set_gain_from_prediction_shape(pred_shape, gain, targets.device, logger)
        
        # Process targets if any exist
        if nt > 0:
            # Create a copy of targets with the right device
            t = targets.clone().to(targets.device)
            
            # Apply gain to normalize to grid coordinates
            t[:, 2:6] *= gain[2:6]
            
            # Match targets to anchors
            target_indices, anchor_indices, filtered_targets = _match_targets_to_anchors(
                t, anchors_i, logger
            )
            
            # Process matched targets
            if target_indices.numel() > 0:
                filtered_targets = filtered_targets[target_indices]
                a = anchor_indices
                
                # Get grid xy coordinates and indices
                gi, gj, b, c = _compute_grid_coordinates(filtered_targets, gain, logger)
                
                # Append to lists
                indices.append((b, a, gj, gi))  # image, anchor, grid y, grid x
                tbox.append(filtered_targets[:, 2:6])  # box coordinates
                anchors_out.append(anchors_i[a])  # anchors
                tcls.append(c)  # class
            else:
                # No matches, append empty tensors
                _append_empty_targets(indices, tbox, anchors_out, tcls, targets.device)
        else:
            # No targets, return empty tensors
            _append_empty_targets(indices, tbox, anchors_out, tcls, targets.device)
    
    return tcls, tbox, indices, anchors_out


def _reshape_predictions_if_necessary(pred: torch.Tensor, logger=None) -> Union[torch.Tensor, List[torch.Tensor]]:
    """
    Reshape flattened prediction tensors into the standard 5D format.
    
    Args:
        pred: Prediction tensor of various shapes
        logger: Logger instance
        
    Returns:
        Reshaped prediction tensor in 5D format or list of 5D tensors
    """
    pred_shape = pred.shape
    
    # If already in 5D format, no reshaping needed
    if len(pred_shape) == 5:
        return pred
        
    # Handle 3D tensors [batch, num_predictions, features]
    if len(pred_shape) == 3:
        # Handle specific YOLOv5 flattened formats
        if pred_shape[1] == 25200 and pred_shape[2] in [12, 22]:
            num_classes = pred_shape[2] - 5
            phase_info = "Phase 1 (7 classes)" if pred_shape[2] == 12 else "Phase 2 (17 classes)"
            if logger:
                logger.info(f"Reshaping [batch, 25200, {pred_shape[2]}] format for {phase_info}")
            
            # Split into 3 scales (80x80, 40x40, 20x20)
            yolo_grid_sizes = [80, 40, 20]
            num_anchors = 3
            
            # Calculate splits
            split_sizes = [gs * gs * num_anchors for gs in yolo_grid_sizes]
            
            # Check if total predictions match expected
            if sum(split_sizes) != pred_shape[1]:
                if logger:
                    logger.warning(f"Prediction size mismatch: expected {sum(split_sizes)}, got {pred_shape[1]}")
                return pred # Return as is if sizes don't match
            
            # Split and reshape each scale
            preds_split = torch.split(pred, split_sizes, dim=1)
            
            reshaped_preds = []
            for i, gs in enumerate(yolo_grid_sizes):
                try:
                    reshaped_scale = preds_split[i].view(
                        pred_shape[0], num_anchors, gs, gs, -1
                    ).contiguous()
                    reshaped_preds.append(reshaped_scale)
                except RuntimeError as e:
                    if logger:
                        logger.error(f"Failed to reshape scale {i} with size {gs}x{gs}: {e}")
                    return pred # Return original on error
            
            return reshaped_preds
            
    # Fallback for other shapes
    return pred


def _get_anchors_for_scale(anchors: torch.Tensor, scale_idx: int, 
                          device: torch.device, logger=None) -> torch.Tensor:
    """Get anchors for a specific scale with bounds checking"""
    if not hasattr(anchors, 'shape') or anchors is None or len(anchors) == 0:
        # If no anchors are defined, use default anchors
        default_anchors = torch.tensor([
            [[10, 13], [16, 30], [33, 23]],      # P3/8
            [[30, 61], [62, 45], [59, 119]],     # P4/16
            [[116, 90], [156, 198], [373, 326]]  # P5/32
        ], device=device).float()
        anchors_i = default_anchors[min(scale_idx, len(default_anchors) - 1)]
        if logger:
            logger.warning(f"No anchors defined. Using default anchors for scale {scale_idx}.")
    elif scale_idx >= len(anchors):
        # If index is out of bounds, use the last available anchors
        if len(anchors) > 0:
            anchors_i = anchors[-1].clone().to(device)
            if logger:
                logger.warning(f"Index {scale_idx} out of bounds for anchors with size {len(anchors)}. Using last available anchors.")
        else:
            # Fallback to default anchors if none available
            default_anchors = torch.tensor([[10, 13], [16, 30], [33, 23]], device=device).float()
            anchors_i = default_anchors
            if logger:
                logger.warning(f"No anchors available for index {scale_idx}. Using default anchors.")
    else:
        # Use the anchors for the current scale
        anchors_i = anchors[scale_idx].clone().to(device)
        
    return anchors_i


def _set_gain_from_prediction_shape(pred_shape: torch.Size, gain: torch.Tensor, 
                                   device: torch.device, logger=None) -> torch.Tensor:
    """Set gain tensor based on prediction shape"""
    if logger:
        logger.debug(f"🔍 Processing prediction tensor with shape: {pred_shape} ({len(pred_shape)}D)")
    
    if len(pred_shape) == 2:  # [num_predictions, features] - missing batch dimension
        if logger:
            logger.warning(f"🚨 Detected 2D prediction tensor {pred_shape}, adding batch dimension")
        # This case would be handled by the caller
        
    if len(pred_shape) == 5:  # [batch, anchors, grid_y, grid_x, features]
        if logger:
            logger.debug(f"📐 Processing 5D tensor: {pred_shape}")
        grid_size = pred_shape[2]  # assuming square grid
        gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                               device=device)
    elif len(pred_shape) == 4:  # [batch, channels, grid_h, grid_w] - our custom detection head format
        if logger:
            logger.debug(f"📐 Processing 4D tensor (custom detection head): {pred_shape}")
        grid_size = pred_shape[2]  # grid height (assuming square grid)
        gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                               device=device)
    elif len(pred_shape) == 3:  # [batch, num_anchors * grid_y * grid_x, features]
        if logger:
            logger.debug(f"📐 Processing 3D tensor: {pred_shape}")
        # Special handling for YOLOv5 output format
        if pred_shape[1] == 25200 and pred_shape[2] in [12, 22]:
            grid_size = 80  # Using the largest grid size
            gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                                   device=device)
        else:
            # Generic fallback for other tensor shapes
            if logger:
                logger.warning(f"Unexpected 3D tensor shape: {pred_shape}. Using generic fallback.")
            grid_size = 80  # default
            gain[2:6] = torch.tensor([grid_size, grid_size, grid_size, grid_size], 
                                   device=device)
    else:
        if logger:
            logger.error(f"🚨 Unexpected prediction shape: {pred_shape} ({len(pred_shape)}D)")
    
    return gain


def _match_targets_to_anchors(targets: torch.Tensor, anchors: torch.Tensor, 
                             logger=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Match targets to anchors based on aspect ratio"""
    # Calculate width/height ratios between targets and anchors
    target_wh = targets[:, 4:6]  # [num_targets, 2]
    r = target_wh.unsqueeze(1) / anchors.unsqueeze(0)  # [num_targets, num_anchors, 2]
    
    # Find anchors with aspect ratios close to target - More lenient for banknotes
    j = torch.max(r, 1. / r).max(-1)[0] < 6  # [num_targets, num_anchors] - Increased from 4 to 6
    
    # Get indices of matching target-anchor pairs
    target_indices, anchor_indices = torch.where(j)
    
    return target_indices, anchor_indices, targets


def _compute_grid_coordinates(targets: torch.Tensor, gain: torch.Tensor, 
                             logger=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute grid coordinates from target coordinates"""
    # Get grid xy coordinates
    gxy = targets[:, 2:4]  # grid xy [num_matches, 2]
    
    # Calculate grid indices
    gij = gxy.long()  # grid indices
    gi, gj = gij.T  # grid x, y indices
    
    # Apply grid size limits
    grid_size = int(gain[2].item())  # assuming square grid
    gi = gi.clamp(0, grid_size - 1)
    gj = gj.clamp(0, grid_size - 1)
    
    # Get the image indices and classes
    b = targets[:, 0].long()  # image index
    c = targets[:, 1].long()  # class
    
    return gi, gj, b, c


def _append_empty_targets(indices: List, tbox: List, anchors_out: List, 
                         tcls: List, device: torch.device) -> None:
    """Append empty tensors when no targets are found"""
    indices.append((torch.zeros(0, dtype=torch.long, device=device), 
                   torch.zeros(0, dtype=torch.long, device=device)))
    tbox.append(torch.zeros((0, 4), device=device))
    anchors_out.append(torch.zeros((0, 2), device=device))
    tcls.append(torch.zeros(0, device=device))


def filter_targets_for_layer(targets: torch.Tensor, layer_name: str) -> torch.Tensor:
    """
    Filter targets by layer detection classes (used by loss coordinator)
    
    Args:
        targets: Target tensor in YOLO format [image_idx, class_id, x, y, w, h]
        layer_name: Layer name (e.g., 'layer_1')
        
    Returns:
        Filtered targets tensor with remapped class IDs
    """
    if not hasattr(targets, 'shape') or targets.numel() == 0:
        return targets
    
    # Phase-aware class filtering
    layer_class_ranges = {
        'layer_1': list(range(0, 7)),    # Layer 1: Classes 0-6 (denomination detection)
        'layer_2': list(range(7, 14)),   # Layer 2: Classes 7-13 (l2_* features)
        'layer_3': list(range(14, 17)),  # Layer 3: Classes 14-16 (l3_* features) 
        # Legacy support
        'banknote': list(range(0, 7)),   # Classes 0-6
        'nominal': list(range(7, 14)),   # Classes 7-13
        'security': list(range(14, 17))  # Classes 14-16
    }
    
    valid_classes = layer_class_ranges.get(layer_name, list(range(0, 7)))   # Default to layer 1 classes
    
    # Filter targets with class matching - safer tensor operations
    if targets.shape[1] < 2:  # Need at least 2 columns for class_id
        return targets
    
    # Extract class IDs safely
    class_ids = targets[:, 1].long()  # Get all class IDs as long tensor
    
    # Create mask for valid classes
    mask = torch.zeros(targets.shape[0], dtype=torch.bool, device=targets.device)
    for class_id in valid_classes:
        mask |= (class_ids == class_id)
    
    filtered_targets = targets[mask].clone()
    
    # Remap class IDs for layer (0-based indexing for the layer)
    if filtered_targets.numel() > 0:
        class_offset = min(valid_classes)
        filtered_targets[:, 1] -= class_offset
    
    return filtered_targets