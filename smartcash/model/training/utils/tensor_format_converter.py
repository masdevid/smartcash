#!/usr/bin/env python3
"""
Tensor Format Converter for YOLO Predictions

This module handles conversion between different YOLO prediction tensor formats:
- Flattened format: [batch, total_predictions, features] 
- YOLO format: [batch, anchors, height, width, features]
"""

import torch
import math
from typing import List, Tuple, Union
from smartcash.common.logger import get_logger

logger = get_logger(__name__)


class TensorFormatConverter:
    """Converts between different YOLO prediction tensor formats."""
    
    # Standard YOLO grid sizes for different scales
    YOLO_GRID_SIZES = [80, 40, 20]  # P3, P4, P5
    NUM_ANCHORS = 3
    
    @staticmethod
    def flatten_to_yolo_format(predictions: Union[torch.Tensor, List[torch.Tensor]], 
                              img_size: int = 640) -> List[torch.Tensor]:
        """
        Convert flattened predictions to YOLO 5D format.
        
        Args:
            predictions: Flattened tensor [batch, total_predictions, features] or list of tensors
            img_size: Input image size
            
        Returns:
            List of tensors in YOLO format [batch, anchors, height, width, features]
        """
        if isinstance(predictions, list):
            # Already in multi-scale format
            return [TensorFormatConverter._ensure_yolo_format(pred, img_size) for pred in predictions]
        
        # Single flattened tensor - split into multiple scales
        return TensorFormatConverter._split_flattened_tensor(predictions, img_size)
    
    @staticmethod
    def _ensure_yolo_format(tensor: torch.Tensor, img_size: int = 640) -> torch.Tensor:
        """Ensure tensor is in YOLO 5D format."""
        if tensor.dim() == 5:
            # Already in correct format [batch, anchors, height, width, features]
            return tensor
        elif tensor.dim() == 3:
            # Flattened format [batch, total_predictions, features] 
            return TensorFormatConverter._reshape_3d_to_5d(tensor, img_size)
        elif tensor.dim() == 2:
            # Fully flattened [batch*total_predictions, features]
            # This is more complex and would need batch size info
            raise ValueError(f"Cannot convert 2D tensor without batch size information: {tensor.shape}")
        else:
            raise ValueError(f"Unsupported tensor dimensions: {tensor.dim()}")
    
    @staticmethod
    def _reshape_3d_to_5d(tensor: torch.Tensor, img_size: int) -> torch.Tensor:
        """
        Reshape 3D flattened tensor to 5D YOLO format.
        
        Args:
            tensor: Input tensor [batch, total_predictions, features]
            img_size: Input image size
            
        Returns:
            Tensor in format [batch, anchors, height, width, features]
        """
        batch_size, total_predictions, num_features = tensor.shape
        
        # Check if this is a YOLOv5 multi-scale format (25200 = 3 * (80*80 + 40*40 + 20*20))
        yolo_grid_sizes = TensorFormatConverter.YOLO_GRID_SIZES  # [80, 40, 20]
        yolo_total_predictions = sum(gs * gs for gs in yolo_grid_sizes) * TensorFormatConverter.NUM_ANCHORS
        
        if total_predictions == yolo_total_predictions:
            # This is a YOLOv5 multi-scale output, split and reshape appropriately
            logger.debug(f"Detected YOLOv5 multi-scale format with {total_predictions} predictions")
            # Split into separate scales and return the first one (largest grid)
            # In practice, this should be handled by _split_flattened_tensor, but we'll handle it here too
            grid_size = max(yolo_grid_sizes)  # Use 80x80 grid (largest)
            predictions_for_largest_grid = grid_size * grid_size * TensorFormatConverter.NUM_ANCHORS
            
            # Take only the first portion for the largest grid
            tensor_subset = tensor[:, :predictions_for_largest_grid, :]
            return tensor_subset.view(batch_size, TensorFormatConverter.NUM_ANCHORS,
                                    grid_size, grid_size, num_features)
        
        # Calculate the most likely grid size for non-YOLOv5 formats
        predictions_per_anchor = total_predictions // TensorFormatConverter.NUM_ANCHORS
        grid_size = int(math.sqrt(predictions_per_anchor))
        
        # Validate that this makes sense
        expected_predictions = grid_size * grid_size * TensorFormatConverter.NUM_ANCHORS
        
        if expected_predictions != total_predictions:
            # Try to find the best matching grid size
            grid_size = TensorFormatConverter._find_best_grid_size(total_predictions)
            expected_predictions = grid_size * grid_size * TensorFormatConverter.NUM_ANCHORS
            
            if expected_predictions != total_predictions:
                logger.warning(f"Cannot perfectly reshape {total_predictions} predictions into grid format. "
                             f"Using grid_size={grid_size}, expected={expected_predictions}")
                
                # Truncate or pad to match expected size
                if total_predictions > expected_predictions:
                    tensor = tensor[:, :expected_predictions, :]
                else:
                    # Pad with zeros
                    padding_size = expected_predictions - total_predictions
                    padding = torch.zeros(batch_size, padding_size, num_features, 
                                        device=tensor.device, dtype=tensor.dtype)
                    tensor = torch.cat([tensor, padding], dim=1)
                
                total_predictions = expected_predictions
        
        # Reshape to 5D format
        try:
            reshaped = tensor.view(batch_size, TensorFormatConverter.NUM_ANCHORS, 
                                 grid_size, grid_size, num_features)
            logger.debug(f"Successfully reshaped {tensor.shape} to {reshaped.shape}")
            return reshaped
            
        except RuntimeError as e:
            logger.error(f"Failed to reshape tensor {tensor.shape} to "
                        f"[{batch_size}, {TensorFormatConverter.NUM_ANCHORS}, {grid_size}, {grid_size}, {num_features}]: {e}")
            
            # Fallback: create a tensor with standard grid size
            fallback_grid = 80  # Use largest standard grid
            fallback_predictions = fallback_grid * fallback_grid * TensorFormatConverter.NUM_ANCHORS
            
            if total_predictions >= fallback_predictions:
                # Use first portion of predictions
                truncated = tensor[:, :fallback_predictions, :]
                return truncated.view(batch_size, TensorFormatConverter.NUM_ANCHORS,
                                    fallback_grid, fallback_grid, num_features)
            else:
                # Create smaller grid that fits
                smaller_grid = int(math.sqrt(total_predictions // TensorFormatConverter.NUM_ANCHORS))
                smaller_grid = max(1, smaller_grid)  # Ensure at least 1
                
                smaller_predictions = smaller_grid * smaller_grid * TensorFormatConverter.NUM_ANCHORS
                if smaller_predictions <= total_predictions:
                    truncated = tensor[:, :smaller_predictions, :]
                    return truncated.view(batch_size, TensorFormatConverter.NUM_ANCHORS,
                                        smaller_grid, smaller_grid, num_features)
                
                # Final fallback: return original tensor with adjusted dimensions
                return tensor.unsqueeze(1)  # Add anchor dimension
    
    @staticmethod
    def _find_best_grid_size(total_predictions: int) -> int:
        """Find the best grid size for given total predictions."""
        predictions_per_anchor = total_predictions // TensorFormatConverter.NUM_ANCHORS
        
        # Try standard YOLO grid sizes first
        for grid_size in TensorFormatConverter.YOLO_GRID_SIZES:
            if grid_size * grid_size == predictions_per_anchor:
                return grid_size
        
        # Find closest square root
        grid_size = int(math.sqrt(predictions_per_anchor))
        
        # Prefer powers of 2 or standard sizes
        standard_sizes = [1, 2, 4, 8, 10, 13, 16, 20, 26, 32, 40, 52, 80]
        
        # Find closest standard size
        best_size = min(standard_sizes, key=lambda x: abs(x - grid_size))
        
        return best_size
    
    @staticmethod
    def _split_flattened_tensor(tensor: torch.Tensor, img_size: int) -> List[torch.Tensor]:
        """
        Split a single flattened tensor into multiple scale tensors.
        
        This is used when the model outputs a single concatenated tensor
        instead of separate tensors for each scale.
        """
        batch_size, total_predictions, num_features = tensor.shape
        
        # Calculate expected predictions for each scale
        scale_predictions = []
        for grid_size in TensorFormatConverter.YOLO_GRID_SIZES:
            scale_preds = grid_size * grid_size * TensorFormatConverter.NUM_ANCHORS
            scale_predictions.append(scale_preds)
        
        total_expected = sum(scale_predictions)
        
        if total_expected != total_predictions:
            logger.warning(f"Total predictions {total_predictions} doesn't match expected {total_expected}. "
                          "Using single scale output.")
            # Return as single scale
            return [TensorFormatConverter._reshape_3d_to_5d(tensor, img_size)]
        
        # Split tensor into multiple scales
        result = []
        start_idx = 0
        
        for i, (grid_size, num_preds) in enumerate(zip(TensorFormatConverter.YOLO_GRID_SIZES, scale_predictions)):
            end_idx = start_idx + num_preds
            scale_tensor = tensor[:, start_idx:end_idx, :]
            
            # Reshape to 5D format
            try:
                reshaped = scale_tensor.view(batch_size, TensorFormatConverter.NUM_ANCHORS,
                                           grid_size, grid_size, num_features)
                result.append(reshaped)
                logger.debug(f"Scale {i+1} ({grid_size}x{grid_size}): reshaped {scale_tensor.shape} to {reshaped.shape}")
            except RuntimeError as e:
                logger.error(f"Failed to reshape scale {i+1} tensor {scale_tensor.shape} to "
                           f"[{batch_size}, {TensorFormatConverter.NUM_ANCHORS}, {grid_size}, {grid_size}, {num_features}]: {e}")
                
                # Calculate what the tensor should be and provide detailed error info
                expected_elements = batch_size * TensorFormatConverter.NUM_ANCHORS * grid_size * grid_size * num_features
                actual_elements = scale_tensor.numel()
                logger.error(f"Expected elements: {expected_elements}, Actual elements: {actual_elements}")
                
                # Try a fallback approach - use the largest possible grid size that fits
                max_spatial_size = int((actual_elements // (batch_size * TensorFormatConverter.NUM_ANCHORS * num_features)) ** 0.5)
                if max_spatial_size > 0:
                    logger.warning(f"Using fallback grid size {max_spatial_size} instead of {grid_size}")
                    reshaped = scale_tensor.view(batch_size, TensorFormatConverter.NUM_ANCHORS,
                                               max_spatial_size, max_spatial_size, num_features)
                    result.append(reshaped)
                else:
                    # If all else fails, skip this scale
                    logger.error(f"Cannot reshape scale {i+1}, skipping...")
                    continue
            start_idx = end_idx
        
        logger.debug(f"Split flattened tensor {tensor.shape} into {len(result)} scales")
        return result
    
    @staticmethod
    def yolo_to_flatten_format(predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Convert YOLO 5D format to flattened format.
        
        Args:
            predictions: List of tensors in YOLO format [batch, anchors, height, width, features]
            
        Returns:
            Flattened tensor [batch, total_predictions, features]
        """
        if not predictions:
            raise ValueError("Empty predictions list")
        
        batch_size = predictions[0].shape[0]
        num_features = predictions[0].shape[-1]
        
        flattened_preds = []
        
        for pred in predictions:
            # Flatten spatial and anchor dimensions
            batch, anchors, height, width, features = pred.shape
            flattened = pred.view(batch, anchors * height * width, features)
            flattened_preds.append(flattened)
        
        # Concatenate all scales
        result = torch.cat(flattened_preds, dim=1)
        logger.debug(f"Converted {len(predictions)} YOLO tensors to flattened {result.shape}")
        
        return result
    
    @staticmethod
    def convert_predictions_for_loss(predictions: Union[torch.Tensor, List[torch.Tensor]], 
                                   img_size: int = 640) -> List[torch.Tensor]:
        """
        Convert predictions to the format expected by YOLOLoss.
        
        Args:
            predictions: Model predictions in any supported format
            img_size: Input image size
            
        Returns:
            List of tensors in YOLO format ready for loss computation
        """
        try:
            if isinstance(predictions, torch.Tensor):
                if predictions.dim() == 3:
                    # Flattened format - needs conversion
                    # Check if this is YOLOv5 multi-scale format (25200 predictions)
                    batch_size, total_predictions, num_features = predictions.shape
                    yolo_grid_sizes = TensorFormatConverter.YOLO_GRID_SIZES  # [80, 40, 20]
                    yolo_total_predictions = sum(gs * gs for gs in yolo_grid_sizes) * TensorFormatConverter.NUM_ANCHORS
                    
                    if total_predictions == yolo_total_predictions:
                        # This is definitely a YOLOv5 multi-scale output, split it properly
                        logger.debug(f"Converting YOLOv5 multi-scale predictions {predictions.shape} to YOLO format")
                        return TensorFormatConverter._split_flattened_tensor(predictions, img_size)
                    else:
                        # Regular flattened format
                        logger.debug(f"Converting flattened predictions {predictions.shape} to YOLO format")
                        return TensorFormatConverter.flatten_to_yolo_format(predictions, img_size)
                elif predictions.dim() == 5:
                    # Already in YOLO format
                    return [predictions]
                elif predictions.dim() == 2:
                    # 2D tensor - missing batch dimension, add it
                    logger.warning(f"🚨 Tensor format converter: Detected 2D tensor {predictions.shape}, adding batch dimension")
                    # Add batch dimension and process as 3D
                    predictions_3d = predictions.unsqueeze(0)  # [1, num_predictions, features]
                    logger.info(f"✅ Tensor format converter: Fixed to 3D tensor: {predictions_3d.shape}")
                    
                    # Now process as 3D tensor
                    batch_size, total_predictions, num_features = predictions_3d.shape
                    yolo_grid_sizes = TensorFormatConverter.YOLO_GRID_SIZES  # [80, 40, 20]
                    yolo_total_predictions = sum(gs * gs for gs in yolo_grid_sizes) * TensorFormatConverter.NUM_ANCHORS
                    
                    if total_predictions == yolo_total_predictions:
                        # This is definitely a YOLOv5 multi-scale output, split it properly
                        logger.debug(f"Converting YOLOv5 multi-scale predictions {predictions_3d.shape} to YOLO format")
                        return TensorFormatConverter._split_flattened_tensor(predictions_3d, img_size)
                    else:
                        # Regular flattened format
                        logger.debug(f"Converting flattened predictions {predictions_3d.shape} to YOLO format")
                        return TensorFormatConverter.flatten_to_yolo_format(predictions_3d, img_size)
                else:
                    logger.warning(f"Unexpected prediction tensor dimensions: {predictions.dim()}")
                    return [predictions]
            
            elif isinstance(predictions, (list, tuple)):
                # List of tensors - ensure each is in correct format
                result = []
                
                # Flatten the list just in case it's nested
                flat_predictions = []
                for item in predictions:
                    if isinstance(item, (list, tuple)):
                        flat_predictions.extend(item)
                    else:
                        flat_predictions.append(item)

                for pred in flat_predictions:
                    if isinstance(pred, torch.Tensor):
                        converted = TensorFormatConverter._ensure_yolo_format(pred, img_size)
                        result.append(converted)
                    else:
                        # Log if it's still not a tensor after flattening
                        if pred is not None:
                            logger.warning(f"Skipping non-tensor prediction after flattening: {type(pred)}")
                
                return result
            
            else:
                logger.error(f"Unsupported prediction type: {type(predictions)}")
                return []
                
        except Exception as e:
            logger.error(f"Error converting predictions format: {e}")
            # Return empty list to avoid breaking the training
            return []


# Convenience functions
def convert_for_yolo_loss(predictions: Union[torch.Tensor, List[torch.Tensor]], 
                         img_size: int = 640) -> List[torch.Tensor]:
    """Convert predictions to format expected by YOLOLoss."""
    return TensorFormatConverter.convert_predictions_for_loss(predictions, img_size)


def ensure_yolo_format(tensor: torch.Tensor, img_size: int = 640) -> torch.Tensor:
    """Ensure single tensor is in YOLO 5D format.""" 
    return TensorFormatConverter._ensure_yolo_format(tensor, img_size)