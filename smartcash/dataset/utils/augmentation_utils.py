"""
File: smartcash/dataset/utils/augmentation_utils.py (Updated)
Deskripsi: Extended augmentation utils dengan inference-specific pipeline dan normalization
"""

import albumentations as A
import cv2
import numpy as np
import torch
from typing import Dict, Any, List, Tuple, Optional


def create_inference_augmentation_pipeline(img_size: int = 416, 
                                         normalize: bool = True) -> A.Compose:
    """
    Create augmentation pipeline khusus untuk inference dengan normalization yang tepat.
    
    Args:
        img_size: Target image size untuk model
        normalize: Apakah apply normalization (default True untuk inference)
        
    Returns:
        Albumentations compose pipeline untuk inference
    """
    
    # Base transforms untuk inference - lebih conservative dari training
    transforms = [
        # Resize dengan maintain aspect ratio
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
        
        # Light augmentations untuk test-time augmentation (TTA) - optional
        A.OneOf([
            A.NoOp(p=0.7),  # 70% chance tidak ada augmentasi tambahan
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=5, p=0.3)
        ], p=1.0),
        
        # Normalization untuk YOLOv5 (ImageNet stats)
        A.Normalize(
            mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225], 
            max_pixel_value=255.0,
            p=1.0
        ) if normalize else A.NoOp(p=1.0)
    ]
    
    return A.Compose(transforms)

def create_test_time_augmentation_pipeline(img_size: int = 416, 
                                         augmentation_level: str = 'light') -> List[A.Compose]:
    """
    Create multiple augmentation pipelines untuk Test-Time Augmentation (TTA).
    
    Args:
        img_size: Target image size
        augmentation_level: Level augmentasi ('light', 'medium', 'heavy')
        
    Returns:
        List of augmentation pipelines untuk TTA
    """
    
    # Base pipeline (no augmentation)
    base_pipeline = create_inference_augmentation_pipeline(img_size, normalize=True)
    
    pipelines = [base_pipeline]
    
    if augmentation_level == 'light':
        # Light TTA - subtle variations
        additional_transforms = [
            # Horizontal flip
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                           max_pixel_value=255.0, p=1.0)
            ]),
            
            # HSV adjustment
            A.Compose([
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=1.0),
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                           max_pixel_value=255.0, p=1.0)
            ])
        ]
        
    else:  # heavy
        # Heavy TTA - maximum variations
        additional_transforms = [
            # Multiple combinations
            A.Compose([
                A.HorizontalFlip(p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=20, val_shift_limit=15, p=1.0),
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                           max_pixel_value=255.0, p=1.0)
            ]),
            
            # Rotation + scale
            A.Compose([
                A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR, border_mode=cv2.BORDER_CONSTANT, 
                        value=(114, 114, 114), p=1.0),
                A.RandomScale(scale_limit=0.15, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
                A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                             border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                           max_pixel_value=255.0, p=1.0)
            ])
        ]
    
    pipelines.extend(additional_transforms)
    return pipelines

def apply_tta_inference(model, image: np.ndarray, tta_pipelines: List[A.Compose], 
                       conf_threshold: float = 0.25, iou_threshold: float = 0.45) -> Dict[str, Any]:
    """
    Apply Test-Time Augmentation untuk inference dengan ensemble predictions.
    
    Args:
        model: YOLOv5 model
        image: Input image (numpy array)
        tta_pipelines: List of augmentation pipelines
        conf_threshold: Confidence threshold
        iou_threshold: IoU threshold untuk NMS
        
    Returns:
        Ensemble prediction results
    """
    import torch
    
    device = next(model.parameters()).device
    all_predictions = []
    
    # Apply each TTA pipeline
    for i, pipeline in enumerate(tta_pipelines):
        # Apply augmentation
        augmented = pipeline(image=image)
        aug_image = augmented['image']
        
        # Convert to tensor dan add batch dimension
        if isinstance(aug_image, np.ndarray):
            aug_tensor = torch.from_numpy(aug_image).permute(2, 0, 1).unsqueeze(0).float()
        else:
            aug_tensor = aug_image.unsqueeze(0)
        
        aug_tensor = aug_tensor.to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(aug_tensor)
            
            # Apply NMS
            from smartcash.ui.evaluation.handlers.evaluation_handler import apply_nms
            processed = apply_nms(outputs, {
                'conf_thresh': conf_threshold,
                'iou_thresh': iou_threshold
            })
            
            # Reverse augmentation pada predictions jika diperlukan
            if i == 1 and len(tta_pipelines) > 1:  # Horizontal flip case
                processed = reverse_horizontal_flip_predictions(processed, aug_image.shape)
            
            all_predictions.extend(processed)
    
    # Ensemble predictions dengan NMS
    if all_predictions:
        ensemble_result = ensemble_predictions(all_predictions, iou_threshold)
        return {'success': True, 'predictions': ensemble_result}
    else:
        return {'success': False, 'predictions': []}

def reverse_horizontal_flip_predictions(predictions: List, image_shape: Tuple[int, int, int]) -> List:
    """Reverse horizontal flip pada predictions untuk TTA"""
    
    reversed_preds = []
    height, width = image_shape[:2]
    
    for pred in predictions:
        if pred is None or len(pred) == 0:
            reversed_preds.append(pred)
            continue
        
        # Clone prediction tensor
        reversed_pred = pred.clone()
        
        # Reverse x coordinates untuk horizontal flip
        # YOLO format: [x_center, y_center, width, height, conf, class]
        reversed_pred[:, 0] = width - pred[:, 0]  # x_center
        
        reversed_preds.append(reversed_pred)
    
    return reversed_preds

def ensemble_predictions(all_predictions: List, iou_threshold: float = 0.45) -> List:
    """Ensemble multiple predictions dengan Weighted Box Fusion atau NMS"""
    
    try:
        import torch
        
        if not all_predictions:
            return []
        
        # Flatten all predictions
        flattened_preds = []
        for pred_batch in all_predictions:
            if pred_batch is None:
                continue
            if isinstance(pred_batch, list):
                flattened_preds.extend(pred_batch)
            else:
                flattened_preds.append(pred_batch)
        
        if not flattened_preds:
            return []
        
        # Concatenate all valid predictions
        valid_preds = [p for p in flattened_preds if p is not None and len(p) > 0]
        
        if not valid_preds:
            return []
        
        # Simple ensemble: concatenate dan apply NMS
        combined = torch.cat(valid_preds, dim=0)
        
        # Apply final NMS untuk remove duplicates
        import torchvision.ops as ops
        
        if len(combined) == 0:
            return []
        
        boxes = combined[:, :4]
        scores = combined[:, 4]
        
        # Apply NMS
        keep = ops.nms(boxes, scores, iou_threshold)
        ensemble_result = combined[keep]
        
        return [ensemble_result]  # Return as list untuk consistency
        
    except Exception:
        # Fallback: return first valid prediction
        for pred_batch in all_predictions:
            if pred_batch is not None and len(pred_batch) > 0:
                return pred_batch
        return []

def preprocess_image_for_yolo(image: np.ndarray, img_size: int = 416, 
                             normalize: bool = True) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """
    Preprocess single image untuk YOLOv5 inference dengan proper scaling dan padding.
    
    Args:
        image: Input image (BGR atau RGB numpy array)
        img_size: Target size untuk model
        normalize: Apply normalization
        
    Returns:
        Tuple of (preprocessed tensor, metadata untuk post-processing)
    """
    # Store original dimensions
    original_shape = image.shape[:2]  # (height, width)
    
    # Convert BGR to RGB jika diperlukan
    if len(image.shape) == 3 and image.shape[2] == 3:
        # Assume BGR dari OpenCV, convert ke RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image.copy()
    
    # Padding ke persegi
    height, width = image_rgb.shape[:2]
    max_dim = max(height, width)
    square_img = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    y_offset, x_offset = (max_dim - height) // 2, (max_dim - width) // 2
    square_img[y_offset:y_offset+height, x_offset:x_offset+width] = image_rgb
    
    # Resize ke target size
    resized_img = cv2.resize(square_img, (img_size, img_size))
    
    # Normalisasi gambar
    if normalize:
        # Normalisasi 0-255 -> 0-1
        normalized_img = resized_img.astype(np.float32) / 255.0
        # Normalisasi dengan mean dan std ImageNet
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        normalized_img = (normalized_img - mean) / std
    else:
        normalized_img = resized_img.astype(np.float32) / 255.0
    
    # Convert ke tensor
    tensor = torch.from_numpy(normalized_img).permute(2, 0, 1).float().unsqueeze(0)
    
    # Metadata untuk post-processing
    metadata = {
        'original_shape': original_shape,
        'processed_shape': (img_size, img_size),
        'scale_factor': min(img_size / original_shape[0], img_size / original_shape[1]),
        'padding': calculate_padding(original_shape, img_size)
    }
    
    return tensor, metadata

def calculate_padding(original_shape: Tuple[int, int], target_size: int) -> Dict[str, int]:
    """Calculate padding yang diapply selama preprocessing"""
    
    h, w = original_shape
    scale = min(target_size / h, target_size / w)
    
    new_h, new_w = int(h * scale), int(w * scale)
    
    pad_h = target_size - new_h
    pad_w = target_size - new_w
    
    return {
        'top': pad_h // 2,
        'bottom': pad_h - (pad_h // 2),
        'left': pad_w // 2,
        'right': pad_w - (pad_w // 2)
    }

def postprocess_yolo_predictions(predictions: torch.Tensor, metadata: Dict[str, Any], 
                                conf_threshold: float = 0.25) -> List[Dict[str, Any]]:
    """
    Post-process YOLOv5 predictions untuk convert kembali ke original image coordinates.
    
    Args:
        predictions: Raw model predictions
        metadata: Metadata dari preprocessing
        conf_threshold: Confidence threshold
        
    Returns:
        List of processed detections dalam original image coordinates
    """
    
    if predictions is None or len(predictions) == 0:
        return []
    
    # Filter by confidence
    mask = predictions[:, 4] > conf_threshold
    filtered_preds = predictions[mask]
    
    if len(filtered_preds) == 0:
        return []
    
    # Convert coordinates ke original image space
    scale_factor = metadata['scale_factor']
    padding = metadata['padding']
    original_shape = metadata['original_shape']
    
    processed_detections = []
    
    for detection in filtered_preds:
        # Extract detection info
        x_center, y_center, width, height = detection[:4]
        confidence = detection[4]
        class_id = int(detection[5]) if len(detection) > 5 else 0
        
        # Remove padding
        x_center_unpadded = x_center - padding['left']
        y_center_unpadded = y_center - padding['top']
        
        # Scale back ke original size
        x_center_orig = x_center_unpadded / scale_factor
        y_center_orig = y_center_unpadded / scale_factor
        width_orig = width / scale_factor
        height_orig = height / scale_factor
        
        # Convert ke bbox format [x1, y1, x2, y2]
        x1 = max(0, x_center_orig - width_orig / 2)
        y1 = max(0, y_center_orig - height_orig / 2)
        x2 = min(original_shape[1], x_center_orig + width_orig / 2)
        y2 = min(original_shape[0], y_center_orig + height_orig / 2)
        
        processed_detections.append({
            'bbox': [float(x1), float(y1), float(x2), float(y2)],
            'confidence': float(confidence),
            'class_id': class_id,
            'center': [float(x_center_orig), float(y_center_orig)],
            'size': [float(width_orig), float(height_orig)]
        })
    
    return processed_detections

def denormalize_image(normalized_image: np.ndarray, 
                     mean: List[float] = [0.485, 0.456, 0.406],
                     std: List[float] = [0.229, 0.224, 0.225]) -> np.ndarray:
    """
    Denormalize image yang sudah di-normalize untuk visualization.
    
    Args:
        normalized_image: Normalized image array
        mean: Normalization mean values
        std: Normalization std values
        
    Returns:
        Denormalized image array (0-255 range)
    """
    
    # Clone array
    denorm_image = normalized_image.copy()
    
    # Denormalize
    for i in range(3):  # RGB channels
        denorm_image[:, :, i] = denorm_image[:, :, i] * std[i] + mean[i]
    
    # Clip ke range [0, 1] dan convert ke [0, 255]
    denorm_image = np.clip(denorm_image, 0, 1)
    denorm_image = (denorm_image * 255).astype(np.uint8)
    
    return denorm_image

# Currency-specific augmentation untuk SmartCash
def create_currency_inference_pipeline(img_size: int = 416, 
                                     lighting_robust: bool = True,
                                     rotation_robust: bool = False) -> A.Compose:
    """
    Create inference pipeline khusus untuk currency detection dengan enhanced robustness.
    
    Args:
        img_size: Target image size
        lighting_robust: Enhanced lighting robustness untuk berbagai kondisi pencahayaan
        rotation_robust: Robustness terhadap slight rotations (currency bisa miring)
        
    Returns:
        Specialized augmentation pipeline untuk currency inference
    """
    
    transforms = [
        # Base resize dan padding
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0)
    ]
    
    # Enhanced lighting robustness untuk currency detection
    if lighting_robust:
        transforms.extend([
            A.OneOf([
                A.NoOp(p=0.6),  # 60% original
                A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.2),  # Contrast enhancement
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2)
            ], p=1.0)
        ])
    
    # Slight rotation robustness jika enabled
    if rotation_robust:
        transforms.append(
            A.OneOf([
                A.NoOp(p=0.8),  # 80% no rotation
                A.Rotate(limit=2, interpolation=cv2.INTER_LINEAR, 
                        border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=0.2)
            ], p=1.0)
        )
    
    # Final normalization
    transforms.append(
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, p=1.0)
    )
    
    return A.Compose(transforms)

def create_multi_scale_inference_pipeline(base_size: int = 416, 
                                        scale_factors: List[float] = [0.8, 1.0, 1.2]) -> List[A.Compose]:
    """Create multi-scale inference pipelines untuk improved detection"""
    
    pipelines = []
    for scale in scale_factors:
        scaled_size = int(base_size * scale)
        pipeline = A.Compose([
            A.LongestMaxSize(max_size=scaled_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=scaled_size, min_width=scaled_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.Resize(height=base_size, width=base_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                       max_pixel_value=255.0, p=1.0)
        ])
        pipelines.append(pipeline)
    
    return pipelines

def batch_inference_with_augmentation(model, image_paths: List[str], aug_pipeline: A.Compose,
                                    batch_size: int = 16, device: str = 'cuda') -> List[Dict[str, Any]]:
    """Batch inference dengan augmentation pipeline"""
    import torch
    from torch.utils.data import DataLoader
    
    # Create dataset
    dataset = InferenceDataset(image_paths, aug_pipeline, 416)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    results = []
    model.eval()
    
    with torch.no_grad():
        for batch_images, batch_paths in dataloader:
            batch_images = batch_images.to(device)
            outputs = model(batch_images)
            
            # Process each image in batch
            for i, (output, path) in enumerate(zip(outputs, batch_paths)):
                results.append({
                    'image_path': path,
                    'predictions': output,
                    'processed': True
                })
    
    return results

class InferenceDataset:
    """Dataset class untuk inference dengan augmentation support"""
    
    def __init__(self, image_paths: List[str], aug_pipeline: Optional[A.Compose], img_size: int = 416):
        self.image_paths = image_paths
        self.aug_pipeline = aug_pipeline
        self.img_size = img_size
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        img_path = self.image_paths[idx]
        
        # Load image
        image = cv2.imread(img_path)
        if image is None:
            # Fallback: create empty image
            image = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Apply augmentation pipeline
        if self.aug_pipeline:
            augmented = self.aug_pipeline(image=image)
            image = augmented['image']
        else:
            # Default preprocessing jika tidak ada pipeline
            image = cv2.resize(image, (self.img_size, self.img_size))
            image = image.astype(np.float32) / 255.0
        
        # Convert to tensor
        if isinstance(image, np.ndarray):
            image_tensor = torch.from_numpy(image).permute(2, 0, 1).float()
        else:
            image_tensor = image
        
        return image_tensor, img_path

def create_robust_inference_transforms(img_size: int = 416, 
                                     robustness_level: str = 'medium') -> Dict[str, A.Compose]:
    """Create set of robust transforms untuk different evaluation scenarios"""
    
    transforms = {}
    
    # Base transform
    transforms['base'] = A.Compose([
        A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
        A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                     border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                   max_pixel_value=255.0, p=1.0)
    ])
    
    if robustness_level in ['medium', 'high']:
        # Lighting robust transform
        transforms['lighting'] = A.Compose([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=1.0),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                       max_pixel_value=255.0, p=1.0)
        ])
        
        # Contrast enhanced transform
        transforms['contrast'] = A.Compose([
            A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=1.0),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                       max_pixel_value=255.0, p=1.0)
        ])
    
    if robustness_level == 'high':
        # Flip transform
        transforms['flip'] = A.Compose([
            A.HorizontalFlip(p=1.0),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                       max_pixel_value=255.0, p=1.0)
        ])
        
        # Rotation transform  
        transforms['rotation'] = A.Compose([
            A.Rotate(limit=5, interpolation=cv2.INTER_LINEAR, 
                    border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.LongestMaxSize(max_size=img_size, interpolation=cv2.INTER_LINEAR, p=1.0),
            A.PadIfNeeded(min_height=img_size, min_width=img_size, 
                         border_mode=cv2.BORDER_CONSTANT, value=(114, 114, 114), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], 
                       max_pixel_value=255.0, p=1.0)
        ])
    
    return transforms

def apply_inference_with_tta_ensemble(model, image: np.ndarray, 
                                    transforms_dict: Dict[str, A.Compose],
                                    conf_threshold: float = 0.25,
                                    iou_threshold: float = 0.45) -> Dict[str, Any]:
    """Apply inference dengan TTA ensemble using multiple transforms"""
    import torch
    
    device = next(model.parameters()).device
    all_predictions = []
    
    for transform_name, transform in transforms_dict.items():
        # Apply transform
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # Convert to tensor
        if isinstance(aug_image, np.ndarray):
            tensor = torch.from_numpy(aug_image).permute(2, 0, 1).unsqueeze(0).float()
        else:
            tensor = aug_image.unsqueeze(0)
        
        tensor = tensor.to(device)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor)
            
            # Apply NMS
            processed = apply_nms_simple(outputs, conf_threshold, iou_threshold)
            
            # Reverse transform effects if needed
            if transform_name == 'flip':
                processed = reverse_horizontal_flip_predictions(processed, aug_image.shape)
            elif transform_name == 'rotation':
                # Could implement rotation reversal here
                pass
            
            all_predictions.extend(processed)
    
    # Ensemble all predictions
    ensemble_result = ensemble_predictions(all_predictions, iou_threshold)
    
    return {
        'success': True,
        'predictions': ensemble_result,
        'individual_results': len(all_predictions),
        'ensemble_count': len(ensemble_result[0]) if ensemble_result else 0
    }

def apply_nms_simple(outputs, conf_threshold: float, iou_threshold: float):
    """Simple NMS application untuk predictions"""
    import torch
    import torchvision.ops as ops
    
    processed = []
    
    for output in outputs:
        if output is None or len(output) == 0:
            processed.append(torch.empty((0, 6)))
            continue
        
        # Filter by confidence
        mask = output[:, 4] > conf_threshold
        filtered = output[mask]
        
        if len(filtered) == 0:
            processed.append(torch.empty((0, 6)))
            continue
        
        # Apply NMS
        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        keep = ops.nms(boxes, scores, iou_threshold)
        
        processed.append(filtered[keep])
    
    return processed

def visualize_inference_augmentations(image: np.ndarray, 
                                    transforms_dict: Dict[str, A.Compose],
                                    save_path: Optional[str] = None) -> None:
    """Visualize effects of different augmentation transforms"""
    import matplotlib.pyplot as plt
    
    n_transforms = len(transforms_dict)
    cols = min(4, n_transforms + 1)  # +1 for original
    rows = (n_transforms + 1 + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    if rows == 1:
        axes = axes.reshape(1, -1)
    
    # Show original
    axes[0, 0].imshow(image)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    # Show transformed versions
    for idx, (name, transform) in enumerate(transforms_dict.items(), 1):
        row, col = divmod(idx, cols)
        
        # Apply transform
        augmented = transform(image=image)
        aug_image = augmented['image']
        
        # Denormalize if needed untuk visualization
        if isinstance(aug_image, np.ndarray) and aug_image.max() <= 1.0:
            display_image = denormalize_image(aug_image)
        else:
            display_image = aug_image
        
        axes[row, col].imshow(display_image)
        axes[row, col].set_title(f'{name.title()} Transform')
        axes[row, col].axis('off')
    
    # Hide unused subplots
    for idx in range(len(transforms_dict) + 1, rows * cols):
        row, col = divmod(idx, cols)
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()

# One-liner utilities untuk common operations
create_standard_inference_pipeline = lambda size: create_inference_augmentation_pipeline(size, normalize=True)
create_tta_pipelines = lambda size, level='light': create_test_time_augmentation_pipeline(size, level)
preprocess_single_image = lambda img, size=416: preprocess_image_for_yolo(img, size, normalize=True)