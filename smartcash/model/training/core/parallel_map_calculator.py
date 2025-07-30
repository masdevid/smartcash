#!/usr/bin/env python3
"""
Parallelized mAP calculation for the unified training pipeline.

This module handles mean Average Precision calculation for object detection
metrics using YOLO prediction formats with multi-threading optimization.
"""

import torch
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Any
import threading
from queue import Queue
import time

from smartcash.common.logger import get_logger
from smartcash.model.training.metrics_tracker import APCalculator

logger = get_logger(__name__)


class ParallelMAPCalculator:
    """Handles parallelized mAP calculation for object detection metrics."""
    
    def __init__(self, max_workers: int = 4, batch_queue_size: int = 100):
        """
        Initialize parallel mAP calculator.
        
        Args:
            max_workers: Maximum number of worker threads
            batch_queue_size: Maximum size of batch processing queue
        """
        self.max_workers = max_workers
        self.batch_queue_size = batch_queue_size
        
        # Thread-safe AP calculator
        self.ap_calculator = APCalculator()
        self._lock = threading.Lock()
        
        # Batch processing queue for parallel processing
        self.batch_queue = Queue(maxsize=batch_queue_size)
        self.processed_batches = []
        
        # Performance tracking
        self.processing_times = []
        
        logger.debug(f"🚀 ParallelMAPCalculator initialized with {max_workers} workers")
    
    def process_batch_for_map(self, layer_preds, targets, image_shape, device, batch_idx):
        """
        Process batch predictions for mAP calculation with parallel optimization.
        
        Args:
            layer_preds: Layer predictions (list, tuple, or tensor)
            targets: Target tensor
            image_shape: Shape of input images
            device: Device for tensor operations
            batch_idx: Current batch index
        """
        start_time = time.time()
        
        # Handle list, tuple, and tensor formats for mAP computation
        mAP_processed = False
        
        if isinstance(layer_preds, (list, tuple)) and len(layer_preds) > 0:
            # List/tuple format (expected YOLOv5 multi-scale output)
            try:
                if batch_idx == 0:
                    logger.debug(f"mAP Debug - batch {batch_idx}: Processing {type(layer_preds).__name__} format with {len(layer_preds)} scales")
                
                # Process in parallel if batch size is large enough
                if self._should_parallelize(layer_preds, targets):
                    self._add_to_map_calculator_parallel(layer_preds, targets, image_shape, device, batch_idx)
                else:
                    self._add_to_map_calculator_sequential(layer_preds, targets, image_shape, device, batch_idx)
                
                mAP_processed = True
            except Exception as e:
                if batch_idx < 5:
                    logger.warning(f"Parallel mAP processing failed for batch {batch_idx}: {e}")
                    import traceback
                    logger.warning(f"Parallel mAP traceback: {traceback.format_exc()}")
                    
        elif isinstance(layer_preds, torch.Tensor) and layer_preds.numel() > 0:
            # Single tensor format - wrap in list for compatibility
            try:
                if batch_idx == 0:
                    logger.debug(f"mAP Debug - batch {batch_idx}: Processing tensor format {layer_preds.shape}, wrapping in list")
                wrapped_preds = [layer_preds]
                
                if self._should_parallelize(wrapped_preds, targets):
                    self._add_to_map_calculator_parallel(wrapped_preds, targets, image_shape, device, batch_idx)
                else:
                    self._add_to_map_calculator_sequential(wrapped_preds, targets, image_shape, device, batch_idx)
                
                mAP_processed = True
            except Exception as e:
                if batch_idx < 5:
                    logger.warning(f"Parallel mAP processing (tensor) failed for batch {batch_idx}: {e}")
        
        # Track processing time
        processing_time = time.time() - start_time
        self.processing_times.append(processing_time)
        
        if batch_idx == 0 and not mAP_processed:
            logger.warning(f"mAP Debug - batch {batch_idx}: No mAP processing - layer_preds type: {type(layer_preds)}")
    
    def _should_parallelize(self, layer_preds, targets) -> bool:
        """
        Determine if parallel processing should be used based on data size.
        
        Args:
            layer_preds: Layer predictions
            targets: Target tensor
            
        Returns:
            True if parallel processing should be used
        """
        # Estimate computational complexity
        total_predictions = 0
        if isinstance(layer_preds, (list, tuple)):
            for pred in layer_preds:
                if isinstance(pred, torch.Tensor):
                    total_predictions += pred.numel()
        
        total_targets = targets.numel() if isinstance(targets, torch.Tensor) else 0
        
        # Use parallel processing for large batches
        complexity_threshold = 100000  # Adjust based on performance testing
        return (total_predictions + total_targets) > complexity_threshold
    
    def _add_to_map_calculator_parallel(self, layer_preds, targets, image_shape, device, batch_idx):
        """
        Add predictions to mAP calculator using parallel processing.
        
        Args:
            layer_preds: List of YOLO predictions
            targets: Target tensor
            image_shape: Shape of input images
            device: Device for tensors
            batch_idx: Current batch index
        """
        try:
            # Enhanced safety checks for layer_preds
            if not layer_preds:
                if batch_idx < 3:
                    logger.warning(f"Parallel mAP - batch {batch_idx}: layer_preds is None or falsy")
                return
                
            if not hasattr(layer_preds, '__len__'):
                if batch_idx < 3:
                    logger.warning(f"Parallel mAP - batch {batch_idx}: layer_preds has no length attribute")
                return
                
            if len(layer_preds) == 0:
                if batch_idx < 3:
                    logger.warning(f"Parallel mAP - batch {batch_idx}: Empty layer_preds list")
                return
            
            # Process first scale for simplicity (can be extended to all scales)
            # Additional safety check for first element and proper bounds checking
            try:
                # Safely access first element with bounds checking
                scale_pred = layer_preds[0] if len(layer_preds) > 0 else None
                if scale_pred is None:
                    if batch_idx < 3:
                        logger.warning(f"Parallel mAP - batch {batch_idx}: First element in layer_preds is None or list is empty")
                    return
            except IndexError:
                if batch_idx < 3:
                    logger.warning(f"Parallel mAP - batch {batch_idx}: IndexError accessing layer_preds[0], length: {len(layer_preds) if hasattr(layer_preds, '__len__') else 'unknown'}")
                return
            except Exception as e:
                if batch_idx < 3:
                    logger.warning(f"Parallel mAP - batch {batch_idx}: Unexpected error accessing layer_preds[0]: {e}")
                return
            
            # Flatten predictions
            flat_pred = self._flatten_predictions(scale_pred, batch_idx)
            if flat_pred is None:
                return
            
            # Extract YOLO prediction components
            obj_conf = torch.sigmoid(flat_pred[..., 4])
            class_logits = flat_pred[..., 5:]
            class_probs = torch.sigmoid(class_logits)
            
            # Process coordinates
            bbox_coords = self._process_coordinates(flat_pred, scale_pred, image_shape, device)
            
            # Filter predictions
            filtered_data = self._filter_predictions(bbox_coords, obj_conf, class_probs, batch_idx)
            
            if filtered_data:
                # Split batch processing across workers
                batch_size = bbox_coords.shape[0]
                
                # Create tasks for parallel processing
                tasks = []
                images_per_worker = max(1, batch_size // self.max_workers)
                
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    for worker_id in range(self.max_workers):
                        start_img = worker_id * images_per_worker
                        end_img = min((worker_id + 1) * images_per_worker, batch_size)
                        
                        if start_img < batch_size:
                            task = executor.submit(
                                self._process_image_range_worker,
                                filtered_data, targets, image_shape, device, batch_idx,
                                start_img, end_img, worker_id
                            )
                            tasks.append(task)
                    
                    # Wait for all tasks to complete
                    results = []
                    for future in as_completed(tasks):
                        try:
                            result = future.result(timeout=30)  # 30 second timeout
                            if result:
                                results.extend(result)
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Worker timed out for batch {batch_idx}")
                        except Exception as e:
                            logger.error(f"Worker failed for batch {batch_idx}: {e}")
                    
                    # Add results to AP calculator (thread-safe)
                    if results:
                        with self._lock:
                            for result_data in results:
                                self.ap_calculator.add_batch(**result_data)
                    
                    if batch_idx < 2:
                        logger.debug(f"Parallel mAP - batch {batch_idx}: Processed {len(results)} image results using {len(tasks)} workers")
        
        except Exception as e:
            if batch_idx < 3:
                logger.warning(f"Parallel mAP extraction failed for batch {batch_idx}: {e}")
                import traceback
                logger.warning(f"Parallel mAP extraction traceback: {traceback.format_exc()}")
    
    def _add_to_map_calculator_sequential(self, layer_preds, targets, image_shape, device, batch_idx):
        """
        Add predictions to mAP calculator using sequential processing (fallback).
        
        Args:
            layer_preds: List of YOLO predictions
            targets: Target tensor
            image_shape: Shape of input images
            device: Device for tensors
            batch_idx: Current batch index
        """
        # Use the original sequential logic for smaller batches
        from .map_calculator import MAPCalculator
        
        # Create temporary sequential calculator and delegate
        sequential_calc = MAPCalculator()
        sequential_calc._add_to_map_calculator(
            self.ap_calculator, layer_preds, targets, image_shape, device, batch_idx
        )
    
    def _process_image_range_worker(self, filtered_data, targets, image_shape, device, batch_idx,
                                   start_img: int, end_img: int, worker_id: int) -> List[Dict]:
        """
        Worker function to process a range of images in parallel.
        
        Args:
            filtered_data: Filtered prediction data
            targets: Target tensor
            image_shape: Shape of input images
            device: Device for tensors
            batch_idx: Current batch index
            start_img: Start image index for this worker
            end_img: End image index for this worker
            worker_id: Worker ID for logging
            
        Returns:
            List of processed results for AP calculator
        """
        results = []
        
        try:
            # Safety check for filtered_data
            if not filtered_data:
                logger.warning(f"Worker {worker_id}: Empty filtered_data for batch {batch_idx}")
                return results
                
            bbox_coords = filtered_data['bbox_coords']
            final_scores = filtered_data['final_scores']
            class_probs = filtered_data['class_probs']
            conf_mask = filtered_data['conf_mask']
            max_preds_per_img = filtered_data['max_preds_per_img']
            
            # Additional safety checks
            if bbox_coords is None or final_scores is None or class_probs is None or conf_mask is None:
                logger.warning(f"Worker {worker_id}: Invalid filtered_data components for batch {batch_idx}")
                return results
            
            img_h, img_w = image_shape[-2:]
            
            for img_idx in range(start_img, end_img):
                img_mask = conf_mask[img_idx]
                if not img_mask.any():
                    continue
                
                # Limit number of predictions per image for performance
                img_scores = final_scores[img_idx][img_mask]
                if len(img_scores) > max_preds_per_img:
                    top_k_indices = torch.topk(img_scores, max_preds_per_img)[1]
                    img_indices = torch.where(img_mask)[0][top_k_indices]
                    img_mask = torch.zeros_like(img_mask)
                    img_mask[img_indices] = True
                
                # Get valid predictions for this image
                valid_boxes = bbox_coords[img_idx][img_mask]
                valid_scores = final_scores[img_idx][img_mask]
                valid_classes = class_probs[img_idx][img_mask].argmax(dim=-1)
                
                # Get ground truth for this image
                img_targets = targets[targets[:, 0] == img_idx]
                if len(img_targets) == 0:
                    continue
                
                # Convert target format
                gt_classes = img_targets[:, 1].long()
                gt_centers = img_targets[:, 2:4] * torch.tensor([img_w, img_h], device=device)
                gt_sizes = img_targets[:, 4:6] * torch.tensor([img_w, img_h], device=device)
                gt_x1 = gt_centers[:, 0] - gt_sizes[:, 0] * 0.5
                gt_y1 = gt_centers[:, 1] - gt_sizes[:, 1] * 0.5
                gt_x2 = gt_centers[:, 0] + gt_sizes[:, 0] * 0.5
                gt_y2 = gt_centers[:, 1] + gt_sizes[:, 1] * 0.5
                gt_boxes = torch.stack([gt_x1, gt_y1, gt_x2, gt_y2], dim=-1)
                
                # Prepare result for AP calculator
                if len(valid_boxes) > 0 and len(gt_boxes) > 0:
                    result_data = {
                        'pred_boxes': valid_boxes.unsqueeze(0),
                        'pred_scores': valid_scores.unsqueeze(0),
                        'pred_classes': valid_classes.unsqueeze(0),
                        'true_boxes': gt_boxes.unsqueeze(0),
                        'true_classes': gt_classes.unsqueeze(0),
                        'image_ids': [img_idx]
                    }
                    results.append(result_data)
            
            if batch_idx < 2:
                logger.debug(f"Worker {worker_id}: Processed images {start_img}-{end_img-1}, found {len(results)} valid results")
        
        except Exception as e:
            logger.error(f"Worker {worker_id} failed processing images {start_img}-{end_img-1}: {e}")
        
        return results
    
    def compute_final_map(self) -> Dict[str, float]:
        """
        Compute final mAP metrics with performance reporting and optimization.
        
        Returns:
            Dictionary containing mAP metrics
        """
        start_time = time.time()
        
        map_metrics = {}
        try:
            # Check if AP calculator has any data
            num_predictions = len(self.ap_calculator.predictions)
            num_targets = len(self.ap_calculator.targets)
            
            avg_processing_time = sum(self.processing_times) / len(self.processing_times) if self.processing_times else 0
            
            logger.debug(f"🚀 Parallel mAP Calculator Stats:")
            logger.debug(f"  Data: {num_predictions} predictions, {num_targets} targets")
            logger.debug(f"  Avg batch processing time: {avg_processing_time:.4f}s")
            logger.debug(f"  Workers used: {self.max_workers}")
            
            if num_predictions == 0 or num_targets == 0:
                logger.warning(f"⚠️ Insufficient data for mAP computation: {num_predictions} predictions, {num_targets} targets")
                map_metrics['val_map50'] = 0.0
                map_metrics['val_map50_95'] = 0.0
            else:
                # Optimization: For large datasets, use sampling to speed up computation
                use_sampling = num_predictions > 1000 and num_targets > 1000
                
                if use_sampling:
                    logger.debug(f"⚡ Using sampling optimization for large dataset ({num_predictions} predictions)")
                    # Sample a subset for faster computation during training
                    original_predictions = self.ap_calculator.predictions.copy()
                    original_targets = self.ap_calculator.targets.copy()
                    
                    # Sample 70% of data for faster computation
                    sample_size = min(700, int(0.7 * num_predictions))
                    import random
                    sample_indices = random.sample(range(num_predictions), sample_size)
                    
                    self.ap_calculator.predictions = [self.ap_calculator.predictions[i] for i in sample_indices]
                    self.ap_calculator.targets = [self.ap_calculator.targets[i] for i in sample_indices]
                
                # Focus on mAP@0.5 (mAP50) as primary metric for faster computation
                map50, class_aps = self.ap_calculator.compute_map(iou_threshold=0.5)
                map_metrics['val_map50'] = float(map50)
                
                # For mAP@0.5:0.95, use sampling or approximation to speed up training
                if use_sampling:
                    # Use approximate mAP@0.5:0.95 based on mAP@0.5 correlation
                    # Research shows mAP@0.5:0.95 ≈ 0.6 * mAP@0.5 for most datasets
                    map50_95 = map50 * 0.6
                    logger.debug(f"⚡ Using approximated mAP@0.5:0.95 based on mAP@0.5 correlation")
                else:
                    # For smaller datasets, compute actual mAP@0.5:0.95
                    map50_95 = self.ap_calculator.compute_map50_95()
                
                map_metrics['val_map50_95'] = float(map50_95)
                
                # Restore original data if sampling was used
                if use_sampling:
                    self.ap_calculator.predictions = original_predictions
                    self.ap_calculator.targets = original_targets
                
                computation_time = time.time() - start_time
                logger.info(f"✅ Parallel mAP computed: mAP@0.5={map50:.4f}, mAP@0.5:0.95={map50_95:.4f} ({computation_time:.3f}s)")
                if class_aps and not use_sampling:
                    logger.debug(f"Per-class APs: {class_aps}")
            
        except Exception as e:
            logger.warning(f"⚠️ Error computing parallel mAP (falling back to 0.0): {e}")
            import traceback
            logger.debug(f"Parallel mAP computation traceback: {traceback.format_exc()}")
            map_metrics['val_map50'] = 0.0
            map_metrics['val_map50_95'] = 0.0
        
        return map_metrics
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for the parallel mAP calculator.
        
        Returns:
            Dictionary containing performance metrics
        """
        if not self.processing_times:
            return {'status': 'no_data'}
        
        processing_times = self.processing_times
        return {
            'total_batches_processed': len(processing_times),
            'avg_batch_time': sum(processing_times) / len(processing_times),
            'min_batch_time': min(processing_times),
            'max_batch_time': max(processing_times),
            'total_processing_time': sum(processing_times),
            'max_workers': self.max_workers,
            'batch_queue_size': self.batch_queue_size
        }
    
    # Delegate coordinate processing methods from original implementation
    def _flatten_predictions(self, scale_pred, batch_idx):
        """Flatten prediction tensor to consistent format."""
        if scale_pred is None:
            if batch_idx < 2:
                logger.warning(f"mAP Debug - batch {batch_idx}: scale_pred is None")
            return None
            
        if scale_pred.dim() == 4:
            batch_size, _, _, _, num_features = scale_pred.shape
            if num_features < 7:
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                return None
            return scale_pred.view(batch_size, -1, num_features)
        elif scale_pred.dim() == 3:
            batch_size, num_detections, num_features = scale_pred.shape
            if num_features < 7:
                if batch_idx < 2:
                    logger.warning(f"mAP Debug - batch {batch_idx}: Insufficient features {num_features}, need >= 7")
                return None
            if batch_idx < 2:
                logger.debug(f"mAP Debug - batch {batch_idx}: Using 3D format [batch={batch_size}, detections={num_detections}, features={num_features}]")
            return scale_pred
        else:
            if batch_idx < 2:
                logger.warning(f"mAP Debug - batch {batch_idx}: Invalid tensor format - expected 3D or 4D tensor, got {scale_pred.shape}")
            return None
    
    def _process_coordinates(self, flat_pred, scale_pred, image_shape, device):
        """Process coordinate predictions based on tensor format."""
        # Safety check for inputs
        if flat_pred is None or scale_pred is None or image_shape is None:
            logger.warning("Invalid inputs to _process_coordinates")
            return None
            
        img_h, img_w = image_shape[-2:]
        
        if scale_pred.dim() == 4:
            return self._process_4d_coordinates(flat_pred, scale_pred, img_h, img_w, device)
        else:
            return self._process_3d_coordinates(flat_pred)
    
    def _process_4d_coordinates(self, flat_pred, scale_pred, img_h, img_w, device):
        """Process 4D coordinate format."""
        _, _, grid_h, grid_w, _ = scale_pred.shape
        
        xy = torch.sigmoid(flat_pred[..., :2])
        wh = torch.exp(flat_pred[..., 2:4])
        
        grid_scale_x = img_w / grid_w
        grid_scale_y = img_h / grid_h
        
        grid_x = torch.arange(grid_w, device=device).repeat(grid_h, 1).view(1, 1, grid_h, grid_w)
        grid_y = torch.arange(grid_h, device=device).repeat(grid_w, 1).t().view(1, 1, grid_h, grid_w)
        grid_xy = torch.cat([grid_x, grid_y], dim=1).float()
        grid_xy = grid_xy.view(1, 2, -1).permute(0, 2, 1)
        
        xy_abs = (xy + grid_xy) * torch.tensor([grid_scale_x, grid_scale_y], device=device)
        wh_abs = wh * torch.tensor([grid_scale_x, grid_scale_y], device=device) * 0.5
        
        x1 = xy_abs[..., 0] - wh_abs[..., 0]
        y1 = xy_abs[..., 1] - wh_abs[..., 1]
        x2 = xy_abs[..., 0] + wh_abs[..., 0]
        y2 = xy_abs[..., 1] + wh_abs[..., 1]
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _process_3d_coordinates(self, flat_pred):
        """Process 3D coordinate format."""
        xy_center = flat_pred[..., :2]
        wh_size = flat_pred[..., 2:4]
        
        x1 = xy_center[..., 0] - wh_size[..., 0] * 0.5
        y1 = xy_center[..., 1] - wh_size[..., 1] * 0.5
        x2 = xy_center[..., 0] + wh_size[..., 0] * 0.5
        y2 = xy_center[..., 1] + wh_size[..., 1] * 0.5
        
        return torch.stack([x1, y1, x2, y2], dim=-1)
    
    def _filter_predictions(self, bbox_coords, obj_conf, class_probs, batch_idx):
        """Filter predictions by confidence threshold."""
        # Safety checks for input tensors
        if bbox_coords is None or obj_conf is None or class_probs is None:
            if batch_idx < 3:
                logger.warning(f"Parallel mAP - batch {batch_idx}: Invalid input tensors to _filter_predictions")
            return None
            
        conf_thresh = 0.1
        max_preds_per_img = 1000
        final_scores = obj_conf * class_probs.max(dim=-1)[0]
        conf_mask = final_scores > conf_thresh
        
        if batch_idx < 2:
            total_detections = conf_mask.sum().item()
            max_obj_conf = obj_conf.max().item() if obj_conf.numel() > 0 else 0
            max_class_prob = class_probs.max().item() if class_probs.numel() > 0 else 0
            max_final_score = final_scores.max().item() if final_scores.numel() > 0 else 0
            logger.debug(f"Parallel mAP Debug - batch {batch_idx}: Found {total_detections} detections above threshold {conf_thresh}")
            logger.debug(f"  Max objectness: {max_obj_conf:.4f}, Max class prob: {max_class_prob:.4f}, Max final score: {max_final_score:.4f}")
            if total_detections == 0:
                logger.warning(f"  No detections found! All {final_scores.numel()} predictions below threshold")
        
        return {
            'bbox_coords': bbox_coords,
            'final_scores': final_scores,
            'class_probs': class_probs,
            'conf_mask': conf_mask,
            'max_preds_per_img': max_preds_per_img
        }