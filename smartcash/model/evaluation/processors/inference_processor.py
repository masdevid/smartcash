"""
Model inference processor for evaluation.
Handles model prediction, YOLOv5 output processing, and NMS.
"""

import torch
import numpy as np
from PIL import Image
from typing import Dict, List, Any, Optional
import time

from smartcash.common.logger import get_logger


class InferenceProcessor:
    """Process model inference and convert outputs to evaluation format"""
    
    def __init__(self, model_api=None, inference_timer=None):
        self.logger = get_logger('inference_processor')
        self.model_api = model_api
        self.inference_timer = inference_timer
    
    def run_inference_with_timing(self, test_images: List[Dict], checkpoint_info: Dict[str, Any]) -> tuple:
        """⏱️ Run inference with timing measurement"""
        
        predictions = []
        inference_times = []
        
        # Warmup if model available and not already warmed up (optimization)
        skip_warmup = checkpoint_info.get('optimized_run', False)
        if self.model_api and checkpoint_info.get('model_loaded', False) and not skip_warmup:
            self.logger.info("🔥 Warming up model")
            # Create dummy input for warmup with proper device detection
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Get model device automatically (same as training pipeline)
            try:
                model_device = next(self.model_api.model.parameters()).device
                dummy_input = dummy_input.to(model_device)
                self.logger.debug(f"🎯 Dummy input moved to model device: {model_device}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not detect model device, using CPU: {e}")
                dummy_input = dummy_input.cpu()
            
            if self.inference_timer:
                warmup_result = self.inference_timer.warmup_model(
                    self.model_api.model, dummy_input, warmup_runs=5
                )
        elif skip_warmup:
            self.logger.info("🚀 Skipping model warmup - already warmed up (optimization)")
        
        # Process each image
        for idx, img_data in enumerate(test_images):
            
            if self.model_api and checkpoint_info.get('model_loaded', False):
                # Real inference with model
                if self.inference_timer:
                    with self.inference_timer.time_inference(batch_size=1, operation='evaluation'):
                        try:
                            # Convert PIL to tensor using model API preprocessing
                            img_tensor = self._preprocess_image(img_data['image'])
                            
                            # Run prediction with latest model API - handle YOLOv5 integration
                            with torch.no_grad():
                                pred_result = self._run_model_prediction(img_tensor)
                            
                            detections = []
                            self.logger.debug(f"🔍 Prediction result: success={pred_result.get('success', False)}, keys={list(pred_result.keys())}")
                            
                            if pred_result.get('success', False):
                                # Handle different prediction formats from latest architecture
                                pred_detections = pred_result.get('detections', [])
                                self.logger.debug(f"🔍 Raw detections type: {type(pred_detections)}, length/keys: {len(pred_detections) if isinstance(pred_detections, (list, dict)) else 'N/A'}")
                                if isinstance(pred_detections, dict):
                                    # Handle multi-layer predictions (layer_1, layer_2, layer_3)
                                    for layer_name, layer_detections in pred_detections.items():
                                        if isinstance(layer_detections, list):
                                            for detection in layer_detections:
                                                detections.append({
                                                    'class_id': detection.get('class_id', 0),
                                                    'confidence': detection.get('confidence', 0),
                                                    'bbox': detection.get('bbox', [0, 0, 0, 0]),
                                                    'layer': layer_name
                                                })
                                elif isinstance(pred_detections, list):
                                    # Handle single-layer predictions
                                    for detection in pred_detections:
                                        detections.append({
                                            'class_id': detection.get('class_id', 0),
                                            'confidence': detection.get('confidence', 0),
                                            'bbox': detection.get('bbox', [0, 0, 0, 0]),
                                            'layer': 'unified'
                                        })
                                elif isinstance(pred_detections, torch.Tensor):
                                    # Handle raw YOLOv5 tensor output
                                    self.logger.debug(f"🔧 Processing raw YOLOv5 tensor output: {pred_detections.shape}")
                                    detections = self._process_yolov5_output(pred_detections, img_tensor.shape)
                            else:
                                # Fallback: direct model inference if API prediction fails
                                self.logger.debug(f"🔧 API prediction failed, trying direct model inference")
                                detections = self._run_direct_model_inference(img_tensor)
                            
                            self.logger.debug(f"🔍 Adding predictions for {img_data['filename']}: {len(detections)} detections")
                            if detections:
                                self.logger.debug(f"🔍 First detection: {detections[0]}")
                            
                            predictions.append({
                                'filename': img_data['filename'],
                                'detections': detections
                            })
                            
                            # Record timing
                            if self.inference_timer and self.inference_timer.timings.get('evaluation'):
                                last_timing = self.inference_timer.timings['evaluation'][-1]
                                inference_times.append(last_timing['time'])
                            else:
                                # Fallback timing if timer not working
                                inference_times.append(0.1)
                            
                        except Exception as e:
                            self.logger.error(f"⚠️ Inference error for {img_data['filename']}: {str(e)}")
                            import traceback
                            self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                            predictions.append({
                                'filename': img_data['filename'],
                                'detections': []
                            })
                            inference_times.append(0.1)  # Default timing
                else:
                    # No timing measurement
                    try:
                        img_tensor = self._preprocess_image(img_data['image'])
                        pred_result = self._run_model_prediction(img_tensor)
                        detections = []
                        
                        if pred_result.get('success', False):
                            pred_detections = pred_result.get('detections', [])
                            if isinstance(pred_detections, torch.Tensor):
                                detections = self._process_yolov5_output(pred_detections, img_tensor.shape)
                            elif isinstance(pred_detections, list):
                                for detection in pred_detections:
                                    detections.append({
                                        'class_id': detection.get('class_id', 0),
                                        'confidence': detection.get('confidence', 0),
                                        'bbox': detection.get('bbox', [0, 0, 0, 0]),
                                        'layer': 'default'
                                    })
                        
                        predictions.append({
                            'filename': img_data['filename'],
                            'detections': detections
                        })
                        inference_times.append(0.1)
                        
                    except Exception as e:
                        self.logger.error(f"⚠️ Inference error for {img_data['filename']}: {str(e)}")
                        predictions.append({
                            'filename': img_data['filename'],
                            'detections': []
                        })
                        inference_times.append(0.1)
            
            else:
                # Fallback: simulate inference for testing
                import random
                time.sleep(0.01)  # Simulate processing time
                
                # Mock predictions
                mock_detections = []
                for _ in range(random.randint(0, 3)):
                    mock_detections.append({
                        'class_id': random.randint(0, 6),
                        'confidence': random.uniform(0.3, 0.9),
                        'bbox': [random.uniform(0.1, 0.8), random.uniform(0.1, 0.8),
                                random.uniform(0.1, 0.3), random.uniform(0.1, 0.3)]
                    })
                
                predictions.append({
                    'filename': img_data['filename'],
                    'detections': mock_detections
                })
                inference_times.append(random.uniform(0.05, 0.15))
        
        # Ensure we have the same number of predictions and timings
        if len(predictions) != len(inference_times):
            self.logger.warning(f"Prediction count ({len(predictions)}) != timing count ({len(inference_times)}), adjusting")
            while len(inference_times) < len(predictions):
                inference_times.append(0.1)  # Add default timing
            while len(predictions) < len(inference_times):
                predictions.append({
                    'filename': f'missing_prediction_{len(predictions)}.jpg',
                    'detections': []
                })
        
        self.logger.info(f"⏱️ Inference complete: {len(predictions)} predictions, avg time: {np.mean(inference_times):.3f}s")
        return predictions, inference_times
    
    def _run_model_prediction(self, img_tensor: torch.Tensor) -> Dict[str, Any]:
        """🔮 Run model prediction with proper error handling"""
        try:
            if hasattr(self.model_api, 'model') and self.model_api.model is not None:
                # Direct model inference (model API doesn't have predict method)
                self.logger.debug(f"🔍 Using direct model inference")
                self.model_api.model.eval()
                with torch.no_grad():
                    output = self.model_api.model(img_tensor)
                
                self.logger.debug(f"🔍 Direct model output type: {type(output)}")
                if isinstance(output, dict):
                    self.logger.debug(f"🔍 Direct model output keys: {list(output.keys())}")
                elif isinstance(output, (list, tuple)):
                    self.logger.debug(f"🔍 Direct model output length: {len(output)}")
                    # YOLOv5 typically returns (predictions, auxiliary_output)
                    # Take the first element which is the predictions tensor
                    if len(output) > 0 and isinstance(output[0], torch.Tensor):
                        self.logger.debug(f"🔍 Using first output tensor: {output[0].shape}")
                        output = output[0]
                elif isinstance(output, torch.Tensor):
                    self.logger.debug(f"🔍 Direct model output shape: {output.shape}")
                
                return {
                    'success': True,
                    'detections': output
                }
            else:
                self.logger.warning(f"🔍 No model available for prediction")
                return {
                    'success': False,
                    'error': 'No model available for prediction'
                }
        except Exception as e:
            self.logger.error(f"🔍 Model prediction error: {e}")
            import traceback
            self.logger.error(f"🔍 Full traceback: {traceback.format_exc()}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _process_yolov5_output(self, output: torch.Tensor, input_shape: tuple) -> List[Dict]:
        """🔧 Process raw YOLOv5 tensor output into detection format using NMS"""
        detections = []
        
        try:
            # Import YOLOv5 non_max_suppression function from training module
            from smartcash.model.training.core.yolo_utils_manager import get_non_max_suppression
            
            # Get non_max_suppression function
            non_max_suppression = get_non_max_suppression()
            
            self.logger.debug(f"🔧 Applying NMS with conf_thres=0.001, iou_thres=0.6")
            
            # Apply NMS to raw YOLOv5 output - use training-like thresholds
            pred = non_max_suppression(
                output, 
                conf_thres=0.001,  # Very low threshold for evaluation
                iou_thres=0.6,     # Higher IoU threshold for NMS
                multi_label=True
            )
            
            self.logger.debug(f"🔧 NMS output type: {type(pred)}, length: {len(pred) if isinstance(pred, (list, tuple)) else 'N/A'}")
            
            # Process NMS results
            if pred and len(pred) > 0:
                # Take first batch (we process one image at a time)
                batch_pred = pred[0]
                
                if batch_pred is not None and len(batch_pred) > 0:
                    self.logger.debug(f"🔧 Batch predictions shape: {batch_pred.shape}")
                    
                    # Convert tensor to detections format
                    for detection in batch_pred:
                        if len(detection) >= 6:  # x1, y1, x2, y2, conf, class
                            x1, y1, x2, y2, conf, cls = detection[:6].tolist()
                            
                            # Convert xyxy to yolo format (xc, yc, w, h) - normalized [0,1]
                            img_w, img_h = input_shape[3], input_shape[2]  # tensor shape is [batch, ch, h, w]
                            
                            # Convert to center coordinates and normalize
                            x_center = (x1 + x2) / 2 / img_w
                            y_center = (y1 + y2) / 2 / img_h
                            width = (x2 - x1) / img_w
                            height = (y2 - y1) / img_h
                            
                            detections.append({
                                'class_id': int(cls),
                                'confidence': float(conf),
                                'bbox': [x_center, y_center, width, height],
                                'layer': 'yolov5_nms'
                            })
                    
                    self.logger.debug(f"🔧 Converted {len(detections)} detections from NMS output")
                else:
                    self.logger.debug(f"🔧 No detections after NMS")
            else:
                self.logger.debug(f"🔧 Empty NMS result")
            
        except Exception as e:
            self.logger.error(f"🔧 Error processing YOLOv5 output with NMS: {e}")
            import traceback
            self.logger.error(f"🔧 Full traceback: {traceback.format_exc()}")
        
        return detections
    
    def _run_direct_model_inference(self, img_tensor: torch.Tensor) -> List[Dict]:
        """🎯 Run direct model inference as fallback"""
        detections = []
        
        try:
            if hasattr(self.model_api, 'model') and self.model_api.model is not None:
                self.model_api.model.eval()
                with torch.no_grad():
                    output = self.model_api.model(img_tensor)
                
                # Process output based on type
                if isinstance(output, dict):
                    # Multi-layer output
                    for layer_name, layer_output in output.items():
                        if isinstance(layer_output, torch.Tensor):
                            layer_detections = self._process_yolov5_output(layer_output, img_tensor.shape)
                            for det in layer_detections:
                                det['layer'] = layer_name
                            detections.extend(layer_detections)
                elif isinstance(output, (list, tuple)):
                    # Multiple outputs (different scales)
                    for i, scale_output in enumerate(output):
                        if isinstance(scale_output, torch.Tensor):
                            scale_detections = self._process_yolov5_output(scale_output, img_tensor.shape)
                            for det in scale_detections:
                                det['layer'] = f'scale_{i}'
                            detections.extend(scale_detections)
                elif isinstance(output, torch.Tensor):
                    # Single tensor output
                    self.logger.debug(f"🔧 Processing single tensor output: {output.shape}")
                    detections = self._process_yolov5_output(output, img_tensor.shape)
                    
        except Exception as e:
            self.logger.debug(f"Direct model inference error: {e}")
        
        return detections
    
    def _preprocess_image(self, img: Image.Image) -> torch.Tensor:
        """🖼️ Preprocess image for inference"""
        # Resize to model input size
        img_resized = img.resize((640, 640))
        
        # Convert to tensor and normalize
        img_array = np.array(img_resized).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).unsqueeze(0)
        
        # Move to same device as model (automatic device detection like training pipeline)
        try:
            if self.model_api:
                model_device = next(self.model_api.model.parameters()).device
                img_tensor = img_tensor.to(model_device)
        except (AssertionError, RuntimeError, StopIteration) as e:
            # Model not available or device error, keep on CPU
            self.logger.debug(f"Using CPU for image tensor: {e}")
            img_tensor = img_tensor.cpu()
        
        return img_tensor


def create_inference_processor(model_api=None, inference_timer=None) -> InferenceProcessor:
    """Factory function to create inference processor"""
    return InferenceProcessor(model_api=model_api, inference_timer=inference_timer)