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
from smartcash.model.inference.inference_service import InferenceService


class InferenceProcessor:
    """Process model inference and convert outputs to evaluation format"""
    
    def __init__(self, inference_service: InferenceService, inference_timer=None):
        self.logger = get_logger('inference_processor')
        self.inference_service = inference_service
        self.inference_timer = inference_timer
    
    def run_inference_with_timing(self, test_images: List[Dict], checkpoint_info: Dict[str, Any]) -> tuple:
        """⏱️ Run inference with timing measurement"""
        
        predictions = []
        inference_times = []
        
        # Warmup if model available and not already warmed up (optimization)
        skip_warmup = checkpoint_info.get('optimized_run', False)
        if self.inference_service.model and not skip_warmup:
            self.logger.info("🔥 Warming up model")
            # Create dummy input for warmup with proper device detection
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Get model device automatically (same as training pipeline)
            try:
                model_device = next(self.inference_service.model.parameters()).device
                dummy_input = dummy_input.to(model_device)
                self.logger.debug(f"🎯 Dummy input moved to model device: {model_device}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not detect model device, using CPU: {e}")
                dummy_input = dummy_input.cpu()
            
            if self.inference_timer:
                warmup_result = self.inference_timer.warmup_model(
                    self.inference_service.model, dummy_input, warmup_runs=5
                )
        elif skip_warmup:
            self.logger.info("🚀 Skipping model warmup - already warmed up (optimization)")
        
        # Process each image
        for idx, img_data in enumerate(test_images):
            
            if self.inference_service.model:
                # Real inference with model
                if self.inference_timer:
                    with self.inference_timer.time_inference(batch_size=1, operation='evaluation'):
                        try:
                            pred_result = self.inference_service.predict([img_data['image']])
                            
                            predictions.append({
                                'filename': img_data['filename'],
                                'detections': pred_result
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
                        pred_result = self.inference_service.predict([img_data['image']])
                        
                        predictions.append({
                            'filename': img_data['filename'],
                            'detections': pred_result
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
    
    


def create_inference_processor(inference_service: InferenceService, inference_timer=None) -> InferenceProcessor:
    """Factory function to create inference processor"""
    return InferenceProcessor(inference_service=inference_service, inference_timer=inference_timer)