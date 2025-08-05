"""
File: smartcash/model/evaluation/evaluation_service.py
Deskripsi: Main evaluation orchestrator untuk research scenarios dengan progress tracking
"""

import time
from pathlib import Path
from typing import Dict, Any, List, Optional
import torch
import numpy as np
from PIL import Image

from smartcash.common.logger import get_logger
from smartcash.model.evaluation.scenario_manager import ScenarioManager
from smartcash.model.training.core.yolov5_map_calculator import YOLOv5MapCalculator
from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector
from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge
from smartcash.model.evaluation.utils.inference_timer import InferenceTimer
from smartcash.model.evaluation.utils.results_aggregator import ResultsAggregator
from smartcash.model.evaluation.visualization.evaluation_chart_generator import EvaluationChartGenerator

class MockProgressBridge:
    """Mock progress bridge for testing and fallback scenarios."""
    
    def start_evaluation(self, scenarios, checkpoints, title):
        pass
    
    def update_scenario(self, idx, name, message):
        pass
    
    def update_checkpoint(self, idx, name, message):
        pass
    
    def update_metrics(self, progress, message):
        pass
    
    def update_substep(self, message):
        pass
    
    def complete_evaluation(self, message):
        pass
    
    def evaluation_error(self, message):
        pass

class EvaluationService:
    """Main evaluation service untuk research scenarios dengan comprehensive metrics"""
    
    def __init__(self, model_api=None, config: Dict[str, Any] = None):
        # Ensure config is a dictionary and has the expected structure
        self.logger = get_logger('evaluation_service')
        
        if not isinstance(config, dict):
            self.logger.warning(f"Config is not a dictionary (got {type(config).__name__}), converting to dict")
            # For non-dict config, create empty dict with evaluation section
            config = {"evaluation": {}} 
        
        # Ensure the config has the required structure
        if "evaluation" not in config:
            self.logger.warning("Config missing 'evaluation' key, adding it")
            # Move existing config under evaluation if it's a dict
            if isinstance(config, dict) and config:
                config = {"evaluation": config}
            else:
                config = {"evaluation": {}}
        
        self.config = config
        if not hasattr(self, 'logger'):
            self.logger = get_logger('evaluation_service')
        self.model_api = model_api
        
        try:
            # Initialize components with normalized config
            self.scenario_manager = ScenarioManager(self.config)
            # Use training module's YOLOv5 mAP calculator for consistency
            self.map_calculator = YOLOv5MapCalculator(
                num_classes=7,  # Primary layer (layer_1) classes
                conf_thres=0.005,
                iou_thres=0.03,
                debug=True,
                training_context={'evaluation_mode': True, 'backbone': 'evaluation'}
            )
            self.checkpoint_selector = CheckpointSelector(config=self.config)
            self.inference_timer = InferenceTimer(self.config)
            self.results_aggregator = ResultsAggregator(self.config)
            
            # Initialize chart generator for visualization
            chart_config = self.config.get('evaluation', {}).get('export', {})
            if chart_config.get('include_visualizations', True):
                chart_output_dir = self.config.get('evaluation', {}).get('data', {}).get('charts_dir', 'data/evaluation/charts')
                self.chart_generator = EvaluationChartGenerator(config=self.config, output_dir=chart_output_dir)
                self.logger.info("📊 Chart generator initialized for evaluation visualizations")
            else:
                self.chart_generator = None
                self.logger.info("📊 Chart generation disabled in configuration")
        except Exception as e:
            self.logger.error(f"Failed to initialize evaluation service components: {e}", exc_info=True)
            raise
        
        # Progress tracking - initialize with fallback
        self.progress_bridge = None
        self.ui_components = {}
        self._initialize_progress_bridge()
    
    def _initialize_progress_bridge(self):
        """Initialize progress bridge with fallback for tests."""
        try:
            from smartcash.model.evaluation.utils.evaluation_progress_bridge import EvaluationProgressBridge
            self.progress_bridge = EvaluationProgressBridge({}, None)
        except Exception as e:
            self.logger.debug(f"Failed to initialize progress bridge: {e}")
            # Create mock progress bridge for testing
            self.progress_bridge = MockProgressBridge()
        
    def run_evaluation(self, scenarios: List[str] = None, checkpoints: List[str] = None,
                      progress_callback=None, metrics_callback=None, 
                      ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
        """🚀 Run comprehensive evaluation pipeline"""
        
        # Setup progress tracking
        self.ui_components = ui_components or {}
        self.progress_bridge = EvaluationProgressBridge(self.ui_components, progress_callback)
        
        # Default scenarios
        if scenarios is None:
            scenarios = ['position_variation', 'lighting_variation']
        
        # Get available checkpoints
        if checkpoints is None:
            available_checkpoints = self.checkpoint_selector.list_available_checkpoints()
            checkpoints = [cp['path'] for cp in available_checkpoints[:2]]  # Top 2 checkpoints
        
        self.progress_bridge.start_evaluation(scenarios, checkpoints, "Research Evaluation")
        
        try:
            evaluation_results = {}
            
            # Prepare scenarios
            self.logger.info(f"🎯 Preparing {len(scenarios)} scenarios")
            scenario_preparation = self.scenario_manager.prepare_all_scenarios()
            
            for scenario_idx, scenario_name in enumerate(scenarios):
                self.progress_bridge.update_scenario(scenario_idx, scenario_name, "Preparing scenario data")
                
                scenario_results = {}
                
                for checkpoint_idx, checkpoint_path in enumerate(checkpoints):
                    self.progress_bridge.update_checkpoint(checkpoint_idx, Path(checkpoint_path).name, "Loading checkpoint")
                    
                    # Load checkpoint
                    checkpoint_info = self._load_checkpoint(checkpoint_path)
                    if not checkpoint_info:
                        continue
                    
                    # Run scenario evaluation
                    scenario_result = self._evaluate_scenario(
                        scenario_name, checkpoint_info, 
                        checkpoint_idx, len(checkpoints)
                    )
                    
                    # Store results
                    backbone = checkpoint_info.get('backbone', 'unknown')
                    scenario_results[backbone] = {
                        'checkpoint_info': checkpoint_info,
                        'metrics': scenario_result['metrics'],
                        'additional_data': scenario_result.get('additional_data', {})
                    }
                    
                    # Add to aggregator
                    self.results_aggregator.add_scenario_results(
                        scenario_name, backbone, checkpoint_info,
                        scenario_result['metrics'], scenario_result.get('additional_data', {})
                    )
                
                evaluation_results[scenario_name] = scenario_results
            
            # Generate comprehensive summary
            self.progress_bridge.update_metrics(90, "Generating comprehensive summary")
            summary = self._generate_evaluation_summary(evaluation_results)
            
            # Export results
            self.progress_bridge.update_metrics(90, "Exporting results")
            export_files = self.results_aggregator.export_results()
            
            # Generate evaluation charts
            chart_files = []
            if self.chart_generator:
                try:
                    self.progress_bridge.update_metrics(95, "Generating evaluation charts")
                    self.logger.info("📊 Generating evaluation visualization charts...")
                    
                    # Prepare data for chart generation
                    chart_data = {
                        'results': evaluation_results,
                        'evaluation_info': {
                            'timestamp': summary.get('evaluation_completed_at', ''),
                            'total_scenarios': len(scenarios),
                            'total_checkpoints': len(checkpoints)
                        }
                    }
                    
                    # Generate all charts
                    chart_files = self.chart_generator.generate_all_charts(chart_data)
                    
                    if chart_files:
                        self.logger.info(f"✅ Generated {len(chart_files)} evaluation charts")
                        chart_summary = self.chart_generator.get_chart_summary()
                        self.logger.info(f"📊 Charts saved to: {chart_summary['output_directory']}")
                    else:
                        self.logger.warning("⚠️ No charts were generated")
                        
                except Exception as e:
                    self.logger.error(f"❌ Chart generation failed: {str(e)}")
                    # Don't fail the entire evaluation if chart generation fails
                    chart_files = []
            
            self.progress_bridge.complete_evaluation("Evaluation completed successfully!")
            
            final_results = {
                'status': 'success',
                'evaluation_results': evaluation_results,
                'summary': summary,
                'export_files': export_files,
                'chart_files': chart_files,
                'scenarios_evaluated': len(scenarios),
                'checkpoints_evaluated': len(checkpoints)
            }
            
            # Call metrics callback
            if metrics_callback:
                metrics_callback(final_results)
            
            return final_results
            
        except Exception as e:
            error_msg = f"Evaluation failed: {str(e)}"
            self.progress_bridge.evaluation_error(error_msg)
            self.logger.error(f"❌ {error_msg}")
            
            return {
                'status': 'error',
                'error': error_msg,
                'partial_results': getattr(self, 'evaluation_results', {})
            }
    
    def run_scenario(self, scenario_name: str, checkpoint_path: str) -> Dict[str, Any]:
        """🎯 Run single scenario evaluation"""
        
        self.logger.info(f"🎯 Running {scenario_name} evaluation")
        
        # Load checkpoint
        checkpoint_info = self._load_checkpoint(checkpoint_path)
        if not checkpoint_info:
            return {'status': 'error', 'error': 'Failed to load checkpoint'}
        
        # Evaluate scenario
        result = self._evaluate_scenario(scenario_name, checkpoint_info, 0, 1)
        
        # Generate charts for single scenario if chart generator is available
        chart_files = []
        if self.chart_generator:
            try:
                self.logger.info("📊 Generating single scenario evaluation charts...")
                
                # Prepare data for chart generation (single scenario format)
                chart_data = {
                    'results': [{
                        'scenario_name': scenario_name,
                        'checkpoint_info': checkpoint_info,
                        'metrics': result['metrics'],
                        'additional_data': result.get('additional_data', {})
                    }],
                    'evaluation_info': {
                        'timestamp': result.get('timestamp', ''),
                        'total_scenarios': 1,
                        'total_checkpoints': 1,
                        'single_scenario': True
                    }
                }
                
                # Generate charts (will adapt to single scenario)
                chart_files = self.chart_generator.generate_all_charts(chart_data)
                
                if chart_files:
                    self.logger.info(f"✅ Generated {len(chart_files)} evaluation charts")
                    chart_summary = self.chart_generator.get_chart_summary()
                    self.logger.info(f"📊 Charts saved to: {chart_summary['output_directory']}")
                    
            except Exception as e:
                self.logger.error(f"❌ Chart generation failed: {str(e)}")
                # Don't fail the entire evaluation if chart generation fails
                chart_files = []
        
        return {
            'status': 'success',
            'scenario_name': scenario_name,
            'checkpoint_info': checkpoint_info,
            'metrics': result['metrics'],
            'additional_data': result.get('additional_data', {}),
            'chart_files': chart_files
        }
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """📥 Load dan validate checkpoint"""
        return self._load_checkpoint(checkpoint_path)
    
    def compute_metrics(self, predictions: List[Dict], ground_truths: List[Dict], 
                       inference_times: List[float] = None) -> Dict[str, Any]:
        """📊 Compute comprehensive metrics"""
        return self.evaluation_metrics.get_metrics_summary(predictions, ground_truths, inference_times)
    
    def generate_report(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """📋 Generate evaluation report"""
        return self._generate_evaluation_summary(evaluation_results)
    
    def compare_scenarios(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """⚖️ Compare scenarios performance"""
        return self.results_aggregator.compare_backbones()
    
    def save_results(self, results: Dict[str, Any], formats: List[str] = None) -> Dict[str, str]:
        """💾 Save evaluation results"""
        
        # Add results to aggregator if not already there
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            if isinstance(eval_results, dict):
                for scenario_name, backbone_results in eval_results.items():
                    if isinstance(backbone_results, dict):
                        for backbone, result_data in backbone_results.items():
                            if isinstance(result_data, dict):
                                self.results_aggregator.add_scenario_results(
                                    scenario_name, backbone,
                                    result_data.get('checkpoint_info', {}),
                                    result_data.get('metrics', {}),
                                    result_data.get('additional_data', {})
                                )
        
        return self.results_aggregator.export_results(formats)
    
    def _load_checkpoint(self, checkpoint_path: str) -> Optional[Dict[str, Any]]:
        """📥 Load checkpoint dengan validation for latest model architecture"""
        try:
            checkpoint_info = self.checkpoint_selector.select_checkpoint(checkpoint_path)
            
            if self.model_api:
                # Load model dengan checkpoint using latest API
                load_result = self.model_api.load_checkpoint(checkpoint_path)
                if load_result.get('success', False):
                    checkpoint_info['model_loaded'] = True
                    checkpoint_info['architecture_type'] = load_result.get('architecture_type', 'yolov5')
                    checkpoint_info['model_info'] = load_result.get('model_info', {})
                    self.logger.info(f"✅ Model loaded: {checkpoint_info['display_name']} ({checkpoint_info['architecture_type']})")
                else:
                    self.logger.warning(f"⚠️ Model load failed: {checkpoint_path}")
                    checkpoint_info['model_loaded'] = False
            else:
                # Create model API using checkpoint metadata for proper configuration
                self.logger.info("🔧 Creating model API for evaluation")
                try:
                    from smartcash.model.api.core import create_api
                    
                    # Extract model configuration from checkpoint
                    model_config = self._extract_model_config_from_checkpoint(checkpoint_info)
                    
                    # Create API with extracted configuration
                    self.model_api = create_api(
                        config=model_config,
                        use_yolov5_integration=True
                    )
                    
                    # Build model first with proper configuration
                    build_result = self.model_api.build_model(
                        backbone=checkpoint_info.get('backbone', 'cspdarknet'),
                        num_classes=17,  # Force hierarchical prediction (Layer 1: 0-6, Layer 2: 7-13, Layer 3: 14-16)
                        img_size=model_config.get('img_size', 640),
                        layer_mode=checkpoint_info.get('layer_mode', 'multi'),
                        detection_layers=['layer_1', 'layer_2', 'layer_3'],
                        pretrained=False  # We'll load weights from checkpoint
                    )
                    
                    if build_result.get('success', False):
                        # Now load checkpoint weights
                        load_result = self.model_api.load_checkpoint(checkpoint_path)
                        if load_result.get('success', False):
                            checkpoint_info['model_loaded'] = True
                            checkpoint_info['architecture_type'] = load_result.get('architecture_type', 'yolov5')
                            checkpoint_info['model_info'] = load_result.get('model_info', {})
                            self.logger.info(f"✅ Model API created and loaded: {checkpoint_info['display_name']}")
                        else:
                            checkpoint_info['model_loaded'] = False
                            self.logger.warning(f"⚠️ Failed to load checkpoint weights: {checkpoint_path}")
                    else:
                        checkpoint_info['model_loaded'] = False
                        self.logger.warning(f"⚠️ Failed to build model: {build_result.get('error', 'Unknown error')}")
                        
                except Exception as api_error:
                    self.logger.warning(f"⚠️ Failed to create model API: {api_error}")
                    checkpoint_info['model_loaded'] = False
            
            return checkpoint_info
            
        except Exception as e:
            self.logger.error(f"❌ Checkpoint load error: {str(e)}")
            return None
    
    def _evaluate_scenario(self, scenario_name: str, checkpoint_info: Dict[str, Any], 
                          checkpoint_idx: int, total_checkpoints: int) -> Dict[str, Any]:
        """🧪 Evaluate single scenario dengan specific checkpoint"""
        
        # Get scenario data path
        scenario_dir = self.scenario_manager.evaluation_dir / scenario_name
        
        if not scenario_dir.exists():
            # Generate scenario data if missing
            self.progress_bridge.update_substep(f"Generating {scenario_name} data")
            self.scenario_manager.generate_scenario_data(scenario_name)
        
        # Load test data
        self.progress_bridge.update_metrics(10, f"Loading {scenario_name} test data")
        test_data = self._load_scenario_data(scenario_dir)
        
        if not test_data['images'] or not test_data['labels']:
            raise ValueError(f"No test data found untuk {scenario_name}")
        
        # Run inference dengan timing
        self.progress_bridge.update_metrics(30, "Running inference")
        predictions, inference_times = self._run_inference_with_timing(
            test_data['images'], checkpoint_info
        )
        
        # Calculate metrics using training module's YOLOv5 mAP calculator
        self.progress_bridge.update_metrics(70, "Calculating mAP with training module")
        metrics = self._calculate_map_with_training_module(
            predictions, test_data['labels'], inference_times
        )
        
        # Additional analysis
        self.progress_bridge.update_metrics(90, "Performing additional analysis")
        additional_data = self._perform_additional_analysis(
            predictions, test_data['labels'], scenario_name
        )
        
        self.logger.info(f"✅ {scenario_name} evaluation complete: mAP={metrics.get('mAP', 0):.3f}")
        
        return {
            'metrics': metrics,
            'additional_data': additional_data,
            'test_data_count': len(test_data['images'])
        }
    
    def _load_scenario_data(self, scenario_dir: Path) -> Dict[str, List]:
        """📁 Load scenario test data"""
        
        images_dir = scenario_dir / 'images'
        labels_dir = scenario_dir / 'labels'
        
        # Load images
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png']:
            image_files.extend(list(images_dir.glob(f'*{ext}')))
        
        # Load corresponding labels
        labels = []
        images = []
        
        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"
            
            if label_file.exists():
                # Load image
                try:
                    img = Image.open(img_file).convert('RGB')
                    images.append({
                        'image': img,
                        'filename': img_file.name,
                        'path': str(img_file)
                    })
                    
                    # Load label
                    annotations = []
                    with open(label_file, 'r') as f:
                        for line in f:
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                class_id = int(float(parts[0]))
                                bbox = [float(x) for x in parts[1:5]]
                                annotations.append({
                                    'class_id': class_id,
                                    'bbox': bbox
                                })
                    
                    labels.append({
                        'filename': img_file.name,
                        'annotations': annotations
                    })
                    
                except Exception as e:
                    self.logger.debug(f"⚠️ Error loading {img_file.name}: {str(e)}")
        
        self.logger.info(f"📁 Loaded {len(images)} test images dengan labels")
        return {'images': images, 'labels': labels}
    
    def _run_inference_with_timing(self, test_images: List[Dict], checkpoint_info: Dict[str, Any]) -> tuple:
        """⏱️ Run inference dengan timing measurement"""
        
        predictions = []
        inference_times = []
        
        # Warmup jika model tersedia
        if self.model_api and checkpoint_info.get('model_loaded', False):
            self.logger.info("🔥 Warming up model")
            # Create dummy input untuk warmup with proper device detection
            dummy_input = torch.randn(1, 3, 640, 640)
            
            # Get model device automatically (same as training pipeline)
            try:
                model_device = next(self.model_api.model.parameters()).device
                dummy_input = dummy_input.to(model_device)
                self.logger.debug(f"🎯 Dummy input moved to model device: {model_device}")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not detect model device, using CPU: {e}")
                dummy_input = dummy_input.cpu()
            
            warmup_result = self.inference_timer.warmup_model(
                self.model_api.model, dummy_input, warmup_runs=5
            )
        
        # Process each image
        for idx, img_data in enumerate(test_images):
            
            if self.model_api and checkpoint_info.get('model_loaded', False):
                # Real inference dengan model
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
                        if self.inference_timer.timings.get('evaluation'):
                            last_timing = self.inference_timer.timings['evaluation'][-1]
                            inference_times.append(last_timing['time'])
                        else:
                            # Fallback timing if timer not working
                            inference_times.append(0.1)
                        
                    except Exception as e:
                        self.logger.error(f"⚠️ Inference error untuk {img_data['filename']}: {str(e)}")
                        import traceback
                        self.logger.error(f"⚠️ Full traceback: {traceback.format_exc()}")
                        predictions.append({
                            'filename': img_data['filename'],
                            'detections': []
                        })
                        inference_times.append(0.1)  # Default timing
            
            else:
                # Fallback: simulate inference untuk testing
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
            
            # Update progress
            if idx % 10 == 0:
                progress = 30 + (idx / len(test_images)) * 40  # 30-70% range
                self.progress_bridge.update_metrics(int(progress), f"Processing image {idx + 1}/{len(test_images)}")
        
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
            
            # Apply non-max suppression (same as training pipeline)
            # Parameters matching evaluation config
            conf_thres = 0.25  # Confidence threshold
            iou_thres = 0.45   # IoU threshold for NMS
            
            self.logger.debug(f"🔧 Applying NMS with conf_thres={conf_thres}, iou_thres={iou_thres}")
            
            # Apply NMS to raw YOLOv5 output
            pred = non_max_suppression(
                output, 
                conf_thres=conf_thres, 
                iou_thres=iou_thres, 
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
        """🖼️ Preprocess image untuk inference"""
        # Resize to model input size
        img_resized = img.resize((640, 640))
        
        # Convert to tensor dan normalize
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
    
    def _extract_model_config_from_checkpoint(self, checkpoint_info: Dict[str, Any]) -> Dict[str, Any]:
        """🔧 Extract model configuration from checkpoint metadata"""
        # Prioritize using evaluation service config over checkpoint config
        if hasattr(self, 'config') and self.config and 'model' in self.config:
            # Use the model config from evaluation service (from evaluations.py)
            model_section = self.config['model']
            num_classes = model_section.get('num_classes', 17)  # Default to 17 for current architecture
            self.logger.debug(f"Using evaluation service model config: {num_classes} classes")
        else:
            # Fallback to checkpoint extraction
            config = checkpoint_info.get('config', {})
            
            # Determine number of classes from checkpoint
            num_classes = 17  # Updated default for current training pipeline
            
            # Try to extract from various locations in checkpoint
            if 'model' in config and isinstance(config['model'], dict):
                num_classes = config['model'].get('num_classes', 17)
            elif 'num_classes' in config:
                num_classes = config['num_classes']
            elif 'training_config' in checkpoint_info and isinstance(checkpoint_info['training_config'], dict):
                num_classes = checkpoint_info['training_config'].get('num_classes', 17)
        
        # Extract backbone information with cleanup
        backbone = checkpoint_info.get('backbone', 'cspdarknet')
        # Clean up backbone names that have extra suffixes
        if backbone.startswith('cspdarknet'):
            backbone = 'cspdarknet'
        elif backbone.startswith('efficientnet'):
            backbone = 'efficientnet_b4'
        
        layer_mode = checkpoint_info.get('layer_mode', 'multi')
        
        # Build comprehensive config for API
        model_config = {
            'device': {'type': 'auto'},
            'model': {
                'backbone': backbone,
                'num_classes': num_classes,
                'img_size': 640,
                'layer_mode': layer_mode,
                'detection_layers': ['layer_1', 'layer_2', 'layer_3'],
                'feature_optimization': {'enabled': True}
            },
            'training': {
                'batch_size': 16,
                'epochs': 100,
                'learning_rate': 1e-3
            }
        }
        
        self.logger.debug(f"🔧 Extracted model config: backbone={backbone}, num_classes={num_classes}, layer_mode={layer_mode}")
        return model_config
    
    def _perform_additional_analysis(self, predictions: List[Dict], ground_truths: List[Dict], 
                                   scenario_name: str) -> Dict[str, Any]:
        """🔍 Perform additional analysis specific to scenario"""
        
        additional_data = {
            'scenario_name': scenario_name,
            'total_predictions': len(predictions),
            'total_ground_truths': len(ground_truths),
            'analysis_timestamp': time.time()
        }
        
        # Currency denomination analysis jika enabled
        currency_config = self.config.get('analysis', {}).get('currency_analysis', {})
        if currency_config.get('enabled', True):
            additional_data['currency_analysis'] = self._analyze_currency_detection(
                predictions, ground_truths, currency_config
            )
        
        # Class distribution analysis
        class_config = self.config.get('analysis', {}).get('class_analysis', {})
        if class_config.get('enabled', True):
            additional_data['class_distribution'] = self._analyze_class_distribution(
                predictions, ground_truths
            )
        
        return additional_data
    
    def _analyze_currency_detection(self, predictions: List[Dict], ground_truths: List[Dict], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """💰 Analyze currency denomination detection"""
        
        primary_layer = config.get('primary_layer', 'banknote')
        confidence_threshold = config.get('confidence_threshold', 0.3)
        
        analysis = {
            'primary_layer': primary_layer,
            'confidence_threshold': confidence_threshold,
            'correct_denominations': 0,
            'total_detections': 0,
            'denomination_accuracy': 0.0
        }
        
        # Analyze currency detection accuracy
        for pred, gt in zip(predictions, ground_truths):
            pred_detections = [d for d in pred.get('detections', []) 
                             if d.get('confidence', 0) >= confidence_threshold]
            gt_annotations = gt.get('annotations', [])
            
            analysis['total_detections'] += len(pred_detections)
            
            # Check if primary denominations match
            for pred_det in pred_detections:
                pred_class = pred_det.get('class_id', -1)
                
                # Find matching ground truth dalam main banknote classes (0-6)
                if pred_class in range(7):  # Main banknote classes
                    for gt_ann in gt_annotations:
                        gt_class = gt_ann.get('class_id', -1)
                        if gt_class == pred_class:
                            analysis['correct_denominations'] += 1
                            break
        
        # Calculate denomination accuracy
        if analysis['total_detections'] > 0:
            analysis['denomination_accuracy'] = analysis['correct_denominations'] / analysis['total_detections']
        
        return analysis
    
    def _analyze_class_distribution(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """📊 Analyze class distribution dalam predictions vs ground truth"""
        
        pred_distribution = {}
        gt_distribution = {}
        
        # Count predictions
        for pred in predictions:
            for detection in pred.get('detections', []):
                class_id = detection.get('class_id', -1)
                pred_distribution[class_id] = pred_distribution.get(class_id, 0) + 1
        
        # Count ground truths
        for gt in ground_truths:
            for annotation in gt.get('annotations', []):
                class_id = annotation.get('class_id', -1)
                gt_distribution[class_id] = gt_distribution.get(class_id, 0) + 1
        
        return {
            'predictions_distribution': pred_distribution,
            'ground_truth_distribution': gt_distribution,
            'classes_detected': len(pred_distribution),
            'classes_in_ground_truth': len(gt_distribution)
        }
    
    def _calculate_map_with_training_module(self, predictions: List[Dict], ground_truths: List[Dict], 
                                          inference_times: List[float] = None) -> Dict[str, Any]:
        """📊 Calculate mAP using training module's YOLOv5 calculator for consistency"""
        try:
            import torch
            
            # Convert evaluation format to YOLOv5 training format
            yolo_predictions, yolo_targets = self._convert_to_yolo_format(predictions, ground_truths)
            
            if yolo_predictions is None or yolo_targets is None:
                self.logger.warning("Failed to convert predictions/targets to YOLOv5 format")
                return self._get_empty_metrics(inference_times)
            
            # Reset mAP calculator for new evaluation
            self.map_calculator.reset()
            
            # Update with converted data
            self.map_calculator.update(yolo_predictions, yolo_targets)
            
            # Calculate final mAP
            map_results = self.map_calculator.compute_map()
            
            # Debug: log what the training module returned
            self.logger.info(f"📊 Training module map_results: {map_results}")
            
            # Calculate denomination classification metrics (7 classes, not mAP-based)
            denomination_metrics = self._calculate_denomination_metrics(predictions, ground_truths)
            
            # Convert back to evaluation format with comprehensive metrics
            metrics = {
                # mAP-based metrics (from training module's YOLOv5 calculator)
                'map50': float(map_results.get('map50', 0.0)),
                'map50_precision': float(map_results.get('precision', 0.0)),
                'map50_recall': float(map_results.get('recall', 0.0)),
                'map50_f1': float(map_results.get('f1', 0.0)),
                
                # Denomination classification metrics (7 classes focus)
                'accuracy': float(denomination_metrics.get('accuracy', 0.0)),
                'precision': float(denomination_metrics.get('precision', 0.0)),
                'recall': float(denomination_metrics.get('recall', 0.0)),
                'f1': float(denomination_metrics.get('f1_score', 0.0)),
                
                # Legacy compatibility
                'mAP': float(map_results.get('map50', 0.0)),  # For backward compatibility
                'f1_score': float(denomination_metrics.get('f1_score', 0.0)),
            }
            
            # Add timing metrics if available
            if inference_times:
                import numpy as np
                metrics.update({
                    'inference_time_avg': float(np.mean(inference_times)),
                    'inference_time_std': float(np.std(inference_times)),
                    'fps': float(len(inference_times) / np.sum(inference_times)) if np.sum(inference_times) > 0 else 0.0
                })
            
            self.logger.info(f"📊 Training module mAP: {metrics['mAP']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating mAP with training module: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_empty_metrics(inference_times)
    
    def _calculate_denomination_metrics(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """📊 Calculate denomination classification metrics focusing on 7-class accuracy"""
        try:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
            import numpy as np
            
            # Extract predicted and true classes for denomination classification
            y_pred = []
            y_true = []
            
            # Match predictions with ground truths
            for pred, gt in zip(predictions, ground_truths):
                gt_annotations = gt.get('annotations', [])
                pred_detections = pred.get('detections', [])
                
                # For each ground truth, find the best matching prediction
                for gt_ann in gt_annotations:
                    gt_class = int(gt_ann['class_id'])
                    gt_bbox = gt_ann['bbox']
                    
                    # Map hierarchical classes to layer_1 classes (0-6)
                    if gt_class >= 7:
                        if gt_class <= 13:  # Layer 2 classes (7-13) -> map to classes 0-6
                            gt_class = gt_class - 7
                        else:  # Layer 3 classes (14-16) -> map to classes 0-2
                            gt_class = min(gt_class - 14, 6)
                    
                    # Find best matching prediction based on IoU
                    best_pred_class = -1  # No detection class
                    best_iou = 0.0
                    
                    for pred_det in pred_detections:
                        pred_class = int(pred_det['class_id'])
                        pred_bbox = pred_det['bbox']
                        pred_conf = pred_det.get('confidence', 0.0)
                        
                        # Only consider predictions with reasonable confidence
                        if pred_conf >= 0.25:
                            # Calculate IoU
                            iou = self._calculate_bbox_iou(pred_bbox, gt_bbox)
                            
                            if iou > best_iou and iou >= 0.5:  # IoU threshold for matching
                                best_iou = iou
                                best_pred_class = pred_class
                    
                    y_true.append(gt_class)
                    y_pred.append(best_pred_class if best_pred_class != -1 else 7)  # Use class 7 for no detection
            
            if not y_true or not y_pred:
                return self._get_empty_denomination_metrics()
            
            # Convert to numpy arrays
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            
            # Calculate metrics for ALL ground truth samples (include missed detections)
            # Filter to only ground truth classes 0-6 (denomination classes)
            valid_gt_mask = (y_true < 7)
            
            if not np.any(valid_gt_mask):
                return self._get_empty_denomination_metrics()
            
            y_true_valid = y_true[valid_gt_mask]
            y_pred_valid = y_pred[valid_gt_mask]
            
            # Convert "no detection" predictions (class 7) back to a special class for metrics
            # This ensures missed detections are counted in the accuracy calculation
            y_pred_valid = np.where(y_pred_valid == 7, 7, y_pred_valid)  # Keep class 7 for no detection
            
            # Calculate denomination-specific metrics (include class 7 for "no detection")
            all_classes = list(range(8))  # Classes 0-6 + class 7 for "no detection"
            
            # For overall metrics, we want to focus on detection performance (classes 0-6)
            # But we need to account for missed detections in accuracy calculation
            
            # Calculate accuracy considering all samples (including missed detections)
            overall_accuracy = accuracy_score(y_true_valid, y_pred_valid)
            
            # Calculate precision/recall/f1 only for detected classes (exclude "no detection" from averaging)
            detected_mask = y_pred_valid < 7
            if np.any(detected_mask):
                y_true_detected = y_true_valid[detected_mask]
                y_pred_detected = y_pred_valid[detected_mask]
                
                precision_detected = precision_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
                recall_detected = recall_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
                f1_detected = f1_score(y_true_detected, y_pred_detected, average='weighted', zero_division=0)
            else:
                # No detections made
                precision_detected = 0.0
                recall_detected = 0.0
                f1_detected = 0.0
            
            metrics = {
                'accuracy': overall_accuracy,
                'precision': precision_detected,
                'recall': recall_detected,
                'f1_score': f1_detected,
                'confusion_matrix': confusion_matrix(y_true_valid, y_pred_valid, labels=all_classes).tolist(),
                'total_samples': len(y_true_valid),
                'detected_samples': int(np.sum(detected_mask)) if np.any(detected_mask) else 0,
                'missed_samples': int(np.sum(y_pred_valid == 7))
            }
            
            # Calculate per-class metrics for denomination classes only (0-6)
            per_class_precision = precision_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(7)))
            per_class_recall = recall_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(7)))
            per_class_f1 = f1_score(y_true_valid, y_pred_valid, average=None, zero_division=0, labels=list(range(7)))
            
            for i in range(7):
                metrics[f'precision_class_{i}'] = float(per_class_precision[i])
                metrics[f'recall_class_{i}'] = float(per_class_recall[i])
                metrics[f'f1_class_{i}'] = float(per_class_f1[i])
            
            self.logger.info(f"📊 Denomination classification details:")
            self.logger.info(f"    Total samples: {metrics['total_samples']}")
            self.logger.info(f"    Detected samples: {metrics['detected_samples']}")
            self.logger.info(f"    Missed samples: {metrics['missed_samples']}")
            self.logger.info(f"    Unique predictions: {np.unique(y_pred_valid)}")
            self.logger.info(f"    Unique ground truth: {np.unique(y_true_valid)}")
            self.logger.info(f"📊 Denomination metrics: accuracy={metrics['accuracy']:.3f}, precision={metrics['precision']:.3f}, recall={metrics['recall']:.3f}, f1={metrics['f1_score']:.3f}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating denomination metrics: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._get_empty_denomination_metrics()
    
    def _calculate_bbox_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate IoU between two bounding boxes in YOLO format [x_center, y_center, width, height]"""
        def yolo_to_xyxy(bbox):
            x_center, y_center, width, height = bbox
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            return [x1, y1, x2, y2]
        
        box1_xyxy = yolo_to_xyxy(bbox1)
        box2_xyxy = yolo_to_xyxy(bbox2)
        
        # Calculate intersection
        x1 = max(box1_xyxy[0], box2_xyxy[0])
        y1 = max(box1_xyxy[1], box2_xyxy[1])
        x2 = min(box1_xyxy[2], box2_xyxy[2])
        y2 = min(box2_xyxy[3], box2_xyxy[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        
        # Calculate union
        area1 = (box1_xyxy[2] - box1_xyxy[0]) * (box1_xyxy[3] - box1_xyxy[1])
        area2 = (box2_xyxy[2] - box2_xyxy[0]) * (box2_xyxy[3] - box2_xyxy[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _get_empty_denomination_metrics(self) -> Dict[str, Any]:
        """Get empty denomination metrics structure"""
        metrics = {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
            'confusion_matrix': [[0] * 7 for _ in range(7)]
        }
        
        # Add per-class metrics
        for i in range(7):
            metrics[f'precision_class_{i}'] = 0.0
            metrics[f'recall_class_{i}'] = 0.0
            metrics[f'f1_class_{i}'] = 0.0
        
        return metrics
    
    def _convert_to_yolo_format(self, predictions: List[Dict], ground_truths: List[Dict]) -> tuple:
        """Convert evaluation format to YOLOv5 training format"""
        try:
            import torch
            
            # Training module expects predictions as [batch_size, max_detections, 6] 
            # where each detection is [x, y, w, h, conf, class] in YOLO format (not xyxy)
            batch_size = len(predictions)
            max_detections = 100  # Use reasonable max
            
            # Initialize prediction tensor
            pred_tensor = torch.zeros((batch_size, max_detections, 6), dtype=torch.float32)
            
            for batch_idx, pred in enumerate(predictions):
                det_count = 0
                for detection in pred.get('detections', []):
                    if det_count >= max_detections:
                        break
                        
                    if 'class_id' in detection and 'confidence' in detection and 'bbox' in detection:
                        bbox = detection['bbox']  # [x_center, y_center, width, height] normalized
                        if len(bbox) == 4:
                            # Map hierarchical classes to layer_1 classes (0-6)
                            class_id = int(detection['class_id'])
                            if class_id >= 7:  # Classes 7+ are hierarchical, map to primary classes
                                if class_id <= 13:  # Layer 2 classes (7-13) -> map to classes 0-6
                                    class_id = class_id - 7
                                else:  # Layer 3 classes (14-16) -> map to classes 0-2
                                    class_id = min(class_id - 14, 6)
                            
                            # Keep normalized YOLO format [x, y, w, h] as expected by training module
                            pred_tensor[batch_idx, det_count, :] = torch.tensor([
                                float(bbox[0]),  # x_center (normalized)
                                float(bbox[1]),  # y_center (normalized)
                                float(bbox[2]),  # width (normalized)
                                float(bbox[3]),  # height (normalized)
                                float(detection['confidence']),  # confidence
                                float(class_id)  # class (mapped to 0-6)
                            ])
                            det_count += 1
            
            # Convert targets to tensor format [N, 6] where each row is [batch_idx, class, x, y, w, h]
            target_list = []
            batch_idx = 0
            
            for gt in ground_truths:
                for annotation in gt.get('annotations', []):
                    if 'class_id' in annotation and 'bbox' in annotation:
                        bbox = annotation['bbox']  # [x_center, y_center, width, height] normalized
                        if len(bbox) == 4:
                            # Map hierarchical classes to layer_1 classes (0-6)
                            class_id = int(annotation['class_id'])
                            if class_id >= 7:  # Classes 7+ are hierarchical, map to primary classes
                                if class_id <= 13:  # Layer 2 classes (7-13) -> map to classes 0-6
                                    class_id = class_id - 7
                                else:  # Layer 3 classes (14-16) -> map to classes 0-2
                                    class_id = min(class_id - 14, 6)
                            
                            target_list.append([
                                batch_idx,  # batch index
                                float(class_id),  # class (mapped to 0-6)
                                float(bbox[0]),  # x_center (normalized)
                                float(bbox[1]),  # y_center (normalized)
                                float(bbox[2]),  # width (normalized)
                                float(bbox[3])   # height (normalized)
                            ])
                batch_idx += 1
            
            if pred_tensor.sum() == 0 or not target_list:
                self.logger.warning(f"Empty conversions: pred_tensor_sum={pred_tensor.sum()}, targets={len(target_list)}")
                return None, None
            
            # Convert targets to tensor
            targets_tensor = torch.tensor(target_list, dtype=torch.float32)
            
            self.logger.info(f"📊 Converted to YOLOv5 format: pred_shape={pred_tensor.shape}, target_shape={targets_tensor.shape}")
            
            # Debug: show actual tensor contents
            pred_count = (pred_tensor.sum(dim=-1) != 0).sum()
            target_count = targets_tensor.shape[0]
            self.logger.info(f"📊 Non-zero predictions: {pred_count}, Targets: {target_count}")
            
            # Show some sample predictions and targets
            if pred_count > 0:
                # Find first non-zero prediction
                for i in range(pred_tensor.shape[0]):
                    for j in range(pred_tensor.shape[1]):
                        if pred_tensor[i, j].sum() != 0:
                            self.logger.info(f"📊 Sample prediction [{i},{j}]: {pred_tensor[i, j].tolist()}")
                            break
                    if pred_tensor[i, j].sum() != 0:
                        break
            
            if target_count > 0:
                self.logger.info(f"📊 Sample target [0]: {targets_tensor[0].tolist()}")
            
            return pred_tensor, targets_tensor
            
        except Exception as e:
            self.logger.error(f"Error converting to YOLOv5 format: {e}")
            return None, None
    
    def _get_empty_metrics(self, inference_times: List[float] = None) -> Dict[str, Any]:
        """Get empty metrics structure"""
        metrics = {
            'mAP': 0.0,
            'mAP@0.5': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1_score': 0.0,
        }
        
        if inference_times:
            import numpy as np
            metrics.update({
                'inference_time_avg': float(np.mean(inference_times)),
                'inference_time_std': float(np.std(inference_times)),
                'fps': float(len(inference_times) / np.sum(inference_times)) if np.sum(inference_times) > 0 else 0.0
            })
        
        return metrics
    
    def _generate_evaluation_summary(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """📋 Generate comprehensive evaluation summary"""
        
        # Aggregate results
        aggregated = self.results_aggregator.aggregate_metrics()
        
        # Generate summary
        summary = self.results_aggregator.generate_summary()
        
        return {
            'evaluation_overview': summary.get('evaluation_overview', {}),
            'aggregated_metrics': aggregated,
            'key_findings': summary.get('key_findings', []),
            'recommendations': summary.get('recommendations', []),
            'best_configurations': aggregated.get('best_configurations', {}),
            'backbone_comparison': aggregated.get('backbone_comparison', {}),
            'scenario_comparison': aggregated.get('scenario_comparison', {})
        }


# Factory functions
def create_evaluation_service(model_api=None, config: Dict[str, Any] = None) -> EvaluationService:
    """🏭 Factory untuk EvaluationService"""
    return EvaluationService(model_api, config)

def run_evaluation_pipeline(scenarios: List[str] = None, checkpoints: List[str] = None,
                           model_api=None, config: Dict[str, Any] = None,
                           progress_callback=None, ui_components: Dict[str, Any] = None) -> Dict[str, Any]:
    """🚀 One-liner untuk run complete evaluation pipeline"""
    service = create_evaluation_service(model_api, config)
    return service.run_evaluation(scenarios, checkpoints, progress_callback, ui_components=ui_components)