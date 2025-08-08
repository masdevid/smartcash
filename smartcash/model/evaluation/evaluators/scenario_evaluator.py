"""
Single scenario evaluator.
Handles evaluation of a single scenario (e.g., position_variation).
"""

from pathlib import Path
from typing import Dict, Any, List, Optional
import time

from smartcash.common.logger import get_logger
from smartcash.model.training.core.yolov5_map_calculator import YOLOv5MapCalculator
from smartcash.model.evaluation.processors.data_loader import create_evaluation_data_loader
from smartcash.model.evaluation.processors.inference_processor import create_inference_processor
from smartcash.model.evaluation.metrics.hierarchical_metrics_calculator import create_hierarchical_metrics_calculator
from smartcash.model.evaluation.converters.yolov5_format_converter import create_yolov5_format_converter


class ScenarioEvaluator:
    """Evaluate a single scenario with a specific checkpoint"""
    
    def __init__(self, scenario_manager=None, inference_timer=None, num_classes: int = 17):
        self.logger = get_logger('scenario_evaluator')
        self.scenario_manager = scenario_manager
        self.inference_timer = inference_timer
        self.num_classes = num_classes
        
        # Initialize components
        self.data_loader = create_evaluation_data_loader()
        self.hierarchical_metrics = create_hierarchical_metrics_calculator(num_classes=num_classes)
        self.yolov5_converter = create_yolov5_format_converter(num_classes=num_classes)
        
        # Use training module's YOLOv5 mAP calculator for consistency
        self.map_calculator = YOLOv5MapCalculator(
            num_classes=num_classes,  # Full hierarchical classification
            conf_thres=0.005,
            iou_thres=0.03,
            debug=True,
            training_context={'evaluation_mode': True, 'backbone': 'evaluation'}
        )
    
    def evaluate_scenario(self, scenario_name: str, checkpoint_info: Dict[str, Any], 
                         model_api=None, progress_callback=None) -> Dict[str, Any]:
        """🧪 Evaluate single scenario with specific checkpoint"""
        
        # Get scenario data path
        if self.scenario_manager:
            scenario_dir = self.scenario_manager.evaluation_dir / scenario_name
        else:
            scenario_dir = Path('data/evaluation/scenarios') / scenario_name
        
        if not scenario_dir.exists():
            # Generate scenario data if missing and scenario manager available
            if self.scenario_manager:
                if progress_callback:
                    progress_callback(10, f"Generating {scenario_name} data")
                self.scenario_manager.generate_scenario_data(scenario_name)
            else:
                raise ValueError(f"Scenario directory does not exist: {scenario_dir}")
        
        # Load test data
        if progress_callback:
            progress_callback(20, f"Loading {scenario_name} test data")
        
        test_data = self.data_loader.load_scenario_data(scenario_dir)
        
        if not test_data['images'] or not test_data['labels']:
            raise ValueError(f"No test data found for {scenario_name}")
        
        # Run inference with timing
        if progress_callback:
            progress_callback(30, "Running inference")
        
        inference_processor = create_inference_processor(model_api=model_api, inference_timer=self.inference_timer)
        predictions, inference_times = inference_processor.run_inference_with_timing(
            test_data['images'], checkpoint_info
        )
        
        # Calculate metrics using training module's YOLOv5 mAP calculator
        if progress_callback:
            progress_callback(70, "Calculating mAP with training module")
        
        metrics = self._calculate_map_with_training_module(
            predictions, test_data['labels'], inference_times
        )
        
        # Additional analysis
        if progress_callback:
            progress_callback(90, "Performing additional analysis")
        
        additional_data = self._perform_additional_analysis(
            predictions, test_data['labels'], scenario_name
        )
        
        self.logger.info(f"✅ {scenario_name} evaluation complete: mAP={metrics.get('mAP', 0):.3f}")
        
        return {
            'metrics': metrics,
            'additional_data': additional_data,
            'test_data_count': len(test_data['images']),
            'timestamp': time.time()
        }
    
    def _calculate_map_with_training_module(self, predictions: List[Dict], ground_truths: List[Dict], 
                                          inference_times: List[float] = None) -> Dict[str, Any]:
        """📊 Calculate mAP using training module's YOLOv5 calculator for consistency"""
        try:
            # Convert evaluation format to YOLOv5 training format
            yolo_predictions, yolo_targets = self.yolov5_converter.convert_to_yolo_format(predictions, ground_truths)
            
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
            
            # Calculate denomination classification metrics (hierarchical classes)
            denomination_metrics = self.hierarchical_metrics.calculate_denomination_metrics(predictions, ground_truths)
            
            # Convert back to evaluation format with comprehensive metrics
            metrics = {
                # mAP-based metrics (from training module's YOLOv5 calculator)
                'map50': float(map_results.get('map50', 0.0)),
                'map50_precision': float(map_results.get('precision', 0.0)),
                'map50_recall': float(map_results.get('recall', 0.0)),
                'map50_f1': float(map_results.get('f1', 0.0)),
                
                # Denomination classification metrics (hierarchical classes focus)
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
    
    def _perform_additional_analysis(self, predictions: List[Dict], ground_truths: List[Dict], 
                                   scenario_name: str) -> Dict[str, Any]:
        """🔍 Perform additional analysis specific to scenario"""
        
        additional_data = {
            'scenario_name': scenario_name,
            'total_predictions': len(predictions),
            'total_ground_truths': len(ground_truths),
            'analysis_timestamp': time.time()
        }
        
        # Currency denomination analysis
        currency_config = {'enabled': True, 'primary_layer': 'banknote', 'confidence_threshold': 0.3}
        additional_data['currency_analysis'] = self._analyze_currency_detection(
            predictions, ground_truths, currency_config
        )
        
        # Class distribution analysis
        additional_data['class_distribution'] = self._analyze_class_distribution(
            predictions, ground_truths
        )
        
        return additional_data
    
    def _analyze_currency_detection(self, predictions: List[Dict], ground_truths: List[Dict], 
                                  config: Dict[str, Any]) -> Dict[str, Any]:
        """💰 Analyze currency denomination detection"""
        
        confidence_threshold = config.get('confidence_threshold', 0.3)
        
        analysis = {
            'confidence_threshold': confidence_threshold,
            'correct_denominations': 0,
            'total_detections': 0,
            'denomination_accuracy': 0.0
        }
        
        # Analyze currency detection accuracy using hierarchical logic
        for pred, gt in zip(predictions, ground_truths):
            pred_detections = [d for d in pred.get('detections', []) 
                             if d.get('confidence', 0) >= confidence_threshold]
            gt_annotations = gt.get('annotations', [])
            
            analysis['total_detections'] += len(pred_detections)
            
            # Check if denominations match using hierarchical logic
            for pred_det in pred_detections:
                pred_class = pred_det.get('class_id', -1)
                
                # Find matching ground truth
                for gt_ann in gt_annotations:
                    gt_class = gt_ann.get('class_id', -1)
                    
                    # Use hierarchical matching logic
                    if (pred_class == gt_class or 
                        self.hierarchical_metrics._is_hierarchical_match(gt_class, pred_class)):
                        analysis['correct_denominations'] += 1
                        break
        
        # Calculate denomination accuracy
        if analysis['total_detections'] > 0:
            analysis['denomination_accuracy'] = analysis['correct_denominations'] / analysis['total_detections']
        
        return analysis
    
    def _analyze_class_distribution(self, predictions: List[Dict], ground_truths: List[Dict]) -> Dict[str, Any]:
        """📊 Analyze class distribution in predictions vs ground truth"""
        
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


def create_scenario_evaluator(scenario_manager=None, inference_timer=None, num_classes: int = 17) -> ScenarioEvaluator:
    """Factory function to create scenario evaluator"""
    return ScenarioEvaluator(scenario_manager=scenario_manager, inference_timer=inference_timer, num_classes=num_classes)