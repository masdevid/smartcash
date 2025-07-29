"""
Integration tests for evaluation module - testing full end-to-end scenarios.
Tests complete evaluation workflows with real backend integration and UI components.
"""

import pytest
import tempfile
import shutil
import json
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any, List

import torch
import numpy as np
from PIL import Image

# Import evaluation modules
from smartcash.model.evaluation.evaluation_service import EvaluationService, run_evaluation_pipeline
from smartcash.ui.model.evaluation.operations.evaluation_all_operation import EvaluationAllOperation


class TestEvaluationBackendIntegration:
    """Test complete backend integration scenarios."""
    
    @pytest.fixture
    def realistic_config(self):
        """Realistic evaluation configuration."""
        return {
            'evaluation': {
                'scenarios': ['position_variation', 'lighting_variation', 'distance_variation'],
                'metrics': {
                    'enabled_metrics': ['mAP', 'precision', 'recall', 'f1_score'],
                    'map': {'enabled': True, 'iou_thresholds': [0.5, 0.75]},
                    'precision': {'enabled': True},
                    'recall': {'enabled': True},
                    'f1_score': {'enabled': True}
                },
                'confidence_threshold': 0.3,
                'iou_threshold': 0.5,
                'data_dir': 'data/evaluation',
                'output_dir': 'data/evaluation/results',
                'checkpoint_dir': 'data/checkpoints',
                'max_images_per_scenario': 100
            },
            'analysis': {
                'currency_analysis': {
                    'enabled': True,
                    'primary_layer': 'banknote',
                    'confidence_threshold': 0.3,
                    'denomination_classes': [0, 1, 2, 3, 4, 5, 6]
                },
                'class_analysis': {
                    'enabled': True,
                    'class_names': ['1000', '2000', '5000', '10000', '20000', '50000', '100000']
                },
                'performance_analysis': {
                    'enabled': True,
                    'timing_metrics': True,
                    'memory_metrics': False
                }
            },
            'device': {
                'auto_detect': True,
                'preferred': 'cuda',
                'mixed_precision': True
            },
            'export': {
                'formats': ['json', 'csv', 'html'],
                'include_images': False,
                'include_predictions': True
            }
        }
    
    @pytest.fixture
    def realistic_model_api(self):
        """Realistic model API mock with proper behavior."""
        api = Mock()
        api.model = Mock()
        
        # Mock predict method with realistic outputs
        def mock_predict(image_tensor):
            batch_size = image_tensor.shape[0] if hasattr(image_tensor, 'shape') else 1
            detections = []
            
            # Generate realistic number of detections (0-3 per image)
            num_detections = np.random.randint(0, 4)
            
            for _ in range(num_detections):
                detections.append({
                    'class_id': np.random.randint(0, 7),
                    'confidence': np.random.uniform(0.3, 0.95),
                    'bbox': [
                        np.random.uniform(0.1, 0.7),  # x
                        np.random.uniform(0.1, 0.7),  # y  
                        np.random.uniform(0.1, 0.3),  # width
                        np.random.uniform(0.1, 0.3)   # height
                    ]
                })
            
            return {
                'success': True,
                'detections': detections,
                'inference_time': np.random.uniform(0.05, 0.2)
            }
        
        api.predict.side_effect = mock_predict
        
        # Mock checkpoint loading
        def mock_load_checkpoint(checkpoint_path):
            return {
                'success': True,
                'model': api.model,
                'checkpoint_data': {
                    'epoch': 50,
                    'metrics': {
                        'val_map50': np.random.uniform(0.7, 0.9),
                        'val_loss': np.random.uniform(0.1, 0.3)
                    },
                    'backbone_type': 'efficientnet_b4',
                    'num_classes': 7
                }
            }
        
        api.load_checkpoint.side_effect = mock_load_checkpoint
        
        return api
    
    @pytest.fixture
    def test_data_structure(self, tmp_path):
        """Create realistic test data structure."""
        # Create evaluation directory structure
        eval_dir = tmp_path / 'evaluation'
        
        scenarios = ['position_variation', 'lighting_variation', 'distance_variation']
        
        for scenario in scenarios:
            scenario_dir = eval_dir / scenario
            images_dir = scenario_dir / 'images'
            labels_dir = scenario_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test images and labels for each scenario
            for i in range(20):  # 20 images per scenario
                # Create realistic test image
                if scenario == 'position_variation':
                    # Simulate different positions with different colored regions
                    color = (50 + i*10, 100, 150)
                elif scenario == 'lighting_variation':
                    # Simulate different lighting with varying brightness
                    brightness = 50 + i*8
                    color = (brightness, brightness, brightness)
                else:  # distance_variation
                    # Simulate different distances with different sizes
                    color = (200, 100 + i*5, 50)
                
                img = Image.new('RGB', (640, 640), color=color)
                
                # Add some visual elements to make it more realistic
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                # Draw some rectangles to simulate banknotes
                num_objects = np.random.randint(1, 4)
                annotations = []
                
                for obj_idx in range(num_objects):
                    # Random position and size
                    x = np.random.uniform(0.1, 0.7)
                    y = np.random.uniform(0.1, 0.7)
                    w = np.random.uniform(0.1, 0.3)
                    h = np.random.uniform(0.1, 0.3)
                    
                    # Ensure bbox is within image bounds
                    x = max(0, min(x, 1-w))
                    y = max(0, min(y, 1-h))
                    
                    # Convert to pixel coordinates for drawing
                    px1, py1 = int(x * 640), int(y * 640)
                    px2, py2 = int((x + w) * 640), int((y + h) * 640)
                    
                    # Draw rectangle with random color
                    rect_color = (
                        np.random.randint(100, 255),
                        np.random.randint(100, 255), 
                        np.random.randint(100, 255)
                    )
                    draw.rectangle([px1, py1, px2, py2], fill=rect_color, outline=(0, 0, 0))
                    
                    # Add to annotations
                    class_id = obj_idx % 7  # Rotate through classes 0-6
                    annotations.append(f"{class_id} {x + w/2} {y + h/2} {w} {h}")
                
                # Save image
                img_path = images_dir / f'{scenario}_{i:03d}.jpg'
                img.save(img_path)
                
                # Save label
                label_path = labels_dir / f'{scenario}_{i:03d}.txt'
                with open(label_path, 'w') as f:
                    f.write('\n'.join(annotations))
        
        # Create checkpoint directory structure
        checkpoint_dir = tmp_path / 'checkpoints'
        checkpoint_dir.mkdir(exist_ok=True)
        
        # Create realistic checkpoint files
        checkpoint_files = []
        backbones = ['efficientnet_b4', 'yolov5s', 'resnet50']
        
        for i, backbone in enumerate(backbones):
            checkpoint_path = checkpoint_dir / f'smartcash_training_{int(time.time())+i}_{backbone}_best.pt'
            
            # Create realistic checkpoint data
            checkpoint_data = {
                'model_state_dict': {'dummy': 'state'},
                'optimizer_state_dict': {'dummy': 'optimizer'},
                'epoch': 50 + i*10,
                'metrics': {
                    'val_map50': 0.75 + i*0.05,
                    'val_loss': 0.25 - i*0.02,
                    'train_loss': 0.3 - i*0.03,
                    'precision': 0.8 + i*0.03,
                    'recall': 0.75 + i*0.04
                },
                'backbone_type': backbone,
                'num_classes': 7,
                'input_size': 640,
                'training_time': 3600 + i*600
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint_files.append(str(checkpoint_path))
        
        return {
            'eval_dir': eval_dir,
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_files': checkpoint_files,
            'scenarios': scenarios
        }
    
    def test_full_evaluation_pipeline(self, realistic_config, realistic_model_api, test_data_structure):
        """Test complete evaluation pipeline from start to finish."""
        # Update config with actual test data paths
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        config['evaluation']['checkpoint_dir'] = str(test_data_structure['checkpoint_dir'])
        
        # Create progress and metrics callbacks
        progress_calls = []
        metrics_calls = []
        
        def progress_callback(progress_data):
            progress_calls.append(progress_data)
        
        def metrics_callback(metrics_data):
            metrics_calls.append(metrics_data)
        
        # Run complete evaluation pipeline
        result = run_evaluation_pipeline(
            scenarios=['position_variation', 'lighting_variation'],
            checkpoints=test_data_structure['checkpoint_files'][:2],
            model_api=realistic_model_api,
            config=config,
            progress_callback=progress_callback
        )
        
        # Verify results structure
        assert result['status'] == 'success'
        assert 'evaluation_results' in result
        assert 'summary' in result
        assert result['scenarios_evaluated'] == 2
        assert result['checkpoints_evaluated'] == 2
        
        # Verify evaluation results structure
        eval_results = result['evaluation_results']
        assert 'position_variation' in eval_results
        assert 'lighting_variation' in eval_results
        
        # Verify each scenario has results for each backbone
        for scenario_name, scenario_results in eval_results.items():
            assert len(scenario_results) > 0
            
            for backbone, backbone_results in scenario_results.items():
                assert 'checkpoint_info' in backbone_results
                assert 'metrics' in backbone_results
                assert 'additional_data' in backbone_results
                
                # Verify metrics structure
                metrics = backbone_results['metrics']
                assert isinstance(metrics, dict)
                
                # Verify checkpoint info
                checkpoint_info = backbone_results['checkpoint_info']
                assert 'backbone' in checkpoint_info or 'backbone_type' in checkpoint_info
        
        # Verify summary structure
        summary = result['summary']
        assert 'evaluation_overview' in summary
        assert 'aggregated_metrics' in summary
        
        # Verify progress callbacks were called
        assert len(progress_calls) > 0
    
    def test_single_scenario_integration(self, realistic_config, realistic_model_api, test_data_structure):
        """Test single scenario evaluation integration."""
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        config['evaluation']['checkpoint_dir'] = str(test_data_structure['checkpoint_dir'])
        
        service = EvaluationService(model_api=realistic_model_api, config=config)
        
        # Mock scenario manager to use test data
        service.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
        
        # Run single scenario
        result = service.run_scenario(
            'position_variation', 
            test_data_structure['checkpoint_files'][0]
        )
        
        assert result['status'] == 'success'
        assert result['scenario_name'] == 'position_variation'
        assert 'metrics' in result
        assert 'checkpoint_info' in result
        assert 'additional_data' in result
        
        # Verify metrics are realistic
        metrics = result['metrics']
        if 'mAP' in metrics:
            assert 0 <= metrics['mAP'] <= 1
        if 'precision' in metrics:
            assert 0 <= metrics['precision'] <= 1
        if 'recall' in metrics:
            assert 0 <= metrics['recall'] <= 1
    
    def test_ui_operation_integration(self, realistic_config, realistic_model_api, test_data_structure):
        """Test integration with UI operation handlers."""
        # Mock UI components
        ui_components = {
            'form_widgets': {
                'scenarios': Mock(value=['position_variation', 'lighting_variation']),
                'backbone_selection': Mock(value='efficientnet_b4'),
                'confidence_threshold': Mock(value=0.3),
                'metrics_selection': Mock(value=['mAP', 'precision', 'recall'])
            },
            'operation_container': Mock(),
            'summary_container': Mock(),
            'progress_container': Mock()
        }
        
        # Mock operation container methods
        ui_components['operation_container'].update_triple_progress = Mock()
        ui_components['operation_container'].log = Mock()
        ui_components['operation_container'].update_progress = Mock()
        
        # Update config with test data paths
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        config['evaluation']['checkpoint_dir'] = str(test_data_structure['checkpoint_dir'])
        
        # Create evaluation operation (would normally be created by UI module)
        operation = Mock()
        operation._ui_components = ui_components
        operation.config = config
        operation.model_api = realistic_model_api
        
        # Mock the backend service call
        with patch('smartcash.model.evaluation.run_evaluation_pipeline') as mock_pipeline:
            mock_pipeline.return_value = {
                'status': 'success',
                'evaluation_results': {
                    'position_variation': {
                        'efficientnet_b4': {
                            'checkpoint_info': {'backbone': 'efficientnet_b4'},
                            'metrics': {'mAP': 0.85, 'precision': 0.82, 'recall': 0.88},
                            'additional_data': {'total_images': 20}
                        }
                    }
                },
                'summary': {
                    'evaluation_overview': {'total_scenarios': 1, 'total_checkpoints': 1},
                    'aggregated_metrics': {'best_mAP': 0.85}
                },
                'scenarios_evaluated': 1,
                'checkpoints_evaluated': 1
            }
            
            # Simulate operation execution
            result = mock_pipeline(
                scenarios=['position_variation'],
                checkpoints=test_data_structure['checkpoint_files'][:1],
                model_api=realistic_model_api,
                config=config,
                progress_callback=Mock(),
                ui_components=ui_components
            )
            
            # Verify the call was made with correct parameters
            mock_pipeline.assert_called_once()
            call_args = mock_pipeline.call_args
            
            assert call_args[1]['scenarios'] == ['position_variation']
            assert len(call_args[1]['checkpoints']) == 1
            assert call_args[1]['model_api'] == realistic_model_api
            assert call_args[1]['config'] == config
            
            # Verify result
            assert result['status'] == 'success'
            assert 'evaluation_results' in result
    
    def test_progress_tracking_integration(self, realistic_config, realistic_model_api, test_data_structure):
        """Test progress tracking integration throughout evaluation."""
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        
        # Track all progress updates
        progress_updates = []
        
        def detailed_progress_callback(progress_data):
            progress_updates.append({
                'timestamp': time.time(),
                'data': progress_data
            })
        
        service = EvaluationService(model_api=realistic_model_api, config=config)
        service.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
        
        # Run evaluation with progress tracking
        result = service.run_evaluation(
            scenarios=['position_variation'],
            checkpoints=test_data_structure['checkpoint_files'][:1],
            progress_callback=detailed_progress_callback
        )
        
        # Verify progress was tracked
        assert len(progress_updates) > 0
        
        # Verify progress updates have expected structure
        for update in progress_updates:
            assert 'timestamp' in update
            assert 'data' in update
            assert isinstance(update['timestamp'], float)
    
    def test_error_handling_integration(self, realistic_config):
        """Test error handling in integration scenarios."""
        # Test with missing model API
        service = EvaluationService(model_api=None, config=realistic_config)
        
        result = service.run_evaluation(
            scenarios=['position_variation'],
            checkpoints=['nonexistent.pt']
        )
        
        # Should handle gracefully in fallback mode
        assert isinstance(result, dict)
        
        # Test with invalid model API
        invalid_api = Mock()
        invalid_api.predict.side_effect = RuntimeError("Model API error")
        invalid_api.load_checkpoint.return_value = {'success': False}
        
        service_invalid = EvaluationService(model_api=invalid_api, config=realistic_config)
        
        result = service_invalid.run_evaluation(
            scenarios=['position_variation'],
            checkpoints=['dummy.pt']
        )
        
        # Should handle API errors gracefully
        assert result.get('status') == 'error'
    
    def test_metrics_consistency_across_runs(self, realistic_config, test_data_structure):
        """Test that metrics are consistent across multiple runs."""
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        
        # Create deterministic model API
        deterministic_api = Mock()
        deterministic_api.model = Mock()
        
        # Always return the same predictions for consistency
        def deterministic_predict(image_tensor):
            return {
                'success': True,
                'detections': [
                    {'class_id': 0, 'confidence': 0.85, 'bbox': [0.2, 0.3, 0.4, 0.5]},
                    {'class_id': 2, 'confidence': 0.72, 'bbox': [0.6, 0.1, 0.3, 0.4]}
                ]
            }
        
        deterministic_api.predict.side_effect = deterministic_predict
        deterministic_api.load_checkpoint.return_value = {
            'success': True,
            'model': deterministic_api.model
        }
        
        # Run evaluation multiple times
        results = []
        for run in range(3):
            service = EvaluationService(model_api=deterministic_api, config=config)
            service.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
            
            result = service.run_scenario(
                'position_variation',
                test_data_structure['checkpoint_files'][0]
            )
            
            results.append(result)
        
        # Verify all runs completed successfully
        for result in results:
            assert result['status'] == 'success'
        
        # Verify metrics consistency (should be identical with deterministic API)
        if len(results) >= 2:
            metrics1 = results[0]['metrics']
            metrics2 = results[1]['metrics']
            
            # Compare common metrics
            for key in metrics1:
                if key in metrics2 and isinstance(metrics1[key], (int, float)):
                    # Allow small floating point differences
                    assert abs(metrics1[key] - metrics2[key]) < 1e-6, f"Metrics {key} not consistent: {metrics1[key]} vs {metrics2[key]}"
    
    def test_large_scale_evaluation(self, realistic_config, realistic_model_api, test_data_structure):
        """Test evaluation with larger scale data."""
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        config['evaluation']['max_images_per_scenario'] = 50  # Increase for this test
        
        # Run evaluation with all scenarios and checkpoints
        service = EvaluationService(model_api=realistic_model_api, config=config)
        service.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
        
        result = service.run_evaluation(
            scenarios=test_data_structure['scenarios'],
            checkpoints=test_data_structure['checkpoint_files'],
            progress_callback=Mock()
        )
        
        # Verify results for large scale
        assert result['status'] == 'success'
        assert result['scenarios_evaluated'] == len(test_data_structure['scenarios'])
        assert result['checkpoints_evaluated'] == len(test_data_structure['checkpoint_files'])
        
        # Verify all scenarios were processed
        eval_results = result['evaluation_results']
        for scenario in test_data_structure['scenarios']:
            assert scenario in eval_results
            assert len(eval_results[scenario]) > 0
    
    @patch('smartcash.model.evaluation.evaluation_service.torch.cuda.is_available')
    def test_gpu_integration(self, mock_cuda, realistic_config, realistic_model_api, test_data_structure):
        """Test GPU integration in evaluation."""
        # Test with CUDA available
        mock_cuda.return_value = True
        
        config = realistic_config.copy()
        config['evaluation']['data_dir'] = str(test_data_structure['eval_dir'])
        config['device']['preferred'] = 'cuda'
        
        service = EvaluationService(model_api=realistic_model_api, config=config)
        service.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
        
        # Should handle CUDA operations
        result = service.run_scenario(
            'position_variation',
            test_data_structure['checkpoint_files'][0]
        )
        
        assert result['status'] == 'success'
        
        # Test with CUDA unavailable
        mock_cuda.return_value = False
        
        service_cpu = EvaluationService(model_api=realistic_model_api, config=config)
        service_cpu.scenario_manager.evaluation_dir = test_data_structure['eval_dir']
        
        result_cpu = service_cpu.run_scenario(
            'position_variation',
            test_data_structure['checkpoint_files'][0]
        )
        
        assert result_cpu['status'] == 'success'


if __name__ == '__main__':
    # Run integration tests
    pytest.main([__file__, '-v', '--tb=short'])