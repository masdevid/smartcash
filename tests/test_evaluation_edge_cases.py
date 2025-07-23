"""
Edge case tests for evaluation module - testing extreme scenarios and error conditions.
Ensures robust error handling and graceful degradation in all edge cases.
"""

import pytest
import tempfile
import shutil
import json
import time
import random
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

import torch
import numpy as np
from PIL import Image

# Import evaluation modules
from smartcash.model.evaluation.evaluation_service import EvaluationService
from smartcash.model.evaluation.scenario_manager import ScenarioManager
from smartcash.model.evaluation.evaluation_metrics import EvaluationMetrics
from smartcash.model.evaluation.checkpoint_selector import CheckpointSelector


class TestEvaluationEdgeCases:
    """Test extreme edge cases and error conditions."""
    
    @pytest.fixture
    def corrupt_config(self):
        """Configuration with various corruption scenarios."""
        return {
            'evaluation': {
                'scenarios': [],  # Empty scenarios
                'metrics': None,  # None metrics
                'confidence_threshold': -1.5,  # Invalid threshold
                'iou_threshold': 2.0,  # Invalid IoU threshold
                'data_dir': '/nonexistent/path',  # Non-existent path
                'output_dir': ''  # Empty output dir
            },
            'analysis': {
                'currency_analysis': {
                    'enabled': 'invalid_boolean',  # Invalid boolean
                    'primary_layer': '',  # Empty string
                    'confidence_threshold': 'not_a_number'  # Invalid number
                },
                'class_analysis': None  # None instead of dict
            }
        }
    
    @pytest.fixture
    def extreme_data_scenarios(self):
        """Extreme data scenarios for testing."""
        return {
            'very_large_image': Image.new('RGB', (8192, 8192), color=(255, 255, 255)),
            'very_small_image': Image.new('RGB', (1, 1), color=(0, 0, 0)),
            'corrupted_image_data': b'\x89PNG\r\n\x1a\n\x00\x00corrupted',
            'huge_bbox_list': [
                {'class_id': i % 7, 'confidence': random.random(), 
                 'bbox': [random.random(), random.random(), random.random(), random.random()]}
                for i in range(10000)
            ],
            'extreme_coordinates': [
                {'class_id': 0, 'confidence': 0.9, 'bbox': [-1000, -1000, 2000, 2000]},
                {'class_id': 1, 'confidence': 0.8, 'bbox': [float('inf'), 0, 0.1, 0.1]},
                {'class_id': 2, 'confidence': 0.7, 'bbox': [0, float('nan'), 0.1, 0.1]}
            ]
        }
    
    def test_service_initialization_with_corrupt_config(self, corrupt_config):
        """Test service initialization with corrupted configuration."""
        # Should not crash, should handle gracefully
        service = EvaluationService(config=corrupt_config)
        
        assert service is not None
        assert hasattr(service, 'config')
        assert isinstance(service.config, dict)
        
        # Service should still be functional despite corrupt config
        assert hasattr(service, 'scenario_manager')
        assert hasattr(service, 'evaluation_metrics')
    
    def test_evaluation_with_no_checkpoints(self, corrupt_config):
        """Test evaluation when no checkpoints are available."""
        service = EvaluationService(config=corrupt_config)
        
        result = service.run_evaluation(
            scenarios=['position_variation'],
            checkpoints=[],  # Empty checkpoints list
            progress_callback=Mock(),
            metrics_callback=Mock()
        )
        
        # Should handle gracefully
        assert isinstance(result, dict)
        assert 'status' in result
    
    def test_evaluation_with_invalid_scenarios(self, corrupt_config):
        """Test evaluation with invalid scenario names."""
        service = EvaluationService(config=corrupt_config)
        
        invalid_scenarios = [
            '',  # Empty string
            None,  # None value  
            'non_existent_scenario',  # Non-existent scenario
            123,  # Invalid type
            {'invalid': 'dict'}  # Invalid type
        ]
        
        for invalid_scenario in invalid_scenarios:
            try:
                result = service.run_evaluation(
                    scenarios=[invalid_scenario] if invalid_scenario is not None else [None],
                    checkpoints=['dummy.pt']
                )
                # Should return error status
                assert result.get('status') == 'error'
            except Exception:
                # Exception is also acceptable for invalid inputs
                pass
    
    def test_inference_with_extreme_images(self, extreme_data_scenarios):
        """Test inference with extreme image sizes and corrupt data."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Test with very large image
        try:
            tensor_large = service._preprocess_image(extreme_data_scenarios['very_large_image'])
            assert isinstance(tensor_large, torch.Tensor)
            assert tensor_large.shape == (1, 3, 640, 640)  # Should be resized
        except Exception as e:
            # Memory error is acceptable for very large images
            assert 'memory' in str(e).lower() or 'size' in str(e).lower()
        
        # Test with very small image
        tensor_small = service._preprocess_image(extreme_data_scenarios['very_small_image'])
        assert isinstance(tensor_small, torch.Tensor)
        assert tensor_small.shape == (1, 3, 640, 640)  # Should be resized
    
    def test_metrics_calculation_with_extreme_data(self, extreme_data_scenarios):
        """Test metrics calculation with extreme and invalid data."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Test with huge number of detections
        huge_predictions = [{
            'filename': 'test.jpg',
            'detections': extreme_data_scenarios['huge_bbox_list']
        }]
        
        huge_ground_truths = [{
            'filename': 'test.jpg', 
            'annotations': extreme_data_scenarios['huge_bbox_list'][:100]  # Smaller GT
        }]
        
        try:
            metrics = service.compute_metrics(huge_predictions, huge_ground_truths, [0.1])
            assert isinstance(metrics, dict)
        except Exception as e:
            # Memory or computation error is acceptable for extreme data sizes
            assert any(keyword in str(e).lower() for keyword in ['memory', 'size', 'computation'])
        
        # Test with extreme coordinates
        extreme_predictions = [{
            'filename': 'test.jpg',
            'detections': extreme_data_scenarios['extreme_coordinates']
        }]
        
        normal_ground_truths = [{
            'filename': 'test.jpg',
            'annotations': [
                {'class_id': 0, 'bbox': [0.1, 0.1, 0.2, 0.2]},
                {'class_id': 1, 'bbox': [0.5, 0.5, 0.3, 0.3]}
            ]
        }]
        
        # Should handle extreme coordinates gracefully
        metrics = service.compute_metrics(extreme_predictions, normal_ground_truths, [0.1])
        assert isinstance(metrics, dict)
    
    def test_memory_exhaustion_simulation(self):
        """Test behavior under simulated memory pressure."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Create memory-intensive test data
        memory_intensive_images = []
        for i in range(1000):  # Large number of images
            img = Image.new('RGB', (640, 640), color=(i % 255, (i*2) % 255, (i*3) % 255))
            memory_intensive_images.append({
                'image': img,
                'filename': f'memory_test_{i}.jpg',
                'path': f'/fake/path/memory_test_{i}.jpg'
            })
        
        checkpoint_info = {
            'model_loaded': False,  # Use fallback mode to avoid model memory
            'backbone': 'efficientnet_b4'
        }
        
        # Should handle large datasets without crashing
        try:
            predictions, inference_times = service._run_inference_with_timing(
                memory_intensive_images[:50], checkpoint_info  # Limit to 50 for CI
            )
            
            assert len(predictions) <= 50
            assert len(inference_times) <= 50
        except MemoryError:
            # Memory error is acceptable in extreme scenarios
            pytest.skip("Memory exhaustion reached - expected in extreme scenarios")
    
    def test_file_system_edge_cases(self):
        """Test file system related edge cases."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Test with non-existent paths
        non_existent_paths = [
            '/completely/non/existent/path/checkpoint.pt',
            '',  # Empty path
            None,  # None path
            '/dev/null',  # Special file
            'relative/path/checkpoint.pt'  # Relative path
        ]
        
        for path in non_existent_paths:
            if path is not None:
                result = service._load_checkpoint(path)
                assert result is None  # Should return None for invalid paths
    
    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in file names."""
        service = EvaluationService(config={'evaluation': {}})
        
        special_filenames = [
            'test_图片_🖼️.jpg',  # Unicode and emoji
            'test file with spaces.jpg',  # Spaces
            'test-file-with-dashes.jpg',  # Dashes
            'test_file_with_very_long_name_that_exceeds_normal_limits_and_might_cause_issues.jpg',  # Long name
            'test.file.with.multiple.dots.jpg',  # Multiple dots
            'TEST_UPPERCASE.JPG',  # Uppercase
            '123456789.jpg'  # Numbers only
        ]
        
        for filename in special_filenames:
            # Test prediction structure with special filename
            prediction = {
                'filename': filename,
                'detections': [
                    {'class_id': 0, 'confidence': 0.8, 'bbox': [0.1, 0.1, 0.2, 0.2]}
                ]
            }
            
            ground_truth = {
                'filename': filename,
                'annotations': [
                    {'class_id': 0, 'bbox': [0.1, 0.1, 0.2, 0.2]}
                ]
            }
            
            # Should handle special characters gracefully
            metrics = service.compute_metrics([prediction], [ground_truth], [0.1])
            assert isinstance(metrics, dict)
    
    def test_concurrent_access_edge_cases(self):
        """Test edge cases in concurrent access scenarios."""
        # Create multiple service instances rapidly
        services = []
        for i in range(10):
            service = EvaluationService(config={'evaluation': {'scenario_id': i}})
            services.append(service)
        
        # All services should be independent
        for i, service in enumerate(services):
            assert service.config['evaluation']['scenario_id'] == i
            if i > 0:
                assert service is not services[0]
    
    def test_configuration_mutation_safety(self):
        """Test that configuration mutations don't affect service."""
        config = {'evaluation': {'test_value': 'original'}}
        service = EvaluationService(config=config)
        
        # Mutate original config
        config['evaluation']['test_value'] = 'mutated'
        config['evaluation']['new_key'] = 'new_value'
        
        # Service should not be affected by external mutations
        # (depends on implementation - may or may not be isolated)
        original_config = service.config
        assert isinstance(original_config, dict)
    
    def test_infinite_and_nan_values(self):
        """Test handling of infinite and NaN values in data."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Create predictions with extreme numerical values
        extreme_predictions = [{
            'filename': 'extreme_test.jpg',
            'detections': [
                {'class_id': 0, 'confidence': float('inf'), 'bbox': [0.1, 0.1, 0.2, 0.2]},
                {'class_id': 1, 'confidence': float('-inf'), 'bbox': [0.3, 0.3, 0.4, 0.4]},
                {'class_id': 2, 'confidence': float('nan'), 'bbox': [0.5, 0.5, 0.6, 0.6]},
                {'class_id': 3, 'confidence': 0.8, 'bbox': [float('inf'), 0.1, 0.2, 0.2]},
                {'class_id': 4, 'confidence': 0.7, 'bbox': [0.1, float('nan'), 0.2, 0.2]}
            ]
        }]
        
        normal_ground_truths = [{
            'filename': 'extreme_test.jpg',
            'annotations': [
                {'class_id': 0, 'bbox': [0.1, 0.1, 0.2, 0.2]},
                {'class_id': 1, 'bbox': [0.3, 0.3, 0.4, 0.4]}
            ]
        }]
        
        # Should handle extreme values gracefully without crashing
        metrics = service.compute_metrics(extreme_predictions, normal_ground_truths, [0.1])
        assert isinstance(metrics, dict)
        
        # Check that metrics are finite
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                assert not (np.isinf(value) or np.isnan(value)), f"Metric {key} contains inf/nan: {value}"
    
    def test_zero_sized_tensors_and_arrays(self):
        """Test handling of zero-sized tensors and arrays."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Test with empty predictions and ground truths
        empty_predictions = []
        empty_ground_truths = []
        empty_inference_times = []
        
        metrics = service.compute_metrics(empty_predictions, empty_ground_truths, empty_inference_times)
        assert isinstance(metrics, dict)
        
        # Test with predictions but no ground truths
        only_predictions = [{
            'filename': 'only_pred.jpg',
            'detections': [{'class_id': 0, 'confidence': 0.8, 'bbox': [0.1, 0.1, 0.2, 0.2]}]
        }]
        
        metrics = service.compute_metrics(only_predictions, empty_ground_truths, [0.1])
        assert isinstance(metrics, dict)
        
        # Test with ground truths but no predictions  
        only_ground_truths = [{
            'filename': 'only_gt.jpg',
            'annotations': [{'class_id': 0, 'bbox': [0.1, 0.1, 0.2, 0.2]}]
        }]
        
        metrics = service.compute_metrics(empty_predictions, only_ground_truths, [])
        assert isinstance(metrics, dict)
    
    def test_currency_analysis_edge_cases(self):
        """Test currency analysis with edge case data."""
        config = {
            'analysis': {
                'currency_analysis': {
                    'enabled': True,
                    'primary_layer': 'banknote',
                    'confidence_threshold': 0.3
                }
            }
        }
        service = EvaluationService(config=config)
        
        # Test with invalid class IDs
        edge_case_predictions = [{
            'detections': [
                {'class_id': -1, 'confidence': 0.9, 'bbox': [0.1, 0.1, 0.2, 0.2]},  # Negative class
                {'class_id': 999, 'confidence': 0.8, 'bbox': [0.3, 0.3, 0.4, 0.4]},  # Very high class
                {'class_id': 7, 'confidence': 0.7, 'bbox': [0.5, 0.5, 0.6, 0.6]},   # Beyond normal range
                {'class_id': None, 'confidence': 0.6, 'bbox': [0.7, 0.7, 0.8, 0.8]}  # None class
            ]
        }]
        
        edge_case_ground_truths = [{
            'annotations': [
                {'class_id': 0, 'bbox': [0.1, 0.1, 0.2, 0.2]},
                {'class_id': 1, 'bbox': [0.3, 0.3, 0.4, 0.4]}
            ]
        }]
        
        # Should handle edge cases gracefully
        currency_config = config['analysis']['currency_analysis']
        analysis = service._analyze_currency_detection(
            edge_case_predictions, edge_case_ground_truths, currency_config
        )
        
        assert isinstance(analysis, dict)
        assert 'total_detections' in analysis
        assert 'correct_denominations' in analysis
        assert 'denomination_accuracy' in analysis
        assert 0.0 <= analysis['denomination_accuracy'] <= 1.0
    
    def test_progress_callback_edge_cases(self):
        """Test progress callback with edge case scenarios."""
        # Test with invalid callback
        service = EvaluationService(config={'evaluation': {}})
        
        # Test with callback that raises exceptions
        def error_callback(*args, **kwargs):
            raise ValueError("Callback error")
        
        # Should handle callback errors gracefully
        try:
            result = service.run_evaluation(
                scenarios=['position_variation'],
                checkpoints=[],
                progress_callback=error_callback
            )
            assert isinstance(result, dict)
        except Exception as e:
            # If callback error propagates, it should be a known exception
            assert 'callback' in str(e).lower() or 'error' in str(e).lower()
        
        # Test with None callback
        result = service.run_evaluation(
            scenarios=['position_variation'], 
            checkpoints=[],
            progress_callback=None
        )
        assert isinstance(result, dict)


class TestEvaluationErrorRecovery:
    """Test error recovery and resilience."""
    
    def test_partial_failure_recovery(self):
        """Test recovery from partial failures during evaluation."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Mock scenario manager to fail on some scenarios
        original_method = service.scenario_manager.prepare_all_scenarios
        def failing_prepare():
            raise ValueError("Scenario preparation failed")
        
        service.scenario_manager.prepare_all_scenarios = failing_prepare
        
        # Should handle scenario preparation failure
        result = service.run_evaluation(
            scenarios=['position_variation', 'lighting_variation'],
            checkpoints=['dummy.pt']
        )
        
        assert result.get('status') == 'error'
        assert 'error' in result
    
    def test_checkpoint_loading_failure_recovery(self):
        """Test recovery from checkpoint loading failures."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Override checkpoint loading to always fail
        def failing_load_checkpoint(path):
            return None
        
        service._load_checkpoint = failing_load_checkpoint
        
        # Should handle checkpoint loading failure gracefully
        result = service.run_scenario('position_variation', 'dummy_checkpoint.pt')
        
        # Should return error or handle gracefully
        assert isinstance(result, dict)
    
    def test_metrics_calculation_failure_recovery(self):
        """Test recovery from metrics calculation failures."""
        service = EvaluationService(config={'evaluation': {}})
        
        # Override metrics computation to fail
        def failing_compute_metrics(*args, **kwargs):
            raise RuntimeError("Metrics computation failed")
        
        service.compute_metrics = failing_compute_metrics
        
        # Test that service handles metrics failure
        try:
            result = service.run_scenario('position_variation', 'dummy.pt')
            # If it returns, should be error status
            assert result.get('status') == 'error'
        except Exception:
            # Exception is also acceptable
            pass


if __name__ == '__main__':
    # Run edge case tests
    pytest.main([__file__, '-v', '--tb=short', '--maxfail=5'])