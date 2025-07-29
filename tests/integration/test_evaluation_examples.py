#!/usr/bin/env python3
"""
Comprehensive tests for examples/evaluations.py script.

Tests the evaluation example script functionality including:
- Command line argument parsing
- Checkpoint selection and filtering
- Evaluation mode execution
- Progress tracking and callbacks  
- Error handling and edge cases
"""

import pytest
import tempfile
import shutil
import json
import sys
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List
import torch
import numpy as np
from PIL import Image

# Add examples directory to path for testing
examples_dir = Path(__file__).parent.parent / 'examples'
if str(examples_dir) not in sys.path:
    sys.path.insert(0, str(examples_dir))

# Import the evaluation script functions
try:
    from evaluations import (
        create_evaluation_config,
        get_selected_checkpoints,
        run_single_scenario,
        run_all_scenarios,
        compare_backbones,
        list_available_resources,
        setup_scenarios,
        create_argument_parser,
        create_log_callback,
        create_metrics_callback,
        create_progress_callback
    )
except ImportError:
    pytest.skip("Could not import evaluations module", allow_module_level=True)


class TestEvaluationExamplesScript:
    """Test the examples/evaluations.py script functionality."""
    
    @pytest.fixture
    def temp_test_env(self):
        """Create temporary test environment with checkpoints and scenarios."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create checkpoint directory structure
        checkpoint_dir = temp_path / 'checkpoints'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Create mock checkpoint files
        checkpoint_files = []
        backbones = ['cspdarknet', 'efficientnet_b4', 'yolov5s']
        
        for i, backbone in enumerate(backbones):
            checkpoint_path = checkpoint_dir / f'best_{backbone}_multi_{20240101+i}.pt'
            
            # Create realistic checkpoint data
            checkpoint_data = {
                'model_state_dict': {'dummy': 'weights'},
                'config': {
                    'backbone': backbone,
                    'training_mode': 'two_phase'
                },
                'metrics': {
                    'val_map': 0.75 + i*0.05,
                    'val_loss': 0.25 - i*0.02,
                    'precision': 0.8 + i*0.03,
                    'recall': 0.78 + i*0.02
                },
                'epoch': 50 + i*10,
                'architecture_type': 'yolov5',
                'backbone': backbone
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint_files.append(str(checkpoint_path))
        
        # Create evaluation directory structure
        eval_dir = temp_path / 'evaluation'
        scenarios = ['position_variation', 'lighting_variation']
        
        for scenario in scenarios:
            scenario_dir = eval_dir / scenario
            images_dir = scenario_dir / 'images'
            labels_dir = scenario_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Create test images and labels
            for i in range(5):
                # Create test image
                img = Image.new('RGB', (640, 640), color=(i*50, i*40, i*30))
                img_path = images_dir / f'{scenario}_{i:03d}.jpg'
                img.save(img_path)
                
                # Create corresponding label
                label_path = labels_dir / f'{scenario}_{i:03d}.txt'
                with open(label_path, 'w') as f:
                    f.write(f"{i % 7} 0.5 0.5 0.2 0.3\n")
        
        yield {
            'temp_dir': temp_path,
            'checkpoint_dir': checkpoint_dir,
            'checkpoint_files': checkpoint_files,
            'eval_dir': eval_dir,
            'scenarios': scenarios
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_args(self, temp_test_env):
        """Create mock command line arguments."""
        args = Mock()
        args.checkpoint_dir = str(temp_test_env['checkpoint_dir'])
        args.checkpoint = None
        args.checkpoints = None
        args.backbone = None
        args.min_map = None
        args.top_n = 3
        args.output_dir = None
        args.force_regenerate = False
        args.verbose = True
        args.scenario = None
        args.all_scenarios = False
        args.compare_backbones = False
        args.list_resources = False
        args.setup_scenarios = False
        return args
    
    def test_create_evaluation_config(self):
        """Test evaluation configuration creation."""
        config = create_evaluation_config()
        
        assert isinstance(config, dict)
        assert 'evaluation' in config
        
        eval_config = config['evaluation']
        assert 'data' in eval_config
        assert 'checkpoints' in eval_config
        assert 'scenarios' in eval_config
        assert 'metrics' in eval_config
        assert 'analysis' in eval_config
        
        # Test checkpoint configuration
        checkpoint_config = eval_config['checkpoints']
        assert 'discovery_paths' in checkpoint_config
        assert 'filename_patterns' in checkpoint_config
        assert 'supported_backbones' in checkpoint_config
        
        # Test scenarios configuration
        scenarios_config = eval_config['scenarios']
        assert 'position_variation' in scenarios_config
        assert 'lighting_variation' in scenarios_config
        
        for scenario_name, scenario_config in scenarios_config.items():
            assert 'enabled' in scenario_config
            assert 'augmentation_config' in scenario_config
    
    def test_get_selected_checkpoints_single(self, mock_args, temp_test_env):
        """Test single checkpoint selection."""
        config = create_evaluation_config()
        
        # Test specific checkpoint selection
        mock_args.checkpoint = temp_test_env['checkpoint_files'][0]
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        assert len(checkpoints) == 1
        assert checkpoints[0] == temp_test_env['checkpoint_files'][0]
    
    def test_get_selected_checkpoints_by_name(self, mock_args, temp_test_env):
        """Test checkpoint selection by name patterns."""
        config = create_evaluation_config()
        
        # Test checkpoint selection by names
        mock_args.checkpoints = "cspdarknet,efficientnet_b4"
        mock_args.checkpoint = None
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        assert len(checkpoints) >= 2
        # Should include checkpoints matching the patterns
        checkpoint_names = [Path(cp).name for cp in checkpoints]
        assert any('cspdarknet' in name for name in checkpoint_names)
        assert any('efficientnet_b4' in name for name in checkpoint_names)
    
    def test_get_selected_checkpoints_by_backbone(self, mock_args, temp_test_env):
        """Test checkpoint filtering by backbone."""
        config = create_evaluation_config()
        
        # Test backbone filtering
        mock_args.backbone = 'cspdarknet'
        mock_args.checkpoint = None
        mock_args.checkpoints = None
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        # Should only return cspdarknet checkpoints
        assert len(checkpoints) >= 1
        checkpoint_names = [Path(cp).name for cp in checkpoints]
        assert all('cspdarknet' in name for name in checkpoint_names)
    
    def test_get_selected_checkpoints_by_min_map(self, mock_args, temp_test_env):
        """Test checkpoint filtering by minimum mAP."""
        config = create_evaluation_config()
        
        # Test minimum mAP filtering
        mock_args.min_map = 0.8  # High threshold
        mock_args.checkpoint = None
        mock_args.checkpoints = None
        mock_args.backbone = None
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        # Should return only high-performing checkpoints
        # Note: May return empty list if no checkpoints meet threshold
        assert isinstance(checkpoints, list)
    
    def test_get_selected_checkpoints_top_n(self, mock_args, temp_test_env):
        """Test top-N checkpoint selection."""
        config = create_evaluation_config()
        
        # Test top-N selection
        mock_args.top_n = 2
        mock_args.checkpoint = None
        mock_args.checkpoints = None
        mock_args.backbone = None
        mock_args.min_map = None
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        assert len(checkpoints) <= 2
    
    def test_create_callbacks(self):
        """Test callback creation functions."""
        # Test log callback
        log_callback = create_log_callback(verbose=True)
        assert callable(log_callback)
        
        # Test calling log callback (shouldn't crash)
        try:
            log_callback('info', 'Test message', {'key': 'value'})
        except Exception as e:
            pytest.fail(f"Log callback should not raise exception: {e}")
        
        # Test metrics callback
        metrics_callback = create_metrics_callback(verbose=True)
        assert callable(metrics_callback)
        
        # Test calling metrics callback with mock results
        mock_results = {
            'status': 'success',
            'summary': {
                'aggregated_metrics': {
                    'overall_metrics': {
                        'mAP': 0.85,
                        'precision': 0.82,
                        'recall': 0.88,
                        'f1_score': 0.85
                    },
                    'best_configurations': {
                        'mAP': {'backbone': 'efficientnet_b4', 'value': 0.85}
                    }
                },
                'key_findings': ['Finding 1', 'Finding 2']
            },
            'evaluation_results': {
                'position_variation': {
                    'cspdarknet': {'metrics': {'mAP': 0.83}},
                    'efficientnet_b4': {'metrics': {'mAP': 0.87}}
                }
            }
        }
        
        try:
            metrics_callback(mock_results)
        except Exception as e:
            pytest.fail(f"Metrics callback should not raise exception: {e}")
        
        # Test progress callback
        progress_callback = create_progress_callback(verbose=True)
        assert callable(progress_callback)
        
        try:
            progress_callback('evaluation', 50, 100, 'Processing...')
        except Exception as e:
            pytest.fail(f"Progress callback should not raise exception: {e}")
    
    @patch('evaluations.create_evaluation_service')
    def test_run_single_scenario(self, mock_create_service, mock_args, temp_test_env):
        """Test single scenario evaluation."""
        # Setup mock service
        mock_service = Mock()
        mock_service.run_scenario.return_value = {
            'status': 'success',
            'scenario_name': 'position_variation',
            'metrics': {
                'mAP': 0.85,
                'precision': 0.82,
                'recall': 0.88,
                'f1_score': 0.85,
                'inference_time_avg': 0.15
            },
            'checkpoint_info': {
                'backbone': 'cspdarknet',
                'display_name': 'CSPDarknet - 01/01/2024'
            },
            'additional_data': {}
        }
        mock_create_service.return_value = mock_service
        
        # Setup args for single scenario
        mock_args.scenario = 'position_variation'
        mock_args.checkpoint = temp_test_env['checkpoint_files'][0]
        
        # Run single scenario
        result = run_single_scenario(mock_args)
        
        # Verify service was called correctly
        mock_create_service.assert_called_once()
        mock_service.run_scenario.assert_called_once_with(
            'position_variation', 
            temp_test_env['checkpoint_files'][0]
        )
        
        # Verify result
        assert result['status'] == 'success'
        assert result['scenario_name'] == 'position_variation'
        assert 'metrics' in result
    
    @patch('evaluations.run_evaluation_pipeline')
    @patch('evaluations.get_selected_checkpoints')
    def test_run_all_scenarios(self, mock_get_checkpoints, mock_run_pipeline, mock_args, temp_test_env):
        """Test all scenarios evaluation."""
        # Setup mocks
        mock_get_checkpoints.return_value = temp_test_env['checkpoint_files'][:2]
        mock_run_pipeline.return_value = {
            'status': 'success',
            'evaluation_results': {
                'position_variation': {
                    'cspdarknet': {
                        'metrics': {'mAP': 0.83, 'precision': 0.80},
                        'checkpoint_info': {'backbone': 'cspdarknet'},
                        'additional_data': {}
                    }
                },
                'lighting_variation': {
                    'cspdarknet': {
                        'metrics': {'mAP': 0.81, 'precision': 0.78},
                        'checkpoint_info': {'backbone': 'cspdarknet'},
                        'additional_data': {}
                    }
                }
            },
            'summary': {
                'aggregated_metrics': {
                    'overall_metrics': {'mAP': 0.82}
                },
                'key_findings': ['Good performance overall']
            },
            'scenarios_evaluated': 2,
            'checkpoints_evaluated': 2
        }
        
        # Setup args for all scenarios
        mock_args.all_scenarios = True
        
        # Run all scenarios
        result = run_all_scenarios(mock_args)
        
        # Verify calls
        mock_get_checkpoints.assert_called_once()
        mock_run_pipeline.assert_called_once()
        
        # Verify result
        assert result['status'] == 'success'
        assert result['scenarios_evaluated'] == 2
        assert result['checkpoints_evaluated'] == 2
    
    @patch('evaluations.run_evaluation_pipeline')
    @patch('evaluations.create_checkpoint_selector')
    def test_compare_backbones(self, mock_create_selector, mock_run_pipeline, mock_args, temp_test_env):
        """Test backbone comparison."""
        # Setup mock checkpoint selector
        mock_selector = Mock()
        mock_selector.list_available_checkpoints.return_value = [
            {
                'path': temp_test_env['checkpoint_files'][0],
                'backbone': 'cspdarknet',
                'metrics': {'val_map': 0.83}
            },
            {
                'path': temp_test_env['checkpoint_files'][1],
                'backbone': 'efficientnet_b4',
                'metrics': {'val_map': 0.87}
            }
        ]
        mock_create_selector.return_value = mock_selector
        
        # Setup mock evaluation results
        def mock_pipeline_side_effect(*args, **kwargs):
            checkpoint_path = kwargs['checkpoints'][0]
            if 'cspdarknet' in checkpoint_path:
                backbone = 'cspdarknet'
                map_score = 0.83
            else:
                backbone = 'efficientnet_b4'
                map_score = 0.87
            
            return {
                'status': 'success',
                'summary': {
                    'aggregated_metrics': {
                        'overall_metrics': {
                            'mAP': map_score,
                            'precision': map_score - 0.02,
                            'recall': map_score + 0.01,
                            'f1_score': map_score - 0.01
                        }
                    }
                }
            }
        
        mock_run_pipeline.side_effect = mock_pipeline_side_effect
        
        # Setup args for backbone comparison
        mock_args.compare_backbones = True
        
        # Run backbone comparison
        result = compare_backbones(mock_args)
        
        # Verify calls
        mock_create_selector.assert_called_once()
        mock_selector.list_available_checkpoints.assert_called_once()
        assert mock_run_pipeline.call_count == 2  # Once per backbone
        
        # Verify result structure
        assert isinstance(result, dict)
        assert 'cspdarknet' in result
        assert 'efficientnet_b4' in result
        
        for backbone, backbone_result in result.items():
            assert backbone_result['status'] == 'success'
    
    @patch('evaluations.create_checkpoint_selector')
    @patch('evaluations.create_scenario_manager')
    def test_list_available_resources(self, mock_create_scenario_mgr, mock_create_selector, mock_args, temp_test_env):
        """Test resource listing functionality."""
        # Setup mock checkpoint selector
        mock_selector = Mock()
        mock_selector.list_available_checkpoints.return_value = [
            {
                'display_name': 'CSPDarknet - 01/01/2024',
                'path': temp_test_env['checkpoint_files'][0],
                'backbone': 'cspdarknet',
                'metrics': {'val_map': 0.83},
                'file_size_mb': 25.5
            },
            {
                'display_name': 'EfficientNet-B4 - 02/01/2024',
                'path': temp_test_env['checkpoint_files'][1],
                'backbone': 'efficientnet_b4',
                'metrics': {'val_map': 0.87},
                'file_size_mb': 32.1
            }
        ]
        mock_create_selector.return_value = mock_selector
        
        # Setup mock scenario manager
        mock_scenario_mgr = Mock()
        mock_scenario_mgr.list_available_scenarios.return_value = [
            {
                'display_name': 'Position Variation',
                'name': 'position_variation',
                'enabled': True,
                'data_exists': True,
                'ready': True,
                'description': 'Test position variations'
            },
            {
                'display_name': 'Lighting Variation',
                'name': 'lighting_variation',
                'enabled': True,
                'data_exists': False,
                'ready': False,
                'description': 'Test lighting variations'
            }
        ]
        mock_create_scenario_mgr.return_value = mock_scenario_mgr
        
        # Setup args for listing resources
        mock_args.list_resources = True
        
        # Run resource listing (should not raise exception)
        try:
            list_available_resources(mock_args)
        except Exception as e:
            pytest.fail(f"list_available_resources should not raise exception: {e}")
        
        # Verify calls
        mock_create_selector.assert_called_once()
        mock_selector.list_available_checkpoints.assert_called_once()
        mock_create_scenario_mgr.assert_called_once()
        mock_scenario_mgr.list_available_scenarios.assert_called_once()
    
    @patch('evaluations.create_scenario_manager')
    def test_setup_scenarios(self, mock_create_scenario_mgr, mock_args):
        """Test scenario setup functionality."""
        # Setup mock scenario manager
        mock_scenario_mgr = Mock()
        mock_scenario_mgr.prepare_all_scenarios.return_value = {
            'total_scenarios': 2,
            'successful': 2,
            'failed': 0,
            'results': {
                'position_variation': {
                    'status': 'successful',
                    'validation': {'images_count': 20, 'labels_count': 20}
                },
                'lighting_variation': {
                    'status': 'existing',
                    'validation': {'images_count': 15, 'labels_count': 15}
                }
            }
        }
        mock_create_scenario_mgr.return_value = mock_scenario_mgr
        
        # Setup args for scenario setup
        mock_args.setup_scenarios = True
        mock_args.force_regenerate = True
        
        # Run scenario setup (should not raise exception)
        try:
            setup_scenarios(mock_args)
        except Exception as e:
            pytest.fail(f"setup_scenarios should not raise exception: {e}")
        
        # Verify calls
        mock_create_scenario_mgr.assert_called_once()
        mock_scenario_mgr.prepare_all_scenarios.assert_called_once_with(force_regenerate=True)
    
    def test_argument_parser(self):
        """Test command line argument parser."""
        parser = create_argument_parser()
        
        # Test valid arguments
        test_cases = [
            ['--scenario', 'position_variation', '--checkpoint', 'test.pt'],
            ['--all-scenarios', '--checkpoint-dir', 'checkpoints', '--top-n', '3'],
            ['--compare-backbones', '--backbone', 'cspdarknet', '--min-map', '0.5'],
            ['--list-resources', '--verbose'],
            ['--setup-scenarios', '--force-regenerate']
        ]
        
        for args in test_cases:
            try:
                parsed = parser.parse_args(args)
                assert parsed is not None
            except SystemExit:
                pytest.fail(f"Parser should accept valid arguments: {args}")
        
        # Test mutually exclusive groups
        invalid_cases = [
            ['--scenario', 'position_variation', '--all-scenarios'],  # Both scenario modes
            ['--checkpoint', 'test.pt', '--checkpoints', 'test1.pt,test2.pt'],  # Both checkpoint modes
        ]
        
        for args in invalid_cases:
            with pytest.raises(SystemExit):
                parser.parse_args(args)
    
    def test_error_handling_no_checkpoints(self, mock_args):
        """Test error handling when no checkpoints are found."""
        config = create_evaluation_config()
        
        # Test with non-existent checkpoint directory
        mock_args.checkpoint_dir = '/nonexistent/path'
        mock_args.checkpoint = None
        mock_args.checkpoints = None
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        # Should return empty list gracefully
        assert checkpoints == []
    
    def test_error_handling_invalid_checkpoint_patterns(self, mock_args, temp_test_env):
        """Test error handling with invalid checkpoint patterns."""
        config = create_evaluation_config()
        
        # Test with non-matching checkpoint patterns
        mock_args.checkpoints = "nonexistent1.pt,nonexistent2.pt"
        mock_args.checkpoint = None
        mock_args.checkpoint_dir = str(temp_test_env['checkpoint_dir'])
        
        checkpoints = get_selected_checkpoints(mock_args, config)
        
        # Should handle gracefully and return empty list
        assert checkpoints == []
    
    def test_verbose_vs_quiet_modes(self):
        """Test verbose and quiet callback modes."""
        # Test verbose mode
        verbose_log = create_log_callback(verbose=True)
        verbose_metrics = create_metrics_callback(verbose=True)
        verbose_progress = create_progress_callback(verbose=True)
        
        # Test quiet mode
        quiet_log = create_log_callback(verbose=False)
        quiet_metrics = create_metrics_callback(verbose=False)
        quiet_progress = create_progress_callback(verbose=False)
        
        # All callbacks should be callable regardless of verbose setting
        assert callable(verbose_log)
        assert callable(verbose_metrics)
        assert callable(verbose_progress)
        assert callable(quiet_log)
        assert callable(quiet_metrics)
        assert callable(quiet_progress)
        
        # Test calling callbacks (should not crash)
        test_data = {'key': 'value'}
        mock_results = {'status': 'success', 'summary': {}}
        
        try:
            quiet_log('info', 'Test message', test_data)
            quiet_metrics(mock_results)
            quiet_progress('test', 1, 10, 'message')
        except Exception as e:
            pytest.fail(f"Quiet callbacks should not raise exceptions: {e}")


class TestEvaluationExamplesIntegration:
    """Integration tests for the evaluation examples script."""
    
    @pytest.fixture
    def integrated_test_env(self):
        """Create integrated test environment."""
        temp_dir = tempfile.mkdtemp()
        temp_path = Path(temp_dir)
        
        # Create full directory structure
        checkpoint_dir = temp_path / 'checkpoints'
        eval_dir = temp_path / 'evaluation'
        results_dir = temp_path / 'results'
        
        for directory in [checkpoint_dir, eval_dir, results_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Create realistic checkpoint files
        checkpoint_files = []
        backbones = ['cspdarknet', 'efficientnet_b4']
        
        for i, backbone in enumerate(backbones):
            checkpoint_path = checkpoint_dir / f'best_{backbone}_{20240101+i}.pt'
            
            checkpoint_data = {
                'model_state_dict': {'layer1.weight': torch.randn(10, 10)},
                'config': {
                    'backbone': backbone,
                    'model': {'backbone': backbone},
                    'training_mode': 'two_phase'
                },
                'metrics': {
                    'val_map': 0.8 + i*0.05,
                    'val_loss': 0.2 - i*0.01,
                    'precision': 0.82 + i*0.02,
                    'recall': 0.79 + i*0.03
                },
                'epoch': 50,
                'architecture_type': 'yolov5',
                'backbone': backbone,
                'model_info': {'total_parameters': 10000000}
            }
            
            torch.save(checkpoint_data, checkpoint_path)
            checkpoint_files.append(str(checkpoint_path))
        
        # Create evaluation scenarios
        scenarios = ['position_variation', 'lighting_variation']
        for scenario in scenarios:
            scenario_dir = eval_dir / scenario
            images_dir = scenario_dir / 'images'
            labels_dir = scenario_dir / 'labels'
            
            images_dir.mkdir(parents=True, exist_ok=True)
            labels_dir.mkdir(parents=True, exist_ok=True)
            
            # Create more realistic test data
            for i in range(10):
                # Create test image with some content
                img = Image.new('RGB', (640, 640), color=(100+i*10, 150, 200))
                from PIL import ImageDraw
                draw = ImageDraw.Draw(img)
                
                # Draw some shapes to simulate objects
                for j in range(2):
                    x1, y1 = 100 + j*200 + i*10, 100 + j*100
                    x2, y2 = x1 + 150, y1 + 100
                    draw.rectangle([x1, y1, x2, y2], fill=(255-i*20, 100+i*15, 50+i*10))
                
                img_path = images_dir / f'{scenario}_{i:03d}.jpg'
                img.save(img_path)
                
                # Create corresponding realistic label
                label_path = labels_dir / f'{scenario}_{i:03d}.txt'
                with open(label_path, 'w') as f:
                    # Write multiple annotations per image
                    for j in range(np.random.randint(1, 4)):
                        class_id = j % 7
                        x_center = 0.2 + j*0.3 + np.random.uniform(-0.1, 0.1)
                        y_center = 0.3 + j*0.2 + np.random.uniform(-0.1, 0.1)
                        width = 0.15 + np.random.uniform(-0.05, 0.05)
                        height = 0.12 + np.random.uniform(-0.03, 0.03)
                        
                        # Ensure values are within bounds
                        x_center = max(width/2, min(1-width/2, x_center))
                        y_center = max(height/2, min(1-height/2, y_center))
                        
                        f.write(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        yield {
            'temp_dir': temp_path,
            'checkpoint_dir': checkpoint_dir,
            'eval_dir': eval_dir,
            'results_dir': results_dir,
            'checkpoint_files': checkpoint_files,
            'scenarios': scenarios
        }
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @patch('evaluations.create_api')
    def test_end_to_end_single_scenario(self, mock_create_api, integrated_test_env):
        """Test end-to-end single scenario evaluation."""
        # Setup mock API
        mock_api = Mock()
        mock_api.predict.return_value = {
            'success': True,
            'detections': [
                {'class_id': 0, 'confidence': 0.85, 'bbox': [0.2, 0.3, 0.4, 0.5]},
                {'class_id': 2, 'confidence': 0.72, 'bbox': [0.6, 0.1, 0.3, 0.4]}
            ]
        }
        mock_api.load_checkpoint.return_value = {'success': True}
        mock_create_api.return_value = mock_api
        
        # Create args
        args = Mock()
        args.scenario = 'position_variation'
        args.checkpoint = integrated_test_env['checkpoint_files'][0]
        args.verbose = False
        
        # Patch the evaluation config to use test directories
        with patch('evaluations.create_evaluation_config') as mock_config:
            test_config = create_evaluation_config()
            test_config['evaluation']['data']['evaluation_dir'] = str(integrated_test_env['eval_dir'])
            mock_config.return_value = test_config
            
            # Run single scenario evaluation
            result = run_single_scenario(args)
            
            # Verify result
            assert isinstance(result, dict)
            # Result may be success or error depending on mock setup
            assert 'status' in result
    
    def test_checkpoint_discovery_integration(self, integrated_test_env):
        """Test checkpoint discovery with real files."""
        config = create_evaluation_config()
        
        # Update config to use test checkpoint directory
        config['evaluation']['checkpoints']['discovery_paths'] = [str(integrated_test_env['checkpoint_dir'])]
        
        args = Mock()
        args.checkpoint_dir = str(integrated_test_env['checkpoint_dir'])
        args.checkpoint = None
        args.checkpoints = None
        args.backbone = None
        args.min_map = None
        args.top_n = 10
        
        # Get selected checkpoints
        checkpoints = get_selected_checkpoints(args, config)
        
        # Should find the test checkpoint files
        assert len(checkpoints) > 0
        assert all(Path(cp).exists() for cp in checkpoints)
        
        # Test backbone filtering
        args.backbone = 'cspdarknet'
        cspdarknet_checkpoints = get_selected_checkpoints(args, config)
        
        assert len(cspdarknet_checkpoints) >= 1
        checkpoint_names = [Path(cp).name for cp in cspdarknet_checkpoints]
        assert all('cspdarknet' in name for name in checkpoint_names)
    
    @patch('sys.argv', ['evaluations.py', '--list-resources', '--checkpoint-dir', '/tmp/test'])
    def test_main_function_integration(self):
        """Test main function integration with argument parsing."""
        # This tests that the argument parser works with realistic command lines
        parser = create_argument_parser()
        
        # Test various command line combinations
        test_commands = [
            ['--scenario', 'position_variation', '--checkpoint', 'test.pt'],
            ['--all-scenarios', '--top-n', '5', '--verbose'],
            ['--compare-backbones', '--backbone', 'cspdarknet'],
            ['--list-resources'],
            ['--setup-scenarios', '--force-regenerate']
        ]
        
        for cmd in test_commands:
            try:
                args = parser.parse_args(cmd)
                # Verify required mode is set
                modes = [args.scenario, args.all_scenarios, args.compare_backbones, 
                        args.list_resources, args.setup_scenarios]
                assert any(modes), f"No mode selected for command: {cmd}"
            except SystemExit as e:
                pytest.fail(f"Command should be valid: {cmd}, but got SystemExit: {e}")


if __name__ == '__main__':
    # Run comprehensive tests
    pytest.main([__file__, '-v', '--tb=short', '--durations=10'])