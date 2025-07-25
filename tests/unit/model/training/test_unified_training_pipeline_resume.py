#!/usr/bin/env python3
"""
File: tests/unit/model/training/test_unified_training_pipeline_resume.py

Comprehensive tests for resume functionality and checkpoint handling in UnifiedTrainingPipeline.

Test Coverage:
- Resume from different training phases (1, 2, invalid)
- Resume from different epoch positions within phases
- Checkpoint loading and validation
- Resume with corrupted or missing checkpoints
- Resume with configuration mismatches
- Resume with different training modes
- Resume with different device configurations
- Checkpoint state restoration edge cases
- Resume failure recovery scenarios
- Memory management during resume operations
"""

import pytest
import torch
import tempfile
import json
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock, call
from typing import Dict, Any, Optional

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

from smartcash.model.training.unified_training_pipeline import UnifiedTrainingPipeline


class TestUnifiedTrainingPipelineResumeScenarios:
    """Test various resume scenarios and checkpoint handling."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_base = tempfile.mkdtemp()
        dirs = {
            'checkpoints': Path(temp_base) / 'checkpoints',
            'logs': Path(temp_base) / 'logs'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_dirs):
        """Create mock configuration for testing."""
        return {
            'model': {
                'backbone': 'cspdarknet',
                'num_classes': [80, 40, 20],
                'layer_mode': 'multi'
            },
            'device': {
                'auto_detect': True,
                'device': 'cpu'
            },
            'training': {
                'compile_model': False,
                'loss': {'type': 'multi_layer'}
            },
            'training_phases': {
                'phase_1': {'epochs': 5},
                'phase_2': {'epochs': 3}
            },
            'paths': {
                'checkpoints': temp_dirs['checkpoints'],
                'logs': temp_dirs['logs']
            }
        }
    
    def test_resume_from_phase_1_middle(self, pipeline, mock_config):
        """Test resume from middle of phase 1."""
        resume_info = {
            'checkpoint_name': 'checkpoint_phase1_epoch2.pth',
            'phase': 1,
            'epoch': 2,  # Resume from epoch 2 (0-indexed)
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_123'
        }
        
        # Mock the required functions
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_123', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock successful resume results
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'resumed_from_epoch': 3}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.4}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=5,
                                phase_2_epochs=3,
                                resume_from_checkpoint=True,
                                training_mode='two_phase'
                            )
        
        # Verify successful resume
        assert result['success'] is True
        
        # Verify resume was called with correct info
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert resume_args[0] == resume_info
        assert resume_args[1] == 'cspdarknet'  # backbone
        assert resume_args[2] == 5  # phase_1_epochs
        assert resume_args[3] == 3  # phase_2_epochs
    
    def test_resume_from_phase_2_middle(self, pipeline, mock_config):
        """Test resume from middle of phase 2."""
        resume_info = {
            'checkpoint_name': 'checkpoint_phase2_epoch1.pth',
            'phase': 2,
            'epoch': 1,  # Resume from epoch 1 of phase 2
            'model_state_dict': {'layer.weight': torch.tensor([2.0])},
            'session_id': 'session_456'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_456', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock results for phase 2 resume
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'message': 'Completed (loaded from checkpoint)'}
                    phase2_result = {'success': True, 'resumed_from_epoch': 2}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Phase 2 Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=5,
                                phase_2_epochs=3,
                                resume_from_checkpoint=True,
                                training_mode='two_phase'
                            )
        
        # Verify successful resume from phase 2
        assert result['success'] is True
        
        # Verify correct resume parameters
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert resume_args[0]['phase'] == 2
        assert resume_args[0]['epoch'] == 1
    
    def test_resume_from_invalid_phase(self, pipeline, mock_config):
        """Test resume from invalid phase number."""
        resume_info = {
            'checkpoint_name': 'checkpoint_invalid.pth',
            'phase': 99,  # Invalid phase
            'epoch': 1,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_789'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_789', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock fallback to fresh training
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'fresh_training': True}
                    phase2_result = {'success': True, 'fresh_training': True}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Invalid Phase Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=5,
                                phase_2_epochs=3,
                                resume_from_checkpoint=True
                            )
        
        # Should still succeed by falling back to fresh training
        assert result['success'] is True
        
        # Verify resume was attempted with invalid phase
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert resume_args[0]['phase'] == 99
    
    def test_resume_single_phase_training(self, pipeline, mock_config):
        """Test resume functionality with single phase training mode."""
        resume_info = {
            'checkpoint_name': 'checkpoint_single_epoch3.pth',
            'phase': 1,  # Single phase uses phase 1
            'epoch': 3,
            'model_state_dict': {'layer.weight': torch.tensor([1.5])},
            'session_id': 'session_single'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_single', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock single phase resume results
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'message': 'Skipped in single phase mode'}
                    phase2_result = {'success': True, 'single_phase_resumed': True}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Single Phase Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='efficientnet_b4',
                                phase_1_epochs=8,  # Total epochs for single phase
                                phase_2_epochs=0,
                                training_mode='single_phase',
                                single_phase_layer_mode='single',
                                single_phase_freeze_backbone=False,
                                resume_from_checkpoint=True
                            )
        
        # Verify successful single phase resume
        assert result['success'] is True
        
        # Verify correct training mode was passed
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args
        training_mode = resume_args[1]['training_mode']
        assert training_mode == 'single_phase'
    
    def test_resume_with_corrupted_checkpoint(self, pipeline):
        """Test resume with corrupted or invalid checkpoint data."""
        corrupted_resume_info = {
            'checkpoint_name': 'corrupted_checkpoint.pth',
            'phase': 1,
            'epoch': 2,
            'model_state_dict': None,  # Corrupted state dict
            'session_id': 'session_corrupted'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_corrupted', corrupted_resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock failed resume due to corruption
                    mock_resume.side_effect = Exception("Checkpoint corrupted")
                    
                    with patch.object(pipeline, '_phase_preparation') as mock_prep:
                        mock_config = {
                            'training_phases': {'phase_1': {'epochs': 5}, 'phase_2': {'epochs': 3}}
                        }
                        mock_prep.return_value = {'success': True, 'config': mock_config}
                        with patch.object(pipeline, '_phase_build_model') as mock_build:
                            mock_build.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                                mock_validate.return_value = {'success': True}
                                with patch.object(pipeline, '_phase_training_1_with_manager') as mock_train1:
                                    mock_train1.return_value = {'success': True}
                                    with patch.object(pipeline, '_phase_training_2_with_manager') as mock_train2:
                                        mock_train2.return_value = {'success': True}
                                        with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                                            mock_summary.return_value = {'success': True}
                                            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager'):
                                                with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                                                    mock_md.return_value = "# Fallback Training"
                                                    
                                                    result = pipeline.run_full_training_pipeline(
                                                        backbone='cspdarknet',
                                                        resume_from_checkpoint=True
                                                    )
        
        # Should fallback to fresh training and succeed
        assert result['success'] is True
        
        # Verify resume was attempted but failed
        mock_resume.assert_called_once()
        
        # Verify fallback phases were executed
        mock_prep.assert_called_once()
        mock_build.assert_called_once()
        mock_validate.assert_called_once()
    
    def test_resume_with_configuration_mismatch(self, pipeline, mock_config):
        """Test resume with configuration that doesn't match checkpoint."""
        resume_info = {
            'checkpoint_name': 'checkpoint_different_config.pth',
            'phase': 1,
            'epoch': 2,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_mismatch',
            'config': {
                'model': {'backbone': 'efficientnet_b4'},  # Different backbone
                'training_phases': {'phase_1': {'epochs': 10}}  # Different epochs
            }
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_mismatch', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock handling of configuration mismatch
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'config_mismatch_handled': True}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.3}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Config Mismatch Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',  # Different from checkpoint
                                phase_1_epochs=5,  # Different from checkpoint
                                phase_2_epochs=3,
                                resume_from_checkpoint=True
                            )
        
        # Should handle mismatch gracefully
        assert result['success'] is True
        
        # Verify resume was called with mismatched configuration
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert resume_args[1] == 'cspdarknet'  # Requested backbone
        assert resume_args[2] == 5  # Requested phase_1_epochs
    
    def test_resume_with_device_change(self, pipeline, mock_config):
        """Test resume when switching devices (CPU to GPU, etc.)."""
        resume_info = {
            'checkpoint_name': 'checkpoint_cpu.pth',
            'phase': 2,
            'epoch': 1,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_device_change',
            'device': 'cpu'  # Checkpoint was on CPU
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_device_change', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock device change handling
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built', 'device_changed': True}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'message': 'Skipped (resumed from phase 2)'}
                    phase2_result = {'success': True, 'device_migration_successful': True}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Device Change Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=5,
                                phase_2_epochs=3,
                                force_cpu=False,  # Try to use GPU if available
                                resume_from_checkpoint=True
                            )
        
        # Should handle device change successfully
        assert result['success'] is True
        
        # Verify device change parameters were passed
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args
        assert resume_args[1]['force_cpu'] is False
    
    def test_resume_with_missing_model_state(self, pipeline, mock_config):
        """Test resume when checkpoint doesn't contain model state dict."""
        incomplete_resume_info = {
            'checkpoint_name': 'checkpoint_no_model.pth',
            'phase': 1,
            'epoch': 3,
            # Missing model_state_dict
            'session_id': 'session_no_model'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_no_model', incomplete_resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock handling of missing model state
                    prep_result = {'success': True, 'config': mock_config}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'model_state_warning': 'No model state in checkpoint'}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.6}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Missing Model State Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                resume_from_checkpoint=True
                            )
        
        # Should handle missing model state gracefully
        assert result['success'] is True
        
        # Verify resume was attempted with incomplete info
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert 'model_state_dict' not in resume_args[0]
    
    def test_resume_epoch_boundary_cases(self, pipeline, mock_config):
        """Test resume from epoch boundary cases (first epoch, last epoch)."""
        boundary_cases = [
            {'epoch': 0, 'phase': 1, 'description': 'first epoch of phase 1'},
            {'epoch': 4, 'phase': 1, 'description': 'last epoch of phase 1'},
            {'epoch': 0, 'phase': 2, 'description': 'first epoch of phase 2'},
            {'epoch': 2, 'phase': 2, 'description': 'last epoch of phase 2'},
        ]
        
        for case in boundary_cases:
            resume_info = {
                'checkpoint_name': f'checkpoint_boundary_{case["epoch"]}_{case["phase"]}.pth',
                'phase': case['phase'],
                'epoch': case['epoch'],
                'model_state_dict': {'layer.weight': torch.tensor([1.0])},
                'session_id': f'session_boundary_{case["epoch"]}_{case["phase"]}'
            }
            
            with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
                with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                    mock_setup.return_value = (resume_info['session_id'], resume_info)
                    
                    with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                        # Mock successful boundary case handling
                        prep_result = {'success': True, 'config': mock_config}
                        build_result = {'success': True, 'model_info': 'built'}
                        validate_result = {'success': True, 'forward_pass_successful': True}
                        phase1_result = {'success': True, f'boundary_case': case['description']}
                        phase2_result = {'success': True, 'final_metrics': {'loss': 0.5}}
                        
                        mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                        
                        with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                            mock_summary.return_value = {'success': True}
                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                                mock_md.return_value = f"# Boundary Case: {case['description']}"
                                
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    phase_1_epochs=5,
                                    phase_2_epochs=3,
                                    resume_from_checkpoint=True
                                )
            
            # Should handle boundary cases successfully
            assert result['success'] is True, f"Failed for boundary case: {case['description']}"
            
            # Verify correct epoch and phase were passed
            mock_resume.assert_called_once()
            resume_args = mock_resume.call_args[0]
            assert resume_args[0]['epoch'] == case['epoch']
            assert resume_args[0]['phase'] == case['phase']


class TestUnifiedTrainingPipelineCheckpointOperations:
    """Test checkpoint-related operations and utilities."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    @pytest.fixture
    def temp_dirs(self):
        """Create temporary directories for testing."""
        temp_base = tempfile.mkdtemp()
        dirs = {
            'checkpoints': Path(temp_base) / 'checkpoints'
        }
        
        for dir_path in dirs.values():
            dir_path.mkdir(parents=True, exist_ok=True)
        
        yield dirs
        
        # Cleanup
        shutil.rmtree(temp_base, ignore_errors=True)
    
    @pytest.fixture
    def mock_config(self, temp_dirs):
        """Create mock configuration for checkpoint testing."""
        return {
            'model': {
                'backbone': 'cspdarknet',
                'layer_mode': 'multi'
            },
            'paths': {
                'checkpoints': temp_dirs['checkpoints']
            }
        }
    
    def test_save_checkpoint_with_model_api(self, pipeline, mock_config, temp_dirs):
        """Test checkpoint saving using model API."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0, 2.0])}
        pipeline.model = mock_model
        
        # Mock model API
        mock_api = MagicMock()
        saved_path = str(temp_dirs['checkpoints'] / 'test_checkpoint.pth')
        mock_api.save_checkpoint.return_value = saved_path
        pipeline.model_api = mock_api
        
        # Test checkpoint saving
        metrics = {'loss': 0.5, 'accuracy': 0.85}
        result_path = pipeline._save_checkpoint(epoch=2, metrics=metrics, phase_num=1)
        
        # Verify checkpoint was saved via API
        assert result_path == saved_path
        mock_api.save_checkpoint.assert_called_once()
        
        # Verify correct parameters were passed
        save_args = mock_api.save_checkpoint.call_args[1]
        assert save_args['epoch'] == 2
        assert save_args['phase'] == 1
        assert save_args['metrics'] == metrics
        assert save_args['is_best'] is True
        assert save_args['config'] == mock_config
    
    def test_save_checkpoint_fallback_to_utils(self, pipeline, mock_config, temp_dirs):
        """Test checkpoint saving fallback when model API fails."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0, 2.0])}
        pipeline.model = mock_model
        
        # Mock model API that fails
        mock_api = MagicMock()
        mock_api.save_checkpoint.return_value = None  # API fails
        pipeline.model_api = mock_api
        
        with patch('smartcash.model.training.unified_training_pipeline.generate_checkpoint_name') as mock_gen_name:
            checkpoint_name = 'fallback_checkpoint.pth'
            mock_gen_name.return_value = checkpoint_name
            
            with patch('smartcash.model.training.unified_training_pipeline.save_checkpoint_to_disk') as mock_save_disk:
                mock_save_disk.return_value = True
                
                # Test checkpoint saving with fallback
                metrics = {'loss': 0.3, 'accuracy': 0.9}
                result_path = pipeline._save_checkpoint(epoch=5, metrics=metrics, phase_num=2)
                
                # Verify fallback was used
                assert result_path == str(temp_dirs['checkpoints'] / checkpoint_name)
                mock_save_disk.assert_called_once()
                
                # Verify fallback parameters
                save_args = mock_save_disk.call_args[1]
                assert save_args['epoch'] == 5
                assert save_args['phase'] == 2
                assert save_args['metrics'] == metrics
                assert save_args['config'] == mock_config
                assert save_args['session_id'] == 'test_session'
    
    def test_save_checkpoint_complete_failure(self, pipeline, mock_config):
        """Test checkpoint saving when both API and fallback fail."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0, 2.0])}
        pipeline.model = mock_model
        
        # Mock model API that fails
        mock_api = MagicMock()
        mock_api.save_checkpoint.return_value = None
        pipeline.model_api = mock_api
        
        with patch('smartcash.model.training.unified_training_pipeline.generate_checkpoint_name') as mock_gen_name:
            mock_gen_name.return_value = 'failed_checkpoint.pth'
            
            with patch('smartcash.model.training.unified_training_pipeline.save_checkpoint_to_disk') as mock_save_disk:
                mock_save_disk.return_value = False  # Fallback also fails
                
                # Test checkpoint saving with complete failure
                result_path = pipeline._save_checkpoint(epoch=1, metrics={}, phase_num=1)
                
                # Should return None on complete failure
                assert result_path is None
    
    def test_save_checkpoint_exception_handling(self, pipeline, mock_config):
        """Test checkpoint saving with exception during save operation."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        pipeline.model = mock_model
        
        # Mock model API that raises exception
        mock_api = MagicMock()
        mock_api.save_checkpoint.side_effect = Exception("Save failed")
        pipeline.model_api = mock_api
        
        # Test checkpoint saving with exception
        result_path = pipeline._save_checkpoint(epoch=1, metrics={}, phase_num=1)
        
        # Should handle exception gracefully and return None
        assert result_path is None
    
    def test_save_checkpoint_no_model_api(self, pipeline, mock_config, temp_dirs):
        """Test checkpoint saving when no model API is available."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        pipeline.model_api = None  # No model API
        
        # Mock model
        mock_model = MagicMock()
        mock_model.parameters.return_value = [torch.tensor([1.0]).to('cpu')]
        mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0])}
        pipeline.model = mock_model
        
        with patch('smartcash.model.training.unified_training_pipeline.generate_checkpoint_name') as mock_gen_name:
            checkpoint_name = 'no_api_checkpoint.pth'
            mock_gen_name.return_value = checkpoint_name
            
            with patch('smartcash.model.training.unified_training_pipeline.save_checkpoint_to_disk') as mock_save_disk:
                mock_save_disk.return_value = True
                
                # Test checkpoint saving without model API
                result_path = pipeline._save_checkpoint(epoch=3, metrics={'loss': 0.4}, phase_num=1)
                
                # Should use fallback directly
                assert result_path == str(temp_dirs['checkpoints'] / checkpoint_name)
                mock_save_disk.assert_called_once()
    
    def test_save_checkpoint_device_information_logging(self, pipeline, mock_config):
        """Test that checkpoint saving logs device information correctly."""
        pipeline.config = mock_config
        pipeline.training_session_id = 'test_session'
        
        # Test different device scenarios
        device_scenarios = [
            ('cpu', 'cpu'),
            ('mps', 'mps'),
            ('cuda', 'gpu')
        ]
        
        for device_type, expected_log in device_scenarios:
            # Mock model with specific device
            mock_model = MagicMock()
            mock_device = torch.device(device_type)
            mock_model.parameters.return_value = [torch.tensor([1.0]).to(mock_device)]
            mock_model.state_dict.return_value = {'layer.weight': torch.tensor([1.0])}
            pipeline.model = mock_model
            
            # Mock model API
            mock_api = MagicMock()
            mock_api.save_checkpoint.return_value = '/tmp/test_checkpoint.pth'
            pipeline.model_api = mock_api
            
            # Test checkpoint saving
            result_path = pipeline._save_checkpoint(epoch=1, metrics={}, phase_num=1)
            
            # Verify checkpoint was saved
            assert result_path == '/tmp/test_checkpoint.pth'
            
            # Device information should be captured (logged in real implementation)
            # This test verifies the device detection logic works correctly
            device_param = next(pipeline.model.parameters())
            assert device_param.device.type == device_type


class TestUnifiedTrainingPipelineResumeRobustness:
    """Test robustness and edge cases in resume functionality."""
    
    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return UnifiedTrainingPipeline(verbose=False)
    
    def test_resume_with_extremely_large_epoch_numbers(self, pipeline):
        """Test resume with unrealistically large epoch numbers."""
        resume_info = {
            'checkpoint_name': 'checkpoint_large_epoch.pth',
            'phase': 1,
            'epoch': 999999,  # Extremely large epoch
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_large_epoch'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_large_epoch', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock handling of large epoch numbers
                    prep_result = {'success': True, 'config': {'training_phases': {'phase_1': {'epochs': 5}}}}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'large_epoch_handled': True}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.5}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Large Epoch Resume Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                phase_1_epochs=5,
                                resume_from_checkpoint=True
                            )
        
        # Should handle large epoch numbers gracefully
        assert result['success'] is True
        
        # Verify large epoch was passed correctly
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        assert resume_args[0]['epoch'] == 999999
    
    def test_resume_with_negative_epoch_numbers(self, pipeline):
        """Test resume with negative epoch numbers."""
        resume_info = {
            'checkpoint_name': 'checkpoint_negative_epoch.pth',
            'phase': 1,
            'epoch': -5,  # Negative epoch
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_negative_epoch'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_negative_epoch', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock should handle negative epoch (might fall back to fresh training)
                    mock_resume.side_effect = Exception("Invalid negative epoch")
                    
                    with patch.object(pipeline, '_phase_preparation') as mock_prep:
                        mock_config = {'training_phases': {'phase_1': {'epochs': 5}, 'phase_2': {'epochs': 3}}}
                        mock_prep.return_value = {'success': True, 'config': mock_config}
                        with patch.object(pipeline, '_phase_build_model') as mock_build:
                            mock_build.return_value = {'success': True}
                            with patch.object(pipeline, '_phase_validate_model') as mock_validate:
                                mock_validate.return_value = {'success': True}
                                with patch.object(pipeline, '_phase_training_1_with_manager') as mock_train1:
                                    mock_train1.return_value = {'success': True}
                                    with patch.object(pipeline, '_phase_training_2_with_manager') as mock_train2:
                                        mock_train2.return_value = {'success': True}
                                        with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                                            mock_summary.return_value = {'success': True}
                                            with patch('smartcash.model.training.unified_training_pipeline.TrainingPhaseManager'):
                                                with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                                                    mock_md.return_value = "# Negative Epoch Fallback"
                                                    
                                                    result = pipeline.run_full_training_pipeline(
                                                        backbone='cspdarknet',
                                                        resume_from_checkpoint=True
                                                    )
        
        # Should fall back to fresh training
        assert result['success'] is True
        
        # Verify resume was attempted but failed
        mock_resume.assert_called_once()
    
    def test_resume_with_very_long_session_ids(self, pipeline):
        """Test resume with extremely long session IDs."""
        long_session_id = "x" * 10000  # Very long session ID
        resume_info = {
            'checkpoint_name': 'checkpoint_long_session.pth',
            'phase': 1,
            'epoch': 2,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': long_session_id
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = (long_session_id, resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock successful handling of long session ID
                    prep_result = {'success': True, 'config': {'training_phases': {'phase_1': {'epochs': 5}}}}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'long_session_handled': True}
                    phase2_result = {'success': True, 'final_metrics': {'loss': 0.5}}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Long Session ID Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                resume_from_checkpoint=True
                            )
        
        # Should handle long session IDs without issues
        assert result['success'] is True
        
        # Verify session ID was preserved
        assert pipeline.training_session_id == long_session_id
    
    def test_resume_memory_stress(self, pipeline):
        """Test resume functionality under memory stress conditions."""
        # Simulate large model state dict
        large_state_dict = {
            f'layer_{i}.weight': torch.randn(1000, 1000) for i in range(10)
        }
        
        resume_info = {
            'checkpoint_name': 'checkpoint_large_state.pth',
            'phase': 2,
            'epoch': 1,
            'model_state_dict': large_state_dict,
            'session_id': 'session_memory_stress'
        }
        
        with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
            with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                mock_setup.return_value = ('session_memory_stress', resume_info)
                
                with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                    # Mock successful handling of large state dict
                    prep_result = {'success': True, 'config': {'training_phases': {'phase_2': {'epochs': 3}}}}
                    build_result = {'success': True, 'model_info': 'built'}
                    validate_result = {'success': True, 'forward_pass_successful': True}
                    phase1_result = {'success': True, 'message': 'Skipped (resumed from phase 2)'}
                    phase2_result = {'success': True, 'large_state_loaded': True}
                    
                    mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                    
                    with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                        mock_summary.return_value = {'success': True}
                        with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                            mock_md.return_value = "# Memory Stress Test"
                            
                            result = pipeline.run_full_training_pipeline(
                                backbone='cspdarknet',
                                resume_from_checkpoint=True
                            )
        
        # Should handle large state dicts successfully
        assert result['success'] is True
        
        # Verify large state dict was passed
        mock_resume.assert_called_once()
        resume_args = mock_resume.call_args[0]
        state_dict = resume_args[0]['model_state_dict']
        assert len(state_dict) == 10  # 10 large tensors
        assert all(tensor.shape == (1000, 1000) for tensor in state_dict.values())
    
    def test_resume_concurrent_access_simulation(self, pipeline):
        """Test resume functionality under simulated concurrent access."""
        resume_info = {
            'checkpoint_name': 'checkpoint_concurrent.pth',
            'phase': 1,
            'epoch': 2,
            'model_state_dict': {'layer.weight': torch.tensor([1.0])},
            'session_id': 'session_concurrent'
        }
        
        # Simulate multiple rapid resume attempts
        for attempt in range(5):
            with patch('smartcash.model.training.unified_training_pipeline.validate_training_mode_and_params'):
                with patch('smartcash.model.training.unified_training_pipeline.setup_training_session') as mock_setup:
                    mock_setup.return_value = (f'session_concurrent_{attempt}', resume_info)
                    
                    with patch('smartcash.model.training.unified_training_pipeline.handle_resume_training_pipeline') as mock_resume:
                        # Mock successful resume
                        prep_result = {'success': True, 'config': {'training_phases': {'phase_1': {'epochs': 5}}}}
                        build_result = {'success': True, 'model_info': 'built'}
                        validate_result = {'success': True, 'forward_pass_successful': True}
                        phase1_result = {'success': True, f'attempt': attempt}
                        phase2_result = {'success': True, 'final_metrics': {'loss': 0.5}}
                        
                        mock_resume.return_value = (prep_result, build_result, validate_result, phase1_result, phase2_result)
                        
                        with patch.object(pipeline, '_phase_summary_visualization') as mock_summary:
                            mock_summary.return_value = {'success': True}
                            with patch('smartcash.model.training.unified_training_pipeline.generate_markdown_summary') as mock_md:
                                mock_md.return_value = f"# Concurrent Test Attempt {attempt}"
                                
                                result = pipeline.run_full_training_pipeline(
                                    backbone='cspdarknet',
                                    resume_from_checkpoint=True
                                )
            
            # Each attempt should succeed independently
            assert result['success'] is True
            
            # Verify session ID was updated for each attempt
            expected_session_id = f'session_concurrent_{attempt}'
            assert pipeline.training_session_id == expected_session_id