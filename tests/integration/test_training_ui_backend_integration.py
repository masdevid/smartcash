"""
Integration tests for training UI with backend configuration and environment detection.
"""

import pytest
import unittest.mock as mock
from typing import Dict, Any
from smartcash.ui.model.training.configs.training_config_handler import TrainingConfigHandler
from smartcash.ui.model.training.components.unified_training_form import create_unified_training_form
from smartcash.ui.model.training.utils.environment_detector import (
    TrainingEnvironmentDetector, 
    detect_environment,
    should_force_cpu_training,
    get_recommended_training_config
)


class TestTrainingUIBackendIntegration:
    """Integration tests for training UI with backend configuration."""
    
    @pytest.fixture
    def mock_local_environment(self):
        """Mock local environment with GPU."""
        return {
            'is_colab': False,
            'is_local': True,
            'has_gpu': True,
            'has_mps': False,
            'cuda_available': True,
            'mps_available': False,
            'device_count': 1,
            'device_names': ['NVIDIA GeForce RTX 3080'],
            'platform': 'local',
            'force_cpu_recommended': False
        }
    
    @pytest.fixture
    def mock_colab_no_gpu_environment(self):
        """Mock Colab environment without GPU."""
        return {
            'is_colab': True,
            'is_local': False,
            'has_gpu': False,
            'has_mps': False,
            'cuda_available': False,
            'mps_available': False,
            'device_count': 0,
            'device_names': [],
            'platform': 'colab',
            'force_cpu_recommended': True
        }
    
    @pytest.fixture
    def mock_colab_with_gpu_environment(self):
        """Mock Colab environment with GPU."""
        return {
            'is_colab': True,
            'is_local': False,
            'has_gpu': True,
            'has_mps': False,
            'cuda_available': True,
            'mps_available': False,
            'device_count': 1,
            'device_names': ['Tesla T4'],
            'platform': 'colab',
            'force_cpu_recommended': False
        }
    
    @pytest.fixture
    def mock_apple_mps_environment(self):
        """Mock Apple Silicon environment with MPS."""
        return {
            'is_colab': False,
            'is_local': True,
            'has_gpu': True,
            'has_mps': True,
            'cuda_available': False,
            'mps_available': True,
            'device_count': 1,
            'device_names': ['Apple MPS'],
            'platform': 'local',
            'force_cpu_recommended': False
        }
    
    def test_environment_detector_with_mixin(self):
        """Test that TrainingEnvironmentDetector properly uses EnvironmentMixin."""
        detector = TrainingEnvironmentDetector()
        
        # Verify it has mixin methods
        assert hasattr(detector, 'get_environment_info')
        assert hasattr(detector, 'is_colab')
        assert hasattr(detector, 'detect_training_environment')
        assert hasattr(detector, 'should_force_cpu_training')
        assert hasattr(detector, 'get_recommended_training_config')
        
        # Test the training-specific method
        env_info = detector.detect_training_environment()
        assert isinstance(env_info, dict)
        assert 'force_cpu_recommended' in env_info
        assert 'platform' in env_info
        assert 'cuda_available' in env_info
        assert 'mps_available' in env_info
    
    def test_config_handler_applies_local_environment(self, mock_local_environment):
        """Test that config handler applies local environment settings."""
        with mock.patch('smartcash.ui.model.training.configs.training_config_handler.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.configs.training_config_handler.get_recommended_training_config') as mock_get_config:
                mock_detect.return_value = mock_local_environment
                mock_get_config.return_value = {
                    'force_cpu': False,
                    'batch_size': 16,
                }
                
                # Create config handler (should auto-detect environment)
                handler = TrainingConfigHandler()
                config = handler.get_current_config()
                
                # Should not force CPU for local GPU environment
                assert config['training']['force_cpu'] == False
                assert config['training']['batch_size'] == 16  # Local CUDA default
                
                mock_detect.assert_called_once()
    
    def test_config_handler_applies_colab_no_gpu_environment(self, mock_colab_no_gpu_environment):
        """Test that config handler applies Colab without GPU settings."""
        with mock.patch('smartcash.ui.model.training.configs.training_config_handler.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.configs.training_config_handler.get_recommended_training_config') as mock_get_config:
                mock_detect.return_value = mock_colab_no_gpu_environment
                mock_get_config.return_value = {
                    'force_cpu': True,
                    'batch_size': 2,
                    'phase_1_epochs': 1,
                    'phase_2_epochs': 1
                }
                
                # Create config handler (should auto-detect environment)
                handler = TrainingConfigHandler()
                config = handler.get_current_config()
                
                # Should force CPU for Colab without GPU
                assert config['training']['force_cpu'] == True
                assert config['training']['batch_size'] == 2  # Small batch for CPU
                assert config['training']['phase_1_epochs'] == 1  # Reduced epochs
                assert config['training']['phase_2_epochs'] == 1  # Reduced epochs
                
                mock_detect.assert_called_once()
    
    def test_config_handler_applies_colab_with_gpu_environment(self, mock_colab_with_gpu_environment):
        """Test that config handler applies Colab with GPU settings."""
        with mock.patch('smartcash.ui.model.training.configs.training_config_handler.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.configs.training_config_handler.get_recommended_training_config') as mock_get_config:
                mock_detect.return_value = mock_colab_with_gpu_environment
                mock_get_config.return_value = {
                    'force_cpu': False,
                    'batch_size': 8,
                    'phase_1_epochs': 1,
                    'phase_2_epochs': 1
                }
                
                # Create config handler (should auto-detect environment)
                handler = TrainingConfigHandler()
                config = handler.get_current_config()
                
                # Should not force CPU for Colab with GPU
                assert config['training']['force_cpu'] == False
                assert config['training']['batch_size'] == 8  # Moderate batch for Colab GPU
                assert config['training']['phase_1_epochs'] == 1  # Still reduced for Colab
                assert config['training']['phase_2_epochs'] == 1  # Still reduced for Colab
                
                mock_detect.assert_called_once()
    
    def test_config_handler_applies_apple_mps_environment(self, mock_apple_mps_environment):
        """Test that config handler applies Apple MPS settings."""
        with mock.patch('smartcash.ui.model.training.configs.training_config_handler.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.configs.training_config_handler.get_recommended_training_config') as mock_get_config:
                mock_detect.return_value = mock_apple_mps_environment
                mock_get_config.return_value = {
                    'force_cpu': False,
                    'batch_size': 4
                }
                
                # Create config handler (should auto-detect environment)
                handler = TrainingConfigHandler()
                config = handler.get_current_config()
                
                # Should not force CPU for MPS
                assert config['training']['force_cpu'] == False
                assert config['training']['batch_size'] == 4  # Conservative for unified memory
                
                mock_detect.assert_called_once()
    
    def test_training_form_shows_environment_info(self, mock_colab_no_gpu_environment):
        """Test that training form displays environment information."""
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = mock_colab_no_gpu_environment
                mock_should_force.return_value = (True, "Colab environment without GPU detected")
                
                # Create form with default config
                config = {'training': {}}
                form_widget = create_unified_training_form(config)
                
                # Verify form structure
                assert hasattr(form_widget, 'children')
                assert len(form_widget.children) == 2  # Environment display + accordion
                
                # First child should be environment display
                env_display = form_widget.children[0]
                assert hasattr(env_display, 'value')
                assert 'Colab' in env_display.value
                assert 'CPU only' in env_display.value
                assert 'Colab environment without GPU detected' in env_display.value
                
                # Second child should be form accordion
                form_accordion = form_widget.children[1]
                assert hasattr(form_accordion, 'children')
                
                mock_detect.assert_called_once()
                mock_should_force.assert_called_once()
    
    def test_training_form_force_cpu_checkbox_recommended(self, mock_colab_no_gpu_environment):
        """Test that force CPU checkbox shows recommendation."""
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = mock_colab_no_gpu_environment
                mock_should_force.return_value = (True, "Colab environment without GPU detected")
                
                # Create form with default config
                config = {'training': {'force_cpu': False}}
                form_widget = create_unified_training_form(config)
                
                # Get form widgets
                widgets_dict = form_widget._widgets
                force_cpu_widget = widgets_dict['force_cpu']
                
                # Should be checked due to recommendation
                assert force_cpu_widget.value == True
                assert "Recommended:" in force_cpu_widget.description
                assert "Colab environment without GPU detected" in force_cpu_widget.description
                
                mock_detect.assert_called_once()
                mock_should_force.assert_called_once()
    
    def test_training_form_force_cpu_checkbox_not_recommended(self, mock_local_environment):
        """Test that force CPU checkbox works normally when not recommended."""
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = mock_local_environment
                mock_should_force.return_value = (False, "GPU/MPS available for training")
                
                # Create form with default config
                config = {'training': {'force_cpu': False}}
                form_widget = create_unified_training_form(config)
                
                # Get form widgets
                widgets_dict = form_widget._widgets
                force_cpu_widget = widgets_dict['force_cpu']
                
                # Should not be checked and no recommendation text
                assert force_cpu_widget.value == False
                assert "Recommended:" not in force_cpu_widget.description
                assert force_cpu_widget.description == "Force CPU Training"
                
                mock_detect.assert_called_once()
                mock_should_force.assert_called_once()
    
    def test_convenience_functions_work(self):
        """Test that convenience functions work without errors."""
        # These should not raise errors
        env_info = detect_environment()
        assert isinstance(env_info, dict)
        
        force_cpu, reason = should_force_cpu_training(env_info)
        assert isinstance(force_cpu, bool)
        assert isinstance(reason, str)
        
        recommended_config = get_recommended_training_config(env_info)
        assert isinstance(recommended_config, dict)
    
    def test_config_extraction_from_form(self, mock_local_environment):
        """Test that configuration can be extracted from form correctly."""
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = mock_local_environment
                mock_should_force.return_value = (False, "GPU/MPS available for training")
                
                # Create form with test config (using valid epoch values)
                test_config = {
                    'training': {
                        'backbone': 'efficientnet_b4',
                        'training_mode': 'single_phase',
                        'phase_1_epochs': 25,  # Valid minimum is 20
                        'phase_2_epochs': 55,  # Valid minimum is 50
                        'force_cpu': False,
                        'pretrained': True,
                        'feature_optimization': False
                    }
                }
                form_widget = create_unified_training_form(test_config)
                
                # Extract configuration
                extracted_config = form_widget.get_form_values()
                
                # Verify extracted values
                assert extracted_config['backbone'] == 'efficientnet_b4'
                assert extracted_config['training_mode'] == 'single_phase'
                assert extracted_config['phase_1_epochs'] == 25
                assert extracted_config['phase_2_epochs'] == 55
                assert isinstance(extracted_config['force_cpu'], bool)
                assert extracted_config['pretrained'] == True
                assert extracted_config['feature_optimization'] == False
                
                mock_detect.assert_called_once()
    
    def test_form_update_from_config(self, mock_local_environment):
        """Test that form can be updated from configuration."""
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = mock_local_environment
                mock_should_force.return_value = (False, "GPU/MPS available for training")
                
                # Create form with default config
                form_widget = create_unified_training_form({'training': {}})
                
                # Update with new config (using valid epoch values)
                new_config = {
                    'training': {
                        'backbone': 'cspdarknet',
                        'training_mode': 'two_phase',
                        'phase_1_epochs': 30,  # Valid minimum is 20
                        'phase_2_epochs': 60,  # Valid minimum is 50
                        'force_cpu': True,
                        'pretrained': False,
                        'feature_optimization': True
                    }
                }
                form_widget.update_from_config(new_config)
                
                # Verify widgets were updated
                widgets_dict = form_widget._widgets
                assert widgets_dict['backbone'].value == 'cspdarknet'
                assert widgets_dict['training_mode'].value == 'two_phase'
                assert widgets_dict['phase_1_epochs'].value == 30
                assert widgets_dict['phase_2_epochs'].value == 60
                assert widgets_dict['force_cpu'].value == True
                assert widgets_dict['pretrained'].value == False
                assert widgets_dict['feature_optimization'].value == True
                
                mock_detect.assert_called_once()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])