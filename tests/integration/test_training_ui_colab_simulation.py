"""
Simulated Colab environment tests for training UI functionality.
"""

import pytest
import unittest.mock as mock
import os
from contextlib import contextmanager
from smartcash.ui.model.training.utils.environment_detector import (
    TrainingEnvironmentDetector,
    detect_environment,
    should_force_cpu_training,
    get_recommended_training_config
)
from smartcash.ui.model.training.configs.training_config_handler import TrainingConfigHandler
from smartcash.ui.model.training.components.unified_training_form import create_unified_training_form


class TestTrainingUIColabSimulation:
    """Test training UI in simulated Colab environments."""
    
    @contextmanager
    def simulate_colab_environment(self, has_gpu=False):
        """Simulate Colab environment by mocking necessary components."""
        # Mock the google.colab import
        mock_colab = mock.MagicMock()
        
        # Mock torch CUDA and MPS availability
        with mock.patch.dict('sys.modules', {'google.colab': mock_colab}):
            with mock.patch('torch.cuda.is_available', return_value=has_gpu):
                with mock.patch('torch.cuda.device_count', return_value=1 if has_gpu else 0):
                    with mock.patch('torch.backends.mps.is_available', return_value=False):
                        with mock.patch('smartcash.ui.model.training.utils.environment_detector.TrainingEnvironmentDetector.is_colab', new_callable=mock.PropertyMock) as mock_is_colab:
                            mock_is_colab.return_value = True
                            if has_gpu:
                                with mock.patch('torch.cuda.get_device_name', return_value='Tesla T4'):
                                    yield
                            else:
                                yield
    
    def test_colab_without_gpu_forces_cpu(self):
        """Test that Colab without GPU forces CPU training."""
        with self.simulate_colab_environment(has_gpu=False):
            detector = TrainingEnvironmentDetector()
            env_info = detector.detect_training_environment()
            
            # Should detect Colab
            assert env_info['is_colab'] == True
            assert env_info['cuda_available'] == False
            assert env_info['has_gpu'] == False
            
            # Should recommend CPU training
            force_cpu, reason = detector.should_force_cpu_training(env_info)
            assert force_cpu == True
            assert 'Colab' in reason or 'GPU' in reason
            
            # Should get appropriate config
            config = detector.get_recommended_training_config(env_info)
            assert config['force_cpu'] == True
            assert config['batch_size'] <= 4  # Small batch for CPU
    
    def test_colab_with_gpu_allows_gpu_training(self):
        """Test that Colab with GPU allows GPU training."""
        with self.simulate_colab_environment(has_gpu=True):
            detector = TrainingEnvironmentDetector()
            env_info = detector.detect_training_environment()
            
            # Should detect Colab with GPU
            assert env_info['is_colab'] == True
            assert env_info['cuda_available'] == True
            assert env_info['has_gpu'] == True
            
            # Should not force CPU training
            force_cpu, reason = detector.should_force_cpu_training(env_info)
            assert force_cpu == False
            
            # Should get appropriate config
            config = detector.get_recommended_training_config(env_info)
            assert config.get('force_cpu', False) == False
            assert config['batch_size'] <= 16  # Moderate batch for Colab
    
    def test_config_handler_in_simulated_colab_no_gpu(self):
        """Test config handler behavior in simulated Colab without GPU."""
        with self.simulate_colab_environment(has_gpu=False):
            # Mock the environment detection in config handler
            with mock.patch('smartcash.ui.model.training.configs.training_config_handler.detect_environment') as mock_detect:
                mock_detect.return_value = {
                    'is_colab': True,
                    'cuda_available': False,
                    'has_gpu': False,
                    'platform': 'colab',
                    'force_cpu_recommended': True
                }
                
                with mock.patch('smartcash.ui.model.training.configs.training_config_handler.get_recommended_training_config') as mock_get_config:
                    mock_get_config.return_value = {
                        'force_cpu': True,
                        'batch_size': 2,
                        'phase_1_epochs': 1,
                        'phase_2_epochs': 1
                    }
                    
                    handler = TrainingConfigHandler()
                    config = handler.get_current_config()
                    
                    # Should apply Colab-specific settings
                    assert config['training']['force_cpu'] == True
                    assert config['training']['batch_size'] == 2
                    assert config['training']['phase_1_epochs'] == 1
                    assert config['training']['phase_2_epochs'] == 1
    
    def test_training_form_in_simulated_colab_no_gpu(self):
        """Test training form behavior in simulated Colab without GPU."""
        # Mock environment detection for the form
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = {
                    'is_colab': True,
                    'cuda_available': False,
                    'has_gpu': False,
                    'platform': 'colab',
                    'device_count': 0,
                    'device_names': []
                }
                mock_should_force.return_value = (True, "Colab environment without GPU detected")
                
                config = {'training': {'force_cpu': False}}
                form_widget = create_unified_training_form(config)
                
                # Environment display should show Colab
                env_display = form_widget.children[0]
                assert 'Colab' in env_display.value
                assert 'CPU only' in env_display.value
                assert 'Colab environment without GPU detected' in env_display.value
                
                # Force CPU checkbox should be checked and show recommendation
                widgets_dict = form_widget._widgets
                force_cpu_widget = widgets_dict['force_cpu']
                assert force_cpu_widget.value == True
                assert 'Recommended:' in force_cpu_widget.description
    
    def test_training_form_in_simulated_colab_with_gpu(self):
        """Test training form behavior in simulated Colab with GPU."""
        # Mock environment detection for the form
        with mock.patch('smartcash.ui.model.training.components.unified_training_form.detect_environment') as mock_detect:
            with mock.patch('smartcash.ui.model.training.components.unified_training_form.should_force_cpu_training') as mock_should_force:
                mock_detect.return_value = {
                    'is_colab': True,
                    'cuda_available': True,
                    'has_gpu': True,
                    'platform': 'colab',
                    'device_count': 1,
                    'device_names': ['Tesla T4']
                }
                mock_should_force.return_value = (False, "GPU/MPS available for training")
                
                config = {'training': {'force_cpu': False}}
                form_widget = create_unified_training_form(config)
                
                # Environment display should show Colab with GPU
                env_display = form_widget.children[0]
                assert 'Colab' in env_display.value
                assert 'CUDA GPU' in env_display.value
                assert 'Colab environment without GPU detected' not in env_display.value
                
                # Force CPU checkbox should not be checked
                widgets_dict = form_widget._widgets
                force_cpu_widget = widgets_dict['force_cpu']
                assert force_cpu_widget.value == False
                assert 'Recommended:' not in force_cpu_widget.description
    
    def test_environment_variables_colab_detection(self):
        """Test environment variable-based Colab detection."""
        # Test with COLAB_GPU environment variable
        with mock.patch.dict(os.environ, {'COLAB_GPU': '1'}):
            with mock.patch('torch.cuda.is_available', return_value=False):
                with mock.patch.dict('sys.modules', {'google.colab': mock.MagicMock()}):
                    with mock.patch('smartcash.ui.model.training.utils.environment_detector.TrainingEnvironmentDetector.is_colab', new_callable=mock.PropertyMock) as mock_is_colab:
                        mock_is_colab.return_value = True
                        detector = TrainingEnvironmentDetector()
                        env_info = detector.detect_training_environment()
                        
                        # Should detect problematic Colab setup
                        assert env_info['is_colab'] == True
                        assert env_info['cuda_available'] == False
                        assert env_info['force_cpu_recommended'] == True
    
    def test_recommendations_are_appropriate_for_environment(self):
        """Test that recommendations are appropriate for different environments."""
        # Test scenarios
        scenarios = [
            {
                'name': 'Colab without GPU',
                'env': {'is_colab': True, 'has_gpu': False, 'cuda_available': False},
                'expected_force_cpu': True,
                'expected_small_batch': True
            },
            {
                'name': 'Colab with GPU',
                'env': {'is_colab': True, 'has_gpu': True, 'cuda_available': True},
                'expected_force_cpu': False,
                'expected_small_batch': False
            },
            {
                'name': 'Local without GPU',
                'env': {'is_colab': False, 'has_gpu': False, 'cuda_available': False, 'mps_available': False},
                'expected_force_cpu': True,
                'expected_small_batch': True
            }
        ]
        
        for scenario in scenarios:
            detector = TrainingEnvironmentDetector()
            
            force_cpu, reason = detector.should_force_cpu_training(scenario['env'])
            config = detector.get_recommended_training_config(scenario['env'])
            
            assert force_cpu == scenario['expected_force_cpu'], f"Force CPU failed for {scenario['name']}"
            
            if scenario['expected_small_batch']:
                assert config.get('batch_size', 32) <= 4, f"Batch size too large for {scenario['name']}"
            
            if scenario['expected_force_cpu']:
                assert config.get('force_cpu', False) == True, f"Force CPU not set for {scenario['name']}"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])