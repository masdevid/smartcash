"""
Basic integration tests for training UI functionality.
"""

import pytest
from smartcash.ui.model.training.configs.training_config_handler import TrainingConfigHandler
from smartcash.ui.model.training.components.unified_training_form import create_unified_training_form
from smartcash.ui.model.training.utils.environment_detector import (
    TrainingEnvironmentDetector, 
    detect_environment,
    should_force_cpu_training,
    get_recommended_training_config
)


class TestTrainingUIBasicIntegration:
    """Basic integration tests for training UI functionality."""
    
    def test_environment_detector_basic_functionality(self):
        """Test that environment detector basic functionality works."""
        # Test the class-based approach
        detector = TrainingEnvironmentDetector()
        env_info = detector.detect_training_environment()
        
        # Verify basic structure
        assert isinstance(env_info, dict)
        assert 'platform' in env_info
        assert 'force_cpu_recommended' in env_info
        assert 'cuda_available' in env_info
        assert 'mps_available' in env_info
        assert 'device_count' in env_info
        
        # Test convenience functions
        force_cpu, reason = detector.should_force_cpu_training(env_info)
        assert isinstance(force_cpu, bool)
        assert isinstance(reason, str)
        
        recommended_config = detector.get_recommended_training_config(env_info)
        assert isinstance(recommended_config, dict)
    
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
        
        # Verify required keys exist
        if 'force_cpu' in recommended_config:
            assert isinstance(recommended_config['force_cpu'], bool)
        if 'batch_size' in recommended_config:
            assert isinstance(recommended_config['batch_size'], int)
    
    def test_config_handler_initialization(self):
        """Test that config handler initializes correctly with environment detection."""
        handler = TrainingConfigHandler()
        config = handler.get_current_config()
        
        # Verify basic structure
        assert isinstance(config, dict)
        assert 'training' in config
        assert isinstance(config['training'], dict)
        
        # Verify environment-applied settings exist
        training_config = config['training']
        assert 'force_cpu' in training_config
        assert isinstance(training_config['force_cpu'], bool)
        assert 'batch_size' in training_config
        assert training_config['batch_size'] is None or isinstance(training_config['batch_size'], int)
    
    def test_training_form_creation(self):
        """Test that training form creates correctly with environment info."""
        # Create form with default config
        config = {'training': {}}
        form_widget = create_unified_training_form(config)
        
        # Verify basic structure
        assert hasattr(form_widget, 'children')
        assert len(form_widget.children) == 2  # Environment display + accordion
        
        # First child should be environment display (HTML widget)
        env_display = form_widget.children[0]
        assert hasattr(env_display, 'value')
        assert isinstance(env_display.value, str)
        assert 'Environment:' in env_display.value
        assert 'Compute:' in env_display.value
        
        # Second child should be form accordion
        form_accordion = form_widget.children[1]
        assert hasattr(form_accordion, 'children')
        
        # Verify form has required methods
        assert hasattr(form_widget, 'get_form_values')
        assert hasattr(form_widget, 'update_from_config')
        assert hasattr(form_widget, '_widgets')
    
    def test_training_form_widgets_exist(self):
        """Test that all required widgets exist in the form."""
        config = {'training': {}}
        form_widget = create_unified_training_form(config)
        
        # Get form widgets
        widgets_dict = form_widget._widgets
        
        # Verify required widgets exist
        required_widgets = [
            'backbone', 'training_mode', 'phase_1_epochs', 'phase_2_epochs',
            'loss_type', 'head_lr_p1', 'head_lr_p2', 'backbone_lr',
            'force_cpu', 'verbose', 'pretrained', 'feature_optimization'
        ]
        
        for widget_name in required_widgets:
            assert widget_name in widgets_dict, f"Missing widget: {widget_name}"
        
        # Verify force_cpu widget has description (may include recommendation)
        force_cpu_widget = widgets_dict['force_cpu']
        assert hasattr(force_cpu_widget, 'description')
        assert 'Force CPU Training' in force_cpu_widget.description
    
    def test_config_extraction_basic(self):
        """Test basic configuration extraction from form."""
        # Create form with some test values
        test_config = {
            'training': {
                'backbone': 'efficientnet_b4',
                'training_mode': 'single_phase'
            }
        }
        form_widget = create_unified_training_form(test_config)
        
        # Extract configuration
        extracted_config = form_widget.get_form_values()
        
        # Verify it's a dictionary with expected keys
        assert isinstance(extracted_config, dict)
        
        # Check required keys exist
        required_keys = [
            'backbone', 'training_mode', 'phase_1_epochs', 'phase_2_epochs',
            'force_cpu', 'pretrained', 'feature_optimization'
        ]
        
        for key in required_keys:
            assert key in extracted_config, f"Missing extracted config key: {key}"
    
    def test_environment_detector_handles_errors_gracefully(self):
        """Test that environment detector handles errors gracefully."""
        try:
            detector = TrainingEnvironmentDetector()
            env_info = detector.detect_training_environment()
            
            # Should not raise exceptions and should return valid data
            assert isinstance(env_info, dict)
            assert 'force_cpu_recommended' in env_info
            
        except Exception as e:
            pytest.fail(f"Environment detector should handle errors gracefully, but raised: {e}")
    
    def test_environment_shows_different_info_types(self):
        """Test that environment display shows appropriate info for different setups."""
        env_info = detect_environment()
        
        # Check that we get either CUDA, MPS, or CPU info
        has_compute_info = (
            env_info.get('cuda_available', False) or 
            env_info.get('mps_available', False) or 
            not env_info.get('has_gpu', True)  # CPU only
        )
        assert has_compute_info, "Should detect some type of compute capability"
        
        # Platform should be detected
        assert env_info.get('platform') in ['local', 'colab', 'unknown']
    
    def test_config_handler_respects_environment(self):
        """Test that config handler respects environment settings."""
        # Get current environment
        env_info = detect_environment()
        force_cpu_recommended = should_force_cpu_training(env_info)[0]
        
        # Create handler
        handler = TrainingConfigHandler()
        config = handler.get_current_config()
        
        # If environment recommends CPU, config should reflect that
        if force_cpu_recommended:
            assert config['training']['force_cpu'] == True
            # Batch size should be small for CPU
            if config['training']['batch_size'] is not None:
                assert config['training']['batch_size'] <= 4
        
        # Batch size should be reasonable (not None and positive)
        batch_size = config['training']['batch_size']
        if batch_size is not None:
            assert isinstance(batch_size, int)
            assert batch_size > 0
            assert batch_size <= 32  # Reasonable upper limit


if __name__ == '__main__':
    pytest.main([__file__, '-v'])