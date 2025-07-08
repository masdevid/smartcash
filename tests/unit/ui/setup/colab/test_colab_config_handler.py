"""
Test module for colab configuration handler.
"""

import pytest
import os
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any

from smartcash.ui.setup.colab.configs.colab_config_handler import ColabConfigHandler
from smartcash.ui.setup.colab.configs.colab_defaults import get_default_colab_config


class TestColabConfigHandler:
    """Test cases for ColabConfigHandler."""
    
    @pytest.fixture
    def config_handler(self):
        """Create a ColabConfigHandler instance for testing."""
        return ColabConfigHandler()
    
    @pytest.fixture
    def sample_config(self):
        """Sample configuration for testing."""
        return {
            'environment': {
                'type': 'colab',
                'auto_mount_drive': True,
                'project_name': 'SmartCash',
                'gpu_enabled': True,
                'gpu_type': 't4'
            },
            'setup': {
                'stages': ['environment_detection', 'drive_mount', 'gpu_setup'],
                'auto_start': False,
                'stop_on_error': True,
                'max_retries': 3
            }
        }
    
    def test_init(self, config_handler):
        """Test ColabConfigHandler initialization."""
        assert config_handler is not None
        assert hasattr(config_handler, '_config')
        assert hasattr(config_handler, '_available_environments')
        assert hasattr(config_handler, '_setup_stages')
        assert hasattr(config_handler, '_gpu_configurations')
    
    def test_get_config(self, config_handler):
        """Test getting configuration."""
        config = config_handler.get_config()
        
        assert isinstance(config, dict)
        assert 'environment' in config
        assert 'setup' in config
        
        # Should return a copy, not the original
        config['environment']['type'] = 'modified'
        original_config = config_handler.get_config()
        assert original_config['environment']['type'] != 'modified'
    
    def test_update_config_valid(self, config_handler, sample_config):
        """Test updating configuration with valid data."""
        result = config_handler.update_config(sample_config)
        
        assert result is True
        updated_config = config_handler.get_config()
        assert updated_config['environment']['type'] == 'colab'
        assert updated_config['environment']['gpu_enabled'] is True
    
    def test_update_config_invalid_environment(self, config_handler):
        """Test updating configuration with invalid environment."""
        invalid_config = {
            'environment': {
                'type': 'invalid_environment'
            }
        }
        
        result = config_handler.update_config(invalid_config)
        assert result is False
    
    def test_validate_config_valid(self, config_handler, sample_config):
        """Test configuration validation with valid config."""
        result = config_handler.validate_config(sample_config)
        assert result is True
    
    def test_validate_config_invalid_environment(self, config_handler):
        """Test configuration validation with invalid environment."""
        invalid_config = {
            'environment': {
                'type': 'nonexistent_environment'
            }
        }
        
        result = config_handler.validate_config(invalid_config)
        assert result is False
    
    def test_validate_config_invalid_setup_stage(self, config_handler):
        """Test configuration validation with invalid setup stage."""
        invalid_config = {
            'setup': {
                'stages': ['invalid_stage']
            }
        }
        
        result = config_handler.validate_config(invalid_config)
        assert result is False
    
    def test_reset_to_defaults(self, config_handler):
        """Test resetting configuration to defaults."""
        # First modify the config
        config_handler.set_environment_type('local')
        
        # Reset to defaults
        config_handler.reset_to_defaults()
        
        config = config_handler.get_config()
        default_config = get_default_colab_config()
        assert config['environment']['type'] == default_config['environment']['type']
    
    def test_set_environment_type_valid(self, config_handler):
        """Test setting valid environment type."""
        result = config_handler.set_environment_type('kaggle')
        assert result is True
        
        config = config_handler.get_config()
        assert config['environment']['type'] == 'kaggle'
        assert config['environment']['base_path'] == '/kaggle/working'
    
    def test_set_environment_type_invalid(self, config_handler):
        """Test setting invalid environment type."""
        result = config_handler.set_environment_type('invalid_environment')
        assert result is False
    
    def test_set_setup_stages_valid(self, config_handler):
        """Test setting valid setup stages."""
        stages = ['environment_detection', 'gpu_setup', 'verify']
        result = config_handler.set_setup_stages(stages)
        assert result is True
        
        config = config_handler.get_config()
        assert config['setup']['stages'] == stages
    
    def test_set_setup_stages_invalid(self, config_handler):
        """Test setting invalid setup stages."""
        stages = ['invalid_stage']
        result = config_handler.set_setup_stages(stages)
        assert result is False
    
    def test_set_gpu_enabled(self, config_handler):
        """Test setting GPU configuration."""
        result = config_handler.set_gpu_enabled(True, 't4')
        assert result is True
        
        config = config_handler.get_config()
        assert config['environment']['gpu_enabled'] is True
        assert config['environment']['gpu_type'] == 't4'
    
    def test_set_gpu_enabled_invalid_type(self, config_handler):
        """Test setting GPU with invalid type."""
        result = config_handler.set_gpu_enabled(True, 'invalid_gpu')
        assert result is False
    
    def test_set_auto_mount_drive(self, config_handler):
        """Test setting auto mount drive."""
        result = config_handler.set_auto_mount_drive(False)
        assert result is True
        
        config = config_handler.get_config()
        assert config['environment']['auto_mount_drive'] is False
    
    def test_set_project_name_valid(self, config_handler):
        """Test setting valid project name."""
        result = config_handler.set_project_name('TestProject')
        assert result is True
        
        config = config_handler.get_config()
        assert config['environment']['project_name'] == 'TestProject'
    
    def test_set_project_name_empty(self, config_handler):
        """Test setting empty project name."""
        result = config_handler.set_project_name('')
        assert result is False
        
        result = config_handler.set_project_name('   ')
        assert result is False
    
    def test_get_current_environment(self, config_handler):
        """Test getting current environment."""
        env_type = config_handler.get_current_environment()
        assert env_type in ['colab', 'kaggle', 'local']
    
    def test_get_current_gpu_config(self, config_handler):
        """Test getting current GPU configuration."""
        gpu_config = config_handler.get_current_gpu_config()
        
        assert isinstance(gpu_config, dict)
        assert 'display_name' in gpu_config
        assert 'memory_gb' in gpu_config
    
    @patch('os.path.exists')
    def test_detect_environment_colab(self, mock_exists, config_handler):
        """Test environment detection for Colab."""
        mock_exists.return_value = False
        
        with patch('smartcash.ui.setup.colab.configs.colab_config_handler.google.colab'):
            config_handler._detect_environment()
            assert config_handler.get_current_environment() == 'colab'
    
    @patch('os.path.exists')
    def test_detect_environment_kaggle(self, mock_exists, config_handler):
        """Test environment detection for Kaggle."""
        mock_exists.return_value = True
        
        with patch('smartcash.ui.setup.colab.configs.colab_config_handler.google.colab', side_effect=ImportError):
            config_handler._detect_environment()
            assert config_handler.get_current_environment() == 'kaggle'
    
    @patch('os.path.exists')
    def test_detect_environment_local(self, mock_exists, config_handler):
        """Test environment detection for local."""
        mock_exists.return_value = False
        
        with patch('smartcash.ui.setup.colab.configs.colab_config_handler.google.colab', side_effect=ImportError):
            config_handler._detect_environment()
            assert config_handler.get_current_environment() == 'local'
    
    def test_is_valid_path(self, config_handler):
        """Test path validation."""
        # Valid absolute paths
        assert config_handler._is_valid_path('/content/test') is True
        assert config_handler._is_valid_path('/kaggle/working') is True
        
        # Invalid relative paths
        assert config_handler._is_valid_path('relative/path') is False
        
        # Invalid paths
        assert config_handler._is_valid_path('') is False
    
    def test_update_project_paths(self, config_handler):
        """Test updating project paths."""
        # Set to colab environment
        config_handler.set_environment_type('colab')
        config_handler._update_project_paths('NewProject')
        
        config = config_handler.get_config()
        assert 'NewProject' in config['paths']['drive_base']
        assert 'NewProject' in config['paths']['colab_base']
    
    def test_deep_merge(self, config_handler):
        """Test deep merge functionality."""
        dict1 = {
            'a': {'b': 1, 'c': 2},
            'd': 3
        }
        dict2 = {
            'a': {'b': 10, 'e': 4},
            'f': 5
        }
        
        result = config_handler._deep_merge(dict1, dict2)
        
        assert result['a']['b'] == 10  # Overwritten
        assert result['a']['c'] == 2   # Preserved
        assert result['a']['e'] == 4   # Added
        assert result['d'] == 3        # Preserved
        assert result['f'] == 5        # Added
    
    def test_get_available_environments(self, config_handler):
        """Test getting available environments."""
        environments = config_handler.get_available_environments()
        
        assert isinstance(environments, dict)
        assert 'colab' in environments
        assert 'kaggle' in environments
        assert 'local' in environments
    
    def test_get_setup_stages_config(self, config_handler):
        """Test getting setup stages configuration."""
        stages_config = config_handler.get_setup_stages_config()
        
        assert isinstance(stages_config, dict)
        assert 'environment_detection' in stages_config
        assert 'drive_mount' in stages_config
        assert 'verify' in stages_config
    
    def test_get_gpu_configurations(self, config_handler):
        """Test getting GPU configurations."""
        gpu_configs = config_handler.get_gpu_configurations()
        
        assert isinstance(gpu_configs, dict)
        assert 'none' in gpu_configs
        assert 't4' in gpu_configs
        assert 'v100' in gpu_configs