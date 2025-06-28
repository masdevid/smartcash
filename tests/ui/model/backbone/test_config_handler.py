"""
Tests for BackboneConfigHandler class.
"""
import os
import tempfile
import shutil
import pytest
from unittest.mock import MagicMock
from smartcash.ui.model.backbone.handlers.config_handler import BackboneConfigHandler

class TestBackboneConfigHandler:
    """Test cases for BackboneConfigHandler."""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create a temporary config directory for testing."""
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        os.chdir(temp_dir)
        os.makedirs('config', exist_ok=True)
        
        yield temp_dir
        
        # Cleanup
        os.chdir(original_dir)
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_logger(self):
        """Create a mock logger."""
        return MagicMock()
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = BackboneConfigHandler.get_default_config()
        assert 'model' in config
        assert config['model']['backbone'] == 'efficientnet_b4'
        assert config['model']['model_name'] == 'smartcash_yolov5'
    
    def test_init_creates_config_file(self, temp_config_dir, mock_logger):
        """Test that init creates config file if it doesn't exist."""
        # Expected config file path
        expected_filename = 'model_config.yaml'
        expected_path = os.path.join('config', expected_filename)
        
        # Ensure the file doesn't exist before we start
        if os.path.exists(expected_path):
            os.remove(expected_path)
            
        # Create the handler - this should create the config file
        handler = BackboneConfigHandler(mock_logger)
        
        # Verify config file was created with the correct name pattern
        assert os.path.exists(expected_path), f"Expected file {expected_path} does not exist"
        assert hasattr(handler, 'file_path')
        assert os.path.basename(handler.file_path) == expected_filename, \
            f"Expected filename {expected_filename}, got {os.path.basename(handler.file_path)}"
            
        # Verify the file contains the default config
        with open(handler.file_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            assert 'model' in config, "Config file missing 'model' key"
            assert config['model']['backbone'] == 'efficientnet_b4', \
                f"Expected default backbone 'efficientnet_b4', got {config['model'].get('backbone')}"
            
        # Verify the file contains the default config
        with open(handler.file_path, 'r') as f:
            import yaml
            config = yaml.safe_load(f)
            assert 'model' in config, "Config file missing 'model' key"
            assert config['model']['backbone'] == 'efficientnet_b4', \
                f"Expected default backbone 'efficientnet_b4', got {config['model'].get('backbone')}"
    
    def test_save_and_load_config(self, temp_config_dir, mock_logger):
        """Test saving and loading configuration."""
        handler = BackboneConfigHandler(mock_logger)
        test_config = {
            'model': {
                'backbone': 'test_backbone',
                'model_name': 'test_model'
            }
        }
        
        # Update config and save
        handler.config = test_config
        result = handler.save()
        assert result is True, "Save operation failed"
        
        # Clear and reload config
        handler.config = {}
        loaded = handler.load()
        assert loaded is not None, "Loaded config is None"
        assert 'model' in loaded, "'model' key not in loaded config"
        assert loaded['model']['backbone'] == 'test_backbone', \
            f"Expected 'test_backbone', got {loaded['model'].get('backbone')}"
        
        # Verify success message
        expected_path = os.path.abspath(os.path.join('config', 'model_config.yaml'))
        mock_logger.success.assert_called_with(
            f"✅ Configuration saved to {expected_path}"
        )
    
    def test_validate_config(self, mock_logger):
        """Test configuration validation."""
        handler = BackboneConfigHandler(mock_logger)
        
        # Test valid config
        valid_config = {
            'model': {
                'backbone': 'efficientnet_b4',
                'detection_layers': ['banknote'],
                'layer_mode': 'single'
            }
        }
        is_valid, message = handler.validate_config(valid_config)
        assert is_valid is True
        
        # Test invalid config
        invalid_config = {'model': {}}
        is_valid, message = handler.validate_config(invalid_config)
        assert is_valid is False
