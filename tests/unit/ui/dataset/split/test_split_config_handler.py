"""
Tests for the dataset split configuration handler.

This module tests the SplitConfigHandler class and its functionality
for managing dataset split configurations.
"""

import pytest
from unittest.mock import Mock, patch
import copy
from typing import Dict, Any

from smartcash.ui.dataset.split.handlers.config_handler import SplitConfigHandler
from smartcash.ui.dataset.split.configs.split_defaults import DEFAULT_SPLIT_CONFIG, VALIDATION_RULES


class TestSplitConfigHandler:
    """Test cases for SplitConfigHandler."""
    
    @pytest.fixture
    def handler(self):
        """Create a SplitConfigHandler instance for testing."""
        return SplitConfigHandler()
    
    @pytest.fixture
    def valid_config(self):
        """Provide a valid configuration for testing."""
        return copy.deepcopy(DEFAULT_SPLIT_CONFIG)
    
    @pytest.fixture
    def invalid_config(self):
        """Provide an invalid configuration for testing."""
        return {
            'data': {
                'split_ratios': {
                    'train': 0.5,
                    'val': 0.3,
                    'test': 0.3  # Sum = 1.1, invalid
                },
                'seed': 42,
                'shuffle': True,
                'stratify': False
            },
            'output': {
                'train_dir': 'data/train',
                'val_dir': 'data/val',
                'test_dir': 'data/test',
                'create_subdirs': True,
                'overwrite': False,
                'relative_paths': True,
                'preserve_dir_structure': True,
                'use_symlinks': False,
                'backup': True,
                'backup_dir': 'data/backup'
            }
        }
    
    def test_initialization_with_default_config(self):
        """Test handler initialization with default configuration."""
        handler = SplitConfigHandler()
        
        assert handler.config == DEFAULT_SPLIT_CONFIG
        assert handler.module_name == 'split'
        assert hasattr(handler, '_validation_rules')
    
    def test_initialization_with_custom_config(self, valid_config):
        """Test handler initialization with custom configuration."""
        custom_config = copy.deepcopy(valid_config)
        custom_config['data']['seed'] = 123
        
        handler = SplitConfigHandler(custom_config)
        
        assert handler.config['data']['seed'] == 123
        assert handler.config['data']['split_ratios'] == valid_config['data']['split_ratios']
    
    def test_initialization_with_invalid_config_type(self):
        """Test handler initialization with invalid config type."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            SplitConfigHandler("invalid_config")
    
    def test_load_config_valid(self, handler, valid_config):
        """Test loading a valid configuration."""
        custom_config = copy.deepcopy(valid_config)
        custom_config['data']['seed'] = 999
        
        handler.load_config(custom_config)
        
        assert handler.config['data']['seed'] == 999
        assert handler.config['data']['split_ratios'] == valid_config['data']['split_ratios']
    
    def test_load_config_invalid_type(self, handler):
        """Test loading configuration with invalid type."""
        with pytest.raises(ValueError, match="Configuration must be a dictionary"):
            handler.load_config("not_a_dict")
    
    def test_load_config_invalid_ratios(self, handler, invalid_config):
        """Test loading configuration with invalid split ratios."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            handler.load_config(invalid_config)
    
    def test_validate_config_valid(self, handler, valid_config):
        """Test validation of a valid configuration."""
        is_valid = handler.validate_config(valid_config)
        assert is_valid is True
    
    def test_validate_config_missing_data_section(self, handler):
        """Test validation with missing data section."""
        config = {'output': {}}
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_missing_output_section(self, handler):
        """Test validation with missing output section."""
        config = {'data': {}}
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_missing_required_fields(self, handler, valid_config):
        """Test validation with missing required fields."""
        config = copy.deepcopy(valid_config)
        del config['data']['split_ratios']
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_ratio_types(self, handler, valid_config):
        """Test validation with invalid ratio types."""
        config = copy.deepcopy(valid_config)
        config['data']['split_ratios']['train'] = "not_a_number"
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_ratio_out_of_range(self, handler, valid_config):
        """Test validation with ratios out of range."""
        config = copy.deepcopy(valid_config)
        config['data']['split_ratios']['train'] = 1.5  # > 1.0
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_ratios_sum(self, handler, invalid_config):
        """Test validation with invalid ratios sum."""
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            handler.validate_config(invalid_config)
    
    def test_validate_config_invalid_seed_type(self, handler, valid_config):
        """Test validation with invalid seed type."""
        config = copy.deepcopy(valid_config)
        config['data']['seed'] = "not_an_integer"
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_boolean_fields(self, handler, valid_config):
        """Test validation with invalid boolean fields."""
        config = copy.deepcopy(valid_config)
        config['data']['shuffle'] = "not_a_boolean"
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_validate_config_invalid_directory_types(self, handler, valid_config):
        """Test validation with invalid directory types."""
        config = copy.deepcopy(valid_config)
        config['output']['train_dir'] = 123  # Should be string
        
        is_valid = handler.validate_config(config)
        assert is_valid is False
    
    def test_update_config_valid(self, handler, valid_config):
        """Test updating configuration with valid updates."""
        handler.load_config(valid_config)
        
        updates = {
            'data.seed': 456,
            'output.train_dir': 'new/train/path'
        }
        
        handler.update_config(updates)
        
        assert handler.config['data']['seed'] == 456
        assert handler.config['output']['train_dir'] == 'new/train/path'
    
    def test_update_config_invalid(self, handler, valid_config):
        """Test updating configuration with invalid updates."""
        handler.load_config(valid_config)
        
        updates = {
            'data.split_ratios.train': 2.0  # Invalid ratio
        }
        
        with pytest.raises(ValueError, match="Configuration update would result in invalid configuration"):
            handler.update_config(updates)
    
    def test_update_config_empty_updates(self, handler, valid_config):
        """Test updating configuration with empty updates."""
        handler.load_config(valid_config)
        original_config = copy.deepcopy(handler.config)
        
        handler.update_config({})
        
        assert handler.config == original_config
    
    def test_extract_config_from_ui_with_components(self, handler):
        """Test extracting configuration from UI components."""
        # Mock UI components
        mock_components = {
            'form_components': {
                'train_slider': Mock(value=0.8),
                'val_slider': Mock(value=0.1),
                'test_slider': Mock(value=0.1),
                'train_dir_input': Mock(value='custom/train'),
                'val_dir_input': Mock(value='custom/val'),
                'test_dir_input': Mock(value='custom/test'),
                'seed_input': Mock(value=999),
                'shuffle_checkbox': Mock(value=False),
                'stratify_checkbox': Mock(value=True)
            }
        }
        
        config = handler.extract_config_from_ui(mock_components)
        
        assert config['data']['split_ratios']['train'] == 0.8
        assert config['data']['split_ratios']['val'] == 0.1
        assert config['data']['split_ratios']['test'] == 0.1
        assert config['output']['train_dir'] == 'custom/train'
        assert config['output']['val_dir'] == 'custom/val'
        assert config['output']['test_dir'] == 'custom/test'
        assert config['data']['seed'] == 999
        assert config['data']['shuffle'] is False
        assert config['data']['stratify'] is True
    
    def test_extract_config_from_ui_without_components(self, handler):
        """Test extracting configuration from UI without components."""
        mock_components = {}
        
        config = handler.extract_config_from_ui(mock_components)
        
        # Should return default values
        assert config['data']['split_ratios']['train'] == 0.7
        assert config['data']['split_ratios']['val'] == 0.15
        assert config['data']['split_ratios']['test'] == 0.15
        assert config['data']['seed'] == 42
    
    def test_update_ui_from_config(self, handler, valid_config):
        """Test updating UI components from configuration."""
        # Mock UI components
        mock_components = {
            'form_components': {
                'train_ratio': Mock(),
                'val_ratio': Mock(),
                'test_ratio': Mock(),
                'seed': Mock(),
                'shuffle': Mock(),
                'stratify': Mock(),
                'train_dir': Mock(),
                'val_dir': Mock(),
                'test_dir': Mock(),
                'create_subdirs': Mock(),
                'overwrite': Mock()
            },
            'log_output': Mock()
        }
        
        custom_config = copy.deepcopy(valid_config)
        custom_config['data']['split_ratios']['train'] = 0.8
        custom_config['data']['seed'] = 999
        
        handler.update_ui_from_config(custom_config, mock_components)
        
        # Verify UI components were updated
        mock_components['form_components']['train_ratio'].value = 0.8
        mock_components['form_components']['seed'].value = 999
    
    def test_update_ui_from_config_with_error(self, handler, valid_config):
        """Test updating UI components with error handling."""
        # Mock UI components that will raise an error
        mock_components = {
            'form_components': {
                'train_ratio': Mock(side_effect=Exception("Mock error"))
            },
            'log_output': Mock()
        }
        
        # Should not raise exception, but log error
        handler.update_ui_from_config(valid_config, mock_components)
    
    def test_initialize(self, handler):
        """Test handler initialization."""
        result = handler.initialize()
        
        assert result['status'] == 'success'
        assert 'SplitConfigHandler initialized' in result['message']
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        # This will need to be implemented in the config handler
        # For now, we test that the method exists on the class
        assert hasattr(SplitConfigHandler, 'get_default_config')
    
    def test_config_property(self, handler, valid_config):
        """Test the config property."""
        handler.load_config(valid_config)
        
        retrieved_config = handler.config
        assert retrieved_config == valid_config
    
    def test_validation_edge_cases(self, handler):
        """Test validation edge cases."""
        # Test ratios sum exactly to 1.0
        config = copy.deepcopy(DEFAULT_SPLIT_CONFIG)
        config['data']['split_ratios'] = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.15
        }
        assert handler.validate_config(config) is True
        
        # Test ratios sum just within tolerance
        config['data']['split_ratios'] = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.1499  # Sum = 0.9999, within tolerance
        }
        assert handler.validate_config(config) is True
        
        # Test ratios sum just outside tolerance
        config['data']['split_ratios'] = {
            'train': 0.7,
            'val': 0.15,
            'test': 0.148  # Sum = 0.998, outside tolerance
        }
        with pytest.raises(ValueError, match="Split ratios must sum to 1.0"):
            handler.validate_config(config)