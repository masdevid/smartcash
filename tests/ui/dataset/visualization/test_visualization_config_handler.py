"""
File: tests/ui/dataset/visualization/test_visualization_config_handler.py
Description: Tests for the VisualizationConfigHandler class.
"""

import pytest
from unittest.mock import MagicMock, patch

from smartcash.ui.dataset.visualization.configs.visualization_config_handler import (
    VisualizationConfigHandler
)
from smartcash.ui.dataset.visualization.configs.visualization_defaults import (
    VisualizationDefaults
)

class TestVisualizationConfigHandler:
    """Test cases for VisualizationConfigHandler."""
    
    @pytest.fixture
    def default_config(self):
        """Return default configuration as a dictionary."""
        return {
            'SPLITS': ['train', 'valid', 'test'],
            'COLORS': {'train': '#4CAF50', 'valid': '#2196F3', 'test': '#FF9800'},
            'REFRESH_INTERVAL': 0,
            'PERCENTAGE_FORMAT': '{:.1f}%',
            'SHOW_LOG_ACCORDION': True,
            'SHOW_REFRESH_BUTTON': True,
            'DEFAULT_DATASET_PATH': None
        }
    
    @pytest.fixture
    def handler(self, default_config):
        """Create a VisualizationConfigHandler instance for testing."""
        with patch('smartcash.ui.dataset.visualization.configs.visualization_config_handler.get_module_logger') as mock_logger:
            mock_logger.return_value = MagicMock()
            return VisualizationConfigHandler()
    
    def test_initialization(self, handler, default_config):
        """Test that handler initializes with default config."""
        assert handler is not None
        assert handler.get_config() == default_config
    
    def test_update_config_valid(self, handler):
        """Test updating configuration with valid values."""
        new_config = {
            'REFRESH_INTERVAL': 5,
            'SHOW_LOG_ACCORDION': False
        }
        
        result = handler.update_config(new_config)
        assert result is True
        assert handler['REFRESH_INTERVAL'] == 5
        assert handler['SHOW_LOG_ACCORDION'] is False
    
    def test_update_config_invalid_type(self, handler):
        """Test updating configuration with invalid types."""
        # Invalid REFRESH_INTERVAL type
        with pytest.raises(ValueError, match="REFRESH_INTERVAL must be a non-negative number"):
            handler.update_config({'REFRESH_INTERVAL': -1})
            
        # Invalid SPLITS type
        with pytest.raises(ValueError, match="SPLITS must be a list"):
            handler.update_config({'SPLITS': 'not-a-list'})
            
        # Invalid COLORS type
        with pytest.raises(ValueError, match="COLORS must be a dictionary"):
            handler.update_config({'COLORS': 'not-a-dict'})
    
    def test_reset_to_defaults(self, handler, default_config):
        """Test resetting configuration to default values."""
        # Modify some values
        handler['REFRESH_INTERVAL'] = 10
        handler['SHOW_LOG_ACCORDION'] = False
        
        # Reset to defaults
        handler.reset_to_defaults()
        
        # Verify reset
        assert handler.get_config() == default_config
    
    def test_ui_integration(self, handler):
        """Test UI integration methods."""
        # Mock UI components
        mock_ui = {
            'splits_dropdown': MagicMock(value=['train', 'test']),
            'colors_picker': {
                'train': MagicMock(value='#4CAF50'),
                'test': MagicMock(value='#FF9800')
            },
            'refresh_interval_slider': MagicMock(value=5)
        }
        
        # Test update_from_ui
        handler.update_from_ui(mock_ui)
        assert handler['SPLITS'] == ['train', 'test']
        assert handler['COLORS'] == {'train': '#4CAF50', 'test': '#FF9800'}
        assert handler['REFRESH_INTERVAL'] == 5
        
        # Test update_ui_from_config
        new_config = {
            'SPLITS': ['valid', 'test'],
            'COLORS': {'valid': '#2196F3', 'test': '#FF9800'},
            'REFRESH_INTERVAL': 10
        }
        
        handler.update_ui_from_config(mock_ui, new_config)
        
        # Verify UI was updated
        mock_ui['splits_dropdown'].value = ['valid', 'test']
        mock_ui['colors_picker']['valid'].value = '#2196F3'
        mock_ui['colors_picker']['test'].value = '#FF9800'
        mock_ui['refresh_interval_slider'].value = 10
    
    def test_serialization(self, handler):
        """Test serialization to and from dictionary."""
        # Get current config
        config = handler.to_dict()
        
        # Create new handler from serialized config
        new_handler = VisualizationConfigHandler.from_dict(config)
        
        # Verify configs match
        assert new_handler.to_dict() == config
        
        # Verify it's a deep copy
        new_handler['REFRESH_INTERVAL'] = 10
        assert handler['REFRESH_INTERVAL'] != 10
