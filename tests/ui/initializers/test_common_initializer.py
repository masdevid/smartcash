"""
Tests for the CommonInitializer class to ensure it returns display widgets.
"""
from typing import Dict, Any
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.initializers.common_initializer import CommonInitializer
from smartcash.ui.handlers.config_handlers import ConfigHandler

class TestCommonInitializer(CommonInitializer):
    """Test implementation of CommonInitializer for testing purposes."""
    
    def _create_ui_components(self, config: Dict[str, Any] = None, **kwargs) -> dict:
        """Create test UI components.
        
        Args:
            config: Configuration dictionary
            **kwargs: Additional keyword arguments
            
        Returns:
            Dictionary of UI components
        """
        return {
            'container': widgets.VBox(),
            'button': widgets.Button(description='Test Button')
        }
    
    def _get_default_config(self) -> dict:
        """Return default config for testing."""
        return {'test': 'config'}

class TestCommonInitializerDisplay:
    """Test that CommonInitializer returns display widgets."""
    
    def test_initialize_returns_widgets(self):
        """Test that initialize() returns widgets instead of dict."""
        # Arrange
        initializer = TestCommonInitializer('test_module')
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), \
            "Should return a widget instance"
    
    def test_initialize_with_error_returns_error_widget(self):
        """Test that initialize() returns an error widget on failure."""
        # Arrange
        initializer = TestCommonInitializer('test_module')
        initializer._create_ui_components = MagicMock(side_effect=Exception("Test error"))
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), \
            "Should return a widget even on error"
        assert 'Error' in str(result), \
            "Error widget should contain error information"
    
    def test_initialize_with_config_handler(self):
        """Test initialization with a config handler."""
        # Arrange
        mock_handler = MagicMock(spec=ConfigHandler)
        initializer = TestCommonInitializer('test_module', config_handler_class=MagicMock(return_value=mock_handler))
        
        # Act
        result = initializer.initialize()
        
        # Assert
        assert isinstance(result, widgets.Widget), \
            "Should return a widget with config handler"
        assert hasattr(initializer, 'config_handler'), \
            "Config handler should be initialized"
