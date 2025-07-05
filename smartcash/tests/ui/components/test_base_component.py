"""
Tests for the BaseUIComponent class.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.components.base_component import BaseUIComponent
from smartcash.ui.core.errors import UIComponentError, ErrorLevel

class TestBaseUIComponent:
    """Test cases for BaseUIComponent class."""
    
    @pytest.fixture
    def mock_component(self):
        """Fixture that creates a testable component instance."""
        class TestComponent(BaseUIComponent):
            def _create_ui_components(self):
                self._ui_components['container'] = widgets.VBox()
                
        return TestComponent("test_component")
    
    def test_initialization(self, mock_component):
        """Test that component initializes with correct attributes."""
        assert mock_component.component_name == "test_component"
        assert not mock_component._initialized
        assert isinstance(mock_component._error_handler, MagicMock)
    
    def test_initialize_success(self, mock_component):
        """Test successful initialization of the component."""
        mock_component.initialize()
        assert mock_component._initialized
        assert 'container' in mock_component._ui_components
    
    def test_show_success(self, mock_component):
        """Test successful display of the component."""
        widget = mock_component.show()
        assert isinstance(widget, widgets.VBox)
    
    def test_show_error_no_container(self):
        """Test error when container widget is missing."""
        class NoContainerComponent(BaseUIComponent):
            def _create_ui_components(self):
                pass  # Don't create a container
                
        component = NoContainerComponent("no_container")
        
        with pytest.raises(UIComponentError) as exc_info:
            component.show()
            
        assert "No container widget found" in str(exc_info.value)
    
    def test_get_component_success(self, mock_component):
        """Test getting a component by name."""
        mock_component.initialize()
        container = mock_component.get_component('container')
        assert isinstance(container, widgets.VBox)
    
    def test_get_component_not_found(self, mock_component):
        """Test getting a non-existent component returns None."""
        mock_component.initialize()
        assert mock_component.get_component('nonexistent') is None
    
    def test_dispose(self, mock_component):
        """Test that dispose cleans up resources."""
        mock_component.initialize()
        mock_component.dispose()
        
        assert not mock_component._initialized
        assert not mock_component._ui_components
    
    @patch('smartcash.ui.components.base_component.handle_errors')
    def test_initialize_error_handling(self, mock_handle_errors, mock_component):
        """Test that errors during initialization are properly handled."""
        # Simulate an error during _create_ui_components
        class ErrorComponent(BaseUIComponent):
            def _create_ui_components(self):
                raise ValueError("Test error")
                
        component = ErrorComponent("error_component")
        
        with pytest.raises(ValueError):
            component.initialize()
            
        # Verify error handler was called
        assert component._error_handler.handle_error.called
