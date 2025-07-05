"""Integration tests for SummaryContainer component.

Tests the functionality and integration of the SummaryContainer component with
its child components and dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the container we're testing
from smartcash.ui.components.summary_container import SummaryContainer, create_summary_container

# Fixtures
@pytest.fixture
def summary_container():
    """Create a SummaryContainer instance for testing."""
    return SummaryContainer(
        component_name="test_summary",
        theme="info",
        title="Test Summary",
        icon="📊"
    )

class TestSummaryContainer:
    """Test suite for SummaryContainer integration."""
    
    def test_initialization(self, summary_container):
        """Test basic initialization with parameters."""
        assert summary_container is not None
        assert summary_container.component_name == "test_summary"
        assert summary_container._title == "Test Summary"
        assert summary_container._icon == "📊"
        assert summary_container._theme == "info"
    
    def test_ui_components_creation(self, summary_container):
        """Test that all UI components are created correctly."""
        # Check main container
        assert hasattr(summary_container, '_ui_components')
        assert 'container' in summary_container._ui_components
        assert 'content' in summary_container._ui_components
        
        # Check container widget
        container = summary_container._ui_components['container']
        assert isinstance(container, widgets.Box)
        
        # Check content widget
        content = summary_container._ui_components['content']
        assert isinstance(content, widgets.HTML)
    
    def test_content_updates(self, summary_container):
        """Test content update functionality."""
        # Test HTML content update
        test_html = "<p>Test HTML content</p>"
        summary_container.update_content(test_html)
        
        content = summary_container._ui_components['content']
        assert content.value == test_html
        
        # Test text content update
        test_text = "Test plain text content"
        summary_container.update_text(test_text)
        assert test_text in content.value
    
    def test_theme_changes(self, summary_container):
        """Test theme changes and their visual effects."""
        # Test theme change
        new_theme = "success"
        summary_container.set_theme(new_theme)
        
        assert summary_container._theme == new_theme
        
        # Verify theme styles are applied
        container = summary_container._ui_components['container']
        assert 'success' in container.layout.background
        
        # Test invalid theme
        with pytest.raises(ValueError):
            summary_container.set_theme("invalid_theme")
    
    def test_title_and_icon_updates(self, summary_container):
        """Test title and icon updates."""
        # Test title update
        new_title = "Updated Title"
        summary_container.set_title(new_title)
        assert summary_container._title == new_title
        
        # Test icon update
        new_icon = "⭐"
        summary_container.set_icon(new_icon)
        assert summary_container._icon == new_icon
        
        # Test title and icon together
        summary_container.set_title("Final Title", "🎯")
        assert summary_container._title == "Final Title"
        assert summary_container._icon == "🎯"
    
    def test_create_summary_container_function(self):
        """Test the create_summary_container convenience function."""
        container = create_summary_container(
            theme="warning",
            title="Function Test",
            icon="⚠️"
        )
        
        assert container is not None
        assert isinstance(container, SummaryContainer)
        assert container._theme == "warning"
        assert container._title == "Function Test"
        assert container._icon == "⚠️"
    
    def test_visibility_control(self, summary_container):
        """Test show/hide functionality."""
        container = summary_container._ui_components['container']
        
        # Initially visible
        assert container.layout.visibility == 'visible'
        
        # Hide
        summary_container.hide()
        assert container.layout.visibility == 'hidden'
        assert container.layout.display == 'none'
        
        # Show
        summary_container.show()
        assert container.layout.visibility == 'visible'
        assert container.layout.display == ''
    
    def test_error_handling(self, summary_container):
        """Test error handling for invalid inputs."""
        # Test invalid theme
        with pytest.raises(ValueError):
            summary_container.set_theme("invalid_theme")
        
        # Test empty content
        summary_container.update_content("")
        content = summary_container._ui_components['content']
        assert content.value == ""
    
    def test_custom_styling(self, summary_container):
        """Test custom styling options."""
        # Test custom styles
        custom_styles = {
            'border_radius': '10px',
            'box_shadow': '0 4px 8px rgba(0,0,0,0.2)',
            'padding': '20px'
        }
        
        summary_container.update_style(**custom_styles)
        
        # Verify styles were applied
        container = summary_container._ui_components['container']
        assert container.layout.border_radius == '10px'
        assert container.layout.box_shadow == '0 4px 8px rgba(0,0,0,0.2)'
        assert container.layout.padding == '20px'
