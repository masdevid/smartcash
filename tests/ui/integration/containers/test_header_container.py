"""Integration tests for HeaderContainer component.

Tests the functionality and integration of the HeaderContainer component with
its child components and dependencies.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

# Import the container we're testing
from smartcash.ui.components.header_container import HeaderContainer

# Fixtures
@pytest.fixture
def header_container():
    """Create a HeaderContainer instance for testing."""
    return HeaderContainer(
        title="Test Title",
        subtitle="Test Subtitle",
        icon="🔧",
        status_message="Initializing...",
        status_type="info"
    )

class TestHeaderContainer:
    """Test suite for HeaderContainer integration."""
    
    def test_initialization(self, header_container):
        """Test basic initialization with parameters."""
        assert header_container is not None
        assert header_container.title == "Test Title"
        assert header_container.subtitle == "Test Subtitle"
        assert header_container.icon == "🔧"
        assert header_container.status_message == "Initializing..."
        assert header_container.status_type == "info"
        assert header_container.show_status_panel is True
    
    def test_ui_components_creation(self, header_container):
        """Test that all UI components are created correctly."""
        # Check main container
        assert hasattr(header_container, 'container')
        assert header_container.container is not None
        
        # Check header elements
        assert hasattr(header_container, 'title_widget')
        assert hasattr(header_container, 'subtitle_widget')
        assert hasattr(header_container, 'status_panel')
        
        # Check initial visibility
        assert header_container.status_panel.layout.visibility == 'visible'
    
    def test_status_updates(self, header_container):
        """Test status update functionality."""
        # Test status update
        new_status = "Operation completed successfully!"
        header_container.update_status(new_status, "success")
        
        assert header_container.status_message == new_status
        assert header_container.status_type == "success"
        
        # Test status panel visibility
        header_container.toggle_status_panel(False)
        assert header_container.status_panel.layout.visibility == 'hidden'
        
        header_container.toggle_status_panel(True)
        assert header_container.status_panel.layout.visibility == 'visible'
    
    def test_title_updates(self, header_container):
        """Test title and subtitle updates."""
        # Test title update
        new_title = "Updated Title"
        header_container.update_title(new_title)
        assert header_container.title == new_title
        
        # Test subtitle update
        new_subtitle = "Updated Subtitle"
        header_container.update_subtitle(new_subtitle)
        assert header_container.subtitle == new_subtitle
        
        # Test icon update
        new_icon = "🚀"
        header_container.update_icon(new_icon)
        assert header_container.icon == new_icon
    
    def test_style_updates(self, header_container):
        """Test style updates."""
        # Test updating styles
        new_style = {
            'margin_bottom': '24px',
            'padding_bottom': '16px',
            'border_bottom': '2px solid #ccc'
        }
        
        header_container.update_style(**new_style)
        
        # Verify styles were updated
        assert header_container.style['margin_bottom'] == '24px'
        assert header_container.style['padding_bottom'] == '16px'
        assert header_container.style['border_bottom'] == '2px solid #ccc'
        
        # Verify styles are applied to container
        container_style = header_container.container.layout
        assert container_style.margin == '0 0 24px 0'
        assert container_style.padding == '0 0 16px 0'
        assert container_style.border_bottom == '2px solid #ccc'
    
    def test_error_handling(self, header_container):
        """Test error handling for invalid inputs."""
        # Test invalid status type
        with pytest.raises(ValueError):
            header_container.update_status("Test", "invalid_type")
        
        # Test valid status types
        valid_types = ["info", "success", "warning", "error"]
        for status_type in valid_types:
            header_container.update_status("Test", status_type)  # Should not raise
    
    def test_initialization_without_optional_params(self):
        """Test initialization without optional parameters."""
        container = HeaderContainer(title="Minimal Header")
        
        assert container is not None
        assert container.title == "Minimal Header"
        assert container.subtitle == ""
        assert container.icon == ""
        assert container.status_message == ""
        assert container.status_type == "info"
        assert container.show_status_panel is True
    
    def test_custom_status_indicators(self):
        """Test custom status indicators."""
        container = HeaderContainer(
            title="Custom Status",
            status_message="Custom Status",
            status_type="custom",
            custom_status_indicators={
                "custom": {"icon": "⚙️", "color": "#800080"}
            }
        )
        
        # Test custom status type
        container.update_status("Custom status working", "custom")
        assert container.status_type == "custom"
        
        # Test reverting to default status type
        container.update_status("Back to info", "info")
        assert container.status_type == "info"
