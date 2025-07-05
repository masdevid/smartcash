"""
Tests for InfoBox component.

This module contains unit tests for the InfoBox class which displays
informational messages in a styled box with different alert levels.
"""
import pytest
from unittest.mock import MagicMock, patch
from ipywidgets import HTML, VBox, HBox
from smartcash.ui.components.info.info_component import InfoBox, ALERT_STYLES

class TestInfoBox:
    """Test suite for InfoBox component."""
    
    @pytest.fixture
    def info_box(self):
        """Create an InfoBox instance for testing."""
        return InfoBox(
            content="Test content",
            style="info",
            title="Test Title",
            padding=10,
            border_radius=5
        )
    
    def test_initialization(self, info_box):
        """Test that InfoBox initializes with provided values."""
        assert info_box.content == "Test content"
        assert info_box.style == "info"
        assert info_box.title == "Test Title"
        assert info_box.padding == 10
        assert info_box.border_radius == 5
        assert hasattr(info_box, 'widget')
        assert hasattr(info_box, '_initialized')
    
    def test_create_ui_components(self, info_box):
        """Test creation of UI components."""
        info_box._create_ui_components()
        assert hasattr(info_box, 'content_widget')
        assert hasattr(info_box, 'widget')
    
    def test_update_content(self, info_box):
        """Test updating the content of the info box."""
        new_content = "Updated content"
        info_box.update_content(new_content)
        assert info_box.content == new_content
        assert info_box.content_widget.value == new_content
    
    def test_update_style(self, info_box):
        """Test updating the style of the info box."""
        new_style = "warning"
        info_box.update_style(new_style)
        assert info_box.style == new_style
        assert info_box.widget.layout.border in ALERT_STYLES[new_style]["border"]
    
    def test_show_hide(self, info_box):
        """Test showing and hiding the info box."""
        # Initially visible by default
        assert info_box.widget.layout.visibility == 'visible'
        
        # Hide and verify
        info_box.hide()
        assert info_box.widget.layout.visibility == 'hidden'
        
        # Show and verify
        info_box.show()
        assert info_box.widget.layout.visibility == 'visible'
    
    def test_different_styles(self):
        """Test initialization with different style options."""
        styles = ["info", "success", "warning", "error"]
        
        for style in styles:
            box = InfoBox("Test", style=style)
            assert box.style == style
            assert hasattr(box, 'widget')
    
    def test_without_title(self):
        """Test initialization without a title."""
        box = InfoBox("Content only")
        assert box.title is None
        assert hasattr(box, 'widget')
