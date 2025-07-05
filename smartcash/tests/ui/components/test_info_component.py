"""
Tests for InfoBox component.

This module contains unit tests for the InfoBox class which displays
informational messages in a styled box with different alert levels.
"""
import pytest
from ipywidgets import HTML

from smartcash.ui.components.info.info_component import InfoBox, ALERT_STYLES
from smartcash.tests.test_helpers import assert_has_trait, assert_widget_visible

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
        # Test basic properties
        assert info_box.content == "Test content"
        assert info_box.style == "info"
        assert info_box.title == "Test Title"
        assert info_box.padding == 10
        assert info_box.border_radius == 5
        
        # Test widget initialization
        assert hasattr(info_box, 'widget')
        assert hasattr(info_box, '_initialized')
        assert info_box._initialized is True
        
        # Test widget properties
        assert_has_trait(info_box.widget, 'children')
        assert_widget_visible(info_box.widget)
    
    def test_create_ui_components(self, info_box):
        """Test creation of UI components."""
        # Test component creation through public interface
        info_box.update_content("New test content")
        assert hasattr(info_box, 'content_widget')
        assert hasattr(info_box, 'widget')
        assert isinstance(info_box.content_widget, HTML)
        assert info_box.content_widget.value == "New test content"
        
        # Test widget structure
        assert_has_trait(info_box.widget, 'children')
        assert len(info_box.widget.children) > 0
    
    def test_update_content(self, info_box):
        """Test updating the content of the info box."""
        # Initial content check
        assert info_box.content == "Test content"
        
        # Update content
        new_content = "Updated content with <b>HTML</b>"
        info_box.update_content(new_content)
        
        # Verify updates
        assert info_box.content == new_content
        assert info_box.content_widget.value == new_content
        
        # Verify widget is updated
        assert info_box.content in str(info_box.widget.layout)
    
    def test_update_style(self, info_box):
        """Test updating the style of the info box."""
        # Test all available styles
        for style in ALERT_STYLES:
            info_box.update_style(style)
            assert info_box.style == style
            assert style in info_box.widget.layout.border
            assert style in info_box.content_widget.style.font_weight
            
        # Test style update with a specific style
        new_style = "warning"
        info_box.update_style(new_style)
        assert info_box.style == new_style
        assert info_box.widget.layout.border.startswith(ALERT_STYLES[new_style]["border"])
    
    def test_show_hide(self, info_box):
        """Test showing and hiding the info box."""
        # Test show
        info_box.show()
        assert info_box.visible is True
        assert info_box.widget.layout.visibility == 'visible'
        assert info_box.widget.layout.display == 'flex'
        
        # Test hide
        info_box.hide()
        assert info_box.visible is False
        assert info_box.widget.layout.visibility == 'hidden'
        assert info_box.widget.layout.display == 'none'
        
        # Test show again
        info_box.show()
        assert info_box.visible is True
        assert info_box.widget.layout.visibility == 'visible'
        assert info_box.widget.layout.display == 'flex'
    
    def test_toggle_visibility(self, info_box):
        """Test toggling the visibility of the info box."""
        # Get initial state
        initial_state = info_box.visible
        
        # Toggle and verify changed
        info_box.toggle_visibility()
        assert info_box.visible is not initial_state
        assert info_box.widget.layout.visibility == ('visible' if not initial_state else 'hidden')
        
        # Toggle back and verify original state
        info_box.toggle_visibility()
        assert info_box.visible is initial_state
        assert info_box.widget.layout.visibility == ('visible' if initial_state else 'hidden')
    
    def test_set_title(self, info_box):
        """Test setting the title of the info box."""
        # Initial title check
        assert info_box.title == "Test Title"
        
        # Set new title
        new_title = "New Test Title with Special Chars: 123!@#"
        info_box.set_title(new_title)
        
        # Verify updates
        assert info_box.title == new_title
        assert new_title in str(info_box.widget.layout)
    
    def test_alert_styles(self):
        """Test all available alert styles."""
        for style, style_props in ALERT_STYLES.items():
            # Create a new instance for each style
            box = InfoBox(
                content=f"Content for {style}",
                style=style,
                title=f"{style.capitalize()} Box"
            )
            
            # Verify style properties
            assert box.style == style
            assert style in box.widget.layout.border
            assert style in box.content_widget.style.font_weight
            
            # Verify style-specific properties
            if 'background' in style_props:
                assert style_props['background'] in box.widget.layout.background
            if 'color' in style_props:
                assert style_props['color'] in box.content_widget.style.font_weight
    
    def test_invalid_style(self, info_box):
        """Test behavior with invalid style."""
        # Should default to 'info' for invalid styles
        info_box.update_style("invalid_style")
        assert info_box.style == 'info'
        
        # Verify widget still works with default style
        assert 'info' in info_box.widget.layout.border
    
    def test_different_styles(self):
        """Test initialization with different style options."""
        styles = ["info", "success", "warning", "danger"]
        for style in styles:
            box = InfoBox("Test", style=style)
            assert box.style == style
            assert box.widget.layout.border.startswith(style)
    
    def test_edge_cases(self):
        """Test edge cases for InfoBox initialization."""
        # Empty content
        box = InfoBox("")
        assert box.content == ""
        assert hasattr(box, 'widget')
        
        # No title
        box = InfoBox("Content only")
        assert box.title == ""
        assert hasattr(box, 'widget')
        
        # Invalid style (should default to 'info')
        box = InfoBox("Test", style="invalid")
        assert box.style == "info"
        assert hasattr(box, 'widget')
        
        # Test with very long content
        long_content = "This is a very long test content " * 50
        box = InfoBox(long_content)
        assert box.content == long_content
        assert hasattr(box, 'content_widget')
        assert long_content in box.content_widget.value
