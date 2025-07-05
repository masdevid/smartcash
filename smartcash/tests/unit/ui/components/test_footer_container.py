"""Tests for FooterContainer log display functionality."""

import pytest
import ipywidgets as widgets
from unittest.mock import MagicMock, patch, PropertyMock

from smartcash.ui.components.footer_container import FooterContainer, PanelConfig, PanelType
from smartcash.tests.test_helpers import (
    assert_has_trait,
    assert_widget_visible,
    assert_has_class,
    assert_widget_children,
    mock_widget,
    mock_vbox,
    mock_button,
    mock_text,
    MockWidget,
    MockButton,
    patch_display,
    patch_widget,
    mock_ui_components,
    create_mock_widget
)

# Create a mock InfoAccordion class that's a proper widget
class MockInfoAccordion(widgets.VBox):
    """Mock InfoAccordion widget for testing."""
    def __init__(self, title="", content="", style="info", open_by_default=False, **kwargs):
        super().__init__(**kwargs)
        self.title = title
        self.content = content
        self.style = style
        self.open_by_default = open_by_default
        self.show = MagicMock(return_value=self)
        
        # Create a simple layout
        self.children = [widgets.HTML(value=content)]
        
    def __call__(self, *args, **kwargs):
        # Allow the mock to be called as a function
        return self


class TestFooterContainer:
    """Test cases for FooterContainer component."""
    
    @pytest.fixture
    def footer_container(self, monkeypatch):
        """Create a FooterContainer instance with a log panel for testing."""
        # Create a mock InfoAccordion
        mock_accordion = MockInfoAccordion()
        
        # Patch the InfoAccordion class to return our mock
        monkeypatch.setattr(
            'smartcash.ui.components.footer_container.InfoAccordion',
            MockInfoAccordion
        )
        
        # Create a log panel configuration
        log_panel = PanelConfig(
            panel_type=PanelType.INFO_ACCORDION,
            title="Logs",
            content="",
            style="info",
            flex="1",
            min_width="300px",
            open_by_default=True,
            panel_id="logs"
        )
        
        # Create footer container with just the log panel
        container = FooterContainer(panels=[log_panel])
        
        # Store the mock accordion on the container for test access
        container._test_mocks = {
            'accordion': mock_accordion
        }
        
        return container
    
    def test_initialization(self, footer_container):
        """Test that the footer container initializes correctly with a log panel."""
        # Verify the container is created
        assert footer_container.container is not None
        
        # Verify the log panel was added
        panel_id = next(iter(footer_container._panels))
        panel = footer_container._panels[panel_id]
        
        # Verify the panel has a widget and config
        assert 'widget' in panel
        assert 'config' in panel
        assert isinstance(panel['widget'], MockInfoAccordion)
    
    def test_panel_initialization(self, footer_container):
        """Test that panels are properly initialized in the footer."""
        # Verify the container is created
        assert footer_container.container is not None
        
        # Check that we have at least one panel
        assert len(footer_container._panels) > 0
        
        # Get the first panel
        panel_id = next(iter(footer_container._panels))
        panel = footer_container._panels[panel_id]
        
        # Verify the panel has a widget and config
        assert 'widget' in panel
        assert 'config' in panel
        
        # Verify the widget is an instance of MockInfoAccordion
        assert isinstance(panel['widget'], MockInfoAccordion)
    
    def test_multiple_panels(self, monkeypatch):
        """Test creating a footer container with multiple panels."""
        # Patch the InfoAccordion class to return our mock
        monkeypatch.setattr(
            'smartcash.ui.components.footer_container.InfoAccordion',
            MockInfoAccordion
        )
        
        # Create panel configurations
        panels = [
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Panel 1",
                content="Content 1",
                panel_id="panel1"
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Panel 2",
                content="Content 2",
                panel_id="panel2"
            )
        ]
        
        # Create footer container with multiple panels
        container = FooterContainer(panels=panels)
        
        # Verify both panels were added
        assert len(container._panels) == 2
        assert 'panel1' in container._panels
        assert 'panel2' in container._panels
        
        # Verify container has children
        assert container.container is not None
        assert len(container.container.children) > 0
    
    def test_panel_operations(self, footer_container):
        """Test various panel operations."""
        # Get the first panel
        panel_id = next(iter(footer_container._panels))
        panel = footer_container._panels[panel_id]
        
        # Test updating panel title
        new_title = "New Panel Title"
        footer_container.update_panel(panel_id, title=new_title)
        assert panel['config'].title == new_title
        assert panel['widget'].title == new_title
        
        # Test updating panel style
        new_style = "warning"
        footer_container.update_panel(panel_id, style=new_style)
        assert panel['config'].style == new_style
        assert panel['widget'].style == new_style
        
        # Test updating panel layout
        new_flex = "2"
        new_min_width = "400px"
        footer_container.update_panel(panel_id, flex=new_flex, min_width=new_min_width)
        assert panel['widget'].layout.flex == new_flex
        assert panel['widget'].layout.min_width == new_min_width
    
    def test_panel_removal(self, footer_container):
        """Test removing a panel from the footer."""
        # Get the first panel ID
        panel_id = next(iter(footer_container._panels))
        
        # Remove the panel
        footer_container.remove_panel(panel_id)
        
        # Verify the panel was removed
        assert panel_id not in footer_container._panels
        
        # Verify the container was updated
        assert footer_container.container is not None

    def test_container_initialization(self, footer_container):
        """Test that the container is properly initialized with layout."""
        # Check container properties
        assert hasattr(footer_container, 'container')
        assert footer_container.container is not None
        
        # Check layout properties
        assert 'display' in footer_container.layout_config
        assert 'flex_flow' in footer_container.layout_config
        assert 'width' in footer_container.layout_config

    def test_panel_content_update(self, footer_container):
        """Test updating panel content."""
        # Get the first panel
        panel_id = next(iter(footer_container._panels))
        
        # Update the panel content
        new_content = "Updated panel content"
        footer_container.update_panel(panel_id, content=new_content)
        
        # Get the updated panel
        panel = footer_container._panels[panel_id]
        
        # Verify the content was updated
        assert panel['config'].content == new_content
        assert panel['widget'].content == new_content

    # Test case for verifying log entry content and count

