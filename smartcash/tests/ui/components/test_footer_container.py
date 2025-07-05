"""
Tests for FooterContainer component.

This module contains unit tests for the FooterContainer class which manages
the footer section of the UI with multiple configurable panels.
"""
import pytest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from smartcash.ui.components.footer_container import (
    FooterContainer, 
    PanelConfig, 
    PanelType
)

class TestFooterContainer:
    """Test suite for FooterContainer component."""
    
    @pytest.fixture
    def sample_panels(self):
        """Create sample panel configurations for testing."""
        return [
            PanelConfig(
                panel_type=PanelType.INFO_BOX,
                title="Test Info",
                content="Test content",
                style="info",
                flex="1",
                min_width="200px"
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Test Accordion",
                content="Accordion content",
                style="warning",
                flex="2",
                min_width="300px",
                open_by_default=True
            )
        ]
    
    @pytest.fixture
    def footer_container(self, sample_panels):
        """Create a FooterContainer instance for testing."""
        return FooterContainer(panels=sample_panels)
    
    def test_initialization(self, footer_container, sample_panels):
        """Test that FooterContainer initializes with provided panels."""
        assert hasattr(footer_container, 'panels')
        assert hasattr(footer_container, 'container')
        assert len(footer_container.panels) == 2
        assert footer_container.container.children[0].value == "<h4>Test Info</h4>"
    
    def test_add_panel(self, footer_container):
        """Test adding a new panel to the footer."""
        initial_count = len(footer_container.panels)
        
        new_panel = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="New Panel",
            content="New content"
        )
        
        footer_container.add_panel(new_panel)
        
        assert len(footer_container.panels) == initial_count + 1
        assert footer_container.panels[-1].title == "New Panel"
    
    def test_remove_panel(self, footer_container):
        """Test removing a panel from the footer."""
        panel_id = footer_container.panels[0].panel_id
        initial_count = len(footer_container.panels)
        
        footer_container.remove_panel(panel_id)
        
        assert len(footer_container.panels) == initial_count - 1
        assert not any(p.panel_id == panel_id for p in footer_container.panels)
    
    def test_update_panel(self, footer_container):
        """Test updating an existing panel."""
        panel_id = footer_container.panels[0].panel_id
        updated_content = "Updated content"
        
        footer_container.update_panel(panel_id, content=updated_content)
        
        updated_panel = next(p for p in footer_container.panels if p.panel_id == panel_id)
        assert updated_panel.content == updated_content
    
    def test_clear_panels(self, footer_container):
        """Test clearing all panels from the footer."""
        footer_container.clear_panels()
        assert len(footer_container.panels) == 0
        assert len(footer_container.container.children) == 0
    
    @patch('ipywidgets.HTML')
    @patch('ipywidgets.VBox')
    def test_show_hide(self, mock_vbox, mock_html, footer_container):
        """Test showing and hiding the footer."""
        # Initially visible
        assert footer_container.container.layout.visibility == 'visible'
        
        # Hide and verify
        footer_container.hide()
        assert footer_container.container.layout.visibility == 'hidden'
        
        # Show and verify
        footer_container.show()
        assert footer_container.container.layout.visibility == 'visible'
