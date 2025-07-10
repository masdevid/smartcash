"""
Tests for footer_container.py module.

This module tests the flexible footer container that supports multiple
info panels (InfoBox or InfoAccordion) with configurable flex layout.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import ipywidgets as widgets
from typing import Dict, Any

# Import the components to test
from smartcash.ui.components.footer_container import (
    FooterContainer,
    PanelConfig,
    PanelType,
    create_footer_container
)


class TestPanelConfig:
    """Test cases for PanelConfig dataclass."""
    
    def test_panel_config_creation(self):
        """Test that PanelConfig can be created with default values."""
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Test Panel"
        )
        
        assert config.panel_type == PanelType.INFO_BOX
        assert config.title == "Test Panel"
        assert config.content == ""
        assert config.style == "info"
        assert config.flex == "1"
        assert config.min_width == "200px"
        assert config.open_by_default is True
        assert config.panel_id is None
    
    def test_panel_config_with_custom_values(self):
        """Test PanelConfig with custom values."""
        config = PanelConfig(
            panel_type=PanelType.INFO_ACCORDION,
            title="Custom Panel",
            content="Custom content",
            style="warning",
            flex="2",
            min_width="300px",
            open_by_default=False,
            panel_id="custom_panel"
        )
        
        assert config.panel_type == PanelType.INFO_ACCORDION
        assert config.title == "Custom Panel"
        assert config.content == "Custom content"
        assert config.style == "warning"
        assert config.flex == "2"
        assert config.min_width == "300px"
        assert config.open_by_default is False
        assert config.panel_id == "custom_panel"


class TestFooterContainer:
    """Test cases for FooterContainer class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.mock_info_box = Mock()
        self.mock_info_accordion = Mock()
        
        # Mock the info components
        self.info_box_patcher = patch('smartcash.ui.components.footer_container.InfoBox')
        self.info_accordion_patcher = patch('smartcash.ui.components.footer_container.InfoAccordion')
        
        self.mock_info_box_class = self.info_box_patcher.start()
        self.mock_info_accordion_class = self.info_accordion_patcher.start()
        
        self.mock_info_box_class.return_value = self.mock_info_box
        self.mock_info_accordion_class.return_value = self.mock_info_accordion
        
        # Mock the show method
        self.mock_info_box.show.return_value = widgets.HTML("Mock InfoBox")
        self.mock_info_accordion.show.return_value = widgets.HTML("Mock InfoAccordion")
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.info_box_patcher.stop()
        self.info_accordion_patcher.stop()
    
    def test_footer_container_initialization(self):
        """Test FooterContainer initialization."""
        footer = FooterContainer()
        
        assert footer._panels == {}
        assert footer.container is None
        assert footer.style == {}
        assert 'display' in footer.layout_config
        assert footer.layout_config['display'] == 'flex'
        assert footer.layout_config['width'] == '100%'
    
    def test_footer_container_with_custom_style_and_layout(self):
        """Test FooterContainer with custom style and layout."""
        custom_style = {"border": "2px solid red"}
        custom_layout = {"background": "#ffffff"}
        
        footer = FooterContainer(style=custom_style, layout=custom_layout)
        
        assert footer.style == custom_style
        assert footer.layout_config["background"] == "#ffffff"
    
    def test_add_info_box_panel(self):
        """Test adding an InfoBox panel."""
        footer = FooterContainer()
        
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Info Panel",
            content="This is info content",
            style="info"
        )
        
        panel_id = footer.add_panel(config)
        
        # Verify panel was added
        assert panel_id in footer._panels
        assert footer._panels[panel_id]['config'] == config
        
        # Verify InfoBox was created with correct parameters
        self.mock_info_box_class.assert_called_once_with(
            title="Info Panel",
            content="This is info content",
            style="info"
        )
    
    def test_add_info_accordion_panel(self):
        """Test adding an InfoAccordion panel."""
        footer = FooterContainer()
        
        config = PanelConfig(
            panel_type=PanelType.INFO_ACCORDION,
            title="Accordion Panel",
            content="This is accordion content",
            style="warning",
            open_by_default=False
        )
        
        panel_id = footer.add_panel(config)
        
        # Verify panel was added
        assert panel_id in footer._panels
        assert footer._panels[panel_id]['config'] == config
        
        # Verify InfoAccordion was created with correct parameters
        self.mock_info_accordion_class.assert_called_once_with(
            title="Accordion Panel",
            content="This is accordion content",
            style="warning",
            open_by_default=False
        )
    
    def test_add_panel_with_custom_id(self):
        """Test adding a panel with custom ID."""
        footer = FooterContainer()
        
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Custom ID Panel",
            panel_id="custom_id"
        )
        
        panel_id = footer.add_panel(config)
        
        assert panel_id == "custom_id"
        assert "custom_id" in footer._panels
    
    def test_add_panel_auto_generated_id(self):
        """Test adding panels with auto-generated IDs."""
        footer = FooterContainer()
        
        config1 = PanelConfig(panel_type=PanelType.INFO_BOX, title="Panel 1")
        config2 = PanelConfig(panel_type=PanelType.INFO_BOX, title="Panel 2")
        
        id1 = footer.add_panel(config1)
        id2 = footer.add_panel(config2)
        
        assert id1 == "panel_1"
        assert id2 == "panel_2"
    
    def test_remove_panel(self):
        """Test removing a panel."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        panel_id = footer.add_panel(config)
        
        assert panel_id in footer._panels
        
        footer.remove_panel(panel_id)
        
        assert panel_id not in footer._panels
    
    def test_remove_nonexistent_panel(self):
        """Test removing a non-existent panel doesn't raise error."""
        footer = FooterContainer()
        
        # Should not raise an error
        footer.remove_panel("nonexistent_id")
    
    def test_update_panel(self):
        """Test updating a panel's configuration."""
        footer = FooterContainer()
        
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Original Title",
            content="Original content"
        )
        panel_id = footer.add_panel(config)
        
        # Update the panel
        footer.update_panel(panel_id, title="Updated Title", content="Updated content")
        
        # Verify config was updated
        updated_config = footer._panels[panel_id]['config']
        assert updated_config.title == "Updated Title"
        assert updated_config.content == "Updated content"
        
        # Verify widget was updated
        assert self.mock_info_box.title == "Updated Title"
        assert self.mock_info_box.content == "Updated content"
    
    def test_update_panel_layout(self):
        """Test updating a panel's layout configuration."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        panel_id = footer.add_panel(config)
        
        # Update layout properties
        footer.update_panel(panel_id, flex="2", min_width="400px")
        
        # Verify config was updated
        updated_config = footer._panels[panel_id]['config']
        assert updated_config.flex == "2"
        assert updated_config.min_width == "400px"
    
    def test_update_nonexistent_panel(self):
        """Test updating a non-existent panel doesn't raise error."""
        footer = FooterContainer()
        
        # Should not raise an error
        footer.update_panel("nonexistent_id", title="New Title")
    
    def test_get_panel(self):
        """Test getting a panel by ID."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        panel_id = footer.add_panel(config)
        
        retrieved_panel = footer.get_panel(panel_id)
        
        assert retrieved_panel == self.mock_info_box
    
    def test_get_nonexistent_panel(self):
        """Test getting a non-existent panel returns None."""
        footer = FooterContainer()
        
        panel = footer.get_panel("nonexistent_id")
        
        assert panel is None
    
    def test_show_panel(self):
        """Test showing/hiding a panel."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        panel_id = footer.add_panel(config)
        
        # Mock the layout attribute
        self.mock_info_box.layout = Mock()
        
        # Show panel
        footer.show_panel(panel_id, True)
        assert self.mock_info_box.layout.display == 'flex'
        
        # Hide panel
        footer.show_panel(panel_id, False)
        assert self.mock_info_box.layout.display == 'none'
    
    def test_toggle_panel(self):
        """Test toggling a panel's visibility."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        panel_id = footer.add_panel(config)
        
        # Mock the layout attribute
        self.mock_info_box.layout = Mock()
        self.mock_info_box.layout.display = 'flex'
        
        # Toggle panel (should hide)
        footer.toggle_panel(panel_id)
        assert self.mock_info_box.layout.display == 'none'
        
        # Toggle again (should show)
        footer.toggle_panel(panel_id)
        assert self.mock_info_box.layout.display == 'flex'
    
    def test_container_creation_with_no_panels(self):
        """Test container creation when no panels are present."""
        footer = FooterContainer()
        
        # Container should be None initially when no panels are present
        assert footer.container is None
        
        # Container should be created when _update_container is called
        footer._update_container()
        assert footer.container is not None
        assert footer.container.layout.display == 'none'
    
    def test_container_creation_with_panels(self):
        """Test container creation with panels."""
        footer = FooterContainer()
        
        config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        footer.add_panel(config)
        
        # Should create a visible container
        assert footer.container is not None
        assert isinstance(footer.container, widgets.VBox)
    
    def test_initialization_with_initial_panels(self):
        """Test FooterContainer initialization with initial panels."""
        panels = [
            PanelConfig(panel_type=PanelType.INFO_BOX, title="Panel 1"),
            PanelConfig(panel_type=PanelType.INFO_ACCORDION, title="Panel 2")
        ]
        
        footer = FooterContainer(panels=panels)
        
        assert len(footer._panels) == 2
        assert "panel_1" in footer._panels
        assert "panel_2" in footer._panels


class TestCreateFooterContainer:
    """Test cases for create_footer_container function."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock the FooterContainer class
        self.footer_container_patcher = patch('smartcash.ui.components.footer_container.FooterContainer')
        self.mock_footer_container_class = self.footer_container_patcher.start()
        self.mock_footer_instance = Mock()
        self.mock_footer_container_class.return_value = self.mock_footer_instance
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.footer_container_patcher.stop()
    
    def test_create_footer_container_basic(self):
        """Test basic footer container creation."""
        footer = create_footer_container()
        
        # Verify FooterContainer was called with correct parameters
        self.mock_footer_container_class.assert_called_once()
        call_args = self.mock_footer_container_class.call_args
        
        assert call_args[1]['panels'] is None
        assert call_args[1]['style'] is None
        assert 'layout' in call_args[1]
        
        # Verify default layout
        layout = call_args[1]['layout']
        assert layout['display'] == 'flex'
        assert layout['flex_flow'] == 'row wrap'
        assert layout['width'] == '100%'
    
    def test_create_footer_container_with_panels(self):
        """Test creating footer container with panels."""
        panels = [
            PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
        ]
        
        footer = create_footer_container(panels=panels)
        
        call_args = self.mock_footer_container_class.call_args
        assert call_args[1]['panels'] == panels
    
    def test_create_footer_container_with_style(self):
        """Test creating footer container with custom style."""
        custom_style = {"border_top": "2px solid #007bff"}
        
        footer = create_footer_container(style=custom_style)
        
        call_args = self.mock_footer_container_class.call_args
        assert call_args[1]['style'] == custom_style
    
    def test_create_footer_container_with_layout_kwargs(self):
        """Test creating footer container with layout kwargs."""
        footer = create_footer_container(
            flex_flow="column",
            justify_content="center",
            background="#ffffff"
        )
        
        call_args = self.mock_footer_container_class.call_args
        layout = call_args[1]['layout']
        
        assert layout['flex_flow'] == "column"
        assert layout['justify_content'] == "center"
        assert layout['background'] == "#ffffff"
    
    def test_create_footer_container_full_example(self):
        """Test creating footer container with full configuration."""
        panels = [
            PanelConfig(
                panel_type=PanelType.INFO_BOX,
                title="Info",
                content="This is an info box",
                flex="1",
                min_width="300px"
            ),
            PanelConfig(
                panel_type=PanelType.INFO_ACCORDION,
                title="Details",
                content="More details here...",
                flex="2",
                min_width="400px"
            )
        ]
        
        footer = create_footer_container(
            panels=panels,
            style={"border_top": "2px solid #007bff"},
            flex_flow="row wrap",
            justify_content="space-between"
        )
        
        call_args = self.mock_footer_container_class.call_args
        
        assert call_args[1]['panels'] == panels
        assert call_args[1]['style'] == {"border_top": "2px solid #007bff"}
        
        layout = call_args[1]['layout']
        assert layout['flex_flow'] == "row wrap"
        assert layout['justify_content'] == "space-between"


class TestFooterContainerIntegration:
    """Integration tests for FooterContainer."""
    
    def test_multiple_panels_layout(self):
        """Test footer container with multiple panels."""
        with patch('smartcash.ui.components.footer_container.InfoBox') as mock_info_box, \
             patch('smartcash.ui.components.footer_container.InfoAccordion') as mock_info_accordion:
            
            # Create mock instances
            mock_box = Mock()
            mock_accordion = Mock()
            mock_info_box.return_value = mock_box
            mock_info_accordion.return_value = mock_accordion
            
            # Mock show methods
            mock_box.show.return_value = widgets.HTML("Box Content")
            mock_accordion.show.return_value = widgets.HTML("Accordion Content")
            
            # Create footer with multiple panels
            panels = [
                PanelConfig(
                    panel_type=PanelType.INFO_BOX,
                    title="Box Panel",
                    content="Box content",
                    flex="1"
                ),
                PanelConfig(
                    panel_type=PanelType.INFO_ACCORDION,
                    title="Accordion Panel",
                    content="Accordion content",
                    flex="2"
                )
            ]
            
            footer = FooterContainer(panels=panels)
            
            # Verify both panels were created
            assert len(footer._panels) == 2
            assert "panel_1" in footer._panels
            assert "panel_2" in footer._panels
            
            # Verify container was created
            assert footer.container is not None
            assert isinstance(footer.container, widgets.VBox)
    
    def test_panel_visibility_management(self):
        """Test managing panel visibility."""
        with patch('smartcash.ui.components.footer_container.InfoBox') as mock_info_box:
            
            mock_box = Mock()
            mock_info_box.return_value = mock_box
            mock_box.show.return_value = widgets.HTML("Box Content")
            mock_box.layout = Mock()
            
            footer = FooterContainer()
            
            config = PanelConfig(panel_type=PanelType.INFO_BOX, title="Test Panel")
            panel_id = footer.add_panel(config)
            
            # Test show/hide functionality
            footer.show_panel(panel_id, True)
            assert mock_box.layout.display == 'flex'
            
            footer.show_panel(panel_id, False)
            assert mock_box.layout.display == 'none'
            
            # Test toggle functionality
            mock_box.layout.display = 'none'
            footer.toggle_panel(panel_id)
            assert mock_box.layout.display == 'flex'
    
    def test_panel_configuration_updates(self):
        """Test updating panel configurations."""
        with patch('smartcash.ui.components.footer_container.InfoBox') as mock_info_box:
            
            mock_box = Mock()
            mock_info_box.return_value = mock_box
            mock_box.show.return_value = widgets.HTML("Box Content")
            mock_box.layout = Mock()
            
            footer = FooterContainer()
            
            config = PanelConfig(
                panel_type=PanelType.INFO_BOX,
                title="Original Title",
                content="Original content",
                style="info"
            )
            panel_id = footer.add_panel(config)
            
            # Update multiple properties
            footer.update_panel(
                panel_id,
                title="Updated Title",
                content="Updated content",
                style="warning",
                flex="2"
            )
            
            # Verify config was updated
            updated_config = footer._panels[panel_id]['config']
            assert updated_config.title == "Updated Title"
            assert updated_config.content == "Updated content"
            assert updated_config.style == "warning"
            assert updated_config.flex == "2"
            
            # Verify widget was updated
            assert mock_box.title == "Updated Title"
            assert mock_box.content == "Updated content"
            assert mock_box.style == "warning"


if __name__ == "__main__":
    pytest.main([__file__])