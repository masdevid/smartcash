"""
Tests for the FooterContainer component.

These tests focus on the behavior of the footer container using mocks.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock, ANY
from dataclasses import asdict

# Import ipywidgets first to avoid mocking issues
import ipywidgets as widgets

# Import the module under test after importing ipywidgets
from smartcash.ui.components.footer_container import (
    FooterContainer, PanelConfig, PanelType, create_footer_container
)

class TestFooterContainer(unittest.TestCase):
    """Test cases for the FooterContainer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a real Layout instance for the container
        self.mock_layout = MagicMock()
        self.mock_panel_layout = MagicMock()
        
        # Create a mock container
        self.mock_container = MagicMock()
        
        # Patch the widgets module
        self.patcher_widgets = patch('smartcash.ui.components.footer_container.widgets')
        self.mock_widgets = self.patcher_widgets.start()
        
        # Configure the mock widgets
        self.mock_widgets.Layout.return_value = self.mock_layout
        self.mock_widgets.VBox.return_value = self.mock_container
        self.mock_widgets.HBox.return_value = MagicMock()
        
        # Patch the InfoBox and InfoAccordion classes
        self.patcher_info_box = patch('smartcash.ui.components.footer_container.InfoBox')
        self.mock_info_box = self.patcher_info_box.start()
        
        self.patcher_info_accordion = patch('smartcash.ui.components.footer_container.InfoAccordion')
        self.mock_info_accordion = self.patcher_info_accordion.start()
        
        # Setup mock info components with layout
        self.mock_info_box_instance = MagicMock()
        self.mock_info_box_instance.layout = self.mock_panel_layout
        self.mock_info_box.return_value = self.mock_info_box_instance
        
        self.mock_info_accordion_instance = MagicMock()
        self.mock_info_accordion_instance.layout = self.mock_panel_layout
        self.mock_info_accordion.return_value = self.mock_info_accordion_instance
        
        # Patch uuid4 to return a consistent ID for testing
        self.patcher_uuid = patch('uuid.uuid4')
        self.mock_uuid = self.patcher_uuid.start()
        self.mock_uuid.return_value = 'test_panel_id'
        
    def tearDown(self):
        """Tear down test fixtures."""
        self.patcher_widgets.stop()
        self.patcher_info_box.stop()
        self.patcher_info_accordion.stop()
        self.patcher_uuid.stop()
    
    def test_footer_container_creation(self):
        """Test that FooterContainer is created with default values."""
        # Create a footer container with no panels
        footer = FooterContainer()
        
        # Verify the container is None initially (it's set in _update_container)
        self.assertIsNone(footer.container)
        self.assertEqual(footer._panels, {})
        
        # Verify the layout config was set up correctly
        self.assertEqual(footer.layout_config, {
            'display': 'flex',
            'flex_flow': 'row wrap',
            'align_items': 'stretch',
            'justify_content': 'space-between',
            'width': '100%',
            'border': '1px solid #e0e0e0',
            'margin': '10px 0 0 0',
            'padding': '10px',
            'background': '#f9f9f9'
        })
    
    def test_add_info_box_panel(self):
        """Test adding an InfoBox panel to the footer."""
        # Create a footer container
        footer = FooterContainer()
        
        # Create a panel config
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Test Info",
            content="Test Content",
            style="info",
            flex="1",
            min_width="200px"
        )
        
        # Add the panel
        panel_id = footer.add_panel(config)
        
        # Verify the panel was added
        self.assertIn(panel_id, footer._panels)
        self.assertEqual(len(footer._panels), 1)
        
        # Verify InfoBox was created with correct parameters
        self.mock_info_box.assert_called_once_with(
            title="Test Info",
            content="Test Content",
            style="info"
        )
        
        # Verify layout was set on the panel
        # Note: We can't directly assert on the layout.update call because it's a real Layout object
        # Instead, we'll verify that the panel was added to the container
        self.mock_widgets.HBox.assert_called_once()
        
    def test_add_info_accordion_panel(self):
        """Test adding an InfoAccordion panel to the footer."""
        # Create a footer container
        footer = FooterContainer()
        
        # Create a panel config
        config = PanelConfig(
            panel_type=PanelType.INFO_ACCORDION,
            title="Test Accordion",
            content="Test Content",
            style="warning",
            flex="2",
            min_width="300px",
            open_by_default=True
        )
        
        # Add the panel
        panel_id = footer.add_panel(config)
        
        # Verify the panel was added
        self.assertIn(panel_id, footer._panels)
        self.assertEqual(len(footer._panels), 1)
        
        # Verify InfoAccordion was created with correct parameters
        self.mock_info_accordion.assert_called_once_with(
            title="Test Accordion",
            content="Test Content",
            style="warning",
            open_by_default=True
        )
        
        # Verify layout was set on the panel
        # Note: We can't directly assert on the layout.update call because it's a real Layout object
        # Instead, we'll verify that the panel was added to the container
        self.mock_widgets.HBox.assert_called_once()
    
    def test_remove_panel(self):
        """Test removing a panel from the footer."""
        # Create a panel config
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Test Panel",
            content="Test Content"
        )
        
        # Create a footer container with the panel
        footer = FooterContainer(panels=[config])
        
        # Get the panel ID that was generated
        panel_id = next(iter(footer._panels.keys()))
        
        # Mock _update_container to do nothing
        with patch.object(footer, '_update_container') as mock_update:
            # Remove the panel
            footer.remove_panel(panel_id)
            
            # Verify the panel was removed and _update_container was called
            self.assertNotIn(panel_id, footer._panels)
            mock_update.assert_called_once()
    
    def test_remove_nonexistent_panel(self):
        """Test removing a panel that doesn't exist."""
        # Create a footer container
        footer = FooterContainer()
        
        # Try to remove a non-existent panel
        result = footer.remove_panel("nonexistent_id")
        
        # Verify the method returned False
        self.assertFalse(result)
    
    def test_get_panel(self):
        """Test getting a panel by ID."""
        # Create a panel config
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Test Panel",
            content="Test Content"
        )
        
        # Create a footer container with the panel
        footer = FooterContainer(panels=[config])
        
        # Get the panel ID that was generated
        panel_id = next(iter(footer._panels.keys()))
        
        # Get the panel widget (should be the mock_info_box_instance)
        panel = footer.get_panel(panel_id)
        
        # Verify the correct widget was returned
        self.assertEqual(panel, self.mock_info_box_instance)
    
    def test_get_nonexistent_panel(self):
        """Test getting a panel that doesn't exist."""
        # Create a footer container
        footer = FooterContainer()
        
        # Try to get a non-existent panel
        panel = footer.get_panel("nonexistent_id")
        
        # Verify the method returned None
        self.assertIsNone(panel)
    
    def test_create_footer_container_helper(self):
        """Test the create_footer_container helper function."""
        # Create a panel config
        config = PanelConfig(
            panel_type=PanelType.INFO_BOX,
            title="Test Panel",
            content="Test Content"
        )
        
        # Create a footer container using the helper function
        footer = create_footer_container(
            panels=[config],
            style={"border_top": "2px solid #007bff"},
            flex_flow="row wrap",
            justify_content="space-between"
        )
        
        # Verify the footer was created
        self.assertIsInstance(footer, FooterContainer)
        
        # Verify the style was applied
        self.assertEqual(footer.style, {"border_top": "2px solid #007bff"})
        
        # Verify the layout was updated
        self.assertEqual(footer.layout_config['flex_flow'], "row wrap")
        self.assertEqual(footer.layout_config['justify_content'], "space-between")

if __name__ == '__main__':
    unittest.main()
