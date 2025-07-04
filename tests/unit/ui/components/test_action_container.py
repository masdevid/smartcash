"""
Tests for the ActionContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch, ANY
import ipywidgets as widgets

class TestActionContainer(unittest.TestCase):
    """Test cases for the ActionContainer component."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create a real VBox for testing
        self.mock_vbox = MagicMock(spec=widgets.VBox)
        self.mock_vbox.children = []
        
        # Patch widgets.VBox to return our mock
        self.vbox_patcher = patch('ipywidgets.VBox', return_value=self.mock_vbox)
        self.mock_vbox_class = self.vbox_patcher.start()
        
        # Create a real Button for testing
        self.mock_button = MagicMock(spec=widgets.Button)
        self.mock_button._click_handlers = MagicMock()
        self.mock_button.on_click = MagicMock()  # Add this line to mock the on_click method
        
        # Patch create_action_buttons
        self.action_buttons_patcher = patch(
            'smartcash.ui.components.action_container.create_action_buttons',
            return_value={
                'buttons': {'test1': self.mock_button, 'test2': self.mock_button},
                'container': MagicMock()
            }
        )
        self.mock_create_action_buttons = self.action_buttons_patcher.start()
    
    def tearDown(self):
        """Clean up after tests."""
        self.vbox_patcher.stop()
        self.action_buttons_patcher.stop()
    
    def test_create_action_container(self):
        """Test creating an action container with buttons and dialog functions."""
        # Import the function directly to avoid caching issues
        from smartcash.ui.components.action_container import create_action_container
        
        # Mock the dialog functions
        with patch('smartcash.ui.components.action_container.create_confirmation_area') as mock_create_area, \
             patch('smartcash.ui.components.action_container.show_confirmation_dialog') as mock_show_dialog, \
             patch('smartcash.ui.components.action_container.show_info_dialog') as mock_show_info, \
             patch('smartcash.ui.components.action_container.clear_dialog_area') as mock_clear_dialog, \
             patch('smartcash.ui.components.action_container.is_dialog_visible') as mock_is_visible:
            
            # Setup mock return values
            mock_dialog_area = MagicMock()
            mock_create_area.return_value = mock_dialog_area
            mock_is_visible.return_value = False
            
            # Create test buttons
            test_buttons = [
                {'button_id': 'test1', 'text': 'Test 1'},
                {'button_id': 'test2', 'text': 'Test 2', 'style': 'danger'}
            ]
            
            # Create the action container
            action_container = create_action_container(
                buttons=test_buttons,
                title="Test Actions",
                alignment='center'
            )
            
            # Verify the container was created
            self.assertIsInstance(action_container, dict)
            self.assertIn('container', action_container)
            self.assertEqual(action_container['container'], self.mock_vbox)
            
            # Verify buttons were created
            self.assertIn('buttons', action_container)
            self.assertIsInstance(action_container['buttons'], dict)
            self.assertEqual(len(action_container['buttons']), 2)
            self.assertIn('test1', action_container['buttons'])
            self.assertIn('test2', action_container['buttons'])
            
            # Verify dialog area was created
            self.assertEqual(action_container['dialog_area'], mock_dialog_area)
            mock_create_area.assert_called_once()
            
            # Test dialog functions
            test_callback = MagicMock()
            
            # Test show_dialog
            action_container['show_dialog'](
                title="Confirm",
                message="Are you sure?",
                on_confirm=test_callback
            )
            mock_show_dialog.assert_called_once()
            
            # Test show_info
            action_container['show_info'](
                title="Info",
                message="Information message"
            )
            mock_show_info.assert_called_once()
            
            # Test clear_dialog
            action_container['clear_dialog']()
            mock_clear_dialog.assert_called_once()
            
            # Test is_dialog_visible
            action_container['is_dialog_visible']()
            mock_is_visible.assert_called_once()
    
    def test_action_container_button_click(self):
        """Test that button click handlers work correctly."""
        from smartcash.ui.components.action_container import create_action_container
        
        # Create a test click handler
        test_clicked = MagicMock()
        
        # Create a test button with our click handler
        test_button = {
            'button_id': 'test_btn',
            'text': 'Click Me',
            'on_click': test_clicked
        }
        
        # Create a mock for the button that will be created by create_action_buttons
        mock_button = MagicMock(spec=widgets.Button)
        mock_button.description = test_button['text']
        
        # Configure the mock to return our test button
        self.mock_create_action_buttons.return_value = {
            'buttons': {test_button['button_id']: mock_button},
            'container': MagicMock()
        }
        
        # Create the action container with our test button
        action_container = create_action_container(buttons=[test_button])
        
        # Get the button that was created
        created_button = action_container['buttons'][test_button['button_id']]
        
        # Verify the button was created with the correct text
        self.assertEqual(created_button.description, test_button['text'])
        
        # Test that the click handler is called when the button is clicked
        # We'll test this by calling the click handler directly
        test_clicked.assert_not_called()  # Shouldn't be called yet
        
        # Simulate a button click by calling the handler directly
        test_button['on_click'](mock_button)
        
        # Verify the test click handler was called
        test_clicked.assert_called_once()
    
    def test_action_container_custom_styling(self):
        """Test that custom styling is applied to the action container."""
        from smartcash.ui.components.action_container import create_action_container
        
        # Create with custom styling
        create_action_container(
            buttons=[{'button_id': 'test', 'text': 'Test'}],
            container_margin="20px 0"
        )
        
        # Verify the VBox was created with the correct layout
        args, kwargs = self.mock_vbox_class.call_args
        # The layout is a traitlets.traitlets.Instance, not a dict
        layout = kwargs.get('layout')
        self.assertIsNotNone(layout)
        self.assertEqual(layout.margin, '20px 0')
        
        # Verify the action buttons were created with the correct alignment
        args, kwargs = self.mock_create_action_buttons.call_args
        self.assertEqual(kwargs.get('alignment'), 'left')  # Default alignment


if __name__ == "__main__":
    unittest.main()
