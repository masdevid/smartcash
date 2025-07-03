"""
Tests for the FormContainer component.
"""
import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

class TestFormContainer(unittest.TestCase):
    """Test cases for the FormContainer component."""
    
    def test_create_form_container_basic(self):
        """Test creating a basic form container with default settings."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create a form container with default settings
        form = create_form_container()
        
        # Verify the container was created
        self.assertIsInstance(form, dict)
        self.assertIn('container', form)
        self.assertIsInstance(form['container'], widgets.VBox)
        
        # Verify form container exists
        self.assertIn('form_container', form)
        self.assertIsInstance(form['form_container'], widgets.VBox)
        
        # Verify buttons were created
        self.assertIn('save_button', form)
        self.assertIn('reset_button', form)
        self.assertIsInstance(form['save_button'], widgets.Button)
        self.assertIsInstance(form['reset_button'], widgets.Button)
        
        # Verify default button labels
        self.assertEqual(form['save_button'].description, 'Simpan')
        self.assertEqual(form['reset_button'].description, 'Reset')
    
    def test_form_container_without_buttons(self):
        """Test creating a form container without save/reset buttons."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create a form container without buttons
        form = create_form_container(show_buttons=False)
        
        # Verify the container was created
        self.assertIsInstance(form, dict)
        self.assertIn('container', form)
        self.assertIsInstance(form['container'], widgets.VBox)
        
        # Verify form container exists
        self.assertIn('form_container', form)
        self.assertIsInstance(form['form_container'], widgets.VBox)
        
        # Verify buttons were not created
        self.assertNotIn('save_button', form)
        self.assertNotIn('reset_button', form)
    
    def test_form_container_with_callbacks(self):
        """Test form container with save and reset callbacks."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create mock callbacks
        save_callback = MagicMock()
        reset_callback = MagicMock()
        
        # Create form with callbacks
        form = create_form_container(
            on_save=save_callback,
            on_reset=reset_callback
        )
        
        # Trigger save button click
        save_button = form['save_button']
        save_button._click_handlers(save_button)
        
        # Verify save callback was called
        save_callback.assert_called_once()
        
        # Trigger reset button click
        reset_button = form['reset_button']
        reset_button._click_handlers(reset_button)
        
        # Verify reset callback was called
        reset_callback.assert_called_once()
    
    def test_form_container_styling(self):
        """Test form container with custom styling."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Create form with custom styling
        form = create_form_container(
            container_margin="20px",
            container_padding="10px",
            form_spacing="15px"
        )
        
        # Verify container styling
        self.assertEqual(form['container'].layout.margin, '20px')
        self.assertEqual(form['container'].layout.padding, '10px')
        
        # Verify form container styling
        form_container = form['form_container']
        self.assertEqual(form_container.layout.margin, '0 0 15px 0')
    
    @patch('smartcash.ui.components.form_container.create_save_reset_buttons')
    @patch('smartcash.ui.components.form_container.widgets')
    def test_form_container_button_creation(self, mock_widgets, mock_create_buttons):
        """Test that save/reset buttons are created with correct parameters."""
        from smartcash.ui.components.form_container import create_form_container
        
        # Setup mock widgets
        mock_form_container = MagicMock()
        mock_button_container = MagicMock()
        
        # Create side effect for VBox to return different mocks for different calls
        def vbox_side_effect(*args, **kwargs):
            if not hasattr(vbox_side_effect, 'call_count'):
                vbox_side_effect.call_count = 0
            vbox_side_effect.call_count += 1
            
            if vbox_side_effect.call_count == 1:
                # First call is for the form container
                return mock_form_container
            else:
                # Second call is for the main container
                return MagicMock()
        
        mock_widgets.VBox.side_effect = vbox_side_effect
        
        # Setup mock buttons
        mock_save_button = MagicMock()
        mock_reset_button = MagicMock()
        
        mock_create_buttons.return_value = {
            'save_button': mock_save_button,
            'reset_button': mock_reset_button,
            'container': mock_button_container
        }
        
        # Create form with custom button labels and tooltips
        form = create_form_container(
            save_label="Save Changes",
            reset_label="Reset Form",
            save_tooltip="Click to save",
            reset_tooltip="Click to reset",
            alignment='center',
            with_sync_info=True,
            sync_message="Auto-save enabled"
        )
        
        # Verify create_save_reset_buttons was called with correct parameters
        args, kwargs = mock_create_buttons.call_args
        self.assertEqual(kwargs['save_label'], 'Save Changes')
        self.assertEqual(kwargs['reset_label'], 'Reset Form')
        self.assertEqual(kwargs['save_tooltip'], 'Click to save')
        self.assertEqual(kwargs['reset_tooltip'], 'Click to reset')
        self.assertEqual(kwargs['alignment'], 'center')
        self.assertTrue(kwargs['with_sync_info'])
        self.assertEqual(kwargs['sync_message'], 'Auto-save enabled')
        
        # Verify VBox was called twice (once for form container, once for main container)
        self.assertEqual(mock_widgets.VBox.call_count, 2)
        
        # Get the main container call (second call to VBox)
        main_container_call = mock_widgets.VBox.call_args_list[1]
        main_container_args = main_container_call[0]
        
        # Verify the main container has the form container and button container as children
        self.assertEqual(len(main_container_args[0]), 2)
        self.assertEqual(main_container_args[0][0], mock_form_container)  # Form container
        self.assertEqual(main_container_args[0][1], mock_button_container)  # Button container


if __name__ == "__main__":
    unittest.main()
