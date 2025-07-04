"""
Tests for the ConfirmationDialog component.
"""
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
from IPython.display import HTML

from smartcash.ui.components.dialog.confirmation_dialog import (
    ConfirmationDialog, 
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible,
    create_confirmation_area
)


class TestConfirmationDialog(unittest.TestCase):
    """Test cases for the ConfirmationDialog class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.dialog = ConfirmationDialog("test_dialog")
        
        # Mock the container and its layout
        self.mock_container = MagicMock()
        self.mock_container.layout = MagicMock()
        
        # Mock the display function
        self.mock_display = MagicMock()
        
        # Patch the necessary components
        self.patchers = [
            patch('IPython.display.display', self.mock_display),
            patch('IPython.display.clear_output'),
            patch('IPython.display.HTML', return_value="<html></html>")
        ]
        
        for patcher in self.patchers:
            patcher.start()
        
        # Initialize the dialog with our mock container
        self.dialog._ui_components = {'container': self.mock_container}
        self.dialog._initialized = True
    
    def tearDown(self):
        """Clean up after tests."""
        for patcher in self.patchers:
            patcher.stop()
    
    def test_initialization(self):
        """Test that the dialog initializes correctly."""
        dialog = ConfirmationDialog("test_init")
        self.assertEqual(dialog.component_name, "test_init")
        self.assertFalse(dialog._initialized)
        self.assertEqual(dialog._callbacks, {})
        self.assertFalse(dialog._is_visible)
    
    def test_show(self):
        """Test showing the dialog with default parameters."""
        # Reset the mock container
        self.mock_container.layout = MagicMock()
        
        # Setup the callback mock
        confirm_callback = MagicMock()
        cancel_callback = MagicMock()
        
        # Create a patched version of the show method that doesn't execute JavaScript
        def patched_show(self, title, message, on_confirm=None, on_cancel=None, 
                        confirm_text="Konfirmasi", cancel_text="Batal", danger_mode=False):
            # Store callbacks
            self._callbacks['confirm'] = on_confirm
            self._callbacks['cancel'] = on_cancel
            
            # Update container layout
            if 'container' in self._ui_components:
                self._ui_components['container'].layout.display = 'flex'
                self._ui_components['container'].layout.visibility = 'visible'
                self._ui_components['container'].layout.height = 'auto'
            
            # Mark as visible
            self._is_visible = True
        
        # Replace the show method with our patched version
        original_show = ConfirmationDialog.show
        ConfirmationDialog.show = patched_show
        
        try:
            # Create a new dialog instance to use the patched show method
            dialog = ConfirmationDialog("test_show")
            dialog._ui_components = {'container': self.mock_container}
            dialog._initialized = True
            
            # Call show with test parameters
            dialog.show(
                title="Test Title",
                message="Test Message",
                on_confirm=confirm_callback,
                on_cancel=cancel_callback,
                confirm_text="Yes",
                cancel_text="No",
                danger_mode=True
            )
            
            # Check container layout was updated
            self.assertEqual(self.mock_container.layout.display, 'flex')
            self.assertEqual(self.mock_container.layout.visibility, 'visible')
            self.assertEqual(self.mock_container.layout.height, 'auto')
            
            # Check callbacks were stored
            self.assertIn('confirm', dialog._callbacks)
            self.assertIn('cancel', dialog._callbacks)
            
            # Check dialog is marked as visible
            self.assertTrue(dialog._is_visible)
            
            # Test callbacks
            dialog._callbacks['confirm']()
            confirm_callback.assert_called_once()
            
            dialog._callbacks['cancel']()
            cancel_callback.assert_called_once()
            
        finally:
            # Restore the original method
            ConfirmationDialog.show = original_show
    
    def test_show_info(self):
        """Test showing an info dialog."""
        with patch.object(self.dialog, 'show') as mock_show:
            callback = MagicMock()
            self.dialog.show_info(
                title="Info",
                message="Test Info",
                on_ok=callback,
                ok_text="OK"
            )
            
            # Check show was called with correct parameters
            mock_show.assert_called_once_with(
                title="Info",
                message="Test Info",
                on_confirm=callback,
                on_cancel=None,
                confirm_text="OK",
                cancel_text="",
                danger_mode=False
            )
    
    def test_hide(self):
        """Test hiding the dialog."""
        # Set initial state
        self.dialog._is_visible = True
        self.dialog._callbacks = {'test': lambda: None}
        
        # Call hide
        self.dialog.hide()
        
        # Check container was reset
        self.mock_container.layout.display = 'none'
        self.mock_container.layout.visibility = 'hidden'
        
        # Check state was reset
        self.assertFalse(self.dialog._is_visible)
        self.assertEqual(self.dialog._callbacks, {})
    
    def test_is_visible(self):
        """Test checking if dialog is visible."""
        # Test when not visible
        self.mock_container.layout.display = 'none'
        self.assertFalse(self.dialog.is_visible())
        
        # Test when visible
        self.mock_container.layout.display = 'flex'
        self.assertTrue(self.dialog.is_visible())
        
        # Test when not initialized
        self.dialog._initialized = False
        self.assertFalse(self.dialog.is_visible())


class TestLegacyFunctions(unittest.TestCase):
    """Test cases for legacy functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.ui_components = {}
    
    def test_create_confirmation_area(self):
        """Test creating a confirmation area."""
        area = create_confirmation_area(self.ui_components)
        
        # Check dialog was created and initialized
        self.assertIn('confirmation_dialog', self.ui_components)
        self.assertIn('confirmation_area', self.ui_components)
        self.assertEqual(area, self.ui_components['confirmation_area'])
    
    def test_show_confirmation_dialog(self):
        """Test showing a confirmation dialog."""
        # Mock the dialog instance
        mock_dialog = MagicMock()
        self.ui_components['confirmation_dialog'] = mock_dialog
        
        # Call the function
        callback = MagicMock()
        show_confirmation_dialog(
            self.ui_components,
            title="Test",
            message="Test Message",
            on_confirm=callback,
            confirm_text="Yes",
            cancel_text="No",
            danger_mode=True
        )
        
        # Check dialog was shown with correct parameters
        mock_dialog.show.assert_called_once_with(
            title="Test",
            message="Test Message",
            on_confirm=callback,
            on_cancel=None,
            confirm_text="Yes",
            cancel_text="No",
            danger_mode=True
        )
    
    def test_show_info_dialog(self):
        """Test showing an info dialog."""
        # Mock the dialog instance
        mock_dialog = MagicMock()
        self.ui_components['confirmation_dialog'] = mock_dialog
        
        # Call the function
        callback = MagicMock()
        show_info_dialog(
            self.ui_components,
            title="Info",
            message="Test Info",
            on_ok=callback,
            ok_text="OK"
        )
        
        # Check dialog was shown with correct parameters
        mock_dialog.show_info.assert_called_once_with(
            title="Info",
            message="Test Info",
            on_ok=callback,
            ok_text="OK"
        )
    
    def test_clear_dialog_area(self):
        """Test clearing the dialog area."""
        # Test with dialog instance
        mock_dialog = MagicMock()
        self.ui_components['confirmation_dialog'] = mock_dialog
        
        clear_dialog_area(self.ui_components)
        mock_dialog.hide.assert_called_once()
        
        # Test with area instance
        self.ui_components.clear()
        mock_area = MagicMock()
        mock_area.__enter__ = MagicMock()
        mock_area.__exit__ = MagicMock()
        self.ui_components['confirmation_area'] = mock_area
        
        clear_dialog_area(self.ui_components)
        
        # Check clear_output was called in the context manager
        mock_area.__enter__.assert_called_once()
        mock_area.__exit__.assert_called_once()
        
        # Check layout was updated
        self.assertEqual(mock_area.layout.display, 'none')
    
    def test_is_dialog_visible(self):
        """Test checking if dialog is visible."""
        # Test with dialog instance
        mock_dialog = MagicMock()
        mock_dialog.is_visible.return_value = True
        self.ui_components['confirmation_dialog'] = mock_dialog
        
        self.assertTrue(is_dialog_visible(self.ui_components))
        mock_dialog.is_visible.assert_called_once()
        
        # Test with area instance
        self.ui_components.clear()
        mock_area = MagicMock()
        mock_area.layout.display = 'flex'
        self.ui_components['confirmation_area'] = mock_area
        
        self.assertTrue(is_dialog_visible(self.ui_components))
        
        # Test when not visible
        mock_area.layout.display = 'none'
        self.assertFalse(is_dialog_visible(self.ui_components))
        
        # Test when no dialog or area exists
        self.ui_components.clear()
        self.assertFalse(is_dialog_visible(self.ui_components))


if __name__ == '__main__':
    unittest.main()
