"""
Tests for the ConfirmationDialog component.

This module contains unit tests for the ConfirmationDialog class, which is a
wrapper around SimpleDialog for backward compatibility.
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets

from smartcash.ui.components.dialog.confirmation_dialog import (
    ConfirmationDialog,
    create_confirmation_area,
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible
)


class TestConfirmationDialog(unittest.TestCase):
    """Test cases for ConfirmationDialog class."""

    def setUp(self):
        """Set up test fixtures."""
        self.dialog = ConfirmationDialog("test_dialog")
        # Initialize the dialog to set up UI components
        self.dialog.initialize()
        self.mock_callback = MagicMock()

    def test_initialization(self):
        """Test that the dialog initializes correctly."""
        self.assertTrue(hasattr(self.dialog, '_ui_components'))
        self.assertIn('container', self.dialog._ui_components)
        self.assertIsInstance(self.dialog._ui_components['container'], widgets.VBox)

    def test_show_confirmation(self):
        """Test showing a confirmation dialog."""
        with patch.object(self.dialog, 'show_confirmation') as mock_show_confirmation:
            self.dialog.show(
                title="Test Title",
                message="Test Message",
                on_confirm=self.mock_callback,
                confirm_text="Yes",
                cancel_text="No",
                danger_mode=True
            )
            
            mock_show_confirmation.assert_called_once_with(
                title="Test Title",
                message="Test Message",
                on_confirm=self.mock_callback,
                on_cancel=None,
                confirm_text="Yes",
                cancel_text="No",
                danger_mode=True
            )

    def test_legacy_functions(self):
        """Test backward compatibility functions."""
        ui_components = {}
        
        # Test create_confirmation_area
        with self.assertWarns(DeprecationWarning):
            create_confirmation_area(ui_components)
        self.assertIn('confirmation_dialog', ui_components)
        
        # Test show_confirmation_dialog
        with self.assertWarns(DeprecationWarning):
            with patch.object(ui_components['confirmation_dialog'], 'show') as mock_show:
                show_confirmation_dialog(
                    ui_components,
                    title="Test",
                    message="Test message",
                    on_confirm=self.mock_callback
                )
                mock_show.assert_called_once()
        
        # Test show_info_dialog
        with self.assertWarns(DeprecationWarning):
            with patch.object(ui_components['confirmation_dialog'], 'show_info') as mock_show_info:
                show_info_dialog(
                    ui_components,
                    title="Info",
                    message="Info message"
                )
                mock_show_info.assert_called_once()
        
        # Test clear_dialog_area
        with self.assertWarns(DeprecationWarning):
            with patch.object(ui_components['confirmation_dialog'], 'hide') as mock_hide:
                clear_dialog_area(ui_components)
                mock_hide.assert_called_once()
        
        # Test is_dialog_visible
        with self.assertWarns(DeprecationWarning):
            with patch.object(ui_components['confirmation_dialog'], 'is_visible', return_value=True):
                self.assertTrue(is_dialog_visible(ui_components))


class TestConfirmationDialogIntegration(unittest.TestCase):
    """Integration tests for ConfirmationDialog with SimpleDialog."""
    
    def test_integration_with_simple_dialog(self):
        """Test that ConfirmationDialog properly integrates with SimpleDialog."""
        from smartcash.ui.components.dialog.simple_dialog import SimpleDialog
        
        # Create a ConfirmationDialog (which is a subclass of SimpleDialog)
        dialog = ConfirmationDialog("integration_test")
        dialog.initialize()
        
        # Test show_confirmation
        with patch.object(SimpleDialog, 'show_confirmation') as mock_show_confirmation:
            dialog.show_confirmation(
                title="Test",
                message="Test message",
                on_confirm=lambda: None,
                danger_mode=True
            )
            mock_show_confirmation.assert_called_once()
        
        # Test show_info
        with patch.object(SimpleDialog, 'show_info') as mock_show_info:
            dialog.show_info(
                title="Info",
                message="Info message",
                info_type="success"
            )
            mock_show_info.assert_called_once()


if __name__ == "__main__":
    unittest.main()
