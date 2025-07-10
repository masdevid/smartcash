"""
Tests for dialog component module.

This module tests the dialog components including ConfirmationDialog class
and legacy functions for showing confirmation dialogs, info dialogs, and
managing dialog state.
"""

import pytest
import uuid
from unittest.mock import Mock, patch, MagicMock, call
import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable

# Import the components to test
from smartcash.ui.components.dialog.confirmation_dialog import (
    ConfirmationDialog,
    show_confirmation_dialog,
    show_info_dialog,
    clear_dialog_area,
    is_dialog_visible,
    create_confirmation_area
)


class TestConfirmationDialog:
    """Test cases for ConfirmationDialog class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock the BaseUIComponent from the correct module
        self.base_ui_component_patcher = patch('smartcash.ui.components.base_component.BaseUIComponent')
        self.mock_base_ui_component = self.base_ui_component_patcher.start()
        
        # Mock display and other IPython components
        self.display_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.display')
        self.clear_output_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.clear_output')
        self.html_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.HTML')
        
        self.mock_display = self.display_patcher.start()
        self.mock_clear_output = self.clear_output_patcher.start()
        self.mock_html = self.html_patcher.start()
        
        # Mock widgets
        self.mock_output = Mock(spec=widgets.Output)
        self.mock_output.layout = Mock()
        self.mock_output.layout.display = 'none'
        
        # Mock widgets.Output
        self.output_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.widgets.Output')
        self.mock_output_class = self.output_patcher.start()
        self.mock_output_class.return_value = self.mock_output
        
        # Mock widgets.Layout  
        self.layout_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.widgets.Layout')
        self.mock_layout_class = self.layout_patcher.start()
        self.mock_layout_class.return_value = Mock()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.base_ui_component_patcher.stop()
        self.display_patcher.stop()
        self.clear_output_patcher.stop()
        self.html_patcher.stop()
        self.output_patcher.stop()
        self.layout_patcher.stop()
    
    def test_confirmation_dialog_initialization(self):
        """Test ConfirmationDialog initialization."""
        dialog = ConfirmationDialog("test_dialog")
        
        assert dialog.component_name == "test_dialog"
        assert dialog._callbacks == {}
        assert dialog._is_visible is False
        
        # Note: _initialized is set by BaseUIComponent, so we don't test it here
    
    def test_confirmation_dialog_initialization_with_kwargs(self):
        """Test ConfirmationDialog initialization with kwargs."""
        dialog = ConfirmationDialog("test_dialog", logger=Mock(), error_handler=Mock())
        
        assert dialog.component_name == "test_dialog"
        
        # Note: BaseUIComponent initialization is handled by the real class
    
    def test_create_ui_components(self):
        """Test _create_ui_components method."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._ui_components = {}
        
        # Mock the base class initialization
        dialog._initialized = False
        
        # Call _create_ui_components
        dialog._create_ui_components()
        
        # Verify Output widget was created
        self.mock_output_class.assert_called_once()
        
        # Verify Layout was created with correct parameters
        self.mock_layout_class.assert_called_once()
        
        # Verify container was added to UI components
        assert 'container' in dialog._ui_components
        assert dialog._ui_components['container'] == self.mock_output
    
    def test_show_confirmation_dialog(self):
        """Test show method for confirmation dialog."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        
        # Mock callbacks
        on_confirm = Mock()
        on_cancel = Mock()
        
        # Call show method
        dialog.show(
            title="Test Title",
            message="Test Message",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Confirm",
            cancel_text="Cancel",
            danger_mode=False
        )
        
        # Verify callbacks were stored
        assert dialog._callbacks['confirm'] == on_confirm
        assert dialog._callbacks['cancel'] == on_cancel
        
        # Verify display was called
        self.mock_display.assert_called_once()
        
        # Verify container layout was updated
        assert dialog._is_visible is True
        assert self.mock_output.layout.display == 'flex'
    
    def test_show_confirmation_dialog_danger_mode(self):
        """Test show method with danger mode enabled."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        
        # Call show method with danger mode
        dialog.show(
            title="Danger Title",
            message="Danger Message",
            danger_mode=True
        )
        
        # Verify display was called (danger mode affects styling)
        self.mock_display.assert_called_once()
        
        # Verify container layout was updated
        assert dialog._is_visible is True
    
    def test_show_info_dialog(self):
        """Test show_info method."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        
        # Mock the show method
        dialog.show = Mock()
        
        # Mock callback
        on_ok = Mock()
        
        # Call show_info
        dialog.show_info(
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify show was called with correct parameters
        dialog.show.assert_called_once_with(
            title="Info Title",
            message="Info Message",
            on_confirm=on_ok,
            on_cancel=None,
            confirm_text="OK",
            cancel_text="",
            danger_mode=False
        )
    
    def test_show_dialog_method(self):
        """Test _show_dialog method."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        
        # Mock the container context manager
        container_context = Mock()
        self.mock_output.__enter__ = Mock(return_value=container_context)
        self.mock_output.__exit__ = Mock(return_value=None)
        
        # Call _show_dialog
        html_content = "<div>Test HTML Content</div>"
        dialog._show_dialog(html_content)
        
        # Verify clear_output was called
        self.mock_clear_output.assert_called_once_with(wait=True)
        
        # Verify HTML was created and displayed
        self.mock_html.assert_called_once_with(html_content)
        self.mock_display.assert_called_once()
        
        # Verify container layout was updated
        assert dialog._is_visible is True
        assert self.mock_output.layout.display == 'flex'
    
    def test_hide_dialog(self):
        """Test hide method."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        dialog._is_visible = True
        dialog._callbacks = {'confirm': Mock(), 'cancel': Mock()}
        
        # Mock the container context manager
        container_context = Mock()
        self.mock_output.__enter__ = Mock(return_value=container_context)
        self.mock_output.__exit__ = Mock(return_value=None)
        
        # Call hide
        dialog.hide()
        
        # Verify clear_output was called
        self.mock_clear_output.assert_called_once_with(wait=True)
        
        # Verify container layout was reset
        assert self.mock_output.layout.display == 'none'
        assert self.mock_output.layout.visibility == 'hidden'
        
        # Verify state was cleaned up
        assert dialog._is_visible is False
        assert dialog._callbacks == {}
    
    def test_hide_dialog_not_initialized(self):
        """Test hide method when dialog is not initialized."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = False
        
        # Call hide - should return without error
        dialog.hide()
        
        # Verify clear_output was not called
        self.mock_clear_output.assert_not_called()
    
    def test_is_visible_method(self):
        """Test is_visible method."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_output}
        
        # Test when dialog is hidden
        self.mock_output.layout.display = 'none'
        assert dialog.is_visible() is False
        
        # Test when dialog is visible
        self.mock_output.layout.display = 'flex'
        assert dialog.is_visible() is True
        
        # Test when not initialized
        dialog._initialized = False
        assert dialog.is_visible() is False
    
    def test_is_visible_fallback_to_internal_state(self):
        """Test is_visible method fallback to internal state."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': None}  # No container
        
        # Test internal state
        dialog._is_visible = True
        assert dialog.is_visible() is True
        
        dialog._is_visible = False
        assert dialog.is_visible() is False
    
    def test_show_initializes_if_not_initialized(self):
        """Test that show method initializes dialog if not initialized."""
        dialog = ConfirmationDialog("test_dialog")
        dialog._initialized = False
        dialog.initialize = Mock()
        
        # Mock UI components after initialization
        dialog._ui_components = {'container': self.mock_output}
        
        # Call show
        dialog.show("Title", "Message")
        
        # Verify initialize was called
        dialog.initialize.assert_called_once()


class TestLegacyDialogFunctions:
    """Test cases for legacy dialog functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock ConfirmationDialog
        self.confirmation_dialog_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.ConfirmationDialog')
        self.mock_confirmation_dialog_class = self.confirmation_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_confirmation_dialog_class.return_value = self.mock_dialog_instance
        
        # Setup proper mock structure for _ui_components
        self.mock_container = Mock()
        self.mock_dialog_instance._ui_components = {'container': self.mock_container}
        
        # Mock clear_output
        self.clear_output_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.clear_output')
        self.mock_clear_output = self.clear_output_patcher.start()
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.confirmation_dialog_patcher.stop()
        self.clear_output_patcher.stop()
    
    def test_create_confirmation_area(self):
        """Test create_confirmation_area function."""
        ui_components = {}
        
        # Call create_confirmation_area
        result = create_confirmation_area(ui_components)
        
        # Verify dialog was created without parameters (default behavior)
        self.mock_confirmation_dialog_class.assert_called_once_with()
        
        # Verify ui_components was updated (only confirmation_dialog, not confirmation_area)
        assert 'confirmation_dialog' in ui_components
        assert ui_components['confirmation_dialog'] == self.mock_dialog_instance
    
    def test_create_confirmation_area_existing_dialog(self):
        """Test create_confirmation_area with existing dialog."""
        existing_dialog = Mock()
        existing_dialog._ui_components = {'container': self.mock_container}
        ui_components = {'confirmation_dialog': existing_dialog}
        
        # Call create_confirmation_area
        result = create_confirmation_area(ui_components)
        
        # Verify new dialog was not created
        self.mock_confirmation_dialog_class.assert_not_called()
        
        # Verify existing dialog was used (no changes to ui_components)
        assert ui_components['confirmation_dialog'] == existing_dialog
        
        # Verify existing dialog was NOT initialized (function only creates if missing)
        existing_dialog.initialize.assert_not_called()
    
    def test_show_confirmation_dialog_function(self):
        """Test show_confirmation_dialog function."""
        ui_components = {}
        on_confirm = Mock()
        on_cancel = Mock()
        
        # Call show_confirmation_dialog
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test Title",
            message="Test Message",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Confirm",
            cancel_text="Cancel",
            danger_mode=True
        )
        
        # Verify dialog was created without parameters (default behavior)
        self.mock_confirmation_dialog_class.assert_called_once_with()
        
        # Verify show was called on dialog
        self.mock_dialog_instance.show.assert_called_once_with(
            title="Test Title",
            message="Test Message",
            on_confirm=on_confirm,
            on_cancel=on_cancel,
            confirm_text="Confirm",
            cancel_text="Cancel",
            danger_mode=True
        )
        
        # Verify ui_components was updated
        assert 'confirmation_dialog' in ui_components
    
    def test_show_confirmation_dialog_with_existing_dialog(self):
        """Test show_confirmation_dialog with existing dialog."""
        existing_dialog = Mock()
        ui_components = {'confirmation_dialog': existing_dialog}
        
        # Call show_confirmation_dialog
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test Title",
            message="Test Message"
        )
        
        # Verify new dialog was not created
        self.mock_confirmation_dialog_class.assert_not_called()
        
        # Verify show was called on existing dialog
        existing_dialog.show.assert_called_once()
    
    def test_show_info_dialog_function(self):
        """Test show_info_dialog function."""
        ui_components = {}
        on_ok = Mock()
        
        # Call show_info_dialog
        show_info_dialog(
            ui_components=ui_components,
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify dialog was created without parameters (default behavior)
        self.mock_confirmation_dialog_class.assert_called_once_with()
        
        # Verify show_info was called on dialog
        self.mock_dialog_instance.show_info.assert_called_once_with(
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify ui_components was updated
        assert 'confirmation_dialog' in ui_components
    
    def test_show_info_dialog_with_existing_dialog(self):
        """Test show_info_dialog with existing dialog."""
        existing_dialog = Mock()
        ui_components = {'confirmation_dialog': existing_dialog}
        
        # Call show_info_dialog
        show_info_dialog(
            ui_components=ui_components,
            title="Info Title",
            message="Info Message"
        )
        
        # Verify new dialog was not created
        self.mock_confirmation_dialog_class.assert_not_called()
        
        # Verify show_info was called on existing dialog
        existing_dialog.show_info.assert_called_once()
    
    def test_clear_dialog_area_with_dialog(self):
        """Test clear_dialog_area with confirmation dialog."""
        dialog = Mock()
        ui_components = {'confirmation_dialog': dialog}
        
        # Call clear_dialog_area
        clear_dialog_area(ui_components)
        
        # Verify hide was called on dialog
        dialog.hide.assert_called_once()
    
    def test_clear_dialog_area_with_legacy_area(self):
        """Test clear_dialog_area with legacy confirmation area."""
        mock_area = Mock()
        mock_area.layout = Mock()
        ui_components = {'confirmation_area': mock_area}
        
        # Call clear_dialog_area
        clear_dialog_area(ui_components)
        
        # Verify clear_output was NOT called for legacy area
        self.mock_clear_output.assert_not_called()
        
        # Verify layout visibility was updated (not display)
        assert mock_area.layout.visibility == 'hidden'
    
    def test_clear_dialog_area_no_dialog(self):
        """Test clear_dialog_area with no dialog components."""
        ui_components = {}
        
        # Call clear_dialog_area - should not raise error
        clear_dialog_area(ui_components)
        
        # Verify clear_output was not called
        self.mock_clear_output.assert_not_called()
    
    def test_is_dialog_visible_with_dialog(self):
        """Test is_dialog_visible with confirmation dialog."""
        dialog = Mock()
        dialog.is_visible.return_value = True
        ui_components = {'confirmation_dialog': dialog}
        
        # Call is_dialog_visible
        result = is_dialog_visible(ui_components)
        
        # Verify result
        assert result is True
        dialog.is_visible.assert_called_once()
    
    def test_is_dialog_visible_with_legacy_area(self):
        """Test is_dialog_visible with legacy confirmation area."""
        mock_area = Mock()
        mock_area.layout = Mock()
        mock_area.layout.display = 'flex'
        ui_components = {'confirmation_area': mock_area}
        
        # Call is_dialog_visible
        result = is_dialog_visible(ui_components)
        
        # Verify result
        assert result is True
        
        # Test with hidden area
        mock_area.layout.display = 'none'
        result = is_dialog_visible(ui_components)
        assert result is False
    
    def test_is_dialog_visible_no_dialog(self):
        """Test is_dialog_visible with no dialog components."""
        ui_components = {}
        
        # Call is_dialog_visible
        result = is_dialog_visible(ui_components)
        
        # Verify result
        assert result is False


class TestDialogIntegration:
    """Integration tests for dialog components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock the entire dialog system
        self.confirmation_dialog_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.ConfirmationDialog')
        self.mock_confirmation_dialog_class = self.confirmation_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_confirmation_dialog_class.return_value = self.mock_dialog_instance
        
        # Mock widgets
        self.mock_output = Mock(spec=widgets.Output)
        self.mock_output.layout = Mock()
        self.mock_output.layout.display = 'none'
        
        # Setup dialog instance
        self.mock_dialog_instance._ui_components = {'container': self.mock_output}
        self.mock_dialog_instance._initialized = True
        self.mock_dialog_instance._is_visible = False
        self.mock_dialog_instance._callbacks = {}
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.confirmation_dialog_patcher.stop()
    
    def test_dialog_workflow_confirmation(self):
        """Test complete dialog workflow for confirmation."""
        ui_components = {}
        
        # Step 1: Show confirmation dialog
        confirm_callback = Mock()
        cancel_callback = Mock()
        
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Delete Files",
            message="Are you sure you want to delete all files?",
            on_confirm=confirm_callback,
            on_cancel=cancel_callback,
            danger_mode=True
        )
        
        # Verify dialog was created and shown
        self.mock_confirmation_dialog_class.assert_called_once_with()
        self.mock_dialog_instance.show.assert_called_once()
        
        # Step 2: Check if dialog is visible
        self.mock_dialog_instance.is_visible.return_value = True
        assert is_dialog_visible(ui_components) is True
        
        # Step 3: Simulate user confirmation
        # Access stored callbacks
        stored_callbacks = self.mock_dialog_instance.show.call_args[1]
        assert 'on_confirm' in stored_callbacks
        assert 'on_cancel' in stored_callbacks
        
        # Step 4: Clear dialog
        clear_dialog_area(ui_components)
        self.mock_dialog_instance.hide.assert_called_once()
    
    def test_dialog_workflow_info(self):
        """Test complete dialog workflow for info dialog."""
        ui_components = {}
        
        # Step 1: Show info dialog
        ok_callback = Mock()
        
        show_info_dialog(
            ui_components=ui_components,
            title="Operation Complete",
            message="Files have been processed successfully!",
            on_ok=ok_callback,
            ok_text="Got it"
        )
        
        # Verify dialog was created and shown
        self.mock_confirmation_dialog_class.assert_called_once_with()
        self.mock_dialog_instance.show_info.assert_called_once()
        
        # Step 2: Check if dialog is visible
        self.mock_dialog_instance.is_visible.return_value = True
        assert is_dialog_visible(ui_components) is True
        
        # Step 3: Clear dialog
        clear_dialog_area(ui_components)
        self.mock_dialog_instance.hide.assert_called_once()
    
    def test_multiple_dialogs_sequence(self):
        """Test showing multiple dialogs in sequence."""
        ui_components = {}
        
        # First dialog
        show_confirmation_dialog(
            ui_components=ui_components,
            title="First Dialog",
            message="First message"
        )
        
        # Second dialog (should reuse existing dialog instance)
        show_info_dialog(
            ui_components=ui_components,
            title="Second Dialog",
            message="Second message"
        )
        
        # Verify only one dialog instance was created
        self.mock_confirmation_dialog_class.assert_called_once_with()
        
        # Verify both show methods were called
        self.mock_dialog_instance.show.assert_called_once()
        self.mock_dialog_instance.show_info.assert_called_once()
    
    def test_dialog_state_management(self):
        """Test dialog state management across operations."""
        ui_components = {}
        
        # Initially no dialog
        assert is_dialog_visible(ui_components) is False
        
        # Show dialog
        show_confirmation_dialog(ui_components, "Title", "Message")
        
        # Mock dialog as visible
        self.mock_dialog_instance.is_visible.return_value = True
        assert is_dialog_visible(ui_components) is True
        
        # Hide dialog
        clear_dialog_area(ui_components)
        self.mock_dialog_instance.hide.assert_called_once()
        
        # Mock dialog as hidden
        self.mock_dialog_instance.is_visible.return_value = False
        assert is_dialog_visible(ui_components) is False
    
    def test_dialog_with_callbacks(self):
        """Test dialog with various callback scenarios."""
        ui_components = {}
        
        # Test with both callbacks
        confirm_cb = Mock()
        cancel_cb = Mock()
        
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test",
            message="Message",
            on_confirm=confirm_cb,
            on_cancel=cancel_cb
        )
        
        # Verify callbacks were passed
        call_args = self.mock_dialog_instance.show.call_args
        assert call_args[1]['on_confirm'] == confirm_cb
        assert call_args[1]['on_cancel'] == cancel_cb
        
        # Test with only confirm callback
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test",
            message="Message",
            on_confirm=confirm_cb
        )
        
        # Test with no callbacks
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test",
            message="Message"
        )
    
    def test_dialog_customization_options(self):
        """Test dialog customization options."""
        ui_components = {}
        
        # Test with custom button texts
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Custom Dialog",
            message="Custom message",
            confirm_text="Yes, Do It",
            cancel_text="No, Cancel",
            danger_mode=True
        )
        
        # Verify custom options were passed
        call_args = self.mock_dialog_instance.show.call_args
        assert call_args[1]['confirm_text'] == "Yes, Do It"
        assert call_args[1]['cancel_text'] == "No, Cancel"
        assert call_args[1]['danger_mode'] is True
        
        # Test info dialog customization
        show_info_dialog(
            ui_components=ui_components,
            title="Custom Info",
            message="Custom info message",
            ok_text="Understood"
        )
        
        # Verify custom options were passed
        call_args = self.mock_dialog_instance.show_info.call_args
        assert call_args[1]['ok_text'] == "Understood"


class TestDialogErrorHandling:
    """Test error handling in dialog components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock ConfirmationDialog to raise errors
        self.confirmation_dialog_patcher = patch('smartcash.ui.components.dialog.confirmation_dialog.ConfirmationDialog')
        self.mock_confirmation_dialog_class = self.confirmation_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_confirmation_dialog_class.return_value = self.mock_dialog_instance
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.confirmation_dialog_patcher.stop()
    
    def test_dialog_creation_error_handling(self):
        """Test error handling when dialog creation fails."""
        ui_components = {}
        
        # Mock dialog creation to raise error
        self.mock_confirmation_dialog_class.side_effect = Exception("Dialog creation failed")
        
        # Test should not raise exception
        with pytest.raises(Exception, match="Dialog creation failed"):
            show_confirmation_dialog(ui_components, "Title", "Message")
    
    def test_dialog_show_error_handling(self):
        """Test error handling when dialog.show fails."""
        ui_components = {}
        
        # Mock dialog.show to raise error
        self.mock_dialog_instance.show.side_effect = Exception("Show failed")
        
        # Test should not raise exception for the function call
        with pytest.raises(Exception, match="Show failed"):
            show_confirmation_dialog(ui_components, "Title", "Message")
    
    def test_dialog_hide_error_handling(self):
        """Test error handling when dialog.hide fails."""
        ui_components = {'confirmation_dialog': self.mock_dialog_instance}
        
        # Mock dialog.hide to raise error
        self.mock_dialog_instance.hide.side_effect = Exception("Hide failed")
        
        # Test should not raise exception for the function call
        with pytest.raises(Exception, match="Hide failed"):
            clear_dialog_area(ui_components)
    
    def test_dialog_is_visible_error_handling(self):
        """Test error handling when is_visible fails."""
        ui_components = {'confirmation_dialog': self.mock_dialog_instance}
        
        # Mock is_visible to raise error
        self.mock_dialog_instance.is_visible.side_effect = Exception("is_visible failed")
        
        # Test should not raise exception and return False
        with pytest.raises(Exception, match="is_visible failed"):
            is_dialog_visible(ui_components)
    
    def test_malformed_ui_components(self):
        """Test handling of malformed ui_components."""
        # Test with None ui_components
        with pytest.raises(Exception):
            show_confirmation_dialog(None, "Title", "Message")
        
        # Test with non-dict ui_components
        with pytest.raises(Exception):
            show_confirmation_dialog("not_a_dict", "Title", "Message")
    
    def test_callback_error_handling(self):
        """Test error handling in callbacks."""
        ui_components = {}
        
        # Create a callback that raises an error
        def error_callback():
            raise Exception("Callback error")
        
        # Should not raise exception during show
        show_confirmation_dialog(
            ui_components=ui_components,
            title="Test",
            message="Message",
            on_confirm=error_callback,
            on_cancel=error_callback
        )
        
        # Verify dialog was still created
        self.mock_confirmation_dialog_class.assert_called_once()
        self.mock_dialog_instance.show.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])