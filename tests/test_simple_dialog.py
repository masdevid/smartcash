"""
Tests for simple_dialog.py module.

This module tests the simplified dialog components that use basic hide/show
functionality without complex animations, JavaScript, or CSS.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
import ipywidgets as widgets
from typing import Dict, Any, Optional, Callable

# Import the components to test
from smartcash.ui.components.dialog.simple_dialog import (
    SimpleDialog,
    create_simple_dialog,
    show_confirmation_dialog,
    show_info_dialog,
    show_success_dialog,
    show_warning_dialog,
    show_error_dialog
)


class TestSimpleDialog:
    """Test cases for SimpleDialog class."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock the BaseUIComponent
        self.base_ui_component_patcher = patch('smartcash.ui.components.dialog.simple_dialog.BaseUIComponent')
        self.mock_base_ui_component = self.base_ui_component_patcher.start()
        
        # Mock widgets
        self.mock_vbox = Mock(spec=widgets.VBox)
        self.mock_hbox = Mock(spec=widgets.HBox)
        self.mock_html = Mock(spec=widgets.HTML)
        self.mock_button = Mock(spec=widgets.Button)
        
        # Mock widget classes
        self.vbox_patcher = patch('smartcash.ui.components.dialog.simple_dialog.widgets.VBox')
        self.hbox_patcher = patch('smartcash.ui.components.dialog.simple_dialog.widgets.HBox')
        self.html_patcher = patch('smartcash.ui.components.dialog.simple_dialog.widgets.HTML')
        self.button_patcher = patch('smartcash.ui.components.dialog.simple_dialog.widgets.Button')
        self.layout_patcher = patch('smartcash.ui.components.dialog.simple_dialog.widgets.Layout')
        
        self.mock_vbox_class = self.vbox_patcher.start()
        self.mock_hbox_class = self.hbox_patcher.start()
        self.mock_html_class = self.html_patcher.start()
        self.mock_button_class = self.button_patcher.start()
        self.mock_layout_class = self.layout_patcher.start()
        
        # Setup return values
        self.mock_vbox_class.return_value = self.mock_vbox
        self.mock_hbox_class.return_value = self.mock_hbox
        self.mock_html_class.return_value = self.mock_html
        self.mock_button_class.return_value = self.mock_button
        self.mock_layout_class.return_value = Mock()
        
        # Setup mock attributes
        self.mock_vbox.layout = Mock()
        self.mock_vbox.layout.display = 'none'
        self.mock_html.value = ""
        self.mock_hbox.children = []
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.base_ui_component_patcher.stop()
        self.vbox_patcher.stop()
        self.hbox_patcher.stop()
        self.html_patcher.stop()
        self.button_patcher.stop()
        self.layout_patcher.stop()
    
    def test_simple_dialog_initialization(self):
        """Test SimpleDialog initialization."""
        dialog = SimpleDialog("test_dialog")
        
        assert dialog.component_name == "test_dialog"
        assert dialog._is_visible is False
        assert dialog._callbacks == {}
        assert dialog._current_buttons == {}
    
    def test_simple_dialog_initialization_with_kwargs(self):
        """Test SimpleDialog initialization with kwargs."""
        dialog = SimpleDialog("test_dialog", logger=Mock())
        
        assert dialog.component_name == "test_dialog"
    
    def test_create_ui_components(self):
        """Test _create_ui_components method."""
        dialog = SimpleDialog("test_dialog")
        dialog._ui_components = {}
        
        # Call _create_ui_components
        dialog._create_ui_components()
        
        # Verify VBox was created for container
        self.mock_vbox_class.assert_called()
        
        # Verify HTML was created for content
        self.mock_html_class.assert_called()
        
        # Verify HBox was created for button area
        self.mock_hbox_class.assert_called()
        
        # Verify components were stored
        assert 'container' in dialog._ui_components
        assert 'content' in dialog._ui_components
        assert 'button_area' in dialog._ui_components
    
    def test_show_confirmation_dialog(self):
        """Test show_confirmation method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Mock callbacks
        on_confirm = Mock()
        on_cancel = Mock()
        
        # Call show_confirmation
        dialog.show_confirmation(
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
        
        # Verify content was updated
        assert "Test Title" in self.mock_html.value
        assert "Test Message" in self.mock_html.value
        
        # Verify buttons were created
        assert self.mock_button_class.call_count >= 2  # Confirm and Cancel buttons
        
        # Verify dialog is visible
        assert dialog._is_visible is True
        assert self.mock_vbox.layout.display == 'block'
    
    def test_show_confirmation_dialog_danger_mode(self):
        """Test show_confirmation with danger mode."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Call show_confirmation with danger mode
        dialog.show_confirmation(
            title="Danger Title",
            message="Danger Message",
            danger_mode=True
        )
        
        # Verify content contains danger styling
        assert "#dc3545" in self.mock_html.value  # Danger color
        
        # Verify dialog is visible
        assert dialog._is_visible is True
    
    def test_show_info_dialog(self):
        """Test show_info method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Mock callback
        on_ok = Mock()
        
        # Call show_info
        dialog.show_info(
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="info"
        )
        
        # Verify callback was stored
        assert dialog._callbacks['ok'] == on_ok
        
        # Verify content was updated
        assert "Info Title" in self.mock_html.value
        assert "Info Message" in self.mock_html.value
        
        # Verify button was created
        self.mock_button_class.assert_called()
        
        # Verify dialog is visible
        assert dialog._is_visible is True
    
    def test_show_info_dialog_different_types(self):
        """Test show_info with different info types."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Test different info types
        info_types = ['success', 'warning', 'error']
        expected_colors = ['#28a745', '#ffc107', '#dc3545']
        
        for info_type, expected_color in zip(info_types, expected_colors):
            dialog.show_info(
                title=f"{info_type.title()} Title",
                message=f"{info_type.title()} Message",
                info_type=info_type
            )
            
            # Verify appropriate color is used
            assert expected_color in self.mock_html.value
    
    def test_handle_confirm(self):
        """Test _handle_confirm method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        
        # Mock callback
        confirm_callback = Mock()
        dialog._callbacks['confirm'] = confirm_callback
        
        # Call _handle_confirm
        dialog._handle_confirm(Mock())
        
        # Verify callback was called
        confirm_callback.assert_called_once()
        
        # Verify dialog was hidden
        assert dialog._is_visible is False
    
    def test_handle_cancel(self):
        """Test _handle_cancel method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        
        # Mock callback
        cancel_callback = Mock()
        dialog._callbacks['cancel'] = cancel_callback
        
        # Call _handle_cancel
        dialog._handle_cancel(Mock())
        
        # Verify callback was called
        cancel_callback.assert_called_once()
        
        # Verify dialog was hidden
        assert dialog._is_visible is False
    
    def test_handle_ok(self):
        """Test _handle_ok method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        
        # Mock callback
        ok_callback = Mock()
        dialog._callbacks['ok'] = ok_callback
        
        # Call _handle_ok
        dialog._handle_ok(Mock())
        
        # Verify callback was called
        ok_callback.assert_called_once()
        
        # Verify dialog was hidden
        assert dialog._is_visible is False
    
    def test_handle_callback_with_exception(self):
        """Test callback handling with exception."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        
        # Mock callback that raises exception
        def error_callback():
            raise Exception("Test error")
        
        dialog._callbacks['confirm'] = error_callback
        
        # Mock print to capture output
        with patch('builtins.print') as mock_print:
            dialog._handle_confirm(Mock())
            
            # Verify error was printed
            mock_print.assert_called_once()
            assert "Error in confirm callback" in str(mock_print.call_args)
        
        # Verify dialog was still hidden
        assert dialog._is_visible is False
    
    def test_hide_dialog(self):
        """Test hide method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        dialog._is_visible = True
        dialog._callbacks = {'confirm': Mock()}
        dialog._current_buttons = {'confirm': Mock()}
        
        # Call hide
        dialog.hide()
        
        # Verify container was hidden
        assert self.mock_vbox.layout.display == 'none'
        
        # Verify state was cleaned up
        assert dialog._is_visible is False
        assert dialog._callbacks == {}
        assert dialog._current_buttons == {}
    
    def test_hide_dialog_not_initialized(self):
        """Test hide method when dialog is not initialized."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = False
        
        # Call hide - should return without error
        dialog.hide()
        
        # No assertions needed - just verify it doesn't crash
    
    def test_is_visible_method(self):
        """Test is_visible method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {'container': self.mock_vbox}
        
        # Test when dialog is hidden
        self.mock_vbox.layout.display = 'none'
        assert dialog.is_visible() is False
        
        # Test when dialog is visible
        self.mock_vbox.layout.display = 'block'
        assert dialog.is_visible() is True
        
        # Test when not initialized
        dialog._initialized = False
        assert dialog.is_visible() is False
    
    def test_is_visible_fallback_to_internal_state(self):
        """Test is_visible method fallback to internal state."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {}  # No container
        
        # Test internal state
        dialog._is_visible = True
        assert dialog.is_visible() is True
        
        dialog._is_visible = False
        assert dialog.is_visible() is False
    
    def test_clear_dialog(self):
        """Test clear method."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = True
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Set some content
        self.mock_html.value = "Some content"
        self.mock_hbox.children = [Mock()]
        
        # Call clear
        dialog.clear()
        
        # Verify content was cleared
        assert self.mock_html.value == ""
        assert self.mock_hbox.children == []
        
        # Verify dialog was hidden
        assert dialog._is_visible is False
    
    def test_show_initializes_if_not_initialized(self):
        """Test that show methods initialize dialog if not initialized."""
        dialog = SimpleDialog("test_dialog")
        dialog._initialized = False
        dialog.initialize = Mock()
        
        # Mock UI components after initialization
        dialog._ui_components = {
            'container': self.mock_vbox,
            'content': self.mock_html,
            'button_area': self.mock_hbox
        }
        
        # Call show_confirmation
        dialog.show_confirmation("Title", "Message")
        
        # Verify initialize was called
        dialog.initialize.assert_called_once()


class TestSimpleDialogFactoryFunctions:
    """Test cases for simple dialog factory functions."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock SimpleDialog class
        self.simple_dialog_patcher = patch('smartcash.ui.components.dialog.simple_dialog.SimpleDialog')
        self.mock_simple_dialog_class = self.simple_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_simple_dialog_class.return_value = self.mock_dialog_instance
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.simple_dialog_patcher.stop()
    
    def test_create_simple_dialog(self):
        """Test create_simple_dialog function."""
        # Call create_simple_dialog
        result = create_simple_dialog("test_dialog")
        
        # Verify SimpleDialog was created
        self.mock_simple_dialog_class.assert_called_once_with("test_dialog")
        
        # Verify dialog was initialized
        self.mock_dialog_instance.initialize.assert_called_once()
        
        # Verify result
        assert result == self.mock_dialog_instance
    
    def test_create_simple_dialog_default_name(self):
        """Test create_simple_dialog with default name."""
        # Call create_simple_dialog with default name
        result = create_simple_dialog()
        
        # Verify SimpleDialog was created with default name
        self.mock_simple_dialog_class.assert_called_once_with("dialog")
    
    def test_show_confirmation_dialog_function(self):
        """Test show_confirmation_dialog function."""
        # Call show_confirmation_dialog
        show_confirmation_dialog(
            dialog=self.mock_dialog_instance,
            title="Test Title",
            message="Test Message",
            confirm_text="Confirm",
            cancel_text="Cancel",
            danger_mode=True
        )
        
        # Verify show_confirmation was called on dialog
        self.mock_dialog_instance.show_confirmation.assert_called_once_with(
            title="Test Title",
            message="Test Message",
            on_confirm=None,
            on_cancel=None,
            confirm_text="Confirm",
            cancel_text="Cancel",
            danger_mode=True
        )
    
    def test_show_info_dialog_function(self):
        """Test show_info_dialog function."""
        on_ok = Mock()
        
        # Call show_info_dialog
        show_info_dialog(
            dialog=self.mock_dialog_instance,
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="info"
        )
        
        # Verify show_info was called on dialog
        self.mock_dialog_instance.show_info.assert_called_once_with(
            title="Info Title",
            message="Info Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="info"
        )
    
    def test_show_success_dialog_function(self):
        """Test show_success_dialog function."""
        on_ok = Mock()
        
        # Call show_success_dialog
        show_success_dialog(
            dialog=self.mock_dialog_instance,
            title="Success Title",
            message="Success Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify show_info was called with success type
        self.mock_dialog_instance.show_info.assert_called_once_with(
            title="Success Title",
            message="Success Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="success"
        )
    
    def test_show_warning_dialog_function(self):
        """Test show_warning_dialog function."""
        on_ok = Mock()
        
        # Call show_warning_dialog
        show_warning_dialog(
            dialog=self.mock_dialog_instance,
            title="Warning Title",
            message="Warning Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify show_info was called with warning type
        self.mock_dialog_instance.show_info.assert_called_once_with(
            title="Warning Title",
            message="Warning Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="warning"
        )
    
    def test_show_error_dialog_function(self):
        """Test show_error_dialog function."""
        on_ok = Mock()
        
        # Call show_error_dialog
        show_error_dialog(
            dialog=self.mock_dialog_instance,
            title="Error Title",
            message="Error Message",
            on_ok=on_ok,
            ok_text="OK"
        )
        
        # Verify show_info was called with error type
        self.mock_dialog_instance.show_info.assert_called_once_with(
            title="Error Title",
            message="Error Message",
            on_ok=on_ok,
            ok_text="OK",
            info_type="error"
        )


class TestSimpleDialogIntegration:
    """Integration tests for simple dialog components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock SimpleDialog class
        self.simple_dialog_patcher = patch('smartcash.ui.components.dialog.simple_dialog.SimpleDialog')
        self.mock_simple_dialog_class = self.simple_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_simple_dialog_class.return_value = self.mock_dialog_instance
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.simple_dialog_patcher.stop()
    
    def test_dialog_workflow_confirmation(self):
        """Test complete dialog workflow for confirmation."""
        # Step 1: Create dialog
        dialog = create_simple_dialog("test_dialog")
        
        # Step 2: Show confirmation dialog
        confirm_callback = Mock()
        cancel_callback = Mock()
        
        show_confirmation_dialog(
            dialog=dialog,
            title="Delete Files",
            message="Are you sure you want to delete all files?",
            on_confirm=confirm_callback,
            on_cancel=cancel_callback,
            danger_mode=True
        )
        
        # Verify dialog was created and shown
        self.mock_simple_dialog_class.assert_called_once_with("test_dialog")
        self.mock_dialog_instance.initialize.assert_called_once()
        self.mock_dialog_instance.show_confirmation.assert_called_once()
        
        # Step 3: Check if dialog is visible
        self.mock_dialog_instance.is_visible.return_value = True
        assert dialog.is_visible() is True
        
        # Step 4: Hide dialog
        dialog.hide()
        self.mock_dialog_instance.hide.assert_called_once()
    
    def test_dialog_workflow_info(self):
        """Test complete dialog workflow for info dialog."""
        # Step 1: Create dialog
        dialog = create_simple_dialog("info_dialog")
        
        # Step 2: Show info dialog
        ok_callback = Mock()
        
        show_info_dialog(
            dialog=dialog,
            title="Operation Complete",
            message="Files have been processed successfully!",
            on_ok=ok_callback,
            ok_text="Got it",
            info_type="success"
        )
        
        # Verify dialog was created and shown
        self.mock_simple_dialog_class.assert_called_once_with("info_dialog")
        self.mock_dialog_instance.show_info.assert_called_once()
        
        # Step 3: Clear dialog
        dialog.clear()
        self.mock_dialog_instance.clear.assert_called_once()
    
    def test_multiple_dialog_types_sequence(self):
        """Test showing multiple dialog types in sequence."""
        dialog = create_simple_dialog("multi_dialog")
        
        # Show confirmation dialog
        show_confirmation_dialog(
            dialog=dialog,
            title="First Dialog",
            message="First message"
        )
        
        # Show success dialog
        show_success_dialog(
            dialog=dialog,
            title="Success",
            message="Operation completed"
        )
        
        # Show warning dialog
        show_warning_dialog(
            dialog=dialog,
            title="Warning",
            message="Please be careful"
        )
        
        # Show error dialog
        show_error_dialog(
            dialog=dialog,
            title="Error",
            message="Something went wrong"
        )
        
        # Verify all show methods were called
        self.mock_dialog_instance.show_confirmation.assert_called_once()
        assert self.mock_dialog_instance.show_info.call_count == 3
        
        # Verify different info types were used
        calls = self.mock_dialog_instance.show_info.call_args_list
        assert calls[0][1]['info_type'] == 'success'
        assert calls[1][1]['info_type'] == 'warning'
        assert calls[2][1]['info_type'] == 'error'
    
    def test_dialog_state_management(self):
        """Test dialog state management across operations."""
        dialog = create_simple_dialog("state_dialog")
        
        # Initially not visible
        self.mock_dialog_instance.is_visible.return_value = False
        assert dialog.is_visible() is False
        
        # Show dialog
        show_confirmation_dialog(dialog, "Title", "Message")
        
        # Mock dialog as visible
        self.mock_dialog_instance.is_visible.return_value = True
        assert dialog.is_visible() is True
        
        # Hide dialog
        dialog.hide()
        self.mock_dialog_instance.hide.assert_called_once()
        
        # Clear dialog
        dialog.clear()
        self.mock_dialog_instance.clear.assert_called_once()


class TestSimpleDialogErrorHandling:
    """Test error handling in simple dialog components."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Mock SimpleDialog to raise errors
        self.simple_dialog_patcher = patch('smartcash.ui.components.dialog.simple_dialog.SimpleDialog')
        self.mock_simple_dialog_class = self.simple_dialog_patcher.start()
        self.mock_dialog_instance = Mock()
        self.mock_simple_dialog_class.return_value = self.mock_dialog_instance
    
    def teardown_method(self):
        """Cleanup test fixtures."""
        self.simple_dialog_patcher.stop()
    
    def test_create_dialog_error_handling(self):
        """Test error handling when dialog creation fails."""
        # Mock dialog creation to raise error
        self.mock_simple_dialog_class.side_effect = Exception("Dialog creation failed")
        
        # Test should raise exception
        with pytest.raises(Exception, match="Dialog creation failed"):
            create_simple_dialog("test_dialog")
    
    def test_show_confirmation_error_handling(self):
        """Test error handling when show_confirmation fails."""
        dialog = create_simple_dialog("test_dialog")
        
        # Mock show_confirmation to raise error
        self.mock_dialog_instance.show_confirmation.side_effect = Exception("Show failed")
        
        # Test should raise exception
        with pytest.raises(Exception, match="Show failed"):
            show_confirmation_dialog(dialog, "Title", "Message")
    
    def test_show_info_error_handling(self):
        """Test error handling when show_info fails."""
        dialog = create_simple_dialog("test_dialog")
        
        # Mock show_info to raise error
        self.mock_dialog_instance.show_info.side_effect = Exception("Show info failed")
        
        # Test should raise exception
        with pytest.raises(Exception, match="Show info failed"):
            show_info_dialog(dialog, "Title", "Message")
    
    def test_hide_error_handling(self):
        """Test error handling when hide fails."""
        dialog = create_simple_dialog("test_dialog")
        
        # Mock hide to raise error
        self.mock_dialog_instance.hide.side_effect = Exception("Hide failed")
        
        # Test should raise exception
        with pytest.raises(Exception, match="Hide failed"):
            dialog.hide()
    
    def test_clear_error_handling(self):
        """Test error handling when clear fails."""
        dialog = create_simple_dialog("test_dialog")
        
        # Mock clear to raise error
        self.mock_dialog_instance.clear.side_effect = Exception("Clear failed")
        
        # Test should raise exception
        with pytest.raises(Exception, match="Clear failed"):
            dialog.clear()


if __name__ == "__main__":
    pytest.main([__file__])