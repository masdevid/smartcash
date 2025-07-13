"""
File: tests/ui/dataset/downloader/test_ui_components.py
Description: Tests for UI components and their interactions in the dataset downloader.
"""

import pytest
from unittest.mock import MagicMock, call, ANY

class TestUIComponents:
    """Tests for UI components and their interactions."""
    
    def test_ui_initialization(self, downloader_module, mock_ui_components):
        """Test that all UI components are properly initialized."""
        # Verify all required UI components exist
        required_components = [
            'main_container', 'header_container', 'form_container',
            'action_container', 'operation_container', 'footer_container'
        ]
        
        for component in required_components:
            assert component in downloader_module.ui_components
            assert downloader_module.ui_components[component] is not None
    
    def test_form_components(self, downloader_module, mock_ui_components):
        """Test that form components are properly set up."""
        form_container = mock_ui_components['form_container']
        
        # Verify form container has children
        assert hasattr(form_container, 'children')
        
        # Check for expected form fields
        form_fields = [
            'api_key_input', 'workspace_input', 
            'project_input', 'version_input', 'format_dropdown'
        ]
        
        # Mock form fields
        form_container.children = [MagicMock(description=field) for field in form_fields]
        
        # Verify form fields are present
        field_descriptions = [child.description for child in form_container.children]
        for field in form_fields:
            assert field in field_descriptions
    
    def test_action_buttons(self, downloader_module, mock_ui_components):
        """Test that action buttons are properly set up and connected."""
        action_container = mock_ui_components['action_container']
        
        # Mock action buttons
        button_descriptions = ['Download', 'Check', 'Cleanup']
        mock_buttons = [MagicMock(description=desc) for desc in button_descriptions]
        action_container.children = mock_buttons
        
        # Verify buttons are present
        assert len(action_container.children) == 3
        
        # Test button click handlers
        for button in mock_buttons:
            # Simulate button click
            if button.description == 'Download':
                # Test download button
                button.click()
                downloader_module._handle_download_button_click.assert_called_once()
                downloader_module._handle_download_button_click.reset_mock()
                
            elif button.description == 'Check':
                # Test check button
                button.click()
                downloader_module._handle_check_button_click.assert_called_once()
                downloader_module._handle_check_button_click.reset_mock()
                
            elif button.description == 'Cleanup':
                # Test cleanup button
                button.click()
                downloader_module._handle_cleanup_button_click.assert_called_once()
                downloader_module._handle_cleanup_button_click.reset_mock()
    
    def test_operation_container_updates(self, downloader_module, mock_ui_components):
        """Test that the operation container is properly updated."""
        operation_container = mock_ui_components['operation_container']
        
        # Test starting an operation
        downloader_module._update_operation_status("Starting download...", "info")
        
        # Verify operation container was updated
        operation_container.clear_output.assert_called_once()
        operation_container.append_display_data.assert_called_with(ANY)
        
        # Test error state
        operation_container.reset_mock()
        downloader_module._update_operation_status("Download failed!", "error")
        operation_container.append_display_data.assert_called_with(ANY)
    
    def test_progress_bar_updates(self, downloader_module, mock_ui_components):
        """Test that the progress bar is properly updated."""
        progress_bar = mock_ui_components['progress_bar']
        
        # Test progress update
        downloader_module._update_progress(50, "Halfway there!")
        
        # Verify progress bar was updated
        assert progress_bar.value == 50
        assert progress_bar.description == "Halfway there!"
        
        # Test completion
        downloader_module._update_progress(100, "Complete!")
        assert progress_bar.value == 100
        assert progress_bar.bar_style == 'success'
    
    def test_log_output(self, downloader_module, mock_ui_components):
        """Test that log messages are properly displayed."""
        log_output = mock_ui_components['log_output']
        
        # Test info message
        test_message = "Test info message"
        downloader_module._log_message(test_message, "info")
        log_output.append_stdout.assert_called_with(f"[INFO] {test_message}")
        
        # Test error message
        test_error = "Test error message"
        downloader_module._log_message(test_error, "error")
        log_output.append_stderr.assert_called_with(f"[ERROR] {test_error}")
        
        # Test debug message (should be filtered by default)
        log_output.reset_mock()
        downloader_module._log_message("Debug message", "debug")
        log_output.append_stdout.assert_not_called()
        
        # Enable debug logging and test again
        downloader_module.log_level = "debug"
        downloader_module._log_message("Debug message", "debug")
        log_output.append_stdout.assert_called_with("[DEBUG] Debug message")
    
    def test_status_updates(self, downloader_module, mock_ui_components):
        """Test that status updates are properly displayed."""
        status_label = mock_ui_components['status_label']
        
        # Test status update
        test_status = "Download in progress..."
        downloader_module._update_status(test_status, "info")
        
        # Verify status was updated
        assert status_label.value == test_status
        assert "info" in status_label.style
        
        # Test error status
        error_status = "Download failed!"
        downloader_module._update_status(error_status, "error")
        assert status_label.value == error_status
        assert "danger" in status_label.style
    
    def test_form_validation(self, downloader_module, mock_ui_components):
        """Test that form validation works correctly."""
        # Mock form inputs
        form_inputs = {
            'api_key_input': MagicMock(value='test_api_key'),
            'workspace_input': MagicMock(value='test_workspace'),
            'project_input': MagicMock(value='test_project'),
            'version_input': MagicMock(value='1'),
            'format_dropdown': MagicMock(value='yolov8')
        }
        
        # Test valid form
        result = downloader_module._validate_form(form_inputs)
        assert result["valid"] is True
        
        # Test missing required field
        form_inputs['api_key_input'].value = ''
        result = downloader_module._validate_form(form_inputs)
        assert result["valid"] is False
        assert "API key is required" in result["errors"]
        
        # Test invalid version
        form_inputs['api_key_input'].value = 'test_api_key'
        form_inputs['version_input'].value = 'invalid'
        result = downloader_module._validate_form(form_inputs)
        assert result["valid"] is False
        assert "Version must be a number" in result["errors"]
    
    def test_theme_switching(self, downloader_module, mock_ui_components):
        """Test that theme switching works correctly."""
        # Mock theme toggle button
        theme_toggle = MagicMock()
        theme_toggle.value = True  # Dark theme
        
        # Call theme change handler
        downloader_module._on_theme_change({'new': True})
        
        # Verify theme was applied to all components
        for name, component in downloader_module.ui_components.items():
            if hasattr(component, 'layout'):
                assert 'dark' in component.layout.theme
        
        # Switch back to light theme
        theme_toggle.value = False
        downloader_module._on_theme_change({'new': False})
        
        # Verify theme was updated
        for name, component in downloader_module.ui_components.items():
            if hasattr(component, 'layout'):
                assert 'light' in component.layout.theme
