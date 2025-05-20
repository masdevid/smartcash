"""
File: smartcash/ui/dataset/download/tests/test_cleanup_handler.py
Deskripsi: Test suite untuk cleanup_handler.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
from typing import Dict, Any
import tempfile
from pathlib import Path
from ipywidgets import Layout

# Tambahkan path ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from smartcash.ui.dataset.download.handlers.cleanup_handler import (
    handle_cleanup_button_click,
    confirm_cleanup,
    execute_cleanup,
    _reset_progress_bar,
    _show_progress,
    _update_progress,
    _disable_buttons,
    _reset_ui_after_cleanup
)

class TestCleanupHandler(unittest.TestCase):
    """Test suite untuk cleanup_handler.py"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock UI components
        self.ui_components = {
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='/tmp/test-output'),
            'log_output': MagicMock(),
            'progress_bar': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'current_progress': MagicMock(),
            'progress_container': MagicMock(layout=MagicMock()),
            'log_accordion': MagicMock(selected_index=0),
            'confirmation_area': MagicMock(clear_output=MagicMock()),
            'status_panel': MagicMock(),
            'download_button': MagicMock(disabled=False),
            'check_button': MagicMock(disabled=False),
            'reset_button': MagicMock(disabled=False),
            'cleanup_button': MagicMock(disabled=False),
            'cleanup_running': False
        }
        
        # Setup mock untuk progress bar dan labels
        for key in ['progress_bar', 'overall_label', 'step_label', 'current_progress']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].value = 0
            self.ui_components[key].description = ""
        
        # Setup mock untuk buttons
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].layout.display = 'inline-block'
            self.ui_components[key].disabled = False
    
    def test_reset_progress_bar(self):
        """Test _reset_progress_bar function"""
        # Call function
        _reset_progress_bar(self.ui_components)
        
        # Verify progress bar reset
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        self.assertEqual(self.ui_components['progress_bar'].description, "Progress: 0%")
        self.assertEqual(self.ui_components['progress_bar'].layout.visibility, 'hidden')
        
        # Verify labels reset
        self.assertEqual(self.ui_components['overall_label'].value, "")
        self.assertEqual(self.ui_components['overall_label'].layout.visibility, 'hidden')
        self.assertEqual(self.ui_components['step_label'].value, "")
        self.assertEqual(self.ui_components['step_label'].layout.visibility, 'hidden')
        self.assertEqual(self.ui_components['current_progress'].value, 0)
        self.assertEqual(self.ui_components['current_progress'].description, "Step 0/0")
        self.assertEqual(self.ui_components['current_progress'].layout.visibility, 'hidden')
    
    def test_show_progress(self):
        """Test _show_progress function"""
        # Call function
        _show_progress(self.ui_components, "Test message")
        
        # Verify progress container visibility
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Verify cleanup_running flag
        self.assertTrue(self.ui_components['cleanup_running'])
        
        # Verify log accordion
        self.assertEqual(self.ui_components['log_accordion'].selected_index, 0)
    
    def test_update_progress(self):
        """Test _update_progress function"""
        # Test with integer value
        _update_progress(self.ui_components, 50, "50% complete")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Test with float value
        _update_progress(self.ui_components, 75.5, "75.5% complete")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        
        # Test with invalid value
        _update_progress(self.ui_components, "invalid", "Invalid progress")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
    
    def test_disable_buttons(self):
        """Test _disable_buttons function"""
        # Test disable buttons
        _disable_buttons(self.ui_components, True)
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertTrue(self.ui_components[key].disabled)
            if key in ['reset_button', 'cleanup_button']:
                self.assertEqual(self.ui_components[key].layout.display, 'none')
        
        # Test enable buttons
        _disable_buttons(self.ui_components, False)
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertFalse(self.ui_components[key].disabled)
            self.assertEqual(self.ui_components[key].layout.display, 'inline-block')
    
    def test_reset_ui_after_cleanup(self):
        """Test _reset_ui_after_cleanup function"""
        # Setup initial state
        self.ui_components['cleanup_running'] = True
        self.ui_components['progress_bar'].value = 100
        
        # Call function
        _reset_ui_after_cleanup(self.ui_components)
        
        # Verify buttons are enabled
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertFalse(self.ui_components[key].disabled)
            self.assertEqual(self.ui_components[key].layout.display, 'inline-block')
        
        # Verify progress bar is reset
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        
        # Verify confirmation area is cleared
        self.ui_components['confirmation_area'].clear_output.assert_called_once()
        
        # Verify cleanup_running flag is reset
        self.assertFalse(self.ui_components['cleanup_running'])
    
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.confirm_cleanup')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_handle_cleanup_button_click(self, mock_notify_log, mock_confirm_cleanup):
        """Test handle_cleanup_button_click function"""
        # Setup mock untuk confirm_cleanup
        mock_confirm_cleanup.return_value = True
        
        # Call function
        handle_cleanup_button_click(self.ui_components, self.ui_components['cleanup_button'])
        
        # Verify button is disabled during operation
        self.ui_components['cleanup_button'].disabled = False  # Button would be re-enabled after async operation
        
        # Verify confirm_cleanup is called
        mock_confirm_cleanup.assert_called_once_with(
            self.ui_components, 
            self.ui_components['output_dir'].value,
            self.ui_components['cleanup_button']
        )
    
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.confirm_cleanup')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_handle_cleanup_button_click_no_output_dir(self, mock_notify_log, mock_confirm_cleanup):
        """Test handle_cleanup_button_click with no output_dir"""
        # Setup mock untuk output_dir
        self.ui_components['output_dir'].value = ""
        
        # Call function
        handle_cleanup_button_click(self.ui_components, self.ui_components['cleanup_button'])
        
        # Verify confirm_cleanup is not called
        mock_confirm_cleanup.assert_not_called()
        
        # Verify notify_log is called with error
        mock_notify_log.assert_called_with(
            sender=self.ui_components,
            message="Direktori output tidak ditentukan",
            level="error"
        )
    
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.confirm_cleanup')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_handle_cleanup_button_click_error(self, mock_notify_log, mock_confirm_cleanup):
        """Test handle_cleanup_button_click with error"""
        # Setup mock untuk confirm_cleanup
        mock_confirm_cleanup.side_effect = Exception("Test error")
        
        # Call function
        handle_cleanup_button_click(self.ui_components, self.ui_components['cleanup_button'])
        
        # Verify notify_log is called with error
        mock_notify_log.assert_called_with(
            sender=self.ui_components,
            message=unittest.mock.ANY,
            level="error"
        )
        
        # Verify button is re-enabled
        self.assertFalse(self.ui_components['cleanup_button'].disabled)
    
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.create_confirmation_dialog')
    @patch('os.path.exists')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_confirm_cleanup(self, mock_notify_log, mock_exists, mock_create_dialog):
        """Test confirm_cleanup function"""
        # Setup mock untuk os.path.exists
        mock_exists.return_value = True
        
        # Setup mock untuk create_confirmation_dialog
        mock_dialog = MagicMock()
        mock_create_dialog.return_value = mock_dialog
        
        # Call function
        result = confirm_cleanup(self.ui_components, "/tmp/test-output", self.ui_components['cleanup_button'])
        
        # Verify result
        self.assertTrue(result)
        
        # Verify dialog is created and displayed
        mock_create_dialog.assert_called_once()
        
        # Verify confirmation area is cleared
        self.ui_components['confirmation_area'].clear_output.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.create_confirmation_dialog')
    @patch('os.path.exists')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_confirm_cleanup_nonexistent_dir(self, mock_notify_log, mock_exists, mock_create_dialog):
        """Test confirm_cleanup with nonexistent directory"""
        # Setup mock untuk os.path.exists
        mock_exists.return_value = False
        
        # Call function
        result = confirm_cleanup(self.ui_components, "/tmp/nonexistent", self.ui_components['cleanup_button'])
        
        # Verify result
        self.assertFalse(result)
        
        # Verify dialog is not created
        mock_create_dialog.assert_not_called()
        
        # Verify notify_log is called with error
        mock_notify_log.assert_called_with(
            sender=self.ui_components,
            message=unittest.mock.ANY,
            level="error"
        )
    
    @patch('smartcash.dataset.manager.DatasetManager.cleanup_dataset')
    @patch('threading.Thread')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_execute_cleanup(self, mock_notify_log, mock_thread, mock_cleanup_dataset):
        """Test execute_cleanup function"""
        # Setup mock untuk cleanup_dataset
        mock_cleanup_dataset.return_value = {
            "status": "success",
            "message": "Dataset berhasil dihapus"
        }
        
        # Call function directly (tidak melalui thread)
        execute_cleanup(self.ui_components, "/tmp/test-output")
        
        # Verify cleanup_dataset is called
        mock_cleanup_dataset.assert_called_once_with(
            "/tmp/test-output",
            backup_before_delete=True,
            show_progress=True
        )
        
        # Verify UI is reset
        self.assertFalse(self.ui_components['cleanup_running'])
        
        # Verify notify_log is called for success message
        mock_notify_log.assert_any_call(
            sender=self.ui_components,
            message=unittest.mock.ANY,
            level="success"
        )
    
    @patch('smartcash.dataset.manager.DatasetManager.cleanup_dataset')
    @patch('smartcash.ui.dataset.download.handlers.cleanup_handler.notify_log')
    def test_execute_cleanup_error(self, mock_notify_log, mock_cleanup_dataset):
        """Test execute_cleanup with error"""
        # Setup mock untuk cleanup_dataset
        mock_cleanup_dataset.side_effect = Exception("Test error")
        
        # Call function
        execute_cleanup(self.ui_components, "/tmp/test-output")
        
        # Verify notify_log is called with error
        mock_notify_log.assert_called_with(
            sender=self.ui_components,
            message=unittest.mock.ANY,
            level="error"
        )
        
        # Verify UI is reset
        self.assertFalse(self.ui_components['cleanup_running'])

if __name__ == '__main__':
    unittest.main() 