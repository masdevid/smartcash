"""
File: smartcash/ui/dataset/download/tests/test_download_handler.py
Deskripsi: Test suite untuk download_handler.py
"""

import unittest
from unittest.mock import MagicMock, patch, call
import sys
import os
from typing import Dict, Any
from ipywidgets import Layout
import ipywidgets as widgets
import tempfile
from pathlib import Path

# Tambahkan path ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../')))

from smartcash.ui.dataset.download.handlers.download_handler import (
    handle_download_button_click,
    execute_download,
    _reset_progress_bar,
    _show_progress,
    _update_progress,
    _reset_ui_after_download,
    _disable_buttons
)

class TestDownloadHandler(unittest.TestCase):
    """Test suite untuk download_handler.py"""
    
    def setUp(self):
        """Setup test environment"""
        # Mock UI components
        self.ui_components = {
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='/tmp/test-output'),
            'validate_dataset': MagicMock(value=True),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='/tmp/test-backup'),
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
            'download_running': False,
            'logger': MagicMock()
        }
        
        # Setup mock untuk progress bar dan labels
        for key in ['progress_bar', 'overall_label', 'step_label', 'current_progress']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].value = 0
            self.ui_components[key].description = ""
        
        # Setup mock untuk buttons
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.ui_components[key].layout = Layout()
            self.ui_components[key].disabled = False
        
        # Import fungsi yang akan ditest
        self.handle_download_button_click = handle_download_button_click
        self.execute_download = execute_download
        self.reset_progress_bar = _reset_progress_bar
        self.show_progress = _show_progress
        self.update_progress = _update_progress
        self.reset_ui_after_download = _reset_ui_after_download
        self.disable_buttons = _disable_buttons
    
    def test_reset_progress_bar(self):
        """Test _reset_progress_bar function"""
        # Call function
        self.reset_progress_bar(self.ui_components)
        
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
        self.show_progress(self.ui_components, "Test message")
        
        # Verify progress container visibility
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Verify download_running flag
        self.assertTrue(self.ui_components['download_running'])
        
        # Verify log accordion
        self.assertEqual(self.ui_components['log_accordion'].selected_index, 0)
    
    def test_update_progress(self):
        """Test _update_progress function"""
        # Test with integer value
        self.update_progress(self.ui_components, 50, "50% complete")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        self.assertEqual(self.ui_components['progress_container'].layout.visibility, 'visible')
        
        # Test with float value
        self.update_progress(self.ui_components, 75.5, "75.5% complete")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
        
        # Test with invalid value
        self.update_progress(self.ui_components, "invalid", "Invalid progress")
        self.assertEqual(self.ui_components['progress_container'].layout.display, 'block')
    
    def test_disable_buttons(self):
        """Test _disable_buttons function"""
        # Test disable buttons
        self.disable_buttons(self.ui_components, True)
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertTrue(self.ui_components[key].disabled)
            if key in ['reset_button', 'cleanup_button']:
                self.assertEqual(self.ui_components[key].layout.display, 'none')
        
        # Test enable buttons
        self.disable_buttons(self.ui_components, False)
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertFalse(self.ui_components[key].disabled)
            self.assertEqual(self.ui_components[key].layout.display, 'inline-block')
    
    def test_reset_ui_after_download(self):
        """Test _reset_ui_after_download function"""
        # Setup initial state
        self.ui_components['download_running'] = True
        self.ui_components['progress_bar'].value = 100
        
        # Call function
        self.reset_ui_after_download(self.ui_components)
        
        # Verify buttons are enabled
        for key in ['download_button', 'check_button', 'reset_button', 'cleanup_button']:
            self.assertFalse(self.ui_components[key].disabled)
            self.assertEqual(self.ui_components[key].layout.display, 'inline-block')
        
        # Verify progress bar is reset
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        
        # Verify confirmation area is cleared
        self.ui_components['confirmation_area'].clear_output.assert_called_once()
        
        # Verify download_running flag is reset
        self.assertFalse(self.ui_components['download_running'])
    
    @patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download')
    def test_handle_download_button_click(self, mock_confirm_download):
        """Test handle_download_button_click function"""
        # Setup mock untuk confirm_download
        mock_confirm_download.return_value = True
        
        # Call function
        self.handle_download_button_click(self.ui_components, self.ui_components['download_button'])
        
        # Tombol download harus kembali enabled setelah proses (karena async/thread)
        self.assertFalse(self.ui_components['download_button'].disabled)
        
        # Verify confirm_download dipanggil
        mock_confirm_download.assert_called_once_with(self.ui_components)
        
        # Verify log output
        self.ui_components['log_output'].append_stdout.assert_called()
    
    @patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download')
    def test_handle_download_button_click_confirmation_dialog_displayed(self, mock_confirm_download):
        """Test bahwa dialog konfirmasi ditampilkan saat tombol download diklik."""
        # Setup mock untuk confirm_download
        mock_confirm_download.return_value = False  # Konfirmasi dibatalkan
        
        # Panggil fungsi
        self.handle_download_button_click(self.ui_components, self.ui_components['download_button'])
        
        # Verifikasi bahwa confirm_download dipanggil
        mock_confirm_download.assert_called_once_with(self.ui_components)
        
        # Verifikasi bahwa tombol download diaktifkan kembali
        self.assertFalse(self.ui_components['download_button'].disabled)
    
    @patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download')
    def test_handle_download_button_click_cancelled(self, mock_confirm_download):
        """Test handle_download_button_click when download is cancelled."""
        # Setup mock untuk confirm_download
        mock_confirm_download.return_value = False
        
        # Panggil fungsi
        self.handle_download_button_click(self.ui_components, self.ui_components['download_button'])
        
        # Verifikasi bahwa confirm_download dipanggil
        mock_confirm_download.assert_called_once_with(self.ui_components)
        
        # Verifikasi bahwa execute_download tidak dipanggil
        with patch('smartcash.ui.dataset.download.handlers.download_handler.execute_download') as mock_execute_download:
            mock_execute_download.assert_not_called()
    
    @patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download')
    def test_handle_download_button_click_error(self, mock_confirm_download):
        """Test handle_download_button_click when error occurs."""
        # Setup mock untuk confirm_download
        mock_confirm_download.side_effect = Exception("Test error")
        
        # Panggil fungsi
        self.handle_download_button_click(self.ui_components, self.ui_components['download_button'])
        
        # Verifikasi bahwa log_output.append_stderr dipanggil
        self.ui_components['log_output'].append_stderr.assert_called_once()
        
        # Verifikasi bahwa tombol download diaktifkan kembali
        self.assertFalse(self.ui_components['download_button'].disabled)
    
    @patch('smartcash.ui.dataset.download.handlers.download_handler._download_from_roboflow')
    def test_execute_download_roboflow(self, mock_download_from_roboflow):
        """Test execute_download with Roboflow endpoint."""
        # Setup mock untuk _download_from_roboflow
        mock_result = {'status': 'success', 'message': 'Download berhasil'}
        mock_download_from_roboflow.return_value = mock_result
        
        # Panggil fungsi
        self.execute_download(self.ui_components, 'Roboflow')
        
        # Verifikasi bahwa _download_from_roboflow dipanggil
        mock_download_from_roboflow.assert_called_once_with(self.ui_components)
    
    @patch('smartcash.ui.dataset.download.handlers.download_handler.notify_log')
    def test_execute_download_invalid_endpoint(self, mock_notify_log):
        """Test execute_download with invalid endpoint"""
        # Call function
        self.execute_download(self.ui_components, 'Invalid')
        
        # Verify notify_log was called with error level
        mock_notify_log.assert_called_with(
            sender=self.ui_components,
            message="Endpoint 'Invalid' tidak didukung",
            level="error"
        )
        
        # Verify UI is reset
        self.assertFalse(self.ui_components['download_running'])

    def test_handle_download_button_click_with_dict(self):
        """Test handle_download_button_click ketika button adalah dict (bukan widget)"""
        with patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download', return_value=False):
            try:
                self.handle_download_button_click(self.ui_components, {'not': 'a button'})
            except Exception as e:
                self.fail(f"handle_download_button_click raised Exception unexpectedly: {e}")

    def test_handle_download_button_click_with_non_dict_ui_components(self):
        """Test handle_download_button_click ketika ui_components bukan dict"""
        with patch('smartcash.ui.dataset.download.handlers.confirmation_handler.confirm_download', return_value=False):
            try:
                # Panggil dengan ui_components bukan dict
                self.handle_download_button_click("not a dict", self.ui_components['download_button'])
            except Exception as e:
                self.fail(f"handle_download_button_click raised Exception unexpectedly: {e}")
            
            # Verify button is re-enabled
            self.assertFalse(self.ui_components['download_button'].disabled)

if __name__ == '__main__':
    unittest.main() 