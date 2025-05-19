"""
File: smartcash/ui/dataset/download/tests/test_confirmation_dialog.py
Deskripsi: Test untuk dialog konfirmasi download dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from IPython.display import display

class TestConfirmationDialog(unittest.TestCase):
    """Test untuk dialog konfirmasi download dataset."""
    
    def setUp(self):
        """Setup untuk test."""
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'output_dir': MagicMock(value='data/test'),
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'source_dropdown': MagicMock(value='roboflow'),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/downloads_backup'),
            'confirmation_area': MagicMock(),
            'status_panel': MagicMock(),
            'download_button': MagicMock(),
            'check_button': MagicMock(),
            'save_button': MagicMock(),
            'progress_bar': MagicMock(),
            'progress_message': MagicMock()
        }
        
        # Tambahkan bind method ke logger mock
        self.ui_components['logger'].bind = MagicMock(return_value=self.ui_components['logger'])
        
        # Import fungsi yang akan ditest
        from smartcash.ui.dataset.download.handlers.confirmation_handler import confirm_download, cancel_download
        self.confirm_download = confirm_download
        self.cancel_download = cancel_download
    
    @patch('smartcash.ui.components.confirmation_dialog.create_confirmation_dialog')
    @patch('smartcash.ui.components.status_panel.update_status_panel')
    def test_confirm_download_shows_dialog(self, mock_update_status_panel, mock_create_dialog):
        """Test bahwa confirm_download menampilkan dialog konfirmasi."""
        # Mock dialog
        mock_dialog = MagicMock()
        mock_create_dialog.return_value = mock_dialog
        
        # Panggil confirm_download
        self.confirm_download(self.ui_components, 'Roboflow', self.ui_components['download_button'])
        
        # Verifikasi bahwa dialog dibuat
        mock_create_dialog.assert_called_once()
        
        # Verifikasi bahwa status panel diupdate
        mock_update_status_panel.assert_called_once()
        
        # Verifikasi bahwa confirmation area dibersihkan
        self.ui_components['confirmation_area'].clear_output.assert_called_once()
        
        # Verifikasi bahwa logger dipanggil
        self.ui_components['logger'].info.assert_called_once()
        
        # Verifikasi bahwa download_logger dibuat
        self.assertIn('download_logger', self.ui_components)
    
    @patch('smartcash.ui.components.status_panel.update_status_panel')
    def test_cancel_download_resets_ui(self, mock_update_status_panel):
        """Test bahwa cancel_download mereset UI."""
        # Panggil cancel_download
        self.cancel_download(self.ui_components, self.ui_components['logger'])
        
        # Verifikasi bahwa confirmation area dibersihkan
        self.ui_components['confirmation_area'].clear_output.assert_called_once()
        
        # Verifikasi bahwa status panel diupdate
        mock_update_status_panel.assert_called_once()
        
        # Verifikasi bahwa logger dipanggil
        self.ui_components['logger'].info.assert_called_once()
    
    @patch('smartcash.ui.dataset.download.handlers.download_handler.execute_download')
    def test_confirm_and_execute_calls_execute_download(self, mock_execute_download):
        """Test bahwa confirm_and_execute memanggil execute_download."""
        # Mock dialog
        mock_dialog = MagicMock()
        
        # Mock create_confirmation_dialog
        with patch('smartcash.ui.components.confirmation_dialog.create_confirmation_dialog') as mock_create_dialog:
            # Setup mock untuk menangkap callback
            def side_effect(title, message, on_confirm, on_cancel):
                # Simpan callback
                self.on_confirm = on_confirm
                return mock_dialog
            
            mock_create_dialog.side_effect = side_effect
            
            # Panggil confirm_download
            self.confirm_download(self.ui_components, 'Roboflow', self.ui_components['download_button'])
            
            # Panggil callback on_confirm
            self.on_confirm()
            
            # Verifikasi bahwa execute_download dipanggil
            mock_execute_download.assert_called_once_with(self.ui_components, 'Roboflow')
            
            # Verifikasi bahwa current_operation diset
            self.assertEqual(self.ui_components['current_operation'], 'download_only')

if __name__ == '__main__':
    unittest.main()
