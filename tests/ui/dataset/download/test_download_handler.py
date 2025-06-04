import unittest
from unittest.mock import MagicMock, patch
from smartcash.ui.dataset.download.handlers import handle_download_click
from smartcash.common.dataset.abstract_manager import AbstractDatasetManager

class TestDownloadHandler(unittest.TestCase):
    
    def test_handle_download_click_success(self):
        """Test handler download sukses"""
        # Mock UI components
        ui_components = {
            'project_input': MagicMock(value='test-project'),
            'version_input': MagicMock(value='1'),
            'api_key_input': MagicMock(value='valid-key'),
            'status_panel': MagicMock(),
            'log_output': MagicMock()
        }
        
        # Mock DatasetManager
        mock_manager = MagicMock(spec=AbstractDatasetManager)
        mock_manager.download_from_roboflow.return_value = True
        
        # Panggil handler
        handle_download_click(ui_components, mock_manager)
        
        # Verifikasi
        mock_manager.download_from_roboflow.assert_called_once_with('test-project', '1', 'valid-key')
        ui_components['status_panel'].append.assert_called_with("✅ Download berhasil")

if __name__ == '__main__':
    unittest.main()
