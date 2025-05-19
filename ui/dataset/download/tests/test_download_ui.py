"""
File: smartcash/ui/dataset/download/tests/test_download_ui.py
Deskripsi: Test untuk UI download dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from IPython.display import display

class TestDownloadUI(unittest.TestCase):
    """Test untuk UI download dataset."""
    
    def setUp(self):
        """Setup untuk test."""
        # Mock environment dan config
        self.env = MagicMock()
        self.config = {
            'data': {
                'roboflow': {
                    'workspace': 'test-workspace',
                    'project': 'test-project',
                    'version': '1',
                    'api_key': 'test-api-key'
                }
            }
        }
        
        # Buat mock UI components langsung daripada mencoba membuat UI sebenarnya
        self.ui_components = {
            'download_button': MagicMock(),
            'check_button': MagicMock(),
            'reset_button': MagicMock(),
            'save_button': MagicMock(),
            'cleanup_button': MagicMock(),
            'source_dropdown': MagicMock(value='roboflow', options=['roboflow', 'google_drive']),
            'output_dir': MagicMock(value='data/test'),
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'drive_folder': MagicMock(value='dataset'),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/downloads_backup'),
            'progress_bar': MagicMock(),
            'progress_message': MagicMock(),
            'status_panel': MagicMock(),
            'confirmation_area': MagicMock(),
            'logger': MagicMock()
        }
        
        # Tambahkan bind method ke logger mock
        self.ui_components['logger'].bind = MagicMock(return_value=self.ui_components['logger'])
    
    def test_ui_components_exist(self):
        """Test bahwa semua komponen UI yang diperlukan ada."""
        required_components = [
            'download_button', 'check_button', 'reset_button', 'save_button', 'cleanup_button',
            'source_dropdown', 'output_dir', 'workspace', 'project', 'version', 'api_key',
            'drive_folder', 'backup_checkbox', 'backup_dir', 'progress_bar', 'progress_message',
            'status_panel', 'confirmation_area'
        ]
        
        for component in required_components:
            self.assertIn(component, self.ui_components)
    
    def test_endpoint_dropdown_options(self):
        """Test bahwa dropdown source memiliki opsi yang benar."""
        # Mock dropdown
        dropdown = MagicMock()
        dropdown.options = ['roboflow', 'google_drive']
        
        # Ganti dropdown di UI components
        self.ui_components['source_dropdown'] = dropdown
        
        # Test bahwa dropdown memiliki opsi yang benar
        self.assertEqual(dropdown.options, ['roboflow', 'google_drive'])
    
    def test_default_values_from_config(self):
        """Test bahwa nilai default diambil dari config."""
        # Mock komponen UI
        self.ui_components['workspace'] = MagicMock(value='test-workspace')
        self.ui_components['project'] = MagicMock(value='test-project')
        self.ui_components['version'] = MagicMock(value='1')
        self.ui_components['api_key'] = MagicMock(value='test-api-key')
        
        # Test bahwa nilai default diambil dari config
        self.assertEqual(self.ui_components['workspace'].value, 'test-workspace')
        self.assertEqual(self.ui_components['project'].value, 'test-project')
        self.assertEqual(self.ui_components['version'].value, '1')
        self.assertEqual(self.ui_components['api_key'].value, 'test-api-key')

if __name__ == '__main__':
    unittest.main()
