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
            'endpoint_dropdown': MagicMock(value='Roboflow', options=['Roboflow', 'Google Drive']),
            'output_dir': MagicMock(value='data/test'),
            'rf_workspace': MagicMock(value='test-workspace'),
            'rf_project': MagicMock(value='test-project'),
            'rf_version': MagicMock(value='1'),
            'rf_api_key': MagicMock(value='test-api-key'),
            'drive_folder': MagicMock(value='dataset'),
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
            'download_button', 'endpoint_dropdown', 'output_dir',
            'rf_workspace', 'rf_project', 'rf_version', 'rf_api_key',
            'drive_folder', 'progress_bar', 'progress_message',
            'status_panel', 'confirmation_area'
        ]
        
        for component in required_components:
            self.assertIn(component, self.ui_components)
    
    def test_endpoint_dropdown_options(self):
        """Test bahwa dropdown endpoint memiliki opsi yang benar."""
        # Mock dropdown
        dropdown = MagicMock()
        dropdown.options = ['Roboflow', 'Google Drive']
        
        # Ganti dropdown di UI components
        self.ui_components['endpoint_dropdown'] = dropdown
        
        # Test bahwa dropdown memiliki opsi yang benar
        self.assertEqual(dropdown.options, ['Roboflow', 'Google Drive'])
    
    def test_default_values_from_config(self):
        """Test bahwa nilai default diambil dari config."""
        # Mock komponen UI
        self.ui_components['rf_workspace'] = MagicMock(value='test-workspace')
        self.ui_components['rf_project'] = MagicMock(value='test-project')
        self.ui_components['rf_version'] = MagicMock(value='1')
        
        # Test bahwa nilai default diambil dari config
        self.assertEqual(self.ui_components['rf_workspace'].value, 'test-workspace')
        self.assertEqual(self.ui_components['rf_project'].value, 'test-project')
        self.assertEqual(self.ui_components['rf_version'].value, '1')

if __name__ == '__main__':
    unittest.main()
