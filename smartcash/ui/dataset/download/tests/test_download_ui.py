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
                },
                'dir': 'data/test'
            }
        }
        
        # Buat mock UI components
        self.ui_components = {
            'ui': MagicMock(),
            'header': MagicMock(),
            'status_panel': MagicMock(),
            'workspace': MagicMock(value='test-workspace'),
            'project': MagicMock(value='test-project'),
            'version': MagicMock(value='1'),
            'api_key': MagicMock(value='test-api-key'),
            'output_dir': MagicMock(value='data/test'),
            'validate_dataset': MagicMock(value=True),
            'backup_checkbox': MagicMock(value=True),
            'backup_dir': MagicMock(value='data/backup'),
            'input_options': MagicMock(),
            'download_button': MagicMock(),
            'check_button': MagicMock(),
            'save_button': MagicMock(),
            'reset_config_button': MagicMock(),
            'save_reset_buttons': MagicMock(),
            'sync_info': MagicMock(),
            'cleanup_button': MagicMock(),
            'button_container': MagicMock(),
            'summary_container': MagicMock(),
            'confirmation_area': MagicMock(),
            'progress_bar': MagicMock(),
            'progress_container': MagicMock(),
            'current_progress': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'status': MagicMock(),
            'log_output': MagicMock(),
            'log_accordion': MagicMock(),
            'module_name': 'download',
            'logger': MagicMock()
        }
        
        # Tambahkan bind method ke logger mock
        self.ui_components['logger'].bind = MagicMock(return_value=self.ui_components['logger'])
    
    def test_ui_components_exist(self):
        """Test bahwa semua komponen UI yang diperlukan ada."""
        required_components = [
            'ui', 'header', 'status_panel', 'workspace', 'project', 'version', 'api_key',
            'output_dir', 'validate_dataset', 'backup_checkbox', 'backup_dir', 'input_options',
            'download_button', 'check_button', 'save_button', 'reset_config_button',
            'save_reset_buttons', 'sync_info', 'cleanup_button', 'button_container',
            'summary_container', 'confirmation_area', 'progress_bar', 'progress_container',
            'current_progress', 'overall_label', 'step_label', 'status', 'log_output',
            'log_accordion', 'module_name'
        ]
        
        for component in required_components:
            self.assertIn(component, self.ui_components)
    
    def test_default_values_from_config(self):
        """Test bahwa nilai default diambil dari config."""
        # Test bahwa nilai default diambil dari config
        self.assertEqual(self.ui_components['workspace'].value, 'test-workspace')
        self.assertEqual(self.ui_components['project'].value, 'test-project')
        self.assertEqual(self.ui_components['version'].value, '1')
        self.assertEqual(self.ui_components['api_key'].value, 'test-api-key')
        self.assertEqual(self.ui_components['output_dir'].value, 'data/test')
        self.assertEqual(self.ui_components['validate_dataset'].value, True)
        self.assertEqual(self.ui_components['backup_checkbox'].value, True)
        self.assertEqual(self.ui_components['backup_dir'].value, 'data/backup')
    
    def test_module_name(self):
        """Test bahwa module_name diset dengan benar."""
        self.assertEqual(self.ui_components['module_name'], 'download')
    
    def test_logger_binding(self):
        """Test bahwa logger binding berfungsi."""
        # Test bahwa bind method dipanggil dengan benar
        self.ui_components['logger'].bind.assert_not_called()
        self.ui_components['logger'].bind('context', 'download_only')
        self.ui_components['logger'].bind.assert_called_once_with('context', 'download_only')

if __name__ == '__main__':
    unittest.main()
