"""
File: tests/ui/dataset/downloader/test_config_updater.py
Deskripsi: Test untuk config_updater.py
"""

import unittest
from unittest.mock import MagicMock, patch, ANY
import ipywidgets as widgets

from smartcash.ui.dataset.downloader.handlers.config_updater import DownloaderConfigUpdater
from smartcash.common.logger import get_logger

class TestDownloaderConfigUpdater(unittest.TestCase):
    """Test case untuk DownloaderConfigUpdater."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        self.logger = get_logger('test_config_updater')
        self.ui_components = {
            'text_field': widgets.Text(value='initial'),
            'checkbox': widgets.Checkbox(value=False),
            'dropdown': widgets.Dropdown(
                options=['option1', 'option2', 'option3'],
                value='option1'
            ),
            'int_slider': widgets.IntSlider(value=5, min=0, max=10, step=1),
            'float_slider': widgets.FloatSlider(value=0.5, min=0.0, max=1.0, step=0.1),
            'disabled_field': widgets.Text(disabled=True, value='disabled')
        }
    
    def test_update_ui_text_field(self):
        """Test update text field."""
        config = {
            'text_field': 'new value'
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertEqual(self.ui_components['text_field'].value, 'new value')
    
    def test_update_ui_checkbox(self):
        """Test update checkbox."""
        config = {
            'checkbox': True
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertTrue(self.ui_components['checkbox'].value)
    
    def test_update_ui_dropdown(self):
        """Test update dropdown."""
        config = {
            'dropdown': 'option2'
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertEqual(self.ui_components['dropdown'].value, 'option2')
    
    def test_update_ui_int_slider(self):
        """Test update integer slider."""
        config = {
            'int_slider': 7
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertEqual(self.ui_components['int_slider'].value, 7)
    
    def test_update_ui_float_slider(self):
        """Test update float slider."""
        config = {
            'float_slider': 0.8
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertAlmostEqual(self.ui_components['float_slider'].value, 0.8)
    
    def test_update_ui_ignores_disabled_fields(self):
        """Test bahwa field yang disabled tidak diupdate."""
        config = {
            'disabled_field': 'new value'
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        self.assertEqual(self.ui_components['disabled_field'].value, 'disabled')
    
    def test_update_ui_with_nonexistent_field(self):
        """Test update dengan field yang tidak ada di UI."""
        config = {
            'nonexistent_field': 'some value'
        }
        
        # Tidak boleh raise exception
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
    
    @patch.object(DownloaderConfigUpdater, '_update_field')
    def test_update_ui_calls_update_field(self, mock_update):
        """Test bahwa update_ui memanggil _update_field dengan benar."""
        config = {
            'text_field': 'test',
            'checkbox': True
        }
        
        DownloaderConfigUpdater.update_ui(self.ui_components, config)
        
        # Pastikan _update_field dipanggil untuk setiap field di config
        self.assertEqual(mock_update.call_count, 2)
        mock_update.assert_any_call(self.ui_components, 'text_field', 'test')
        mock_update.assert_any_call(self.ui_components, 'checkbox', True)


if __name__ == '__main__':
    unittest.main()
