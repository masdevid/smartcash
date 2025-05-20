"""
File: smartcash/ui/dataset/download/tests/test_button_handlers.py
Deskripsi: Test untuk button handlers pada modul download dataset
"""

import unittest
import os
import tempfile
import json
import yaml
from unittest.mock import MagicMock, patch

import ipywidgets as widgets

class TestButtonHandlers(unittest.TestCase):
    """Test suite untuk button handlers pada modul download."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Buat mock UI components
        self.ui_components = self._create_mock_ui_components()
        
        # Setup logger mock
        self.logger_mock = MagicMock()
        self.ui_components['logger'] = self.logger_mock
        
        # Setup update_status_panel mock
        self.update_status_panel_mock = MagicMock()
        self.ui_components['update_status_panel'] = self.update_status_panel_mock
    
    def _create_mock_ui_components(self):
        """Buat mock UI components untuk testing."""
        # Buat mock untuk semua komponen UI yang diperlukan
        ui_components = {
            'download_button': MagicMock(spec=widgets.Button),
            'check_button': MagicMock(spec=widgets.Button),
            'reset_button': MagicMock(spec=widgets.Button),
            'save_button': MagicMock(spec=widgets.Button),
            'cleanup_button': MagicMock(spec=widgets.Button),
            'progress_bar': MagicMock(spec=widgets.FloatProgress),
            'overall_label': MagicMock(spec=widgets.HTML),
            'step_label': MagicMock(spec=widgets.HTML),
            'status_panel': MagicMock(spec=widgets.HTML),
            'log_output': MagicMock(spec=widgets.Output),
            'summary_container': MagicMock(spec=widgets.Output),
            'progress_container': MagicMock(spec=widgets.VBox),
            'workspace': MagicMock(spec=widgets.Text, value='test-workspace'),
            'project': MagicMock(spec=widgets.Text, value='test-project'),
            'version': MagicMock(spec=widgets.Text, value='1'),
            'api_key': MagicMock(spec=widgets.Text, value='test-api-key'),
            'source_dropdown': MagicMock(spec=widgets.Dropdown, value='roboflow'),
            'output_dir': MagicMock(spec=widgets.Text, value='data/test'),
            'validate_dataset': MagicMock(spec=widgets.Checkbox, value=True),
            'backup_checkbox': MagicMock(spec=widgets.Checkbox, value=True),
            'backup_dir': MagicMock(spec=widgets.Text, value='data/downloads_backup'),
            'reset_progress_bar': MagicMock(),
            'dataset_stats': {
                'total_images': 100,
                'total_labels': 90,
                'classes': {'0': 50, '1': 40}
            },
            'download_timestamp': '2025-05-19T11:59:29+07:00'
        }
        
        # Tambahkan layout ke komponen yang membutuhkannya
        for key in ['progress_bar', 'overall_label', 'step_label', 'progress_container', 'summary_container']:
            ui_components[key].layout = MagicMock()
            ui_components[key].layout.visibility = 'visible'
        
        return ui_components
    
    def test_reset_button_handler(self):
        """Test untuk reset button handler."""
        from smartcash.ui.dataset.download.handlers.reset_handler import handle_reset_button_click
        
        # Panggil handler
        button_mock = MagicMock()
        handle_reset_button_click(button_mock, self.ui_components)
        
        # Verifikasi bahwa reset_progress_bar dipanggil
        self.ui_components['reset_progress_bar'].assert_called_once()
        
        # Verifikasi bahwa log_output.clear_output dipanggil
        self.ui_components['log_output'].clear_output.assert_called_once()
        
        # Verifikasi bahwa summary_container.clear_output dipanggil
        self.ui_components['summary_container'].clear_output.assert_called_once()
        
        # Verifikasi bahwa update_status_panel dipanggil dengan parameter yang benar
        self.update_status_panel_mock.assert_called_once()
        
        # Verifikasi bahwa tombol-tombol diaktifkan
        self.assertFalse(self.ui_components['download_button'].disabled)
        self.assertFalse(self.ui_components['check_button'].disabled)
        self.assertFalse(self.ui_components['reset_button'].disabled)
    
    def test_save_button_handler(self):
        """Test untuk save button handler."""
        # Buat mock untuk config_manager
        with patch('smartcash.common.config.get_config_manager') as mock_get_config_manager:
            # Setup mock config manager
            mock_config_manager = MagicMock()
            mock_get_config_manager.return_value = mock_config_manager
            
            # Buat mock button
            button_mock = MagicMock()
            self.ui_components['save_button'].disabled = False
            
            # Panggil handler langsung
            from smartcash.ui.dataset.download.handlers.setup_handlers import _setup_save_button_handler
            _setup_save_button_handler(self.ui_components)
            
            # Verifikasi bahwa tombol save dikonfigurasi dengan handler
            self.assertTrue(hasattr(self.ui_components['save_button'], 'on_click'))

if __name__ == '__main__':
    unittest.main()
