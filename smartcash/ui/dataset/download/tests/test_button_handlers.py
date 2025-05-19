"""
File: smartcash/ui/dataset/download/tests/test_button_handlers.py
Deskripsi: Test untuk button handlers pada modul download dataset
"""

import unittest
import os
import tempfile
import json
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
            'rf_workspace': MagicMock(spec=widgets.Text, value='test-workspace'),
            'rf_project': MagicMock(spec=widgets.Text, value='test-project'),
            'rf_version': MagicMock(spec=widgets.Text, value='1'),
            'rf_apikey': MagicMock(spec=widgets.Text, value='test-api-key'),
            'output_dir': MagicMock(spec=widgets.Text, value='data/test'),
            'validate_dataset': MagicMock(spec=widgets.Checkbox, value=True),
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
        from smartcash.ui.dataset.download.handlers.save_handler import handle_save_button_click, save_config_and_results
        
        # Buat temporary directory untuk test
        with tempfile.TemporaryDirectory() as temp_dir:
            # Update output_dir ke temporary directory
            self.ui_components['output_dir'].value = temp_dir
            
            # Panggil handler dengan mock
            with patch('smartcash.ui.dataset.download.handlers.save_handler.save_config_and_results') as mock_save:
                button_mock = MagicMock()
                handle_save_button_click(button_mock, self.ui_components)
                
                # Verifikasi bahwa save_config_and_results dipanggil
                mock_save.assert_called_once_with(self.ui_components)
                
                # Verifikasi bahwa update_status_panel dipanggil dengan parameter yang benar
                self.update_status_panel_mock.assert_called_once_with(
                    self.ui_components, 
                    "success", 
                    "Konfigurasi dan hasil download berhasil disimpan"
                )
            
            # Test save_config_and_results langsung
            output_path = save_config_and_results(self.ui_components, os.path.join(temp_dir, 'test_config.json'))
            
            # Verifikasi bahwa file berhasil dibuat
            self.assertTrue(os.path.exists(output_path))
            
            # Verifikasi isi file
            with open(output_path, 'r') as f:
                saved_config = json.load(f)
                
                # Verifikasi nilai-nilai kunci
                self.assertEqual(saved_config['workspace'], 'test-workspace')
                self.assertEqual(saved_config['project'], 'test-project')
                self.assertEqual(saved_config['version'], '1')
                self.assertEqual(saved_config['output_dir'], temp_dir)
                self.assertEqual(saved_config['validate_dataset'], True)
                self.assertEqual(saved_config['timestamp'], '2025-05-19T11:59:29+07:00')
                
                # Verifikasi dataset_stats
                self.assertEqual(saved_config['dataset_stats']['total_images'], 100)
                self.assertEqual(saved_config['dataset_stats']['total_labels'], 90)
                self.assertEqual(saved_config['dataset_stats']['classes'], {'0': 50, '1': 40})

if __name__ == '__main__':
    unittest.main()
