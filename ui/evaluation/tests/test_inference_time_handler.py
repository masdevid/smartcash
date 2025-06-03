"""
File: /Users/masdevid/Projects/smartcash/smartcash/ui/evaluation/tests/test_inference_time_handler.py
Deskripsi: Test suite untuk handler inference time dalam evaluasi model
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from smartcash.ui.evaluation.handlers.inference_time_handler import (
    setup_inference_time_handlers, 
    on_inference_time_checkbox_change,
    display_inference_time_metrics
)

class TestInferenceTimeHandler(unittest.TestCase):
    """Test suite untuk handler inference time"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'inference_time_checkbox': MagicMock(),
            'metrics_tabs': MagicMock(),
            'logger': MagicMock()
        }
        
        # Mock config
        self.config = {
            'evaluation': {
                'show_inference_time': True
            }
        }
        
        # Mock metrics
        self.inference_metrics = {
            'avg_inference_time': 0.015,  # 15ms
            'min_inference_time': 0.010,  # 10ms
            'max_inference_time': 0.020,  # 20ms
            'fps': 66.67,
            'inference_times': [0.015, 0.016, 0.014, 0.017, 0.013]
        }
        
        # Mock logger
        self.logger = MagicMock()
    
    def test_setup_inference_time_handlers(self):
        """Test setup handler untuk checkbox inference time"""
        # Test setup dengan checkbox yang ada
        setup_inference_time_handlers(self.ui_components, self.config)
        
        # Verifikasi observe dipanggil pada checkbox
        self.ui_components['inference_time_checkbox'].observe.assert_called_once()
        
        # Test setup tanpa checkbox
        ui_components_no_checkbox = {
            'logger': MagicMock()
        }
        
        # Patch log_to_service untuk menangkap panggilan
        with patch('smartcash.ui.evaluation.handlers.inference_time_handler.log_to_service') as mock_log:
            setup_inference_time_handlers(ui_components_no_checkbox, self.config)
            mock_log.assert_called_once_with(ui_components_no_checkbox['logger'], "⚠️ Checkbox inference time tidak ditemukan", "warning")
    
    def test_on_inference_time_checkbox_change(self):
        """Test handler untuk perubahan nilai checkbox inference time"""
        # Mock change event
        change = {'new': True}
        
        # Test dengan config yang sudah ada
        on_inference_time_checkbox_change(change, self.ui_components, self.config, self.logger)
        
        # Verifikasi config diupdate
        self.assertTrue(self.config['evaluation']['show_inference_time'])
        
        # Test dengan config yang belum ada
        config_empty = {}
        on_inference_time_checkbox_change(change, self.ui_components, config_empty, self.logger)
        
        # Verifikasi config dibuat dan diupdate
        self.assertTrue(config_empty['evaluation']['show_inference_time'])
        
        # Test dengan nilai False
        change = {'new': False}
        on_inference_time_checkbox_change(change, self.ui_components, self.config, self.logger)
        
        # Verifikasi config diupdate
        self.assertFalse(self.config['evaluation']['show_inference_time'])
    
    @patch('ipywidgets.Output')
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    @patch('matplotlib.pyplot.figure')
    @patch('matplotlib.pyplot.show')
    def test_display_inference_time_metrics(self, mock_plt_show, mock_plt_figure, 
                                          mock_clear_output, mock_display, mock_output):
        """Test tampilan metrik waktu inferensi dalam UI hasil evaluasi"""
        # Mock metrics_tabs children
        mock_tab1 = MagicMock()
        mock_tab2 = MagicMock()
        mock_tab3 = MagicMock()
        self.ui_components['metrics_tabs'].children = [mock_tab1, mock_tab2, mock_tab3]
        
        # Mock output widget
        mock_output_instance = MagicMock()
        mock_output.return_value = mock_output_instance
        
        # Test dengan config show_inference_time = True
        display_inference_time_metrics(
            self.inference_metrics, self.ui_components, self.config, self.logger
        )
        
        # Verifikasi tab baru ditambahkan
        self.assertEqual(len(self.ui_components['metrics_tabs'].children), 4)
        self.ui_components['metrics_tabs'].set_title.assert_called_once_with(3, "⏱️ Waktu Inferensi")
        
        # Test dengan config show_inference_time = False
        self.config['evaluation']['show_inference_time'] = False
        
        # Reset mock
        mock_output.reset_mock()
        mock_display.reset_mock()
        mock_clear_output.reset_mock()
        mock_plt_figure.reset_mock()
        mock_plt_show.reset_mock()
        
        # Patch log_to_service untuk menangkap panggilan
        with patch('smartcash.ui.evaluation.handlers.inference_time_handler.log_to_service') as mock_log:
            display_inference_time_metrics(
                self.inference_metrics, self.ui_components, self.config, self.logger
            )
            
            # Verifikasi tidak ada tab yang ditambahkan
            mock_output.assert_not_called()
            
            # Verifikasi pesan info dilog
            mock_log.assert_called_once_with(self.logger, "ℹ️ Metrik waktu inferensi tidak ditampilkan (dinonaktifkan)", "info")
        
        # Test dengan metrics_tabs tidak ada
        ui_components_no_tabs = {
            'logger': MagicMock()
        }
        
        # Reset config untuk test ini
        self.config['evaluation']['show_inference_time'] = True
        
        # Patch log_to_service untuk menangkap panggilan
        with patch('smartcash.ui.evaluation.handlers.inference_time_handler.log_to_service') as mock_log:
            display_inference_time_metrics(
                self.inference_metrics, ui_components_no_tabs, self.config, self.logger
            )
            
            # Verifikasi warning dilog - gunakan pendekatan yang lebih robust
            self.assertEqual(mock_log.call_count, 1, "log_to_service harus dipanggil sekali")
            args, kwargs = mock_log.call_args
            self.assertEqual(args[1], "⚠️ Metrics tabs tidak ditemukan untuk menampilkan waktu inferensi", "Pesan warning tidak sesuai")
            self.assertEqual(args[2], "warning", "Level log tidak sesuai")
        
        # Test dengan inference_metrics tidak valid
        with patch('smartcash.ui.evaluation.handlers.inference_time_handler.log_to_service') as mock_log:
            display_inference_time_metrics(
                None, self.ui_components, self.config, self.logger
            )
            
            # Verifikasi warning dilog
            mock_log.assert_called_with(self.logger, "⚠️ Metrik waktu inferensi tidak valid", "warning")

if __name__ == '__main__':
    unittest.main()
