"""
File: smartcash/ui/dataset/augmentation/tests/test_status_handler.py
Deskripsi: Test untuk status handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any
import pandas as pd

# Patch modul-modul yang dibutuhkan sebelum mengimpor fungsi-fungsi yang akan diuji
import sys
from unittest.mock import MagicMock, patch

# Patch IPython.display.display dan IPython.display.HTML
sys.modules['IPython.display'] = MagicMock()
sys.modules['IPython.display'].display = MagicMock()
sys.modules['IPython.display'].HTML = MagicMock()
sys.modules['IPython.display'].clear_output = MagicMock()

# Patch tqdm
sys.modules['tqdm.auto'] = MagicMock()
sys.modules['tqdm.auto'].tqdm = MagicMock()

# Patch alert_utils
sys.modules['smartcash.ui.utils.alert_utils'] = MagicMock()
sys.modules['smartcash.ui.utils.alert_utils'].create_status_indicator = MagicMock(return_value=MagicMock())

# Sekarang impor fungsi-fungsi yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.status_handler import (
    update_status_panel,
    create_status_panel,
    log_status,
    update_augmentation_info,
    update_status_text,
    update_progress_bar,
    reset_progress_bar,
    register_progress_callback,
    show_augmentation_summary
)

class TestStatusHandler(unittest.TestCase):
    """Test untuk status handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Reset semua mock
        sys.modules['IPython.display'].display.reset_mock()
        sys.modules['IPython.display'].HTML.reset_mock()
        sys.modules['IPython.display'].clear_output.reset_mock()
        sys.modules['smartcash.ui.utils.alert_utils'].create_status_indicator.reset_mock()
        
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'status_panel': MagicMock(),
            'status': MagicMock(),
            'progress_bar': MagicMock(),
            'status_text': MagicMock(),
            'summary_output': MagicMock()
        }
        
        # Mock untuk config handler
        self.patcher1 = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.get_config_from_ui')
        self.mock_get_config = self.patcher1.start()
        
        # Simpan referensi ke mock yang sudah di-patch
        self.mock_display = sys.modules['IPython.display'].display
        self.mock_html = sys.modules['IPython.display'].HTML
        self.mock_clear_output = sys.modules['IPython.display'].clear_output
        self.mock_create_status_indicator = sys.modules['smartcash.ui.utils.alert_utils'].create_status_indicator
        
        # Setup mock split selector
        mock_dropdown = MagicMock()
        mock_dropdown.description = 'Split:'
        mock_dropdown.value = 'train'
        
        mock_child = MagicMock()
        mock_child.children = [mock_dropdown]
        
        self.ui_components['split_selector'] = MagicMock()
        self.ui_components['split_selector'].children = [mock_child]
        
        # Setup mock config
        self.mock_config = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'balance_classes': True
            }
        }
        self.mock_get_config.return_value = self.mock_config
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
    
    def test_update_status_panel(self):
        """Test update panel status"""
        # Panggil fungsi
        update_status_panel(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        self.mock_create_status_indicator.assert_called_once()
    
    def test_update_status_panel_missing(self):
        """Test update panel status ketika panel tidak ada"""
        # Hapus status_panel dari ui_components
        ui_components_no_panel = {k: v for k, v in self.ui_components.items() if k != 'status_panel'}
        
        # Panggil fungsi
        update_status_panel(ui_components_no_panel, "Pesan test", "info")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_create_status_panel(self):
        """Test pembuatan panel status"""
        # Panggil fungsi
        panel = create_status_panel("Judul test", "info")
        
        # Verifikasi hasil
        self.assertIsInstance(panel, widgets.Box)
        self.assertEqual(len(panel.children), 1)
    
    def test_log_status(self):
        """Test log status ke output"""
        # Panggil fungsi
        log_status(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        self.mock_display.assert_called()
    
    def test_log_status_missing(self):
        """Test log status ketika output tidak ada"""
        # Hapus status dari ui_components
        ui_components_no_status = {k: v for k, v in self.ui_components.items() if k != 'status'}
        
        # Panggil fungsi
        log_status(ui_components_no_status, "Pesan test", "info")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_update_augmentation_info(self):
        """Test update informasi augmentasi"""
        # Panggil fungsi
        update_augmentation_info(self.ui_components)
        
        # Verifikasi hasil
        self.mock_create_status_indicator.assert_called()
    
    def test_update_status_text(self):
        """Test update teks status"""
        # Panggil fungsi
        update_status_text(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['status_text'].value.startswith("<div"))
    
    def test_update_status_text_missing(self):
        """Test update teks status ketika status_text tidak ada"""
        # Hapus status_text dari ui_components
        ui_components_no_text = {k: v for k, v in self.ui_components.items() if k != 'status_text'}
        
        # Panggil fungsi
        update_status_text(ui_components_no_text, "Pesan test", "info")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_update_progress_bar(self):
        """Test update progress bar"""
        # Panggil fungsi
        update_progress_bar(self.ui_components, 50, 100, "Progress test")
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['progress_bar'].value, 50)
        self.assertEqual(self.ui_components['progress_bar'].max, 100)
        self.assertEqual(self.ui_components['progress_bar'].description, "Progress test")
    
    def test_update_progress_bar_missing(self):
        """Test update progress bar ketika progress_bar tidak ada"""
        # Hapus progress_bar dari ui_components
        ui_components_no_bar = {k: v for k, v in self.ui_components.items() if k != 'progress_bar'}
        
        # Panggil fungsi
        update_progress_bar(ui_components_no_bar, 50, 100, "Progress test")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_reset_progress_bar(self):
        """Test reset progress bar"""
        # Setup progress bar dengan nilai awal
        self.ui_components['progress_bar'].value = 50
        
        # Panggil fungsi
        reset_progress_bar(self.ui_components, "Reset test")
        
        # Verifikasi hasil
        self.assertEqual(self.ui_components['progress_bar'].value, 0)
        self.assertEqual(self.ui_components['progress_bar'].description, "Reset test")
    
    def test_reset_progress_bar_missing(self):
        """Test reset progress bar ketika progress_bar tidak ada"""
        # Hapus progress_bar dari ui_components
        ui_components_no_bar = {k: v for k, v in self.ui_components.items() if k != 'progress_bar'}
        
        # Panggil fungsi
        reset_progress_bar(ui_components_no_bar, "Reset test")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_register_progress_callback(self):
        """Test registrasi callback untuk progress bar"""
        # Panggil fungsi
        callback = register_progress_callback(self.ui_components, 100)
        
        # Verifikasi hasil
        self.assertTrue(callable(callback))
        
        # Test callback
        callback(50, "Callback test", "info")
        self.assertEqual(self.ui_components['progress_bar'].value, 50)
    
    def test_register_progress_callback_missing(self):
        """Test registrasi callback ketika progress_bar tidak ada"""
        # Hapus progress_bar dari ui_components
        ui_components_no_bar = {k: v for k, v in self.ui_components.items() if k != 'progress_bar'}
        
        # Panggil fungsi
        callback = register_progress_callback(ui_components_no_bar, 100)
        
        # Verifikasi hasil
        self.assertTrue(callable(callback))
        
        # Test callback (tidak akan error)
        callback(50, "Callback test", "info")
        self.assertTrue(True)
    
    def test_show_augmentation_summary_test_format(self):
        """Test tampilkan ringkasan augmentasi dengan format pengujian"""
        # Setup summary dalam format pengujian
        test_summary = {
            'status': 'success',
            'message': 'Augmentasi berhasil',
            'stats': {
                'total_images': 100,
                'augmented_images': 200,
                'classes': {'class1': 50, 'class2': 50},
                'time_taken': 10.5
            }
        }
        
        # Panggil fungsi
        show_augmentation_summary(self.ui_components, test_summary)
        
        # Verifikasi hasil
        self.mock_display.assert_called()
    
    def test_show_augmentation_summary_normal_format(self):
        """Test tampilkan ringkasan augmentasi dengan format normal"""
        # Setup summary dalam format normal
        normal_summary = {
            'total_images': 100,
            'total_augmented': 200,
            'aug_types': ['combined'],
            'class_distribution': {'class1': 50, 'class2': 50},
            'time_taken': 10.5
        }
        
        # Panggil fungsi
        show_augmentation_summary(self.ui_components, normal_summary)
        
        # Verifikasi bahwa mock_display dipanggil
        self.mock_display.assert_called()
    
    def test_show_augmentation_summary_html_widget(self):
        """Test tampilkan ringkasan augmentasi ke HTML widget"""
        # Setup summary dalam format normal
        normal_summary = {
            'total_images': 100,
            'total_augmented': 200,
            'aug_types': ['combined'],
            'class_distribution': {'class1': 50, 'class2': 50},
            'time_taken': 10.5
        }
        
        # Ganti summary_output dengan HTML widget
        self.ui_components['summary_output'] = widgets.HTML()
        
        # Panggil fungsi
        show_augmentation_summary(self.ui_components, normal_summary)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['summary_output'].value.startswith("\n        <h3>"))
    
    def test_show_augmentation_summary_missing(self):
        """Test tampilkan ringkasan augmentasi ketika summary_output tidak ada"""
        # Setup summary dalam format normal
        normal_summary = {
            'total_images': 100,
            'total_augmented': 200,
            'aug_types': ['combined'],
            'class_distribution': {'class1': 50, 'class2': 50},
            'time_taken': 10.5
        }
        
        # Hapus summary_output dari ui_components
        ui_components_no_summary = {k: v for k, v in self.ui_components.items() if k != 'summary_output'}
        
        # Panggil fungsi
        show_augmentation_summary(ui_components_no_summary, normal_summary)
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
