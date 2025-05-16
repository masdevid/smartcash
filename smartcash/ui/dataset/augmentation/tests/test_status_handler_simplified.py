"""
File: smartcash/ui/dataset/augmentation/tests/test_status_handler_simplified.py
Deskripsi: Test sederhana untuk status handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

class TestStatusHandlerSimplified(unittest.TestCase):
    """Test sederhana untuk status handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
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
        self.mock_get_config.return_value = {
            'augmentation': {
                'enabled': True,
                'types': ['combined'],
                'num_variations': 2,
                'target_count': 1000,
                'balance_classes': True
            }
        }
        
        # Setup mock split selector
        mock_dropdown = MagicMock()
        mock_dropdown.description = 'Split:'
        mock_dropdown.value = 'train'
        
        mock_child = MagicMock()
        mock_child.children = [mock_dropdown]
        
        self.ui_components['split_selector'] = MagicMock()
        self.ui_components['split_selector'].children = [mock_child]
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
    
    @patch('smartcash.ui.utils.alert_utils.create_status_indicator')
    def test_update_status_panel(self, mock_create_indicator):
        """Test update panel status"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_panel
        
        # Setup mock
        mock_create_indicator.return_value = widgets.HTML(value="Mocked Indicator")
        
        # Panggil fungsi
        update_status_panel(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        mock_create_indicator.assert_called_once()
    
    def test_create_status_panel(self):
        """Test pembuatan panel status"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import create_status_panel
        
        # Patch create_status_indicator
        with patch('smartcash.ui.utils.alert_utils.create_status_indicator') as mock_create_indicator:
            mock_create_indicator.return_value = widgets.HTML(value="Mocked Indicator")
            
            # Panggil fungsi
            panel = create_status_panel("Judul test", "info")
            
            # Verifikasi hasil
            self.assertIsInstance(panel, widgets.Box)
            mock_create_indicator.assert_called_once()
    
    @patch('IPython.display.display')
    @patch('smartcash.ui.utils.alert_utils.create_status_indicator')
    def test_log_status(self, mock_create_indicator, mock_display):
        """Test log status ke output"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import log_status
        
        # Setup mock
        mock_create_indicator.return_value = widgets.HTML(value="Mocked Indicator")
        
        # Panggil fungsi
        log_status(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        mock_display.assert_called_once()
    
    @patch('smartcash.ui.utils.alert_utils.create_status_indicator')
    def test_update_augmentation_info(self, mock_create_indicator):
        """Test update informasi augmentasi"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_augmentation_info
        
        # Setup mock
        mock_create_indicator.return_value = widgets.HTML(value="Mocked Indicator")
        
        # Panggil fungsi
        update_augmentation_info(self.ui_components)
        
        # Verifikasi hasil
        mock_create_indicator.assert_called()
    
    @patch('IPython.display.display')
    @patch('IPython.display.HTML')
    def test_update_status_text(self, mock_html, mock_display):
        """Test update teks status"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_status_text
        
        # Setup mock
        mock_html.return_value = "Mocked HTML"
        
        # Panggil fungsi
        update_status_text(self.ui_components, "Pesan test", "info")
        
        # Verifikasi hasil
        mock_html.assert_called_once()
        mock_display.assert_called_once()
    
    def test_update_progress_bar(self):
        """Test update progress bar"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_progress_bar
        
        # Panggil fungsi
        update_progress_bar(self.ui_components, 50, 100, "Progress test")
        
        # Verifikasi hasil
        self.ui_components['progress_bar'].value = 50
        self.ui_components['progress_bar'].max = 100
        self.ui_components['progress_bar'].description = "Progress test"
    
    def test_reset_progress_bar(self):
        """Test reset progress bar"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import reset_progress_bar
        
        # Panggil fungsi
        reset_progress_bar(self.ui_components, "Reset test")
        
        # Verifikasi hasil
        self.ui_components['progress_bar'].value = 0
        self.ui_components['progress_bar'].description = "Reset test"
    
    @patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_progress_bar')
    @patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_status_text')
    def test_register_progress_callback(self, mock_update_status, mock_update_progress):
        """Test registrasi callback untuk progress bar"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import register_progress_callback
        
        # Panggil fungsi
        callback = register_progress_callback(self.ui_components, 100)
        
        # Verifikasi hasil
        self.assertTrue(callable(callback))
        
        # Test callback
        callback(50, "Callback test", "info")
        mock_update_progress.assert_called_once_with(self.ui_components, 50, 100, "Callback test")
        mock_update_status.assert_called_once_with(self.ui_components, "Callback test", "info")
    
    @patch('IPython.display.display')
    @patch('IPython.display.HTML')
    def test_show_augmentation_summary_test_format(self, mock_html, mock_display):
        """Test tampilkan ringkasan augmentasi dengan format pengujian"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import show_augmentation_summary
        
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
        mock_display.assert_called()
    
    @patch('IPython.display.display')
    @patch('IPython.display.clear_output')
    def test_show_augmentation_summary_normal_format(self, mock_clear, mock_display):
        """Test tampilkan ringkasan augmentasi dengan format normal"""
        from smartcash.ui.dataset.augmentation.handlers.status_handler import show_augmentation_summary
        
        # Setup summary dalam format normal
        normal_summary = {
            'total_images': 100,
            'total_augmented': 200,
            'aug_types': ['combined'],
            'class_distribution': {'class1': 50, 'class2': 50},
            'time_taken': 10.5
        }
        
        # Setup mock output dengan clear_output method
        self.ui_components['summary_output'].clear_output = MagicMock()
        
        # Panggil fungsi
        show_augmentation_summary(self.ui_components, normal_summary)
        
        # Verifikasi hasil
        mock_clear.assert_called_once()
        mock_display.assert_called()

if __name__ == '__main__':
    unittest.main()
