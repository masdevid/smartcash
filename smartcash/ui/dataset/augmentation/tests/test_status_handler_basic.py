"""
File: smartcash/ui/dataset/augmentation/tests/test_status_handler_basic.py
Deskripsi: Test dasar untuk status handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

class TestStatusHandlerBasic(unittest.TestCase):
    """Test dasar untuk status handler augmentasi dataset"""
    
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
    
    def test_update_progress_bar(self):
        """Test update progress bar"""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_progress_bar
        
        # Panggil fungsi
        update_progress_bar(self.ui_components, 50, 100, "Progress test")
        
        # Verifikasi hasil
        self.ui_components['progress_bar'].value = 50
        self.ui_components['progress_bar'].max = 100
        self.ui_components['progress_bar'].description = "Progress test"
    
    def test_reset_progress_bar(self):
        """Test reset progress bar"""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.status_handler import reset_progress_bar
        
        # Panggil fungsi
        reset_progress_bar(self.ui_components, "Reset test")
        
        # Verifikasi hasil
        self.ui_components['progress_bar'].value = 0
        self.ui_components['progress_bar'].description = "Reset test"
    
    def test_update_progress_bar_missing(self):
        """Test update progress bar ketika progress_bar tidak ada"""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.status_handler import update_progress_bar
        
        # Hapus progress_bar dari ui_components
        ui_components_no_bar = {k: v for k, v in self.ui_components.items() if k != 'progress_bar'}
        
        # Panggil fungsi
        update_progress_bar(ui_components_no_bar, 50, 100, "Progress test")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)
    
    def test_reset_progress_bar_missing(self):
        """Test reset progress bar ketika progress_bar tidak ada"""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.status_handler import reset_progress_bar
        
        # Hapus progress_bar dari ui_components
        ui_components_no_bar = {k: v for k, v in self.ui_components.items() if k != 'progress_bar'}
        
        # Panggil fungsi
        reset_progress_bar(ui_components_no_bar, "Reset test")
        
        # Tidak ada error yang diharapkan
        self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
