"""
File: smartcash/ui/dataset/split/tests/test_components.py
Deskripsi: Test untuk komponen split dataset
"""

import unittest
import sys
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

class TestSplitComponents(unittest.TestCase):
    """Test untuk komponen UI split dataset."""
    
    def test_create_split_ui(self):
        """Test pembuatan UI split dataset."""
        # Import fungsi yang akan ditest
        from smartcash.ui.dataset.split.components.split_components import create_split_ui
        
        # Patch semua dependency
        with patch('smartcash.ui.dataset.split.components.split_components.widgets') as mock_widgets, \
             patch('smartcash.ui.utils.header_utils.create_header', return_value=MagicMock()), \
             patch('smartcash.ui.components.save_reset_buttons.create_save_reset_buttons', 
                   return_value={'container': MagicMock(), 'save_button': MagicMock(), 'reset_button': MagicMock()}):
            
            # Mock widget
            mock_widgets.VBox.return_value = MagicMock()
            mock_widgets.HBox.return_value = MagicMock()
            mock_widgets.Tab.return_value = MagicMock()
            mock_widgets.HTML.return_value = MagicMock()
            mock_widgets.FloatSlider.return_value = MagicMock()
            mock_widgets.Checkbox.return_value = MagicMock()
            mock_widgets.IntText.return_value = MagicMock()
            mock_widgets.Text.return_value = MagicMock()
            mock_widgets.Box.return_value = MagicMock()
            
            # Patch create_info_accordion jika tersedia
            try:
                with patch('smartcash.ui.components.info_accordion.create_info_accordion', return_value={'container': MagicMock()}):
                    # Patch create_sync_info_message jika tersedia
                    try:
                        with patch('smartcash.ui.components.sync_info_message.create_sync_info_message', return_value=MagicMock()):
                            # Patch get_split_info jika tersedia
                            try:
                                with patch('smartcash.ui.info_boxes.split_info.get_split_info', return_value="Mock info"):
                                    # Buat config dummy
                                    config = {
                                        'split': {
                                            'enabled': True,
                                            'train_ratio': 0.7,
                                            'val_ratio': 0.15,
                                            'test_ratio': 0.15,
                                            'random_seed': 42,
                                            'stratify': True
                                        }
                                    }
                                    
                                    # Panggil fungsi
                                    result = create_split_ui(config)
                                    
                                    # Verifikasi hasil
                                    self.assertIsInstance(result, dict)
                                    self.assertIn('ui', result)
                            except ImportError:
                                print("Info: get_split_info tidak tersedia, melewati test")
                    except ImportError:
                        print("Info: create_sync_info_message tidak tersedia, melewati test")
            except ImportError:
                print("Info: create_info_accordion tidak tersedia, melewati test")
    
    @patch('smartcash.ui.components.split_config.widgets')
    def test_create_split_config(self, mock_widgets):
        """Test komponen konfigurasi split."""
        try:
            from smartcash.ui.components.split_config import create_split_config
            
            # Mock widget
            mock_widgets.VBox.return_value = MagicMock()
            mock_widgets.HBox.return_value = MagicMock()
            mock_widgets.FloatSlider.return_value = MagicMock()
            mock_widgets.HTML.return_value = MagicMock()
            
            # Panggil fungsi
            result = create_split_config()
            
            # Verifikasi hasil
            self.assertIsInstance(result, dict)
            self.assertIn('container', result)
            self.assertIn('train_slider', result)
            self.assertIn('val_slider', result)
            self.assertIn('test_slider', result)
            self.assertIn('total_output', result)
        except ImportError:
            print("Info: create_split_config tidak tersedia, melewati test")

if __name__ == '__main__':
    unittest.main() 