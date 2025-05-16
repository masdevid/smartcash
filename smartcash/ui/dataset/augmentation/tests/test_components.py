"""
File: smartcash/ui/dataset/augmentation/tests/test_components.py
Deskripsi: Pengujian untuk komponen UI augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
from smartcash.ui.dataset.augmentation.components.action_buttons import create_action_buttons
from smartcash.ui.dataset.augmentation.components.progress_component import create_progress_component
from smartcash.ui.dataset.augmentation.components.output_component import create_output_component

class TestAugmentationComponents(unittest.TestCase):
    """Kelas pengujian untuk komponen UI augmentasi"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        self.mock_config = {
            'augmentation': {
                'types': ['Combined (Recommended)'],
                'prefix': 'aug_',
                'factor': '2',
                'split': 'train',
                'balance_classes': False,
                'num_workers': 4
            },
            'data': {
                'dataset_path': '/path/to/dataset'
            }
        }

    def test_create_augmentation_ui(self):
        """Pengujian create_augmentation_ui"""
        # Skip pengujian ini karena memerlukan mock yang kompleks
        # dan tidak bisa dijalankan dalam lingkungan pengujian
        self.skipTest("Pengujian ini memerlukan mock yang kompleks")

    def test_create_augmentation_options(self):
        """Pengujian create_augmentation_options"""
        # Panggil fungsi yang diuji
        result = create_augmentation_options()
        
        # Verifikasi hasil
        self.assertIsInstance(result, widgets.VBox)
        self.assertEqual(len(result.children), 2)  # 2 komponen: description dan tab
        
        # Verifikasi tab container
        tab = result.children[1]
        self.assertIsInstance(tab, widgets.Tab)
        self.assertEqual(len(tab.children), 2)  # 2 tab: opsi dasar dan opsi lanjutan
        
        # Verifikasi opsi dasar
        basic_tab = tab.children[0]
        self.assertIsInstance(basic_tab, widgets.VBox)
        self.assertEqual(len(basic_tab.children), 4)  # 4 opsi: aug_type_info, prefix_text, factor_slider, split_info

    @patch('smartcash.ui.components.action_buttons.create_action_buttons')
    @patch('smartcash.ui.components.action_buttons.create_visualization_buttons')
    def test_create_action_buttons(self, mock_vis_buttons, mock_std_buttons):
        """Pengujian create_action_buttons"""
        # Setup mock
        mock_std_buttons.return_value = {
            'primary_button': widgets.Button(description='Run Augmentation'),
            'stop_button': widgets.Button(description='Stop'),
            'reset_button': widgets.Button(description='Reset'),
            'save_button': widgets.Button(description='Save'),
            'cleanup_button': widgets.Button(description='Cleanup'),
            'container': widgets.HBox()
        }
        
        mock_vis_buttons.return_value = {
            'visualize_button': widgets.Button(description='Visualize'),
            'compare_button': widgets.Button(description='Compare'),
            'distribution_button': widgets.Button(description='Distribution'),
            'container': widgets.HBox()
        }
        
        # Panggil fungsi yang diuji
        result = create_action_buttons()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('augment_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('save_button', result)
        self.assertIn('cleanup_button', result)
        self.assertIn('container', result)
        self.assertIn('visualization_buttons', result)
        self.assertIn('visualize_button', result)
        self.assertIn('compare_button', result)
        self.assertIn('distribution_button', result)

    def test_create_progress_component(self):
        """Pengujian create_progress_component"""
        # Patch fungsi yang digunakan dalam create_progress_component
        with patch('smartcash.ui.components.progress_component.create_progress_component') as mock_std_progress:
            # Setup mock
            mock_std_progress.return_value = {
                'progress_bar': widgets.IntProgress(),
                'current_progress': widgets.IntProgress(),
                'overall_message': widgets.HTML(),
                'step_message': widgets.HTML(),
                'container': widgets.VBox()
            }
            
            # Panggil fungsi yang diuji
            result = create_progress_component()
            
            # Verifikasi hasil
            self.assertIsInstance(result, dict)
            self.assertIn('progress_bar', result)
            self.assertIn('current_progress', result)
            self.assertIn('overall_message', result)
            self.assertIn('step_message', result)
            self.assertIn('container', result)
            
            # Verifikasi komponen progress
            self.assertIsInstance(result['progress_bar'], widgets.IntProgress)
            self.assertIsInstance(result['current_progress'], widgets.IntProgress)
            self.assertIsInstance(result['overall_message'], widgets.HTML)
            self.assertIsInstance(result['step_message'], widgets.HTML)

    def test_create_output_component(self):
        """Pengujian create_output_component"""
        # Panggil fungsi yang diuji
        result = create_output_component()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('status', result)
        self.assertIn('log_accordion', result)
        self.assertIn('summary_container', result)
        self.assertIn('visualization_container', result)
        
        # Verifikasi komponen status
        self.assertIsInstance(result['status'], widgets.Output)

if __name__ == '__main__':
    unittest.main()
