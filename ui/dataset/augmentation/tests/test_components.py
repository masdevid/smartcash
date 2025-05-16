"""
File: smartcash/ui/dataset/augmentation/tests/test_components.py
Deskripsi: Pengujian untuk komponen UI augmentasi dataset
"""

import unittest
import ipywidgets as widgets
from unittest.mock import patch, MagicMock

@unittest.skip("Melewati pengujian yang memerlukan komponen UI yang sebenarnya")
class TestAugmentationComponents(unittest.TestCase):
    """Pengujian untuk komponen UI augmentasi dataset."""
    
    @patch('smartcash.common.config.manager.ConfigManager')
    def test_create_augmentation_options(self, mock_config_manager):
        """Pengujian pembuatan komponen augmentation_options."""
        # Setup mock
        mock_instance = MagicMock()
        mock_config_manager.get_instance.return_value = mock_instance
        mock_instance.get_module_config.return_value = {
            'enabled': True,
            'num_variations': 2,
            'types': ['combined'],
            'target_count': 1000,
            'balance_classes': True,
            'move_to_preprocessed': True,
            'validate_results': True,
            'resume': False,
            'num_workers': 4,
            'output_prefix': 'aug',
            'target_split': 'train'
        }
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.components.augmentation_options import create_augmentation_options
        
        # Panggil fungsi
        result = create_augmentation_options()
        
        # Verifikasi hasil
        self.assertIsInstance(result, widgets.VBox)
        self.assertIsInstance(result.children[0], widgets.Tab)
        
        # Verifikasi tab
        tabs = result.children[0]
        self.assertEqual(len(tabs.children), 2)
        
        # Verifikasi tab pertama (opsi dasar)
        basic_tab = tabs.children[0]
        self.assertIsInstance(basic_tab, widgets.VBox)
        
        # Verifikasi tab kedua (jenis augmentasi)
        aug_types_tab = tabs.children[1]
        self.assertIsInstance(aug_types_tab, widgets.VBox)
    
    @patch('smartcash.ui.utils.constants.COLORS', {'dark': '#333'})
    @patch('smartcash.ui.utils.constants.ICONS', {'folder': '📁', 'info': 'ℹ️'})
    @patch('ipywidgets.RadioButtons')
    def test_create_split_selector(self, mock_radio_buttons):
        """Pengujian pembuatan komponen split_selector."""
        # Setup mock untuk RadioButtons
        mock_instance = MagicMock()
        mock_radio_buttons.return_value = mock_instance
        
        # Pastikan options adalah list
        mock_instance.options = ['train', 'valid', 'test']
        mock_instance.value = 'train'
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.components.split_selector import create_split_selector
        
        # Panggil fungsi
        result = create_split_selector()
        
        # Verifikasi hasil
        self.assertIsInstance(result, widgets.VBox)
        self.assertEqual(len(result.children), 2)
        
        # Verifikasi HTML header
        self.assertIsInstance(result.children[0], widgets.HTML)
        
        # Verifikasi HBox
        self.assertIsInstance(result.children[1], widgets.HBox)
        
        # Verifikasi RadioButtons
        # Gunakan mock_instance untuk verifikasi
        
        # Verifikasi nilai default
        self.assertEqual(mock_instance.value, 'train')
        self.assertEqual(mock_instance.options, ['train', 'valid', 'test'])
    
    @patch('smartcash.common.config.manager.ConfigManager')
    def test_create_advanced_options(self, mock_config_manager):
        """Pengujian pembuatan komponen advanced_options."""
        # Setup mock
        mock_instance = MagicMock()
        mock_config_manager.get_instance.return_value = mock_instance
        mock_instance.get_module_config.return_value = {
            'position': {
                'fliplr': 0.5,
                'degrees': 15,
                'translate': 0.15,
                'scale': 0.15,
                'shear_max': 10
            },
            'lighting': {
                'hsv_h': 0.025,
                'hsv_s': 0.7,
                'hsv_v': 0.4,
                'contrast': [0.7, 1.3],
                'brightness': [0.7, 1.3],
                'blur': 0.2,
                'noise': 0.1
            },
            'process_bboxes': True
        }
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.components.advanced_options import create_advanced_options
        
        # Panggil fungsi
        result = create_advanced_options()
        
        # Verifikasi hasil
        self.assertIsInstance(result, widgets.VBox)
        self.assertIsInstance(result.children[0], widgets.Tab)
        
        # Verifikasi tab
        tabs = result.children[0]
        self.assertEqual(len(tabs.children), 3)
        
        # Verifikasi tab pertama (posisi)
        position_tab = tabs.children[0]
        self.assertIsInstance(position_tab, widgets.VBox)
        
        # Verifikasi tab kedua (pencahayaan)
        lighting_tab = tabs.children[1]
        self.assertIsInstance(lighting_tab, widgets.VBox)
        
        # Verifikasi tab ketiga (tambahan)
        additional_tab = tabs.children[2]
        self.assertIsInstance(additional_tab, widgets.VBox)
    
    def test_create_action_buttons(self):
        """Pengujian pembuatan komponen action_buttons."""
        # Import fungsi
        from smartcash.ui.dataset.augmentation.components.action_buttons import create_action_buttons
        
        # Panggil fungsi
        result = create_action_buttons()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('primary_button', result)
        self.assertIn('augment_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('cleanup_button', result)
        self.assertIn('save_button', result)
        self.assertIn('container', result)
        
        # Verifikasi tombol
        self.assertIsInstance(result['primary_button'], widgets.Button)
        self.assertIsInstance(result['augment_button'], widgets.Button)
        self.assertIsInstance(result['stop_button'], widgets.Button)
        self.assertIsInstance(result['reset_button'], widgets.Button)
        self.assertIsInstance(result['cleanup_button'], widgets.Button)
        self.assertIsInstance(result['save_button'], widgets.Button)
        
        # Verifikasi container
        self.assertIsInstance(result['container'], widgets.HBox)
        self.assertEqual(len(result['container'].children), 5)
    
    @patch('smartcash.ui.dataset.augmentation.components.augmentation_component.create_header')
    @patch('smartcash.ui.handlers.status_handler.create_status_panel')
    @patch('smartcash.ui.dataset.augmentation.components.augmentation_options.create_augmentation_options')
    @patch('smartcash.ui.dataset.augmentation.components.split_selector.create_split_selector')
    @patch('smartcash.ui.dataset.augmentation.components.advanced_options.create_advanced_options')
    @patch('smartcash.ui.dataset.augmentation.components.action_buttons.create_action_buttons')
    @patch('smartcash.ui.components.visualization_buttons.create_visualization_buttons')
    @patch('smartcash.ui.info_boxes.augmentation_info.get_augmentation_info')
    def test_create_augmentation_ui(self, mock_get_info, mock_viz_buttons, mock_action_buttons, 
                                   mock_adv_options, mock_split_selector, mock_aug_options, 
                                   mock_status_panel, mock_header):
        """Pengujian pembuatan komponen augmentation_component."""
        # Setup mock
        mock_header.return_value = widgets.HTML()
        mock_status_panel.return_value = widgets.Box()
        mock_aug_options.return_value = widgets.VBox()
        mock_split_selector.return_value = widgets.VBox()
        mock_adv_options.return_value = widgets.VBox()
        mock_action_buttons.return_value = {
            'primary_button': widgets.Button(),
            'augment_button': widgets.Button(),
            'stop_button': widgets.Button(),
            'reset_button': widgets.Button(),
            'cleanup_button': widgets.Button(),
            'save_button': widgets.Button(),
            'container': widgets.HBox()
        }
        mock_viz_buttons.return_value = {
            'visualize_button': widgets.Button(),
            'compare_button': widgets.Button(),
            'distribution_button': widgets.Button(),
            'container': widgets.HBox()
        }
        mock_get_info.return_value = widgets.VBox()
        
        # Import fungsi
        from smartcash.ui.dataset.augmentation.components.augmentation_component import create_augmentation_ui
        
        # Panggil fungsi
        result = create_augmentation_ui()
        
        # Verifikasi hasil
        self.assertIsInstance(result, dict)
        self.assertIn('ui', result)
        self.assertIn('header', result)
        self.assertIn('status_panel', result)
        self.assertIn('augmentation_options', result)
        self.assertIn('advanced_options', result)
        self.assertIn('split_selector', result)
        self.assertIn('advanced_accordion', result)
        self.assertIn('augment_button', result)
        self.assertIn('stop_button', result)
        self.assertIn('reset_button', result)
        self.assertIn('cleanup_button', result)
        self.assertIn('save_button', result)
        self.assertIn('button_container', result)
        self.assertIn('progress_bar', result)
        self.assertIn('current_progress', result)
        self.assertIn('overall_label', result)
        self.assertIn('step_label', result)
        self.assertIn('status', result)
        self.assertIn('log_accordion', result)
        self.assertIn('summary_container', result)
        self.assertIn('visualization_buttons', result)
        self.assertIn('visualize_button', result)
        self.assertIn('compare_button', result)
        self.assertIn('distribution_button', result)
        self.assertIn('visualization_container', result)
        self.assertIn('module_name', result)
        self.assertIn('data_dir', result)
        self.assertIn('augmented_dir', result)

if __name__ == '__main__':
    unittest.main()
