"""
File: smartcash/ui/dataset/augmentation/tests/test_integration.py
Deskripsi: Pengujian integrasi untuk modul augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
import threading
import os

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.augmentation_initializer import initialize_augmentation_ui
from smartcash.ui.dataset.augmentation.handlers.button_handlers import setup_button_handlers
from smartcash.ui.dataset.augmentation.handlers.config_handlers import update_config_from_ui, update_ui_from_config
from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state

class TestAugmentationIntegration(unittest.TestCase):
    """Kelas pengujian integrasi untuk modul augmentasi"""
    
    @patch('smartcash.ui.dataset.augmentation.augmentation_initializer.initialize_module_ui')
    @patch('smartcash.ui.dataset.augmentation.augmentation_initializer.create_augmentation_ui')
    @patch('smartcash.ui.dataset.augmentation.augmentation_initializer.setup_augmentation_config_handler')
    @patch('smartcash.ui.dataset.augmentation.augmentation_initializer.detect_augmentation_state')
    def test_initialize_augmentation_ui(self, mock_detect, mock_setup, mock_create, mock_initialize):
        """Pengujian initialize_augmentation_ui"""
        # Setup mock
        mock_ui_components = {
            'ui': MagicMock(),
            'augment_button': MagicMock(),
            'stop_button': MagicMock(),
            'reset_button': MagicMock(),
            'cleanup_button': MagicMock(),
            'save_button': MagicMock(),
            'aug_options': MagicMock(),
            'progress_bar': MagicMock(),
            'current_progress': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'output': MagicMock(),
            'state': {'running': False, 'completed': False, 'stop_requested': False}
        }
        mock_initialize.return_value = mock_ui_components
        
        # Panggil fungsi yang diuji
        result = initialize_augmentation_ui()
        
        # Verifikasi hasil
        self.assertEqual(result, mock_ui_components)
        mock_initialize.assert_called_once()
        
        # Verifikasi parameter yang diberikan ke initialize_module_ui
        args, kwargs = mock_initialize.call_args
        self.assertEqual(kwargs['module_name'], 'augmentation')
        self.assertEqual(kwargs['create_ui_func'], mock_create)
        self.assertEqual(kwargs['setup_config_handler_func'], mock_setup)
        self.assertEqual(kwargs['detect_state_func'], mock_detect)
        self.assertIn('button_keys', kwargs)
        self.assertIn('multi_progress_config', kwargs)
        self.assertEqual(kwargs['observer_group'], 'augmentation_observers')

    @patch('threading.Thread')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.save_augmentation_config')
    @patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.map_ui_to_config')
    def test_augmentation_workflow(self, mock_map_ui_to_config, mock_save_config, mock_thread):
        """Pengujian alur kerja augmentasi dari klik tombol hingga eksekusi"""
        # Setup mock
        mock_ui_components = {
            'augment_button': widgets.Button(description='Augment'),
            'stop_button': widgets.Button(description='Stop'),
            'reset_button': widgets.Button(description='Reset'),
            'cleanup_button': widgets.Button(description='Cleanup'),
            'status': MagicMock(),
            'logger': MagicMock(),
            'config': {
                'augmentation': {
                    'types': ['Combined (Recommended)'],
                    'prefix': 'aug_',
                    'factor': '2',
                    'split': 'train',
                    'balance_classes': False,
                    'num_workers': 4
                }
            }
        }
        
        # Setup button handlers
        ui_with_handlers = setup_button_handlers(mock_ui_components)
        
        # Verifikasi bahwa handler dikembalikan
        self.assertIn('on_augment_click', ui_with_handlers)
        self.assertIn('on_stop_click', ui_with_handlers)
        self.assertTrue(callable(ui_with_handlers['on_augment_click']))
        self.assertTrue(callable(ui_with_handlers['on_stop_click']))
        
        # Verifikasi bahwa UI components diupdate
        self.assertEqual(ui_with_handlers, mock_ui_components)

    @patch('smartcash.ui.dataset.augmentation.handlers.config_handlers.load_augmentation_config')
    @patch('smartcash.ui.dataset.augmentation.handlers.config_handlers.update_ui_from_config')
    def test_config_persistence_workflow(self, mock_update_ui, mock_load):
        """Pengujian alur kerja persistensi konfigurasi"""
        # Setup mock
        mock_ui_components = {
            'status': MagicMock(),
            'logger': MagicMock(),
            'config': None,
            'aug_options': widgets.VBox([
                widgets.Dropdown(options=['Combined (Recommended)', 'Geometric', 'Color', 'Noise'], value='Combined (Recommended)'),
                widgets.Text(value='aug_'),
                widgets.Text(value='2'),
                widgets.Dropdown(options=['train', 'validation', 'test'], value='train'),
                widgets.Checkbox(value=False),
                widgets.IntText(value=4)
            ])
        }
        
        # Mock konfigurasi yang dimuat
        mock_config = {
            'augmentation': {
                'types': ['Geometric'],
                'prefix': 'custom_',
                'factor': '3',
                'split': 'validation',
                'balance_classes': True,
                'num_workers': 8
            },
            'data': {
                'dataset_path': '/path/to/custom/dataset'
            }
        }
        mock_load.return_value = mock_config
        
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.config_handlers import setup_augmentation_config_handler
        
        # Panggil fungsi yang diuji
        result = setup_augmentation_config_handler(mock_ui_components)
        
        # Verifikasi hasil
        mock_load.assert_called_once()
        mock_update_ui.assert_called_once_with(mock_ui_components, mock_config)
        # Verifikasi config diatur
        self.assertIsNotNone(result['config'])

if __name__ == '__main__':
    unittest.main()
