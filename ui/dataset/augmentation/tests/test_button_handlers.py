"""
File: smartcash/ui/dataset/augmentation/tests/test_button_handlers.py
Deskripsi: Pengujian untuk handler tombol pada modul augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import threading
import ipywidgets as widgets

# Import modul yang akan diuji
from smartcash.ui.dataset.augmentation.handlers.button_handlers import (
    setup_button_handlers,
    disable_ui_during_processing,
    cleanup_ui,
    reset_ui
)

class TestButtonHandlers(unittest.TestCase):
    """Kelas pengujian untuk button_handlers.py"""
    
    def setUp(self):
        """Setup untuk setiap pengujian"""
        # Mock komponen UI
        self.mock_ui_components = {
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
                },
                'data': {
                    'dataset_path': '/path/to/dataset'
                }
            },
            'aug_options': widgets.VBox([
                widgets.Dropdown(options=['Combined (Recommended)', 'Geometric', 'Color', 'Noise'], value='Combined (Recommended)'),
                widgets.Text(value='aug_'),
                widgets.Text(value='2'),
                widgets.Dropdown(options=['train', 'validation', 'test'], value='train'),
                widgets.Checkbox(value=False),
                widgets.IntText(value=4)
            ]),
            'progress_bar': MagicMock(),
            'current_progress': MagicMock(),
            'overall_label': MagicMock(),
            'step_label': MagicMock(),
            'output': MagicMock(),
            'augmentation_step': MagicMock(),
            'state': {
                'running': False,
                'completed': False,
                'stop_requested': False
            }
        }

    def test_setup_button_handlers(self):
        """Pengujian setup_button_handlers"""
        # Mock ensure_ui_persistence
        with patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.ensure_ui_persistence') as mock_ensure_persistence:
            # Panggil fungsi yang diuji
            result = setup_button_handlers(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result, self.mock_ui_components)
            
            # Verifikasi ensure_ui_persistence dipanggil
            mock_ensure_persistence.assert_called_once_with(self.mock_ui_components)
            
            # Verifikasi bahwa handler telah ditambahkan ke komponen UI
            self.assertIn('on_augment_click', self.mock_ui_components)
            self.assertIn('on_stop_click', self.mock_ui_components)
            self.assertIn('on_reset_click', self.mock_ui_components)

    def test_disable_ui_during_processing(self):
        """Pengujian disable_ui_during_processing"""
        # Setup mock UI components dengan struktur yang lebih lengkap
        mock_child = MagicMock()
        mock_child.disabled = False
        
        mock_tab_child = MagicMock()
        mock_tab_child.children = [mock_child]
        
        mock_tab = MagicMock()
        mock_tab.children = [mock_tab_child]
        
        mock_aug_options = MagicMock()
        mock_aug_options.children = [MagicMock(), mock_tab]
        
        self.mock_ui_components['aug_options'] = mock_aug_options
        self.mock_ui_components['save_button'] = MagicMock(disabled=False)
        self.mock_ui_components['reset_button'] = MagicMock(disabled=False)
        
        # Panggil fungsi yang diuji
        disable_ui_during_processing(self.mock_ui_components, True)
        
        # Verifikasi hasil
        self.assertTrue(mock_child.disabled)
        self.assertTrue(self.mock_ui_components['save_button'].disabled)
        self.assertTrue(self.mock_ui_components['reset_button'].disabled)
        
        # Test enable kembali
        disable_ui_during_processing(self.mock_ui_components, False)
        
        # Verifikasi hasil
        self.assertFalse(mock_child.disabled)
        self.assertFalse(self.mock_ui_components['save_button'].disabled)
        self.assertFalse(self.mock_ui_components['reset_button'].disabled)

    def test_cleanup_ui(self):
        """Pengujian cleanup_ui"""
        # Setup mock
        self.mock_ui_components['augment_button'] = MagicMock()
        self.mock_ui_components['augment_button'].layout = MagicMock()
        self.mock_ui_components['stop_button'] = MagicMock()
        self.mock_ui_components['stop_button'].layout = MagicMock()
        
        # Tambahkan progress components
        for element in ['progress_bar', 'current_progress', 'overall_label', 'step_label']:
            self.mock_ui_components[element] = MagicMock()
            self.mock_ui_components[element].layout = MagicMock()
            self.mock_ui_components[element].layout.visibility = 'visible'
            if element in ['progress_bar', 'current_progress']:
                self.mock_ui_components[element].value = 50
        
        # Patch reset_progress_tracking
        with patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.disable_ui_during_processing') as mock_disable:
            # Panggil fungsi yang diuji
            cleanup_ui(self.mock_ui_components)
            
            # Verifikasi hasil
            mock_disable.assert_called_once_with(self.mock_ui_components, False)
            self.assertEqual(self.mock_ui_components['augment_button'].layout.display, 'block')
            self.assertEqual(self.mock_ui_components['stop_button'].layout.display, 'none')

    def test_reset_ui(self):
        """Pengujian reset_ui"""
        # Setup mock
        self.mock_ui_components['visualization_container'] = MagicMock()
        self.mock_ui_components['visualization_container'].layout = MagicMock()
        self.mock_ui_components['summary_container'] = MagicMock()
        self.mock_ui_components['summary_container'].layout = MagicMock()
        self.mock_ui_components['visualization_buttons'] = MagicMock()
        self.mock_ui_components['visualization_buttons'].layout = MagicMock()
        self.mock_ui_components['cleanup_button'] = MagicMock()
        self.mock_ui_components['cleanup_button'].layout = MagicMock()
        self.mock_ui_components['status'] = MagicMock()
        self.mock_ui_components['log_accordion'] = MagicMock()
        
        # Patch cleanup_ui
        with patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.cleanup_ui') as mock_cleanup:
            # Panggil fungsi yang diuji
            reset_ui(self.mock_ui_components)
            
            # Verifikasi hasil
            mock_cleanup.assert_called_once_with(self.mock_ui_components)
            self.assertEqual(self.mock_ui_components['visualization_container'].layout.display, 'none')
            self.assertEqual(self.mock_ui_components['summary_container'].layout.display, 'none')
            self.assertEqual(self.mock_ui_components['visualization_buttons'].layout.display, 'none')
            self.assertEqual(self.mock_ui_components['cleanup_button'].layout.display, 'none')

    def test_setup_button_handlers_config_saving(self):
        """Pengujian setup_button_handlers dengan penyimpanan konfigurasi"""
        # Setup mock
        with patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.ensure_ui_persistence') as mock_ensure_persistence:
            # Panggil fungsi yang diuji
            result = setup_button_handlers(self.mock_ui_components, config={'test': 'config'})
            
            # Verifikasi ensure_ui_persistence dipanggil
            mock_ensure_persistence.assert_called_once_with(self.mock_ui_components)
            
            # Verifikasi bahwa handler telah ditambahkan ke komponen UI
            self.assertIn('on_augment_click', self.mock_ui_components)
            self.assertIn('on_stop_click', self.mock_ui_components)
            self.assertIn('on_reset_click', self.mock_ui_components)

if __name__ == '__main__':
    unittest.main()
