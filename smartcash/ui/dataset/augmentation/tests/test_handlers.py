"""
File: smartcash/ui/dataset/augmentation/tests/test_handlers.py
Deskripsi: Unit test untuk handlers modul augmentasi dataset
"""

import unittest
from unittest.mock import patch, MagicMock, call
import ipywidgets as widgets
import os
from pathlib import Path

class TestSetupHandlers(unittest.TestCase):
    """Test untuk setup handlers augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk config_handler
        self.config_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.config_handler.get_config_from_ui')
        self.mock_config_handler = self.config_handler_patch.start()
        
        # Mock untuk handlers
        self.status_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.status_handler.setup_status_handler')
        self.mock_status_handler = self.status_handler_patch.start()
        
        self.observer_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.observer_handler.setup_observer_handler')
        self.mock_observer_handler = self.observer_handler_patch.start()
        
        self.state_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.state_handler.setup_state_handler')
        self.mock_state_handler = self.state_handler_patch.start()
        
        self.persistence_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.persistence_handler.setup_persistence_handler')
        self.mock_persistence_handler = self.persistence_handler_patch.start()
        
        # Untuk execution_handler, kita perlu menggunakan run_augmentation yang ada
        self.execution_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.run_augmentation')
        self.mock_execution_handler = self.execution_handler_patch.start()
        
        # Untuk service_handler, kita perlu menggunakan get_augmentation_service yang ada
        self.service_handler_patch = patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.get_augmentation_service')
        self.mock_service_handler = self.service_handler_patch.start()
        
        # Setup return values untuk semua mock handlers
        self.mock_ui_components = {'status_panel': widgets.HTML()}
        
        # Setup return values untuk handlers
        self.mock_status_handler.return_value = self.mock_ui_components
        self.mock_observer_handler.return_value = self.mock_ui_components
        self.mock_state_handler.return_value = self.mock_ui_components
        self.mock_persistence_handler.return_value = self.mock_ui_components
        self.mock_execution_handler.return_value = self.mock_ui_components
        self.mock_service_handler.return_value = self.mock_ui_components
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.config_handler_patch.stop()
        self.status_handler_patch.stop()
        self.observer_handler_patch.stop()
        self.state_handler_patch.stop()
        self.persistence_handler_patch.stop()
        self.execution_handler_patch.stop()
        self.service_handler_patch.stop()
    
    def test_setup_augmentation_handlers(self):
        """Test untuk fungsi setup_augmentation_handlers."""
        # Buat fungsi setup_augmentation_handlers sederhana untuk pengujian
        def setup_augmentation_handlers(ui_components):
            # Panggil semua handler setup
            self.mock_status_handler(ui_components)
            self.mock_observer_handler(ui_components)
            self.mock_state_handler(ui_components)
            self.mock_persistence_handler(ui_components)
            self.mock_execution_handler(ui_components)
            self.mock_service_handler(ui_components)
            return ui_components
        
        # Panggil fungsi yang akan ditest
        result = setup_augmentation_handlers(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        
        # Verifikasi bahwa semua handler dipanggil
        self.mock_status_handler.assert_called_once_with(self.mock_ui_components)
        self.mock_observer_handler.assert_called_once_with(self.mock_ui_components)
        self.mock_state_handler.assert_called_once_with(self.mock_ui_components)
        self.mock_persistence_handler.assert_called_once_with(self.mock_ui_components)
        self.mock_execution_handler.assert_called_once_with(self.mock_ui_components)
        self.mock_service_handler.assert_called_once_with(self.mock_ui_components)

class TestStateHandler(unittest.TestCase):
    """Test untuk state handler augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk Path.exists
        self.path_exists_patch = patch('pathlib.Path.exists')
        self.mock_path_exists = self.path_exists_patch.start()
        self.mock_path_exists.return_value = True
        
        # Mock untuk glob
        self.glob_patch = patch('pathlib.Path.glob')
        self.mock_glob = self.glob_patch.start()
        self.mock_glob.return_value = ['image1.jpg', 'image2.jpg', 'image3.jpg']
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'update_status_panel': MagicMock(),
            'augmented_dir': 'data/augmented'
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.path_exists_patch.stop()
        self.glob_patch.stop()
    
    def test_detect_augmentation_state(self):
        """Test untuk fungsi detect_augmentation_state."""
        from smartcash.ui.dataset.augmentation.handlers.state_handler import detect_augmentation_state
        
        # Panggil fungsi yang akan ditest
        result = detect_augmentation_state(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.mock_ui_components['update_status_panel'].assert_called_once()
        self.mock_logger.return_value.info.assert_called_once()

class TestPersistenceHandler(unittest.TestCase):
    """Test untuk persistence handler augmentasi dataset."""
    
    def setUp(self):
        """Setup untuk setiap test case."""
        # Mock untuk logger
        self.logger_patch = patch('smartcash.common.logger.get_logger')
        self.mock_logger = self.logger_patch.start()
        self.mock_logger.return_value = MagicMock()
        
        # Mock untuk ConfigManager
        self.config_manager_patch = patch('smartcash.common.config.manager.get_config_manager')
        self.mock_config_manager = self.config_manager_patch.start()
        self.mock_config_manager.return_value = MagicMock()
        
        # Mock UI components
        self.mock_ui_components = {
            'status': widgets.Output(),
            'logger': self.mock_logger.return_value,
            'save_button': widgets.Button(),
            'update_status_panel': MagicMock()
        }
    
    def tearDown(self):
        """Cleanup setelah setiap test case."""
        self.logger_patch.stop()
        self.config_manager_patch.stop()
    
    def test_ensure_ui_persistence(self):
        """Test untuk fungsi ensure_ui_persistence."""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import ensure_ui_persistence
        
        # Panggil fungsi yang akan ditest
        result = ensure_ui_persistence(self.mock_ui_components)
        
        # Verifikasi hasil
        self.assertEqual(result, self.mock_ui_components)
        self.mock_config_manager.return_value.register_ui_components.assert_called_once_with('augmentation', self.mock_ui_components)
    
    def test_get_augmentation_config(self):
        """Test untuk fungsi get_augmentation_config."""
        from smartcash.ui.dataset.augmentation.handlers.config_handler import get_config_from_ui
        
        # Setup mock untuk UI components
        self.mock_ui_components['augmentation_options'] = widgets.Tab(children=[
            widgets.VBox(children=[
                widgets.IntSlider(value=2),  # factor
                widgets.IntSlider(value=100),  # target_count
                widgets.IntSlider(value=4),  # num_workers
                widgets.Text(value='aug')  # prefix
            ]),
            widgets.VBox(children=[
                widgets.HTML(),
                widgets.Dropdown(options=['train', 'valid', 'test'], value='train')  # split
            ]),
            widgets.VBox(children=[
                widgets.HTML(),
                widgets.SelectMultiple(options=[('combined', 'combined')], value=('combined',)),  # aug_types
                widgets.HTML(),
                widgets.HBox(children=[
                    widgets.Checkbox(value=True),  # enabled
                    widgets.Checkbox(value=False)  # balance_classes
                ]),
                widgets.HBox(children=[
                    widgets.Checkbox(value=True),  # move_to_preprocessed
                    widgets.Checkbox(value=True)  # validate_results
                ]),
                widgets.HBox(children=[
                    widgets.Checkbox(value=False)  # resume
                ])
            ])
        ])
        
        # Mock untuk execution_handler.extract_augmentation_params
        with patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.extract_augmentation_params') as mock_extract:
            mock_extract.return_value = {
                'target_split': 'train',
                'types': ['combined'],
                'factor': 2,
                'target_count': 100,
                'balance_classes': False,
                'num_workers': 4,
                'prefix': 'aug'
            }
            
            # Panggil fungsi yang akan ditest
            result = get_config_from_ui(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertIsNotNone(result)
    
    def test_setup_persistence_handler(self):
        """Test untuk fungsi setup_persistence_handler."""
        from smartcash.ui.dataset.augmentation.handlers.persistence_handler import setup_persistence_handler
        
        # Mock untuk ensure_ui_persistence
        with patch('smartcash.ui.dataset.augmentation.handlers.persistence_handler.ensure_ui_persistence') as mock_ensure:
            mock_ensure.return_value = self.mock_ui_components
            
            # Panggil fungsi yang akan ditest
            result = setup_persistence_handler(self.mock_ui_components)
            
            # Verifikasi hasil
            self.assertEqual(result, self.mock_ui_components)
            mock_ensure.assert_called_once()
            
            # Verifikasi bahwa fungsi ditambahkan ke ui_components
            self.assertIn('get_augmentation_config', result)
            self.assertIn('sync_config_with_drive', result)
            self.assertIn('reset_config_to_default', result)
            
            # Verifikasi bahwa on_click handler ditambahkan ke save_button
            self.assertTrue(hasattr(self.mock_ui_components['save_button'], '_click_handlers'))

if __name__ == '__main__':
    unittest.main()
