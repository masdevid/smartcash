"""
File: smartcash/ui/dataset/preprocessing/tests/test_preprocessing_integration.py
Deskripsi: Integration test untuk preprocessing dataset
"""

import unittest
import os
import shutil
import tempfile
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.dataset.preprocessing import (
    initialize_preprocessing_ui,
    get_preprocessing_ui_components,
    reset_preprocessing_ui,
    get_preprocessing_config,
    execute_preprocessing,
    save_preprocessing_config,
    reset_preprocessing_config
)

class TestPreprocessingIntegration(unittest.TestCase):
    """Integration test untuk modul preprocessing dataset."""
    
    def setUp(self):
        """Setup test environment sebelum setiap test case dijalankan."""
        # Buat temporary directory untuk test dataset
        self.test_dir = tempfile.mkdtemp()
        self.test_img_dir = os.path.join(self.test_dir, "images")
        self.test_labels_dir = os.path.join(self.test_dir, "labels")
        
        # Buat struktur direktori
        os.makedirs(os.path.join(self.test_img_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.test_img_dir, "valid"), exist_ok=True)
        os.makedirs(os.path.join(self.test_labels_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(self.test_labels_dir, "valid"), exist_ok=True)
        
        # Mock config untuk test
        self.test_config = {
            "dataset": {
                "preprocessing": {
                    "enabled": True,
                    "img_size": 640,
                    "splits": ["train", "valid"],
                    "normalization": {
                        "enabled": True,
                        "preserve_aspect_ratio": True
                    },
                    "validate": {
                        "enabled": True,
                        "fix_issues": True,
                        "move_invalid": False,
                        "invalid_dir": "invalid"
                    },
                    "num_workers": 2
                },
                "paths": {
                    "dataset_dir": self.test_dir,
                    "images_dir": self.test_img_dir,
                    "labels_dir": self.test_labels_dir
                }
            }
        }
        
        # Patch get_config_manager untuk testing
        self.config_manager_patcher = patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.get_config_manager')
        self.mock_config_manager = self.config_manager_patcher.start()
        self.mock_config_manager.return_value.get_module_config.return_value = self.test_config["dataset"]
        
        # Patch logger
        self.logger_patcher = patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.get_logger')
        self.mock_logger = self.logger_patcher.start()
        
        # Patch UI logger
        self.ui_logger_patcher = patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.setup_ui_logger')
        self.mock_ui_logger = self.ui_logger_patcher.start()
        self.mock_ui_logger.return_value = {}
        
        # Patch component creation
        self.component_patcher = patch('smartcash.ui.dataset.preprocessing.preprocessing_initializer.create_preprocessing_ui')
        self.mock_component = self.component_patcher.start()
        self.mock_component.return_value = {
            'ui': widgets.VBox(),
            'preprocess_options': MagicMock(children=[
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            ]),
            'validation_options': MagicMock(children=[
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            ]),
            'split_selector': MagicMock(),
            'status': MagicMock(),
            'progress': MagicMock(),
            'logger': MagicMock()
        }
        
    def tearDown(self):
        """Clean up test environment setelah setiap test case selesai."""
        # Hentikan semua patchers
        self.config_manager_patcher.stop()
        self.logger_patcher.stop()
        self.ui_logger_patcher.stop()
        self.component_patcher.stop()
        
        # Hapus direktori temporer
        shutil.rmtree(self.test_dir)
    
    def test_initialize_preprocessing_ui(self):
        """Test inisialisasi UI preprocessing."""
        # Act
        ui = initialize_preprocessing_ui()
        
        # Assert
        self.assertIsNotNone(ui)
        self.mock_component.assert_called_once()
        self.mock_ui_logger.assert_called_once()
    
    def test_get_preprocessing_ui_components(self):
        """Test mendapatkan komponen UI preprocessing yang sudah diinisialisasi."""
        # Arrange
        initialize_preprocessing_ui()
        
        # Act
        components = get_preprocessing_ui_components()
        
        # Assert
        self.assertIsNotNone(components)
        self.assertIn('ui', components)
        self.assertIn('preprocessing_initialized', components)
        self.assertEqual(components['preprocessing_initialized'], True)
    
    def test_reset_preprocessing_ui(self):
        """Test reset UI preprocessing."""
        # Arrange
        initialize_preprocessing_ui()
        
        # Act
        ui = reset_preprocessing_ui()
        
        # Assert
        self.assertIsNotNone(ui)
        self.assertEqual(self.mock_component.call_count, 2)
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.get_preprocessing_config')
    @patch('smartcash.ui.dataset.preprocessing.handlers.button_handler.execute_preprocessing')
    def test_execute_preprocessing(self, mock_execute, mock_get_config):
        """Test eksekusi preprocessing."""
        # Arrange
        mock_get_config.return_value = self.test_config["dataset"]["preprocessing"]
        mock_execute.return_value = {"success": True, "processed_images": 10}
        ui_components = initialize_preprocessing_ui()
        components = get_preprocessing_ui_components()
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.button_handler.confirm_preprocessing') as mock_confirm:
            mock_confirm.return_value = True
            result = execute_preprocessing(components)
        
        # Assert
        self.assertTrue(mock_execute.called)
        self.assertIn("success", result)
        self.assertTrue(result["success"])
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.get_config_manager')
    def test_save_preprocessing_config(self, mock_config_manager):
        """Test menyimpan konfigurasi preprocessing."""
        # Arrange
        mock_config_manager.return_value.get_module_config.return_value = self.test_config["dataset"]
        mock_save = MagicMock()
        mock_config_manager.return_value.save_module_config = mock_save
        ui_components = initialize_preprocessing_ui()
        components = get_preprocessing_ui_components()
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.update_config_from_ui') as mock_update:
            mock_update.return_value = self.test_config["dataset"]
            result = save_preprocessing_config(components)
        
        # Assert
        mock_save.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.get_config_manager')
    def test_reset_preprocessing_config(self, mock_config_manager):
        """Test reset konfigurasi preprocessing."""
        # Arrange
        default_config = {
            "preprocessing": {
                "enabled": True,
                "img_size": 640,
                "splits": ["train"],
                "normalization": {"enabled": True, "preserve_aspect_ratio": True},
                "validate": {"enabled": True, "fix_issues": True, "move_invalid": False, "invalid_dir": "invalid"},
                "num_workers": 4
            }
        }
        mock_config_manager.return_value.get_default_module_config.return_value = default_config
        mock_config_manager.return_value.get_module_config.return_value = self.test_config["dataset"]
        ui_components = initialize_preprocessing_ui()
        components = get_preprocessing_ui_components()
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.update_ui_from_config') as mock_update_ui:
            result = reset_preprocessing_config(components)
        
        # Assert
        mock_update_ui.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.preprocessing_service_handler.handle_preprocessing_service')
    def test_integration_preprocessing_flow(self, mock_service_handler):
        """Test integrasi alur kerja preprocessing dari inisialisasi hingga eksekusi."""
        # Arrange
        mock_service_handler.return_value = {}
        
        # Act - Initialize UI
        ui = initialize_preprocessing_ui()
        components = get_preprocessing_ui_components()
        
        # Assert - Check initialization
        self.assertIsNotNone(ui)
        self.assertIsNotNone(components)
        self.assertTrue(components['preprocessing_initialized'])
        self.assertFalse(components['preprocessing_running'])
        
        # Act - Configure preprocessing
        with patch('smartcash.ui.dataset.preprocessing.handlers.config_handler.update_config_from_ui') as mock_update:
            mock_update.return_value = self.test_config["dataset"]
            updated_config = save_preprocessing_config(components)
        
        # Act - Execute preprocessing
        with patch('smartcash.ui.dataset.preprocessing.handlers.button_handler.execute_preprocessing') as mock_execute:
            mock_execute.return_value = {"success": True, "processed_images": 10, "execution_time": 1.5}
            with patch('smartcash.ui.dataset.preprocessing.handlers.button_handler.confirm_preprocessing') as mock_confirm:
                mock_confirm.return_value = True
                result = execute_preprocessing(components)
        
        # Assert - Check execution result
        self.assertIsNotNone(result)
        self.assertTrue(result["success"])
        self.assertEqual(result["processed_images"], 10)
        self.assertEqual(result["execution_time"], 1.5)


if __name__ == '__main__':
    unittest.main() 