"""
File: smartcash/ui/dataset/preprocessing/tests/test_preprocessing_handlers.py
Deskripsi: Unit test untuk handler di modul preprocessing dataset
"""

import unittest
from unittest.mock import patch, MagicMock
import ipywidgets as widgets

from smartcash.ui.dataset.preprocessing.handlers.preprocessing_service_handler import handle_preprocessing_service
from smartcash.ui.dataset.preprocessing.handlers.execution_handler import run_preprocessing
from smartcash.ui.dataset.preprocessing.handlers.status_handler import update_status_panel

class TestPreprocessingHandlers(unittest.TestCase):
    """Unit tests untuk handler preprocessing."""
    
    def setUp(self):
        """Setup test environment sebelum setiap test case dijalankan."""
        # Mock UI components
        self.ui_components = {
            'preprocess_options': MagicMock(children=[
                MagicMock(), MagicMock(), MagicMock(), MagicMock(), MagicMock()
            ]),
            'validation_options': MagicMock(children=[
                MagicMock(), MagicMock(), MagicMock(), MagicMock()
            ]),
            'split_selector': MagicMock(value="All Splits"),
            'status': MagicMock(),
            'progress': MagicMock(),
            'logger': MagicMock(),
            'preprocessing_initialized': True,
            'preprocessing_running': False
        }
        
        # Mock config
        self.config = {
            'preprocessing': {
                'enabled': True,
                'img_size': 640,
                'splits': ['train', 'valid', 'test'],
                'normalization': {
                    'enabled': True,
                    'preserve_aspect_ratio': True
                },
                'validate': {
                    'enabled': True,
                    'fix_issues': True,
                    'move_invalid': False,
                    'invalid_dir': 'invalid'
                },
                'num_workers': 2
            }
        }
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.preprocessing_service_handler.get_preprocessing_config')
    def test_handle_preprocessing_service(self, mock_get_config):
        """Test handler untuk preprocessing service."""
        # Arrange
        mock_get_config.return_value = self.config
        
        # Act
        result = handle_preprocessing_service(self.ui_components)
        
        # Assert
        self.assertEqual(result, self.ui_components)
        self.ui_components['preprocess_options'].children[0].value = self.config['preprocessing']['img_size']
        self.ui_components['preprocess_options'].children[1].value = self.config['preprocessing']['normalization']['enabled']
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.status_handler.get_logger')
    def test_update_status_panel(self, mock_get_logger):
        """Test update status panel handler."""
        # Arrange
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        
        # Act
        result = update_status_panel(self.ui_components, "info", "Test status message")
        
        # Assert
        self.assertEqual(result, self.ui_components)
        mock_logger.info.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.get_logger')
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.validate_preprocessing_prerequisites')
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.time')
    def test_run_preprocessing_success(self, mock_time, mock_validate, mock_get_logger):
        """Test run preprocessing handler berhasil."""
        # Arrange
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_time.time.side_effect = [100, 105]  # Start time, end time
        
        # Setup prerequisites mock
        mock_validate.return_value = {
            'success': True, 
            'split': 'train',
            'input_dir': '/test/input',
            'output_dir': '/test/output',
            'image_count': 10,
            'preprocess_config': {
                'img_size': 640,
                'normalize': True
            }
        }
        
        # Setup dataset manager mock
        mock_dataset_manager = MagicMock()
        mock_dataset_manager.preprocess_dataset.return_value = {
            'success': True,
            'processed_images': 10,
            'skipped_images': 0
        }
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.validate_preprocessing_prerequisites') as mock_validate_fn:
            mock_validate_fn.return_value = mock_validate.return_value
            with patch('smartcash.ui.dataset.preprocessing.handlers.button_handlers.get_dataset_manager') as mock_get_manager:
                mock_get_manager.return_value = mock_dataset_manager
                with patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.update_status_panel') as mock_update_status:
                    result = run_preprocessing(self.ui_components)
        
        # Assert
        self.assertTrue(result['success'])
        self.assertEqual(result['processed_images'], 10)
        self.assertEqual(result['execution_time'], 5)  # 105 - 100
        mock_dataset_manager.preprocess_dataset.assert_called_once()
    
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.get_logger')
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.validate_preprocessing_prerequisites')
    def test_run_preprocessing_failed_prerequisites(self, mock_validate, mock_get_logger):
        """Test run preprocessing handler dengan kegagalan validasi prasyarat."""
        # Arrange
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_validate.return_value = {
            'success': False,
            'message': 'Dataset tidak valid'
        }
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.update_status_panel') as mock_update_status:
            with self.assertRaises(Exception) as context:
                run_preprocessing(self.ui_components)
        
        # Assert
        self.assertEqual(str(context.exception), 'Dataset tidak valid')
        
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.get_logger')
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.validate_preprocessing_prerequisites')
    @patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.time')
    def test_run_preprocessing_execution_error(self, mock_time, mock_validate, mock_get_logger):
        """Test run preprocessing handler dengan error eksekusi."""
        # Arrange
        mock_logger = MagicMock()
        mock_get_logger.return_value = mock_logger
        mock_time.time.side_effect = [100, 105]  # Start time, end time
        
        # Setup prerequisites mock
        mock_validate.return_value = {
            'success': True, 
            'split': 'train',
            'input_dir': '/test/input',
            'output_dir': '/test/output',
            'image_count': 10,
            'preprocess_config': {
                'img_size': 640,
                'normalize': True
            }
        }
        
        # Setup dataset manager mock with error
        mock_dataset_manager = MagicMock()
        mock_dataset_manager.preprocess_dataset.side_effect = Exception("Preprocessing error")
        
        # Act
        with patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.validate_preprocessing_prerequisites') as mock_validate_fn:
            mock_validate_fn.return_value = mock_validate.return_value
            with patch('smartcash.ui.dataset.preprocessing.handlers.button_handlers.get_dataset_manager') as mock_get_manager:
                mock_get_manager.return_value = mock_dataset_manager
                with patch('smartcash.ui.dataset.preprocessing.handlers.execution_handler.update_status_panel') as mock_update_status:
                    with self.assertRaises(Exception) as context:
                        run_preprocessing(self.ui_components)
        
        # Assert
        self.assertEqual(str(context.exception), 'Preprocessing error')
        mock_update_status.assert_called()
        mock_logger.error.assert_called()


if __name__ == '__main__':
    unittest.main() 