"""
File: smartcash/ui/dataset/augmentation/tests/test_execution_handler.py
Deskripsi: Test untuk execution handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import ipywidgets as widgets
from typing import Dict, Any

class TestExecutionHandler(unittest.TestCase):
    """Test untuk execution handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock UI components
        self.ui_components = {
            'logger': MagicMock(),
            'status': MagicMock(),
            'status_panel': MagicMock(),
            'cleanup_button': MagicMock(),
            'visualization_buttons': MagicMock(),
            'summary_container': MagicMock(),
            'augmentation_running': False,
            'stop_requested': False
        }
        
        # Mock untuk observer.notify
        self.patcher1 = patch('smartcash.components.observer.notify')
        self.mock_notify = self.patcher1.start()
        
        # Mock untuk notification_handler
        self.patcher2 = patch('smartcash.ui.dataset.augmentation.handlers.notification_handler.notify_process_start')
        self.mock_notify_start = self.patcher2.start()
        
        self.patcher3 = patch('smartcash.ui.dataset.augmentation.handlers.notification_handler.notify_process_complete')
        self.mock_notify_complete = self.patcher3.start()
        
        # Mock untuk status_handler
        self.patcher4 = patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_status_panel')
        self.mock_update_status = self.patcher4.start()
        
        # Mock untuk extract_augmentation_params
        self.patcher5 = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.extract_augmentation_params')
        self.mock_extract_params = self.patcher5.start()
        self.mock_extract_params.return_value = {
            'enabled': True,
            'types': ['combined'],
            'num_variations': 2,
            'target_count': 1000,
            'output_prefix': 'aug',
            'split': 'train'
        }
        
        # Mock untuk validate_prerequisites
        self.patcher6 = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.validate_prerequisites')
        self.mock_validate = self.patcher6.start()
        self.mock_validate.return_value = True
        
        # Mock untuk augmentation_service_handler
        self.patcher7 = patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.execute_augmentation')
        self.mock_execute_aug = self.patcher7.start()
        self.mock_execute_aug.return_value = {
            'status': 'success',
            'generated': 100,
            'split': 'train',
            'augmentation_types': ['combined'],
            'output_dir': 'data/augmented'
        }
        
        # Mock untuk display_augmentation_summary
        self.patcher8 = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.display_augmentation_summary')
        self.mock_display_summary = self.patcher8.start()
        
        # Mock untuk IPython.display
        self.patcher9 = patch('IPython.display.display')
        self.mock_display = self.patcher9.start()
        
        self.patcher10 = patch('IPython.display.clear_output')
        self.mock_clear_output = self.patcher10.start()
        
        # Mock untuk create_status_indicator
        self.patcher11 = patch('smartcash.ui.utils.alert_utils.create_status_indicator')
        self.mock_create_indicator = self.patcher11.start()
        self.mock_create_indicator.return_value = MagicMock()
        
        # Mock untuk button_handlers.cleanup_ui
        self.patcher12 = patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.cleanup_ui')
        self.mock_cleanup_ui = self.patcher12.start()
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
        self.patcher5.stop()
        self.patcher6.stop()
        self.patcher7.stop()
        self.patcher8.stop()
        self.patcher9.stop()
        self.patcher10.stop()
        self.patcher11.stop()
        self.patcher12.stop()
    
    def test_execute_augmentation_success(self):
        """Test eksekusi augmentasi dengan hasil sukses"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Panggil fungsi
        execute_augmentation(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['augmentation_running'])
        self.mock_extract_params.assert_called_once_with(self.ui_components)
        self.mock_validate.assert_called_once()
        self.mock_update_status.assert_called()
        self.mock_notify_start.assert_called_once_with(self.ui_components)
        self.mock_execute_aug.assert_called_once()
        self.mock_notify_complete.assert_called_once()
        self.mock_display_summary.assert_called_once()
        self.mock_cleanup_ui.assert_called_once()
    
    def test_execute_augmentation_validation_failed(self):
        """Test eksekusi augmentasi dengan validasi gagal"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Setup validasi gagal
        self.mock_validate.return_value = False
        
        # Panggil fungsi
        execute_augmentation(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['augmentation_running'])
        self.mock_extract_params.assert_called_once_with(self.ui_components)
        self.mock_validate.assert_called_once()
        self.mock_notify_start.assert_not_called()
        self.mock_execute_aug.assert_not_called()
        self.mock_notify_complete.assert_not_called()
        self.mock_display_summary.assert_not_called()
        self.mock_cleanup_ui.assert_called_once()
    
    def test_execute_augmentation_warning(self):
        """Test eksekusi augmentasi dengan hasil warning"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Setup hasil warning
        self.mock_execute_aug.return_value = {
            'status': 'warning',
            'message': 'Augmentasi selesai dengan peringatan',
            'generated': 50,
            'split': 'train',
            'augmentation_types': ['combined'],
            'output_dir': 'data/augmented'
        }
        
        # Panggil fungsi
        execute_augmentation(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['augmentation_running'])
        self.mock_extract_params.assert_called_once_with(self.ui_components)
        self.mock_validate.assert_called_once()
        self.mock_update_status.assert_called()
        self.mock_notify_start.assert_called_once_with(self.ui_components)
        self.mock_execute_aug.assert_called_once()
        self.mock_notify_complete.assert_called_once()
        self.mock_display_summary.assert_not_called()
        self.mock_cleanup_ui.assert_called_once()
    
    def test_execute_augmentation_error(self):
        """Test eksekusi augmentasi dengan hasil error"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Setup hasil error
        self.mock_execute_aug.return_value = {
            'status': 'error',
            'message': 'Augmentasi gagal',
            'generated': 0,
            'split': 'train',
            'augmentation_types': ['combined'],
            'output_dir': 'data/augmented'
        }
        
        # Panggil fungsi
        execute_augmentation(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['augmentation_running'])
        self.mock_extract_params.assert_called_once_with(self.ui_components)
        self.mock_validate.assert_called_once()
        self.mock_update_status.assert_called()
        self.mock_notify_start.assert_called_once_with(self.ui_components)
        self.mock_execute_aug.assert_called_once()
        self.mock_notify_complete.assert_called_once()
        self.mock_display_summary.assert_not_called()
        self.mock_cleanup_ui.assert_called_once()
    
    def test_execute_augmentation_exception(self):
        """Test eksekusi augmentasi dengan exception"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Setup exception
        self.mock_execute_aug.side_effect = Exception("Test exception")
        
        # Panggil fungsi
        execute_augmentation(self.ui_components)
        
        # Verifikasi hasil
        self.assertTrue(self.ui_components['augmentation_running'])
        self.mock_extract_params.assert_called_once_with(self.ui_components)
        self.mock_validate.assert_called_once()
        self.mock_update_status.assert_called()
        self.mock_notify_start.assert_called_once_with(self.ui_components)
        self.mock_execute_aug.assert_called_once()
        self.mock_notify_complete.assert_not_called()
        self.mock_display_summary.assert_not_called()
        self.ui_components['logger'].error.assert_called_once()
        self.mock_cleanup_ui.assert_called_once()
    
    def test_run_augmentation(self):
        """Test menjalankan augmentasi dengan thread terpisah"""
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import run_augmentation
        
        # Mock ThreadPoolExecutor
        with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
            mock_executor_instance = MagicMock()
            mock_executor.return_value.__enter__.return_value = mock_executor_instance
            
            # Panggil fungsi
            run_augmentation(self.ui_components)
            
            # Verifikasi hasil
            mock_executor_instance.submit.assert_called_once()

if __name__ == '__main__':
    unittest.main()
