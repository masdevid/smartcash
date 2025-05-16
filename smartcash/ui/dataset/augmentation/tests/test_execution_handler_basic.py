"""
File: smartcash/ui/dataset/augmentation/tests/test_execution_handler_basic.py
Deskripsi: Test dasar untuk execution handler augmentasi dataset
"""

import unittest
from unittest.mock import MagicMock, patch
import sys

class TestExecutionHandlerBasic(unittest.TestCase):
    """Test dasar untuk execution handler augmentasi dataset"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        # Mock modules untuk menghindari import error
        sys.modules['smartcash.ui.utils.constants'] = MagicMock()
        sys.modules['smartcash.ui.utils.alert_utils'] = MagicMock()
        sys.modules['smartcash.common.logger'] = MagicMock()
        sys.modules['smartcash.components.observer'] = MagicMock()
        sys.modules['smartcash.ui.dataset.augmentation.handlers.status_handler'] = MagicMock()
        sys.modules['smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler'] = MagicMock()
        sys.modules['smartcash.ui.dataset.augmentation.handlers.button_handlers'] = MagicMock()
        
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
        
        # Mock untuk notification_handler
        self.patcher1 = patch('smartcash.ui.dataset.augmentation.handlers.notification_handler.notify_process_start')
        self.mock_notify_start = self.patcher1.start()
        
        self.patcher2 = patch('smartcash.ui.dataset.augmentation.handlers.notification_handler.notify_process_complete')
        self.mock_notify_complete = self.patcher2.start()
        
        # Mock untuk extract_augmentation_params
        self.patcher3 = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.extract_augmentation_params')
        self.mock_extract_params = self.patcher3.start()
        self.mock_extract_params.return_value = {
            'enabled': True,
            'types': ['combined'],
            'num_variations': 2,
            'target_count': 1000,
            'output_prefix': 'aug',
            'split': 'train'
        }
        
        # Mock untuk validate_prerequisites
        self.patcher4 = patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.validate_prerequisites')
        self.mock_validate = self.patcher4.start()
        self.mock_validate.return_value = True
    
    def tearDown(self):
        """Cleanup setelah setiap test case"""
        self.patcher1.stop()
        self.patcher2.stop()
        self.patcher3.stop()
        self.patcher4.stop()
    
    @patch('smartcash.ui.dataset.augmentation.handlers.status_handler.update_status_panel')
    @patch('smartcash.ui.dataset.augmentation.handlers.augmentation_service_handler.execute_augmentation')
    def test_notify_start_called(self, mock_execute_aug, mock_update_status):
        """Test notifikasi proses augmentasi dimulai dipanggil"""
        # Import fungsi yang akan diuji
        from smartcash.ui.dataset.augmentation.handlers.execution_handler import execute_augmentation
        
        # Setup mock
        mock_execute_aug.return_value = {
            'status': 'success',
            'generated': 100,
            'split': 'train',
            'augmentation_types': ['combined'],
            'output_dir': 'data/augmented'
        }
        
        # Patch fungsi lain yang dipanggil
        with patch('smartcash.ui.dataset.augmentation.handlers.execution_handler.display_augmentation_summary'), \
             patch('IPython.display.display'), \
             patch('IPython.display.clear_output'), \
             patch('smartcash.ui.utils.alert_utils.create_status_indicator'), \
             patch('smartcash.ui.dataset.augmentation.handlers.button_handlers.cleanup_ui'):
            
            # Panggil fungsi
            execute_augmentation(self.ui_components)
            
            # Verifikasi hasil
            self.mock_notify_start.assert_called_once_with(self.ui_components)

if __name__ == '__main__':
    unittest.main()
