"""
File: smartcash/ui/setup/dependency_installer/tests/test_helpers.py
Deskripsi: Test untuk helper functions dengan pendekatan DRY
"""

import unittest
from unittest.mock import MagicMock, patch
import time
from typing import Dict, Any

class TestProgressHelper(unittest.TestCase):
    """Test untuk progress_helper functions"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        self.ui_components = {
            'update_progress': MagicMock(),
            'log_message': MagicMock(),
            'reset_progress_bar': MagicMock(),
            'show_for_operation': MagicMock(),
            'complete_operation': MagicMock(),
            'error_operation': MagicMock()
        }
    
    def test_update_progress_step(self):
        """Test update_progress_step function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import update_progress_step
        
        # Test dengan parameter normal
        update_progress_step(self.ui_components, 50, "Test message")
        self.ui_components['update_progress'].assert_called_once_with('step', 50, "Test message", "#007bff")
        
        # Reset mock
        self.ui_components['update_progress'].reset_mock()
        
        # Test dengan custom color
        update_progress_step(self.ui_components, 75, "Test message", "#ff0000")
        self.ui_components['update_progress'].assert_called_once_with('step', 75, "Test message", "#ff0000")
    
    def test_calculate_batch_progress(self):
        """Test calculate_batch_progress function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import calculate_batch_progress
        
        # Test dengan berbagai nilai
        self.assertEqual(calculate_batch_progress(0, 10), 10)  # 1/10 = 10%
        self.assertEqual(calculate_batch_progress(4, 10), 50)  # 5/10 = 50%
        self.assertEqual(calculate_batch_progress(9, 10), 100)  # 10/10 = 100%
        
        # Test dengan total 0 (edge case)
        self.assertEqual(calculate_batch_progress(5, 0), 0)
    
    def test_start_operation(self):
        """Test start_operation function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import start_operation
        
        # Patch log_message di logger_helper module
        with patch('smartcash.ui.setup.dependency_installer.utils.logger_helper.log_message') as mock_log:
            start_operation(self.ui_components, "test operation", 5)
            
            # Verify calls
            self.ui_components['reset_progress_bar'].assert_called_once()
            self.ui_components['show_for_operation'].assert_called_once_with("test operation")
            self.ui_components['update_progress'].assert_called()
            mock_log.assert_called_once()
    
    def test_complete_operation(self):
        """Test complete_operation function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import complete_operation
        
        # Prepare test data
        stats = {
            'success': 8,
            'total': 10,
            'failed': 2,
            'duration': 5.5
        }
        
        # Patch log_message di logger_helper module
        with patch('smartcash.ui.setup.dependency_installer.utils.logger_helper.log_message') as mock_log:
            complete_operation(self.ui_components, "test operation", stats)
            
            # Verify calls
            self.ui_components['update_progress'].assert_called()
            self.ui_components['complete_operation'].assert_called_once_with("test operation", stats)
            mock_log.assert_called_once()
    
    def test_handle_item_error(self):
        """Test handle_item_error function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import handle_item_error
        
        # Patch log_message di logger_helper module
        with patch('smartcash.ui.setup.dependency_installer.utils.logger_helper.log_message') as mock_log:
            handle_item_error(self.ui_components, "test item", "test error")
            
            # Verify calls
            self.ui_components['update_progress'].assert_called_once()
            self.ui_components['error_operation'].assert_called_once_with("test item", "test error")
            mock_log.assert_called_once()
    
    def test_handle_item_success(self):
        """Test handle_item_success function"""
        from smartcash.ui.setup.dependency_installer.utils.progress_helper import handle_item_success
        
        # Patch log_message di logger_helper module
        with patch('smartcash.ui.setup.dependency_installer.utils.logger_helper.log_message') as mock_log:
            handle_item_success(self.ui_components, "test item")
            
            # Verify calls
            self.ui_components['update_progress'].assert_called_once()
            mock_log.assert_called_once()
            
            # Reset mocks
            mock_log.reset_mock()
            self.ui_components['update_progress'].reset_mock()
            
            # Test with message
            handle_item_success(self.ui_components, "test item", "additional info")
            self.ui_components['update_progress'].assert_called_once()
            mock_log.assert_called_once()


class TestObserverHelper(unittest.TestCase):
    """Test untuk observer_helper functions"""
    
    def setUp(self):
        """Setup untuk setiap test case"""
        self.observer_manager = MagicMock()
        self.ui_components = {
            'observer_manager': self.observer_manager
        }
    
    def test_notify_install_start(self):
        """Test notify_install_start function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_install_start
        
        # Test dengan parameter normal
        notify_install_start(self.ui_components, 10)
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_INSTALL_START')
        
        # Reset mock
        self.observer_manager.notify.reset_mock()
        
        # Test dengan custom message
        notify_install_start(self.ui_components, 5, "Custom message")
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_INSTALL_START')
        
        # Test dengan observer_manager None
        self.ui_components['observer_manager'] = None
        notify_install_start(self.ui_components, 5)  # Tidak boleh error
    
    def test_notify_install_progress(self):
        """Test notify_install_progress function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_install_progress
        
        # Test dengan parameter normal
        notify_install_progress(self.ui_components, "test-package", 2, 10)
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_INSTALL_PROGRESS')
        
        # Test dengan observer_manager None
        self.ui_components['observer_manager'] = None
        notify_install_progress(self.ui_components, "test-package", 2, 10)  # Tidak boleh error
    
    def test_notify_install_error(self):
        """Test notify_install_error function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_install_error
        
        # Test dengan parameter normal
        notify_install_error(self.ui_components, "test-package", "test error")
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_INSTALL_ERROR')
        
        # Test dengan observer_manager None
        self.ui_components['observer_manager'] = None
        notify_install_error(self.ui_components, "test-package", "test error")  # Tidak boleh error
    
    def test_notify_install_complete(self):
        """Test notify_install_complete function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_install_complete
        
        # Prepare test data
        stats = {
            'success': 8,
            'total': 10,
            'failed': 2,
            'duration': 5.5,
            'errors': []
        }
        
        # Test dengan parameter normal
        notify_install_complete(self.ui_components, stats)
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_INSTALL_COMPLETE')
        
        # Test dengan observer_manager None
        self.ui_components['observer_manager'] = None
        notify_install_complete(self.ui_components, stats)  # Tidak boleh error
    
    def test_notify_analyze_start(self):
        """Test notify_analyze_start function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_analyze_start
        
        # Reset dan setup observer_manager
        self.ui_components['observer_manager'] = self.observer_manager
        self.observer_manager.notify.reset_mock()
        
        # Test dengan parameter normal
        notify_analyze_start(self.ui_components)
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_ANALYZE_START')
        
        # Reset mock
        self.observer_manager.notify.reset_mock()
        
        # Test dengan custom message
        notify_analyze_start(self.ui_components, "Custom message")
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_ANALYZE_START')
    
    def test_notify_analyze_error(self):
        """Test notify_analyze_error function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_analyze_error
        
        # Reset dan setup observer_manager
        self.ui_components['observer_manager'] = self.observer_manager
        self.observer_manager.notify.reset_mock()
        
        # Test dengan parameter normal
        notify_analyze_error(self.ui_components, "test error")
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_ANALYZE_ERROR')
    
    def test_notify_analyze_complete(self):
        """Test notify_analyze_complete function"""
        from smartcash.ui.setup.dependency_installer.utils.observer_helper import notify_analyze_complete
        
        # Reset dan setup observer_manager
        self.ui_components['observer_manager'] = self.observer_manager
        self.observer_manager.notify.reset_mock()
        
        # Test dengan parameter normal
        notify_analyze_complete(self.ui_components, 5.5, ["package1", "package2"])
        self.observer_manager.notify.assert_called_once()
        args = self.observer_manager.notify.call_args[0]
        self.assertEqual(args[0], 'DEPENDENCY_ANALYZE_COMPLETE')


if __name__ == '__main__':
    unittest.main()
