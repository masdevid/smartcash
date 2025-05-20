"""
File: smartcash/ui/dataset/download/tests/run_tests.py
Deskripsi: Runner untuk test download module
"""

import unittest
import sys
import os

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../'))
sys.path.insert(0, project_root)

# Import test modules
from smartcash.ui.dataset.download.tests.test_download_ui import TestDownloadUI
from smartcash.ui.dataset.download.tests.test_download_initializer import TestDownloadInitializer
from smartcash.ui.dataset.download.tests.test_button_handlers import TestButtonHandlers
from smartcash.ui.dataset.download.tests.test_config_handler import TestDownloadConfigHandler
from smartcash.ui.dataset.download.tests.test_confirmation_dialog import TestConfirmationDialog
from smartcash.ui.dataset.download.tests.test_progress_tracking import TestProgressTracking
from smartcash.ui.dataset.download.tests.test_service_handler_compatibility import TestServiceHandlerCompatibility
from smartcash.ui.dataset.download.tests.test_service_integration import TestServiceIntegration
from smartcash.ui.dataset.download.tests.test_notification_mapping import TestNotificationMapping
from smartcash.ui.dataset.download.tests.test_download_handler import TestDownloadHandler

def run_tests():
    """Run all tests in the download module."""
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test cases
    test_suite.addTest(unittest.makeSuite(TestDownloadUI))
    test_suite.addTest(unittest.makeSuite(TestDownloadInitializer))
    test_suite.addTest(unittest.makeSuite(TestButtonHandlers))
    test_suite.addTest(unittest.makeSuite(TestDownloadConfigHandler))
    test_suite.addTest(unittest.makeSuite(TestConfirmationDialog))
    test_suite.addTest(unittest.makeSuite(TestProgressTracking))
    test_suite.addTest(unittest.makeSuite(TestServiceHandlerCompatibility))
    test_suite.addTest(unittest.makeSuite(TestServiceIntegration))
    test_suite.addTest(unittest.makeSuite(TestNotificationMapping))
    test_suite.addTest(unittest.makeSuite(TestDownloadHandler))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests()) 