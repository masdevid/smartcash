"""
File: smartcash/ui/dataset/visualization/tests/run_tests_with_mock.py
Deskripsi: Script untuk menjalankan tes visualisasi dataset dengan mock modules
"""

import sys
import unittest
from unittest.mock import MagicMock
import ipywidgets as widgets

# Setup mock modules sebelum mengimpor modul lain
def setup_mock_modules():
    """Setup mock modules untuk testing"""
    # Mock untuk header
    header_mock = MagicMock()
    header_mock.create_header = lambda title, description=None, icon=None: widgets.HTML(f"<h2>{title}</h2>")
    sys.modules['smartcash.ui.components.header'] = header_mock
    
    # Mock untuk tabs
    tabs_mock = MagicMock()
    tabs_mock.create_tabs = lambda tabs_list: widgets.Tab()
    sys.modules['smartcash.ui.components.tabs'] = tabs_mock
    
    # Mock untuk alert_utils
    alert_utils_mock = MagicMock()
    alert_utils_mock.create_status_indicator = lambda status_type, message: widgets.HTML(f"<div class='{status_type}'>{message}</div>")
    sys.modules['smartcash.ui.utils.alert_utils'] = alert_utils_mock
    
    # Mock untuk config
    config_mock = MagicMock()
    config_manager = MagicMock()
    config_manager.get_module_config.return_value = {'dataset_path': '/dummy/path'}
    config_mock.get_config_manager = lambda: config_manager
    sys.modules['smartcash.common.config'] = config_mock

# Setup mock modules sebelum mengimpor test cases
setup_mock_modules()

# Sekarang impor test cases
from smartcash.ui.dataset.visualization.tests.test_visualization_display import TestVisualizationDisplay, TestDummyDataGeneration

def run_tests():
    """
    Jalankan tes dengan mock modules.
    """
    # Buat test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Tambahkan test untuk display
    test_suite.addTest(loader.loadTestsFromTestCase(TestVisualizationDisplay))
    test_suite.addTest(loader.loadTestsFromTestCase(TestDummyDataGeneration))
    
    # Jalankan test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code berdasarkan hasil test
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests()) 