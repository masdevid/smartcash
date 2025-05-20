"""
File: smartcash/ui/dataset/visualization/tests/run_tests_standalone.py
Deskripsi: Script untuk menjalankan tes visualisasi dataset secara standalone dengan mock modules
"""

import unittest
import sys
import os

# Import mock modules
from smartcash.ui.dataset.visualization.tests.mock_modules import setup_mock_modules

# Setup mock modules sebelum mengimpor test cases
setup_mock_modules()

# Import test cases setelah setup mock modules
try:
    from smartcash.ui.dataset.visualization.tests.test_visualization_display import TestVisualizationDisplay, TestDummyDataGeneration
    display_tests_available = True
except ImportError as e:
    print(f"Peringatan: Tidak dapat mengimpor test_visualization_display: {e}")
    display_tests_available = False

try:
    from smartcash.ui.dataset.visualization.tests.test_visualization_integration import TestVisualizationIntegration, TestHandlerIntegration, TestRefreshIntegration
    integration_tests_available = True
except ImportError as e:
    print(f"Peringatan: Tidak dapat mengimpor test_visualization_integration: {e}")
    integration_tests_available = False

try:
    from smartcash.ui.dataset.visualization.tests.test_visualization_components import TestDashboardCards, TestSplitStatsCards, TestVisualizationTabs
    components_tests_available = True
except ImportError as e:
    print(f"Peringatan: Tidak dapat mengimpor test_visualization_components: {e}")
    components_tests_available = False

def run_standalone_tests():
    """
    Jalankan tes secara standalone dengan mock modules.
    """
    # Buat test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Tambahkan test untuk display jika tersedia
    if display_tests_available:
        test_suite.addTest(loader.loadTestsFromTestCase(TestVisualizationDisplay))
        test_suite.addTest(loader.loadTestsFromTestCase(TestDummyDataGeneration))
    
    # Tambahkan test untuk integrasi jika tersedia
    if integration_tests_available:
        test_suite.addTest(loader.loadTestsFromTestCase(TestVisualizationIntegration))
        test_suite.addTest(loader.loadTestsFromTestCase(TestHandlerIntegration))
        test_suite.addTest(loader.loadTestsFromTestCase(TestRefreshIntegration))
    
    # Tambahkan test untuk komponen jika tersedia
    if components_tests_available:
        test_suite.addTest(loader.loadTestsFromTestCase(TestDashboardCards))
        test_suite.addTest(loader.loadTestsFromTestCase(TestSplitStatsCards))
        test_suite.addTest(loader.loadTestsFromTestCase(TestVisualizationTabs))
    
    # Jalankan test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code berdasarkan hasil test
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_standalone_tests()) 