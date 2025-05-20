"""
File: smartcash/ui/dataset/visualization/tests/run_tests.py
Deskripsi: Script untuk menjalankan semua test visualisasi dataset
"""

import unittest
import sys
import os

# Import test modules
from smartcash.ui.dataset.visualization.tests.test_simple_visualization import TestSimpleDataGeneration
from smartcash.ui.dataset.visualization.tests.test_components_import import TestComponentsImport

def run_tests():
    """
    Jalankan semua test untuk visualisasi dataset.
    """
    # Buat test suite
    loader = unittest.TestLoader()
    test_suite = unittest.TestSuite()
    
    # Tambahkan test untuk data generation
    test_suite.addTest(loader.loadTestsFromTestCase(TestSimpleDataGeneration))
    
    # Tambahkan test untuk komponen impor
    test_suite.addTest(loader.loadTestsFromTestCase(TestComponentsImport))
    
    # Jalankan test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Return exit code berdasarkan hasil test
    return 0 if result.wasSuccessful() else 1

if __name__ == '__main__':
    sys.exit(run_tests()) 