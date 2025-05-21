"""
File: smartcash/ui/dataset/preprocessing/tests/run_tests.py
Deskripsi: Script untuk menjalankan tests preprocessing dataset
"""

import unittest
import sys
import os

def run_all_tests():
    """
    Menjalankan tests untuk modul preprocessing dataset.
    """
    # Tambahkan project directory ke PYTHONPATH untuk import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../../")))
    
    # Inisialisasi test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Import tests
    try:
        # Import test utilitas preprocessing
        from smartcash.ui.dataset.preprocessing.tests.test_preprocessing_utils import TestPreprocessingUtils
        from smartcash.ui.dataset.preprocessing.tests.test_cell import TestPreprocessingCell
        
        # Tambahkan test cases ke suite
        suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingUtils))
        suite.addTests(loader.loadTestsFromTestCase(TestPreprocessingCell))
        
        # Inisialisasi test runner dan jalankan tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(suite)
        
        return result.wasSuccessful()
    except ImportError as e:
        print(f"❌ Error saat import test modules: {str(e)}")
        return False

if __name__ == "__main__":
    print("🧪 Menjalankan tests untuk modul preprocessing dataset...")
    success = run_all_tests()
    
    if success:
        print("✅ Semua tests berjalan sukses!")
        sys.exit(0)
    else:
        print("❌ Ada tests yang gagal!")
        sys.exit(1) 