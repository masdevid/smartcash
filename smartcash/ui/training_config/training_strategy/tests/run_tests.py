"""
File: smartcash/ui/training_config/training_strategy/tests/run_tests.py
Deskripsi: Runner untuk menjalankan semua test strategi pelatihan
"""

import unittest
import sys
import os

# Tambahkan parent directory ke sys.path untuk import
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

# Import test cases
from smartcash.ui.training_config.training_strategy.tests.test_config import TestTrainingStrategyConfig
from smartcash.ui.training_config.training_strategy.tests.test_ui import TestTrainingStrategyUI
from smartcash.ui.training_config.training_strategy.tests.test_drive_handlers import TestDriveHandlers

def run_tests():
    """Jalankan semua test."""
    # Buat test suite
    test_suite = unittest.TestSuite()
    
    # Tambahkan test cases
    test_suite.addTest(unittest.makeSuite(TestTrainingStrategyConfig))
    test_suite.addTest(unittest.makeSuite(TestTrainingStrategyUI))
    test_suite.addTest(unittest.makeSuite(TestDriveHandlers))
    
    # Jalankan test
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Tampilkan hasil
    print("\n=== Hasil Test ===")
    print(f"Tests run: {result.testsRun}")
    print(f"Errors: {len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    
    # Return True jika semua test berhasil
    return len(result.errors) == 0 and len(result.failures) == 0

if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
