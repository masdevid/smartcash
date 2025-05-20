"""
File: smartcash/ui/dataset/split/tests/run_standalone_tests.py
Deskripsi: Runner untuk menjalankan test standalone untuk konfigurasi split dataset
"""

import unittest
import sys
from pathlib import Path

# Tambahkan root directory ke sys.path agar dapat mengimport modul smartcash
root_dir = str(Path(__file__).parent.parent.parent.parent.parent)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Import test case
from smartcash.ui.dataset.split.tests.test_config_sync_standalone import TestConfigSyncStandalone

def run_tests():
    """Jalankan semua test standalone untuk konfigurasi split dataset."""
    # Buat test suite
    test_suite = unittest.TestSuite()
    
    # Tambahkan test case ke test suite
    test_suite.addTest(unittest.makeSuite(TestConfigSyncStandalone))
    
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
    run_tests() 