"""
File: smartcash/ui/dataset/augmentation/tests/run_tests.py
Deskripsi: Runner untuk menjalankan semua pengujian augmentasi dataset
"""

import unittest
import sys
import os

# Import konfig pengujian untuk mock dependensi eksternal
from smartcash.ui.dataset.augmentation.tests.conftest import mock_all_dependencies

# Mock semua dependensi eksternal sebelum menjalankan pengujian
mock_all_dependencies()

def run_all_tests():
    """
    Jalankan semua pengujian augmentasi dataset.
    
    Returns:
        Boolean menunjukkan apakah semua pengujian berhasil
    """
    # Dapatkan direktori saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Tambahkan direktori root ke path
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    
    # Temukan semua pengujian
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover(current_dir, pattern='test_*.py')
    
    # Jalankan pengujian
    test_runner = unittest.TextTestRunner(verbosity=2)
    result = test_runner.run(test_suite)
    
    # Tampilkan ringkasan
    print(f"\n{'=' * 70}")
    print(f"Hasil Pengujian Augmentasi Dataset:")
    print(f"Jumlah pengujian: {result.testsRun}")
    print(f"Berhasil: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Gagal: {len(result.failures)}")
    print(f"Error: {len(result.errors)}")
    print(f"{'=' * 70}")
    
    # Tampilkan detail kegagalan
    if result.failures:
        print("\nDetail Kegagalan:")
        for failure in result.failures:
            print(f"\n{'-' * 50}")
            print(f"Pengujian: {failure[0]}")
            print(f"Pesan: {failure[1]}")
    
    # Tampilkan detail error
    if result.errors:
        print("\nDetail Error:")
        for error in result.errors:
            print(f"\n{'-' * 50}")
            print(f"Pengujian: {error[0]}")
            print(f"Pesan: {error[1]}")
    
    # Return True jika semua pengujian berhasil
    return len(result.failures) == 0 and len(result.errors) == 0

if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
