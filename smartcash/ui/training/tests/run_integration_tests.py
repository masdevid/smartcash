"""
File: smartcash/ui/training/tests/run_integration_tests.py
Deskripsi: Script untuk menjalankan pengujian integrasi untuk modul training UI
"""

import os
import sys
import unittest
import argparse
from typing import List, Optional

# Emoji untuk status pengujian
EMOJI_SUCCESS = "✅"
EMOJI_FAILED = "❌"
EMOJI_SKIPPED = "⏭️"
EMOJI_RUNNING = "🔄"
EMOJI_SETUP = "🔧"


def setup_test_environment():
    """Setup lingkungan pengujian"""
    print(f"{EMOJI_SETUP} Menyiapkan lingkungan pengujian...")
    
    # Tambahkan root project ke sys.path jika belum ada
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    
    # Import modul yang diperlukan
    try:
        from smartcash.common.logger import get_logger
        logger = get_logger('integration_tests')
        logger.info(f"{EMOJI_SETUP} Lingkungan pengujian siap")
        return logger
    except ImportError as e:
        print(f"{EMOJI_FAILED} Error saat import modul: {str(e)}")
        print(f"Pastikan conda environment 'smartcash_test' aktif dan semua dependensi terinstal")
        sys.exit(1)


def discover_tests(test_pattern: Optional[str] = None) -> unittest.TestSuite:
    """Temukan semua test case yang akan dijalankan
    
    Args:
        test_pattern: Pola nama file test yang akan dijalankan (opsional)
        
    Returns:
        unittest.TestSuite: Suite pengujian yang ditemukan
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    if test_pattern:
        # Jalankan test yang sesuai dengan pola
        pattern = f"test_{test_pattern}*.py" if not test_pattern.startswith("test_") else f"{test_pattern}*.py"
        return unittest.defaultTestLoader.discover(current_dir, pattern=pattern)
    else:
        # Jalankan semua test integrasi
        return unittest.defaultTestLoader.discover(current_dir, pattern="test_*_integration.py")


def run_tests(test_suite: unittest.TestSuite, logger=None) -> bool:
    """Jalankan test suite
    
    Args:
        test_suite: Test suite yang akan dijalankan
        logger: Logger untuk logging
        
    Returns:
        bool: True jika semua test berhasil, False jika ada yang gagal
    """
    # Buat test runner
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Jalankan test
    print(f"\n{EMOJI_RUNNING} Menjalankan pengujian integrasi...")
    result = runner.run(test_suite)
    
    # Tampilkan hasil
    total_tests = result.testsRun
    failed_tests = len(result.failures) + len(result.errors)
    skipped_tests = len(result.skipped) if hasattr(result, 'skipped') else 0
    success_tests = total_tests - failed_tests - skipped_tests
    
    print(f"\n{'=' * 50}")
    print(f"HASIL PENGUJIAN INTEGRASI:")
    print(f"{'=' * 50}")
    print(f"{EMOJI_SUCCESS} Berhasil: {success_tests}")
    print(f"{EMOJI_FAILED} Gagal: {failed_tests}")
    print(f"{EMOJI_SKIPPED} Dilewati: {skipped_tests}")
    print(f"Total: {total_tests}")
    print(f"{'=' * 50}")
    
    if logger:
        logger.info(f"Hasil pengujian: {success_tests} berhasil, {failed_tests} gagal, {skipped_tests} dilewati")
    
    # Tampilkan detail kegagalan
    if result.failures:
        print(f"\nDETAIL KEGAGALAN:")
        for i, (test, traceback) in enumerate(result.failures, 1):
            print(f"\n{EMOJI_FAILED} Kegagalan #{i}: {test}")
            print(f"{'-' * 40}")
            print(traceback)
    
    if result.errors:
        print(f"\nDETAIL ERROR:")
        for i, (test, traceback) in enumerate(result.errors, 1):
            print(f"\n{EMOJI_FAILED} Error #{i}: {test}")
            print(f"{'-' * 40}")
            print(traceback)
    
    # Return True jika semua test berhasil
    return failed_tests == 0


def parse_arguments():
    """Parse argumen command line"""
    parser = argparse.ArgumentParser(description='Jalankan pengujian integrasi untuk modul training UI')
    parser.add_argument('--pattern', type=str, help='Pola nama file test yang akan dijalankan')
    parser.add_argument('--verbose', action='store_true', help='Tampilkan output verbose')
    return parser.parse_args()


def main():
    """Fungsi utama"""
    # Parse argumen
    args = parse_arguments()
    
    # Setup lingkungan pengujian
    logger = setup_test_environment()
    
    # Temukan test
    test_suite = discover_tests(args.pattern)
    
    # Jalankan test
    success = run_tests(test_suite, logger)
    
    # Exit dengan kode yang sesuai
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
