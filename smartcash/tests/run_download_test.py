"""
File: smartcash/tests/run_download_test.py
Deskripsi: Runner untuk menjalankan test integrasi download
"""

import unittest
import sys
from pathlib import Path
from IPython.display import display, HTML
import traceback

# Tambahkan root project ke sys.path untuk import
sys.path.append(str(Path(__file__).parent.parent))

# Import test case
from smartcash.tests.ui.test_download_integration import TestDownloadIntegration

def run_tests():
    """Jalankan semua test dengan output yang informatif"""
    print("🧪 Menjalankan test integrasi download...")
    
    # Setup test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestDownloadIntegration)
    
    # Jalankan test dengan custom result handler
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    
    # Tampilkan hasil
    if result.wasSuccessful():
        print("\n✅ Semua test berhasil!")
        display(HTML(f'<div style="background-color:#d4edda; color:#155724; padding:10px; border-radius:5px; margin:10px 0;">✅ Semua {result.testsRun} test berhasil!</div>'))
    else:
        print(f"\n❌ {len(result.failures) + len(result.errors)} dari {result.testsRun} test gagal.")
        
        if result.failures:
            print("\n🔍 Detail kegagalan test:")
            for test, error in result.failures:
                print(f"\n❌ {test}")
                print(f"{error}")
        
        if result.errors:
            print("\n🔍 Detail error test:")
            for test, error in result.errors:
                print(f"\n❌ {test}")
                print(f"{error}")
        
        display(HTML(f'<div style="background-color:#f8d7da; color:#721c24; padding:10px; border-radius:5px; margin:10px 0;">❌ {len(result.failures) + len(result.errors)} dari {result.testsRun} test gagal.</div>'))

if __name__ == "__main__":
    try:
        run_tests()
    except Exception as e:
        print(f"❌ Error saat menjalankan test: {str(e)}")
        traceback.print_exc()
