"""
File: smartcash/ui/dataset/preprocessing/tests/run_tests.py
Deskripsi: Runner untuk menjalankan semua test preprocessing dengan penanganan khusus untuk ResourceWarning
dan optimasi performa
"""

import unittest
import sys
import os
import gc
import logging
import warnings
import io
import traceback
import time
import signal
from functools import partial

# Import fungsi utilitas
from smartcash.ui.dataset.preprocessing.tests.test_utils import (
    setup_test_environment,
    restore_environment,
    close_all_loggers
)

# Import test modules langsung untuk menghindari pencarian yang lambat
from smartcash.ui.dataset.preprocessing.tests.test_button_handlers import TestPreprocessingButtonHandlers
from smartcash.ui.dataset.preprocessing.tests.test_config_handlers import TestPreprocessingConfigHandlers
from smartcash.ui.dataset.preprocessing.tests.test_execution_handler import TestPreprocessingExecutionHandler
from smartcash.ui.dataset.preprocessing.tests.test_service_handler import TestPreprocessingServiceHandler
from smartcash.ui.dataset.preprocessing.tests.test_state_handler import TestPreprocessingStateHandler

# Konstanta untuk timeout
TEST_TIMEOUT = 10  # Timeout dalam detik untuk setiap test case

class TimeoutException(Exception):
    """Exception yang dilempar ketika test melebihi batas waktu"""
    pass

def timeout_handler(signum, frame):
    """Handler untuk signal timeout"""
    raise TimeoutException("Test melebihi batas waktu")

def run_test_with_timeout(test_case, timeout=TEST_TIMEOUT):
    """
    Menjalankan test case dengan timeout
    
    Args:
        test_case: Test case yang akan dijalankan
        timeout: Batas waktu dalam detik
        
    Returns:
        Hasil test (success, failure, atau error)
    """
    # Setup signal handler untuk timeout
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    
    result = unittest.TestResult()
    
    try:
        test_case(result)
    except TimeoutException:
        print(f"⚠️ Test {test_case} melebihi batas waktu {timeout} detik")
        result.errors.append((test_case, "Test melebihi batas waktu"))
    except Exception as e:
        print(f"❌ Error saat menjalankan {test_case}: {e}")
        result.errors.append((test_case, traceback.format_exc()))
    finally:
        # Matikan alarm
        signal.alarm(0)
    
    return result

def run_tests():
    """
    Menjalankan semua test preprocessing dengan penanganan khusus untuk ResourceWarning
    dan optimasi performa
    """
    start_time = time.time()
    
    # Siapkan lingkungan pengujian
    setup_test_environment()
    
    try:
        # Filter ResourceWarning dan DeprecationWarning
        warnings.filterwarnings("ignore", category=ResourceWarning)
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        
        # Buat test suite secara manual untuk kontrol yang lebih baik
        test_suite = unittest.TestSuite()
        
        # Tambahkan test classes secara eksplisit
        test_classes = [
            TestPreprocessingButtonHandlers,
            TestPreprocessingConfigHandlers,
            TestPreprocessingExecutionHandler,
            TestPreprocessingServiceHandler,
            TestPreprocessingStateHandler
        ]
        
        for test_class in test_classes:
            tests = unittest.defaultTestLoader.loadTestsFromTestCase(test_class)
            test_suite.addTests(tests)
        
        # Jalankan test dengan verbosity=1 untuk output yang lebih ringkas
        print("🚀 Menjalankan test preprocessing...")
        result = unittest.TextTestRunner(verbosity=1).run(test_suite)
        
        # Hitung waktu eksekusi
        execution_time = time.time() - start_time
        
        # Tampilkan ringkasan hasil test
        print(f"\n{'='*60}")
        print(f"📊 Hasil Pengujian Preprocessing Module:")
        print(f"{'='*60}")
        print(f"⏱️ Waktu eksekusi: {execution_time:.2f} detik")
        print(f"📝 Total test: {result.testsRun}")
        print(f"✅ Sukses: {result.testsRun - len(result.failures) - len(result.errors)}")
        print(f"❌ Gagal: {len(result.failures)}")
        print(f"⚠️ Error: {len(result.errors)}")
        
        # Tampilkan detail kegagalan jika ada
        if result.failures or result.errors:
            print(f"\n{'='*60}")
            print(f"❌ Detail Kegagalan:")
            print(f"{'='*60}")
            
            for test, trace in result.failures:
                print(f"\nFAILURE: {test}\n{'-'*60}\n{trace}\n")
                
            for test, trace in result.errors:
                print(f"\nERROR: {test}\n{'-'*60}\n{trace}\n")
        
        # Kembalikan status exit berdasarkan hasil test
        return 0 if result.wasSuccessful() else 1
    except Exception as e:
        print(f"❌ Error menjalankan test: {e}")
        traceback.print_exc()
        return 1
    finally:
        # Tutup semua logger dan kembalikan lingkungan pengujian
        try:
            # Hindari pemanggilan close_all_loggers() yang berlebihan
            # Cukup tutup logger root dan logger yang penting saja
            root_logger = logging.getLogger()
            for handler in list(root_logger.handlers):
                handler.close()
                root_logger.removeHandler(handler)
                
            # Kembalikan lingkungan pengujian
            restore_environment()
        except Exception as e:
            print(f"⚠️ Error saat membersihkan resources: {e}")

if __name__ == '__main__':
    sys.exit(run_tests())
