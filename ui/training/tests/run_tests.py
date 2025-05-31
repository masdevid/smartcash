"""
File: smartcash/ui/training/tests/run_tests.py
Deskripsi: Script untuk menjalankan semua tes UI training
"""

import unittest
import sys
import os
from typing import List, Dict, Any
from tqdm import tqdm
import time

# Import test modules
from smartcash.ui.training.tests.test_training_ui import TestTrainingComponents, TestTrainingInitializer
from smartcash.ui.training.tests.test_components import (
    TestConfigTabs,
    TestMetricsAccordion,
    TestControlButtons,
    TestFallbackComponent
)


def run_all_tests(verbose: bool = True) -> Dict[str, Any]:
    """
    Menjalankan semua tes dengan progress bar dan ringkasan hasil
    
    Args:
        verbose: Apakah akan menampilkan output detail
    
    Returns:
        Dictionary hasil tes
    """
    # Setup test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Tambahkan semua test cases
    print("🔍 Menyiapkan tes UI training...")
    test_cases = [
        TestTrainingComponents,
        TestTrainingInitializer,
        TestConfigTabs, 
        TestMetricsAccordion,
        TestControlButtons,
        TestFallbackComponent
    ]
    
    # Progress bar untuk menyiapkan test
    for test_case in tqdm(test_cases, desc="Menyiapkan test cases", colour="blue"):
        suite.addTests(loader.loadTestsFromTestCase(test_case))
    
    # Setup result handler
    result = unittest.TestResult()
    result.failfast = False
    
    # Run tests dengan progress
    print("\n🚀 Menjalankan tes UI training...")
    test_count = suite.countTestCases()
    
    # Setup tqdm progress
    progress_bar = tqdm(total=test_count, desc="Eksekusi test cases", colour="green")
    
    start_time = time.time()
    
    # Patch unittest untuk update progress bar
    original_stopTest = result.stopTest
    def progress_stop_test(test):
        original_stopTest(test)
        progress_bar.update(1)
    result.stopTest = progress_stop_test
    
    # Run tests
    suite.run(result)
    progress_bar.close()
    
    # Hitung statistik
    run_time = time.time() - start_time
    success_count = test_count - len(result.errors) - len(result.failures)
    success_rate = (success_count / test_count) * 100 if test_count > 0 else 0
    
    # Tampilkan hasil
    print(f"\n{'=' * 60}")
    print(f"📊 HASIL TES UI TRAINING")
    print(f"{'=' * 60}")
    print(f"✅ Total tes: {test_count}")
    print(f"✅ Sukses: {success_count} ({success_rate:.1f}%)")
    print(f"❌ Gagal: {len(result.failures)}")
    print(f"⚠️ Error: {len(result.errors)}")
    print(f"⏱️ Waktu eksekusi: {run_time:.2f} detik")
    print(f"{'=' * 60}")
    
    # Tampilkan detail jika verbose
    if verbose and (result.failures or result.errors):
        if result.failures:
            print("\n❌ DETAIL KEGAGALAN:")
            for i, (test, traceback) in enumerate(result.failures):
                print(f"\n--- Failure {i+1}: {test} ---")
                print(traceback)
        
        if result.errors:
            print("\n⚠️ DETAIL ERROR:")
            for i, (test, traceback) in enumerate(result.errors):
                print(f"\n--- Error {i+1}: {test} ---")
                print(traceback)
    
    # Return hasil untuk analisis lebih lanjut
    return {
        'total': test_count,
        'success': success_count,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'success_rate': success_rate,
        'run_time': run_time
    }


if __name__ == "__main__":
    # Parse argumen command line
    verbose = '--verbose' in sys.argv or '-v' in sys.argv
    
    # Jalankan tes
    print("🧪 Memulai tes untuk UI Training SmartCash...")
    results = run_all_tests(verbose)
    
    # Set exit code
    sys.exit(0 if results['failures'] == 0 and results['errors'] == 0 else 1)
