"""
File: smartcash/ui/dataset/tests/run_all_tests.py
Deskripsi: Script untuk menjalankan semua pengujian persistensi konfigurasi
"""

import unittest
import sys
import os
from pathlib import Path
from tqdm import tqdm
import time

# Tambahkan root directory ke sys.path
current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
sys.path.insert(0, str(root_dir))

def run_tests():
    """Menjalankan semua pengujian persistensi konfigurasi"""
    # Tampilkan banner
    print("=" * 80)
    print("🧪 MENJALANKAN PENGUJIAN PERSISTENSI KONFIGURASI SMARTCASH 🧪")
    print("=" * 80)
    
    # Temukan semua file pengujian
    test_files = []
    
    # Pengujian preprocessing
    preprocessing_test_dir = Path(root_dir, "ui", "dataset", "preprocessing", "tests")
    if preprocessing_test_dir.exists():
        preprocessing_test_files = [
            f for f in preprocessing_test_dir.glob("test_*.py") 
            if f.name.startswith("test_") and f.name.endswith(".py")
        ]
        test_files.extend(preprocessing_test_files)
    
    # Pengujian augmentasi
    augmentation_test_dir = Path(root_dir, "ui", "dataset", "augmentation", "tests")
    if augmentation_test_dir.exists():
        augmentation_test_files = [
            f for f in augmentation_test_dir.glob("test_*.py") 
            if f.name.startswith("test_") and f.name.endswith(".py")
        ]
        test_files.extend(augmentation_test_files)
    
    # Pengujian integrasi
    integration_test_dir = Path(root_dir, "ui", "dataset", "tests")
    if integration_test_dir.exists():
        integration_test_files = [
            f for f in integration_test_dir.glob("test_*.py") 
            if f.name.startswith("test_") and f.name.endswith(".py") and f.name != "run_all_tests.py"
        ]
        test_files.extend(integration_test_files)
    
    # Jalankan semua pengujian
    print(f"📋 Menemukan {len(test_files)} file pengujian")
    
    # Buat progress bar
    progress_bar = tqdm(total=len(test_files), desc="Menjalankan pengujian", unit="file")
    
    # Hasil pengujian
    results = []
    
    for test_file in test_files:
        # Update progress bar
        progress_bar.set_description(f"Menjalankan {test_file.name}")
        progress_bar.refresh()
        
        # Jalankan pengujian
        test_loader = unittest.TestLoader()
        test_suite = test_loader.discover(str(test_file.parent), pattern=test_file.name)
        test_runner = unittest.TextTestRunner(verbosity=0)
        result = test_runner.run(test_suite)
        
        # Simpan hasil
        results.append({
            "file": test_file.name,
            "run": result.testsRun,
            "errors": len(result.errors),
            "failures": len(result.failures),
            "skipped": len(result.skipped)
        })
        
        # Update progress bar
        progress_bar.update(1)
        time.sleep(0.1)  # Untuk visualisasi progress bar
    
    # Tutup progress bar
    progress_bar.close()
    
    # Tampilkan hasil
    print("\n" + "=" * 80)
    print("📊 HASIL PENGUJIAN")
    print("=" * 80)
    
    total_run = 0
    total_errors = 0
    total_failures = 0
    total_skipped = 0
    
    for result in results:
        status = "✅ SUKSES" if result["errors"] == 0 and result["failures"] == 0 else "❌ GAGAL"
        print(f"{status} - {result['file']}: {result['run']} pengujian, {result['errors']} error, {result['failures']} gagal, {result['skipped']} dilewati")
        
        total_run += result["run"]
        total_errors += result["errors"]
        total_failures += result["failures"]
        total_skipped += result["skipped"]
    
    print("-" * 80)
    print(f"TOTAL: {total_run} pengujian, {total_errors} error, {total_failures} gagal, {total_skipped} dilewati")
    
    # Status keseluruhan
    if total_errors == 0 and total_failures == 0:
        print("\n✅ SEMUA PENGUJIAN BERHASIL")
        return 0
    else:
        print("\n❌ BEBERAPA PENGUJIAN GAGAL")
        return 1

if __name__ == "__main__":
    sys.exit(run_tests())
