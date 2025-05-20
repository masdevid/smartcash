#!/usr/bin/env python
"""
File: run_config_tests.py
Deskripsi: Script untuk menjalankan semua test konfigurasi split dataset
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path

# Tambahkan path ke smartcash
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def run_test(test_path, verbose=False):
    """
    Jalankan test.
    
    Args:
        test_path: Path ke file test
        verbose: Tampilkan output secara verbose
        
    Returns:
        Tuple (success, output)
    """
    # Buat environment dengan path ke smartcash
    env = os.environ.copy()
    env["PYTHONPATH"] = str(project_root)
    
    command = [sys.executable, test_path]
    
    try:
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            env=env
        )
        
        success = result.returncode == 0
        output = result.stdout + result.stderr
        
        if verbose:
            print(output)
        
        return success, output
    except Exception as e:
        if verbose:
            print(f"❌ Error saat menjalankan test: {str(e)}")
        return False, str(e)

def run_all_tests(verbose=False):
    """
    Jalankan semua test konfigurasi.
    
    Args:
        verbose: Tampilkan output secara verbose
        
    Returns:
        Boolean yang menunjukkan keberhasilan semua test
    """
    tests = [
        "test_config_simple.py",
        "smartcash/ui/dataset/split/tests/test_config_sync_standalone.py"
    ]
    
    results = []
    
    for test_path in tests:
        print(f"\n🔄 Menjalankan {test_path}...")
        success, output = run_test(test_path, verbose)
        results.append((test_path, success, output))
    
    # Print summary
    print("\n" + "="*50)
    print("📊 HASIL TEST")
    print("="*50)
    
    all_success = True
    for test_path, success, _ in results:
        status = "✅ BERHASIL" if success else "❌ GAGAL"
        print(f"{status}: {test_path}")
        
        if not success:
            all_success = False
    
    # Overall result
    if all_success:
        print("\n✅ Semua test berhasil")
    else:
        print("\n❌ Beberapa test gagal")
    
    return all_success

def main():
    """Fungsi utama."""
    parser = argparse.ArgumentParser(description="Jalankan test konfigurasi split dataset")
    parser.add_argument('-v', '--verbose', action='store_true', help='Tampilkan output test secara verbose')
    args = parser.parse_args()
    
    success = run_all_tests(args.verbose)
    
    # Return exit code berdasarkan keberhasilan test
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
