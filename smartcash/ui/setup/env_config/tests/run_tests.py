"""
File: smartcash/ui/setup/env_config/tests/run_tests.py
Deskripsi: Script untuk menjalankan test UILogger dengan conda environment smartcash_test
"""

import os
import sys
import subprocess
from pathlib import Path

def run_tests():
    """
    Jalankan semua test UILogger dengan conda environment smartcash_test
    """
    print("🚀 Menjalankan test UILogger dengan conda environment smartcash_test...")
    
    # Path ke file test
    test_dir = Path(__file__).parent
    test_file = test_dir / "test_ui_logger.py"
    
    # Perintah untuk menjalankan test dengan conda environment
    cmd = f"conda run -n smartcash_test python {test_file}"
    
    # Jalankan perintah
    try:
        result = subprocess.run(cmd, shell=True, check=True, text=True, capture_output=True)
        print("✅ Test berhasil dijalankan!")
        print("\n--- Output Test ---\n")
        print(result.stdout)
        
        if result.stderr:
            print("\n--- Error Output ---\n")
            print(result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ Error saat menjalankan test: {e}")
        print("\n--- Output Test ---\n")
        print(e.stdout)
        print("\n--- Error Output ---\n")
        print(e.stderr)
        sys.exit(1)

if __name__ == "__main__":
    run_tests() 