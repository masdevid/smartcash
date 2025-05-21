"""
File: smartcash/ui/setup/env_config/tests/run_test.py
Deskripsi: Script untuk menjalankan test di conda environment
"""

import sys
import os
from pathlib import Path
import subprocess

def run_test():
    """
    Jalankan test di conda environment
    """
    # Dapatkan path ke script test
    test_script = Path(__file__).parent / "test_ui_logger_fix.py"
    
    # Cek apakah script test ada
    if not test_script.exists():
        print(f"Error: Script test tidak ditemukan di {test_script}")
        return
    
    # Jalankan test dengan conda environment
    print(f"Menjalankan test di conda environment smartcash_test...")
    
    try:
        # Gunakan conda run untuk menjalankan test di conda environment
        cmd = f"conda run -n smartcash_test python {test_script}"
        
        # Jalankan command
        process = subprocess.Popen(
            cmd, 
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Tampilkan output secara real-time
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
        
        # Dapatkan return code
        return_code = process.poll()
        
        # Tampilkan error jika ada
        if return_code != 0:
            error = process.stderr.read()
            print(f"Error: {error}")
            
        print(f"\nTest selesai dengan return code: {return_code}")
        
    except Exception as e:
        print(f"Error saat menjalankan test: {str(e)}")

if __name__ == "__main__":
    run_test() 