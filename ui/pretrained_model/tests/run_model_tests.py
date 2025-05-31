"""
File: run_model_tests.py
Deskripsi: Script untuk menjalankan test suite modul model dalam conda environment smartcash_test
"""

import os
import sys
import argparse
from subprocess import run
import time

# Emoji dan warna untuk output
ICONS = {
    'test': '🧪',
    'success': '✅',
    'error': '❌',
    'wait': '⏳',
    'python': '🐍'
}
COLORS = {
    'green': '\033[92m',
    'red': '\033[91m',
    'yellow': '\033[93m',
    'blue': '\033[94m',
    'bold': '\033[1m',
    'end': '\033[0m'
}

def print_colored(message, color='blue', icon=None, bold=False):
    """Mencetak pesan berwarna dengan emoji opsional"""
    icon_str = f"{ICONS.get(icon, '')} " if icon else ""
    bold_str = COLORS['bold'] if bold else ""
    print(f"{bold_str}{COLORS.get(color, COLORS['blue'])}{icon_str}{message}{COLORS['end']}")

def run_tests_in_conda():
    """Menjalankan test suite dalam conda environment smartcash_test"""
    print_colored("Memulai pengujian modul model dalam conda environment smartcash_test", "blue", "test", True)
    
    # Daftar test yang akan dijalankan
    test_modules = [
        "smartcash.ui.model.tests.test_model_initializer",
        "smartcash.ui.model.tests.test_ui_components",
        "smartcash.ui.model.tests.test_simple_download"
    ]
    
    # Dapatkan lokasi direktori saat ini
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print_colored(f"Direktori proyek: {current_dir}", "blue")
    
    # Perintah untuk menjalankan test dalam conda environment
    conda_cmd = f"conda run -n smartcash_test python -m unittest {' '.join(test_modules)}"
    
    print_colored(f"Menjalankan perintah: {conda_cmd}", "yellow", "python")
    print_colored("Tunggu sebentar...", "blue", "wait")
    
    # Jalankan test
    try:
        start_time = time.time()
        result = run(conda_cmd, shell=True, check=True)
        elapsed_time = time.time() - start_time
        
        print_colored(f"Test selesai dalam {elapsed_time:.2f} detik", "green", "success", True)
        return True
    except Exception as e:
        print_colored(f"Error saat menjalankan test: {str(e)}", "red", "error")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Menjalankan test suite modul model dalam conda environment")
    parser.add_argument("--create-env", action="store_true", help="Buat conda environment baru jika belum ada")
    args = parser.parse_args()
    
    # Buat conda environment jika diminta
    if args.create_env:
        print_colored("Memeriksa conda environment smartcash_test", "blue")
        
        # Periksa apakah environment sudah ada
        check_cmd = "conda env list | grep smartcash_test"
        if os.system(check_cmd) != 0:
            print_colored("Membuat conda environment smartcash_test", "yellow")
            os.system("conda create -n smartcash_test python=3.8 ipywidgets pytest -y")
            os.system("conda run -n smartcash_test pip install tqdm")
        else:
            print_colored("Conda environment smartcash_test sudah ada", "green", "success")
    
    # Jalankan test
    success = run_tests_in_conda()
    
    # Keluar dengan kode status yang sesuai
    sys.exit(0 if success else 1)
