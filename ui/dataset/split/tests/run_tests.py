"""
File: smartcash/ui/dataset/split/tests/run_tests.py
Deskripsi: Script untuk menjalankan semua test split dataset
"""

import unittest
import sys
import os
import subprocess
import argparse
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import time

# Path absolut ke conda executable
CONDA_PATH = '/opt/anaconda3/bin/conda'

def run_test_module(module_name, conda_env=None):
    """
    Menjalankan modul test tertentu.
    
    Args:
        module_name: Nama modul test yang akan dijalankan
        conda_env: Nama environment conda (opsional)
        
    Returns:
        Tuple (success, output) yang menunjukkan hasil test
    """
    try:
        # Dapatkan path absolut ke direktori root project
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../../..'))
        test_file = os.path.abspath(os.path.join(os.path.dirname(__file__), f'{module_name}.py'))
        
        # Jika conda_env diatur, gunakan conda run
        if conda_env:
            # Jalankan test file langsung
            result = subprocess.run(
                [CONDA_PATH, 'run', '-n', conda_env, 'python', test_file],
                capture_output=True,
                text=True,
                check=False,
                env={'PYTHONPATH': project_root}
            )
        else:
            # Tambahkan project root ke PYTHONPATH
            env = os.environ.copy()
            if 'PYTHONPATH' in env:
                env['PYTHONPATH'] = f"{project_root}:{env['PYTHONPATH']}"
            else:
                env['PYTHONPATH'] = project_root
                
            result = subprocess.run(
                [sys.executable, test_file],
                capture_output=True,
                text=True,
                check=False,
                env=env
            )
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

def print_colored(text, color):
    """
    Mencetak teks dengan warna tertentu.
    
    Args:
        text: Teks yang akan dicetak
        color: Kode warna ANSI
    """
    colors = {
        'green': '\033[92m',
        'red': '\033[91m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'reset': '\033[0m'
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")

def check_conda_env(conda_env):
    """
    Memeriksa apakah environment conda tersedia.
    
    Args:
        conda_env: Nama environment conda
        
    Returns:
        Boolean yang menunjukkan apakah environment tersedia
    """
    try:
        result = subprocess.run(
            [CONDA_PATH, 'env', 'list'],
            capture_output=True,
            text=True,
            check=True
        )
        return conda_env in result.stdout
    except Exception as e:
        print_colored(f"Error saat memeriksa environment conda: {str(e)}", 'red')
        return False

def install_dependencies(conda_env):
    """
    Menginstal dependency yang diperlukan.
    
    Args:
        conda_env: Nama environment conda
        
    Returns:
        Boolean yang menunjukkan keberhasilan instalasi
    """
    try:
        # Buat environment jika belum ada
        if not check_conda_env(conda_env):
            print_colored(f"Environment conda '{conda_env}' tidak ditemukan. Membuat environment baru...", 'yellow')
            subprocess.run(
                [CONDA_PATH, 'create', '-n', conda_env, 'python=3.9', '-y'],
                check=True
            )
        
        # Instal dependency
        print_colored("Menginstal dependency...", 'blue')
        
        # Instal dependency testing
        subprocess.run(
            [CONDA_PATH, 'run', '-n', conda_env, 'pip', 'install', 'pytest', 'pytest-cov', 'ipywidgets', 'tqdm', 'pyyaml'],
            check=True
        )
        
        return True
    except Exception as e:
        print_colored(f"Error saat menginstal dependency: {str(e)}", 'red')
        return False

def run_all_tests(conda_env=None):
    """
    Menjalankan semua test untuk split dataset.
    
    Args:
        conda_env: Nama environment conda (opsional)
    """
    # Daftar modul test yang akan dijalankan
    test_modules = [
        'test_components',
        'test_handlers',
        'test_initializer',
        'test_integration',
        'test_sync'
    ]
    
    print_colored("🧪 Menjalankan test untuk split dataset...", 'blue')
    if conda_env:
        print_colored(f"Menggunakan environment conda: {conda_env}", 'blue')
    
    results = {}
    total_tests = len(test_modules)
    success_count = 0
    
    # Jalankan test secara sekuensial untuk menghindari masalah dengan multiprocessing
    for module in test_modules:
        print_colored(f"Menjalankan {module}...", 'blue')
        success, stdout, stderr = run_test_module(module, conda_env)
        results[module] = (success, stdout, stderr)
        if success:
            success_count += 1
    
    # Tampilkan hasil
    print_colored("\n📊 Hasil Test:", 'blue')
    print(f"Total: {total_tests}, Sukses: {success_count}, Gagal: {total_tests - success_count}")
    
    # Tampilkan detail hasil
    for module, (success, stdout, stderr) in results.items():
        status = "✅ SUKSES" if success else "❌ GAGAL"
        color = "green" if success else "red"
        print_colored(f"\n{status}: {module}", color)
        
        if not success:
            print_colored("Output:", 'yellow')
            print(stdout)
            print_colored("Error:", 'red')
            print(stderr)
    
    # Tampilkan ringkasan
    success_rate = (success_count / total_tests) * 100
    if success_rate == 100:
        print_colored(f"\n🎉 Semua test berhasil! ({success_count}/{total_tests})", 'green')
    else:
        print_colored(f"\n⚠️ {success_count}/{total_tests} test berhasil ({success_rate:.1f}%)", 'yellow')
    
    return success_count == total_tests

if __name__ == '__main__':
    # Parse argumen command line
    parser = argparse.ArgumentParser(description='Menjalankan test untuk split dataset')
    parser.add_argument('--conda-env', type=str, default='smartcash_test', help='Nama environment conda untuk menjalankan test')
    parser.add_argument('--skip-dependency-check', action='store_true', help='Lewati pemeriksaan dan instalasi dependency')
    parser.add_argument('--conda-path', type=str, help='Path ke executable conda')
    args = parser.parse_args()
    
    conda_env = args.conda_env
    
    # Set path conda jika disediakan
    if args.conda_path:
        CONDA_PATH = args.conda_path
    
    # Periksa dan instal dependency jika diperlukan
    if not args.skip_dependency_check:
        if not install_dependencies(conda_env):
            print_colored("Gagal menginstal dependency. Mencoba menjalankan test tanpa instalasi dependency...", 'yellow')
    
    # Jalankan test
    success = run_all_tests(conda_env)
    
    # Keluar dengan kode status yang sesuai
    sys.exit(0 if success else 1) 