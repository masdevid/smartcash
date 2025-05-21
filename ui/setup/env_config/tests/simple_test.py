"""
File: smartcash/ui/setup/env_config/tests/simple_test.py
Deskripsi: Test sederhana untuk perbaikan UILogger dan sinkronisasi environment
"""

import sys
import os
from pathlib import Path
import logging

# Tambahkan path root ke sys.path jika belum ada
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

def test_ui_logger_empty_messages():
    """
    Test untuk memastikan UILogger tidak menampilkan pesan kosong
    """
    print("=== Test UILogger dengan Pesan Kosong ===")
    
    # Impor UILogger secara langsung dari file
    sys.path.insert(0, str(project_root / "smartcash" / "ui" / "utils"))
    from ui_logger import UILogger
    
    # Buat mock UI components
    class MockOutput:
        def __init__(self):
            self.messages = []
            
        def clear_output(self, *args, **kwargs):
            pass
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            pass
    
    # Buat UI components sederhana untuk testing
    output = MockOutput()
    status = MockOutput()
    
    ui_components = {
        'log_output': output,
        'status': status
    }
    
    # Buat logger
    logger = UILogger(ui_components, name="test_logger", log_level=logging.INFO)
    
    # Test dengan pesan kosong dan whitespace
    print("Mencoba log pesan kosong...")
    logger.info("")
    logger.info("   ")
    logger.info("\n")
    
    # Test dengan pesan normal
    print("Mencoba log pesan normal...")
    logger.info("Ini adalah pesan info normal")
    logger.warning("Ini adalah pesan warning")
    logger.error("Ini adalah pesan error")
    
    print("Test UILogger selesai!")
    print("Tidak ada error berarti perbaikan berhasil!")

def test_environment_sync():
    """
    Test untuk memastikan sinkronisasi environment tidak menghasilkan error
    """
    print("\n=== Test Sinkronisasi Environment ===")
    
    # Ubah implementasi sync di environment.py
    env_file = project_root / "smartcash" / "common" / "environment.py"
    
    if not env_file.exists():
        print(f"Error: File environment.py tidak ditemukan di {env_file}")
        return
    
    # Baca file environment.py
    with open(env_file, "r") as f:
        content = f.read()
    
    # Cek apakah masih ada import smartcash.common.config.sync
    if "from smartcash.common.config.sync import sync_all_configs" in content:
        print("❌ Error: Masih ada import sync_all_configs di environment.py")
    else:
        print("✅ Perbaikan sinkronisasi environment berhasil!")
    
    # Cek apakah _sync_config_files_on_drive_connect sudah diubah
    if "_sync_config_files_on_drive_connect(self)" in content and "sync_all_configs" not in content:
        print("✅ Metode _sync_config_files_on_drive_connect sudah diperbaiki!")
    else:
        print("❌ Error: Metode _sync_config_files_on_drive_connect belum diperbaiki")
    
    print("Test sinkronisasi environment selesai!")

def run_all_tests():
    """
    Jalankan semua test
    """
    test_ui_logger_empty_messages()
    test_environment_sync()

if __name__ == "__main__":
    run_all_tests() 