"""
File: smartcash/ui/setup/env_config/tests/test_ui_logger_fix.py
Deskripsi: Script pengujian untuk perbaikan UILogger dan sinkronisasi environment
"""

import sys
import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display

# Tambahkan path root ke sys.path jika belum ada
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.ui.utils.ui_logger import create_ui_logger, UILogger, intercept_stdout_to_ui
from smartcash.common.environment import get_environment_manager

def test_ui_logger_empty_messages():
    """
    Test untuk memastikan UILogger tidak menampilkan pesan kosong
    """
    print("=== Test UILogger dengan Pesan Kosong ===")
    
    # Buat UI components sederhana untuk testing
    output = widgets.Output()
    status = widgets.Output()
    
    ui_components = {
        'log_output': output,
        'status': status
    }
    
    # Buat logger
    logger = create_ui_logger(ui_components, name="test_logger", redirect_stdout=True)
    
    # Tampilkan UI components
    display(widgets.HTML("<h3>Log Output:</h3>"))
    display(output)
    display(widgets.HTML("<h3>Status Output:</h3>"))
    display(status)
    
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
    
    # Test dengan stdout kosong
    print("Mencoba stdout kosong...")
    print("")
    print("   ")
    print("\n")
    
    # Test dengan stdout normal
    print("Mencoba stdout normal...")
    print("Ini adalah stdout normal")
    
    print("Test selesai!")

def test_environment_sync():
    """
    Test untuk memastikan sinkronisasi environment tidak menghasilkan error
    """
    print("=== Test Sinkronisasi Environment ===")
    
    # Buat UI components untuk testing
    output = widgets.Output()
    
    # Tampilkan output
    display(widgets.HTML("<h3>Environment Sync Output:</h3>"))
    display(output)
    
    with output:
        # Dapatkan environment manager
        env_manager = get_environment_manager()
        
        # Test sync_config
        print("Mencoba sync_config...")
        try:
            success, message = env_manager.sync_config()
            print(f"Hasil sync_config: {success}, {message}")
        except Exception as e:
            print(f"Error saat sync_config: {str(e)}")
        
        # Test save_environment_config
        print("\nMencoba save_environment_config...")
        try:
            success, message = env_manager.save_environment_config()
            print(f"Hasil save_environment_config: {success}, {message}")
        except Exception as e:
            print(f"Error saat save_environment_config: {str(e)}")
        
        # Test _sync_config_files_on_drive_connect
        print("\nMencoba _sync_config_files_on_drive_connect...")
        try:
            env_manager._sync_config_files_on_drive_connect()
            print("_sync_config_files_on_drive_connect selesai tanpa error")
        except Exception as e:
            print(f"Error saat _sync_config_files_on_drive_connect: {str(e)}")
    
    print("Test selesai!")

def run_all_tests():
    """
    Jalankan semua test
    """
    test_ui_logger_empty_messages()
    print("\n")
    test_environment_sync()

if __name__ == "__main__":
    run_all_tests() 