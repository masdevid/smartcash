"""
File: smartcash/ui/setup/env_config/tests/test_ui_logger.py
Deskripsi: Script pengujian untuk UILogger yang telah diperbarui
"""

import sys
import os
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display
import logging

# Tambahkan path root ke sys.path jika belum ada
project_root = Path(__file__).resolve().parents[4]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from smartcash.ui.utils.ui_logger import UILogger, create_ui_logger, intercept_stdout_to_ui, restore_stdout, log_to_ui

def test_ui_logger_basic():
    """
    Test dasar untuk UILogger
    """
    print("=== Test Dasar UILogger ===")
    
    # Buat UI components sederhana untuk testing
    output = widgets.Output()
    status = widgets.Output()
    
    ui_components = {
        'log_output': output,
        'status': status
    }
    
    # Buat logger
    logger = create_ui_logger(ui_components, name="test_logger", redirect_stdout=False)
    
    # Tampilkan UI components
    display(widgets.HTML("<h3>Log Output:</h3>"))
    display(output)
    display(widgets.HTML("<h3>Status Output:</h3>"))
    display(status)
    
    # Test dengan berbagai level log
    print("Mencoba berbagai level log...")
    logger.debug("Ini adalah pesan debug")
    logger.info("Ini adalah pesan info")
    logger.success("Ini adalah pesan success")
    logger.warning("Ini adalah pesan warning")
    logger.error("Ini adalah pesan error")
    logger.critical("Ini adalah pesan critical")
    
    # Test dengan pesan kosong (seharusnya tidak muncul)
    print("Mencoba log pesan kosong (tidak akan muncul di UI)...")
    logger.info("")
    logger.info("   ")
    logger.info("\n")
    
    print("Test dasar selesai!")

def test_ui_logger_stdout_redirect():
    """
    Test untuk redirect stdout ke UI
    """
    print("=== Test Redirect Stdout ke UI ===")
    
    # Buat UI components sederhana untuk testing
    output = widgets.Output()
    status = widgets.Output()
    
    ui_components = {
        'log_output': output,
        'status': status
    }
    
    # Tampilkan UI components
    display(widgets.HTML("<h3>Log Output:</h3>"))
    display(output)
    display(widgets.HTML("<h3>Status Output:</h3>"))
    display(status)
    
    # Redirect stdout ke UI
    intercept_stdout_to_ui(ui_components)
    
    # Test dengan print biasa
    print("Ini adalah stdout yang dialihkan ke UI")
    print("Baris kedua dari stdout")
    
    # Test dengan pesan kosong (seharusnya tidak muncul)
    print("")
    print("   ")
    print("\n")
    
    # Kembalikan stdout ke aslinya
    restore_stdout(ui_components)
    
    print("Test redirect stdout selesai! (Pesan ini tidak dialihkan ke UI)")

def test_ui_logger_log_to_ui():
    """
    Test untuk fungsi log_to_ui
    """
    print("=== Test Fungsi log_to_ui ===")
    
    # Buat UI components sederhana untuk testing
    output = widgets.Output()
    status = widgets.Output()
    
    ui_components = {
        'log_output': output,
        'status': status
    }
    
    # Tampilkan UI components
    display(widgets.HTML("<h3>Log Output:</h3>"))
    display(output)
    display(widgets.HTML("<h3>Status Output:</h3>"))
    display(status)
    
    # Test dengan berbagai level log
    print("Mencoba berbagai level log dengan log_to_ui...")
    log_to_ui(ui_components, "Ini adalah pesan info", "info", "ℹ️")
    log_to_ui(ui_components, "Ini adalah pesan success", "success", "✅")
    log_to_ui(ui_components, "Ini adalah pesan warning", "warning", "⚠️")
    log_to_ui(ui_components, "Ini adalah pesan error", "error", "❌")
    log_to_ui(ui_components, "Ini adalah pesan critical", "critical", "🔥")
    
    # Test dengan pesan kosong (seharusnya tidak muncul)
    print("Mencoba log pesan kosong (tidak akan muncul di UI)...")
    log_to_ui(ui_components, "", "info")
    log_to_ui(ui_components, "   ", "info")
    log_to_ui(ui_components, "\n", "info")
    
    print("Test log_to_ui selesai!")

def test_ui_logger_integration_with_dependency_installer():
    """
    Test untuk integrasi UILogger dengan dependency installer
    """
    print("=== Test Integrasi UILogger dengan Dependency Installer ===")
    
    # Buat UI components mirip dependency installer
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='10px',
            overflow='auto'
        )
    )
    
    progress = widgets.IntProgress(
        value=0, 
        min=0, 
        max=100, 
        description='Installing:',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0'
        )
    )
    
    progress_label = widgets.HTML(
        value=""
    )
    
    ui_components = {
        'status': status,
        'install_progress': progress,
        'progress_label': progress_label
    }
    
    # Tampilkan UI components
    display(widgets.HTML("<h3>Status Output:</h3>"))
    display(status)
    display(progress)
    display(progress_label)
    
    # Buat logger
    logger = create_ui_logger(ui_components, name="dependency_installer", redirect_stdout=True)
    
    # Simulasikan instalasi dependencies
    print("Simulasi instalasi dependencies...")
    
    # Update progress
    ui_components['install_progress'].value = 0
    ui_components['progress_label'].value = "Memeriksa dependencies..."
    logger.info("Memeriksa dependencies yang dibutuhkan")
    
    # Update progress
    ui_components['install_progress'].value = 20
    ui_components['progress_label'].value = "Menginstall PyTorch..."
    logger.info("Menginstall PyTorch dan dependencies")
    
    # Update progress
    ui_components['install_progress'].value = 50
    ui_components['progress_label'].value = "Menginstall YOLOv5..."
    logger.info("Menginstall YOLOv5 dan dependencies")
    
    # Update progress dengan warning
    ui_components['install_progress'].value = 70
    ui_components['progress_label'].value = "Menginstall Albumentations..."
    logger.warning("Versi Albumentations mungkin tidak kompatibel")
    
    # Update progress dengan error
    ui_components['install_progress'].value = 80
    ui_components['progress_label'].value = "Menginstall SmartCash utils..."
    logger.error("Gagal menginstall beberapa SmartCash utils")
    
    # Update progress dengan success
    ui_components['install_progress'].value = 100
    ui_components['progress_label'].value = "Instalasi selesai"
    logger.success("Instalasi dependencies selesai")
    
    # Kembalikan stdout ke aslinya
    restore_stdout(ui_components)
    
    print("Test integrasi selesai!")

def run_all_tests():
    """
    Jalankan semua test
    """
    test_ui_logger_basic()
    print("\n")
    test_ui_logger_stdout_redirect()
    print("\n")
    test_ui_logger_log_to_ui()
    print("\n")
    test_ui_logger_integration_with_dependency_installer()

if __name__ == "__main__":
    run_all_tests() 