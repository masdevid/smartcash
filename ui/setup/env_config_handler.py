"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk komponen UI konfigurasi environment dengan integrasi ui_handlers dan perbaikan reset progress
"""

import os
import sys
from typing import Dict, Any, Optional
from IPython.display import display, clear_output
from concurrent.futures import ThreadPoolExecutor

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI environment config dengan integrasi ui_handlers.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    
    # Import komponen dari handlers untuk mengurangi duplikasi
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    from smartcash.ui.setup.env_detection import detect_environment
    from smartcash.ui.setup.drive_handler import handle_drive_connection
    from smartcash.ui.setup.directory_handler import handle_directory_setup
    from smartcash.ui.utils.alert_utils import create_status_indicator
    
    # Setup observer handlers untuk menerima event notifikasi
    ui_components = setup_observer_handlers(ui_components, "env_config_observers")
    
    # Deteksi environment jika belum ada
    ui_components = detect_environment(ui_components, env)
    
    # Helper untuk reset progress setelah operasi selesai/error
    def reset_progress():
        """Reset progress bar dan message ke hidden"""
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'hidden'
            ui_components['progress_message'].layout.visibility = 'hidden'
            ui_components['progress_bar'].value = 0
            ui_components['progress_message'].value = "Siap digunakan"
    
    # Handler untuk tombol Drive dengan error handling yang lebih baik
    def on_drive_button_clicked(b):
        """Handler untuk tombol hubungkan Google Drive menggunakan ThreadPoolExecutor."""
        try:
            # Nonaktifkan tombol selama proses berjalan
            ui_components['drive_button'].disabled = True
            ui_components['directory_button'].disabled = True
            
            # Tampilkan progress bar
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'visible'
                ui_components['progress_message'].layout.visibility = 'visible'
            
            # Jalankan operasi dalam thread terpisah untuk tidak memblokir UI
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(handle_drive_connection, ui_components)
                future.result()  # Tunggu sampai selesai
            
            # Reset progress bar setelah operasi selesai
            reset_progress()
            
            # Re-aktifkan tombol setelah selesai
            ui_components['drive_button'].disabled = False
            ui_components['directory_button'].disabled = False
            
        except Exception as e:
            # Tangani error dan tampilkan di log
            if logger:
                logger.error(f"❌ Error saat menghubungkan Drive: {str(e)}")
                
            with ui_components['status']:
                display(create_status_indicator("error", f"❌ Error saat menghubungkan Drive: {str(e)}"))
                
            # Reset progress setelah error
            reset_progress()
                
            # Re-aktifkan tombol setelah error
            ui_components['drive_button'].disabled = False
            ui_components['directory_button'].disabled = False
            
            # Jalankan cleanup jika ada
            if callable(getattr(ui_components.get('cleanup'), None)):
                ui_components['cleanup']()
    
    # Handler untuk tombol Directory dengan error handling yang lebih baik
    def on_directory_button_clicked(b):
        """Handler untuk tombol setup direktori lokal menggunakan ThreadPoolExecutor."""
        try:
            # Nonaktifkan tombol selama proses berjalan
            ui_components['drive_button'].disabled = True
            ui_components['directory_button'].disabled = True
            
            # Tampilkan progress bar
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'visible'
                ui_components['progress_message'].layout.visibility = 'visible'
            
            # Jalankan operasi dalam thread terpisah untuk tidak memblokir UI
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(handle_directory_setup, ui_components)
                future.result()  # Tunggu sampai selesai
            
            # Reset progress bar setelah operasi selesai
            reset_progress()
            
            # Re-aktifkan tombol setelah selesai
            ui_components['drive_button'].disabled = False
            ui_components['directory_button'].disabled = False
            
        except Exception as e:
            # Tangani error dan tampilkan di log
            if logger:
                logger.error(f"❌ Error saat setup direktori: {str(e)}")
                
            with ui_components['status']:
                display(create_status_indicator("error", f"❌ Error saat setup direktori: {str(e)}"))
                
            # Reset progress setelah error
            reset_progress()
                
            # Re-aktifkan tombol setelah error
            ui_components['drive_button'].disabled = False
            ui_components['directory_button'].disabled = False
            
            # Jalankan cleanup jika ada
            if callable(getattr(ui_components.get('cleanup'), None)):
                ui_components['cleanup']()
    
    # Pastikan tombol tersedia sebelum mendaftarkan handler
    if 'drive_button' in ui_components and ui_components['drive_button'] is not None:
        # Daftarkan handler ke tombol Google Drive
        ui_components['drive_button'].on_click(on_drive_button_clicked)
    
    if 'directory_button' in ui_components and ui_components['directory_button'] is not None:
        # Daftarkan handler ke tombol Directory
        ui_components['directory_button'].on_click(on_directory_button_clicked)
    
    # Definisi cleanup function
    def cleanup_resources():
        """Fungsi untuk membersihkan resources."""
        try:
            # Reset progress
            reset_progress()
            
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            
            # Reset logging
            try:
                from smartcash.ui.utils.logging_utils import reset_logging
                reset_logging()
            except ImportError:
                pass
            
            if logger:
                logger.debug("🧹 Cleanup env_config_handlers berhasil")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat cleanup: {str(e)}")
    
    # Tetapkan fungsi cleanup ke ui_components
    ui_components['cleanup'] = cleanup_resources
    ui_components['reset_progress'] = reset_progress  # Tambahkan helper reset_progress untuk digunakan di luar
    
    # Register cleanup dengan IPython event
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_resources)
    except (ImportError, AttributeError):
        pass
    
    # Log inisialisasi selesai
    if logger:
        logger.info("✅ Environment config handlers berhasil diinisialisasi")
    
    return ui_components