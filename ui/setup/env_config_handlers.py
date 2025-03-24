"""
File: smartcash/ui/setup/env_config_handlers.py
Deskripsi: Handler untuk komponen UI konfigurasi environment dengan delegasi ke handlers khusus
"""

from typing import Dict, Any, Optional
from IPython.display import display
from concurrent.futures import ThreadPoolExecutor

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk komponen UI environment config dengan delegasi.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Tambahkan logger jika belum ada
    logger = ui_components.get('logger')
    
    # Setup observer untuk menerima event notifikasi
    _setup_observers(ui_components)
    
    # Deteksi environment
    _setup_environment_detection(ui_components, env)
    
    # Setup reset progress helper
    ui_components['reset_progress'] = lambda: _reset_progress(ui_components)
    
    # Setup handlers untuk tombol-tombol
    _setup_button_handlers(ui_components)
    
    # Setup cleanup function
    _setup_cleanup(ui_components)
    
    # Log inisialisasi selesai
    if logger:
        logger.info("✅ Environment config handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_observers(ui_components: Dict[str, Any]) -> None:
    """Setup observer handlers."""
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        setup_observer_handlers(ui_components, "env_config_observers")
    except ImportError:
        pass

def _setup_environment_detection(ui_components: Dict[str, Any], env) -> None:
    """Setup environment detection."""
    try:
        from smartcash.ui.setup.environment_detector import detect_environment
        detect_environment(ui_components, env)
    except ImportError:
        # Fallback ke pesan langsung
        if 'status' in ui_components:
            with ui_components['status']:
                from IPython.display import HTML
                display(HTML("<div style='color:orange'>⚠️ Environment detector tidak tersedia</div>"))

def _reset_progress(ui_components: Dict[str, Any]) -> None:
    """Reset progress bar dan message ke hidden."""
    if 'progress_bar' in ui_components and 'progress_message' in ui_components:
        ui_components['progress_bar'].layout.visibility = 'hidden'
        ui_components['progress_message'].layout.visibility = 'hidden'
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Siap digunakan"

def _setup_button_handlers(ui_components: Dict[str, Any]) -> None:
    """Setup handlers untuk tombol-tombol."""
    # Register handlers untuk tombol Drive dan Directory
    if 'drive_button' in ui_components:
        ui_components['drive_button'].on_click(lambda b: _on_drive_button_clicked(b, ui_components))
    
    if 'directory_button' in ui_components:
        ui_components['directory_button'].on_click(lambda b: _on_directory_button_clicked(b, ui_components))

def _on_drive_button_clicked(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol hubungkan Google Drive.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Nonaktifkan tombol selama proses berjalan
        _disable_buttons(ui_components, True)
        
        # Tampilkan progress bar
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 0
            ui_components['progress_message'].value = "Memeriksa Google Drive..."
        
        # Import drive_connector
        from smartcash.ui.setup.drive_connector import connect_google_drive, create_drive_directory_structure, sync_configs_with_drive, create_symlinks_to_drive
        
        # Jalankan operasi dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Connect to drive
            ui_components['progress_bar'].value = 0
            ui_components['progress_message'].value = "Menghubungkan ke Google Drive..."
            future = executor.submit(connect_google_drive, ui_components)
            success, drive_path = future.result()
            
            if success and drive_path:
                # Setup directory structure
                ui_components['progress_bar'].value = 1
                ui_components['progress_message'].value = "Menyiapkan struktur direktori..."
                future = executor.submit(create_drive_directory_structure, drive_path, ui_components)
                success_dir = future.result()
                
                # Symlinks
                ui_components['progress_bar'].value = 2
                ui_components['progress_message'].value = "Membuat symlinks..."
                future = executor.submit(create_symlinks_to_drive, ui_components, drive_path)
                success_symlinks = future.result()
                
                # Sync configs
                ui_components['progress_bar'].value = 3
                ui_components['progress_message'].value = "Sinkronisasi konfigurasi..."
                future = executor.submit(sync_configs_with_drive, ui_components, drive_path)
                success_sync = future.result()
                
                # Semua selesai
                ui_components['progress_bar'].value = 4
                ui_components['progress_message'].value = "Setup Google Drive selesai!"
        
        # Reset progress setelah selesai
        _reset_progress(ui_components)
        
    except Exception as e:
        # Log error
        if logger:
            logger.error(f"❌ Error saat menghubungkan Drive: {str(e)}")
        
        # Tampilkan error di UI
        if 'status' in ui_components:
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator("error", f"❌ Error saat menghubungkan Drive: {str(e)}"))
    finally:
        # Re-aktifkan tombol
        _disable_buttons(ui_components, False)

def _on_directory_button_clicked(b, ui_components: Dict[str, Any]) -> None:
    """
    Handler untuk tombol setup direktori lokal.
    
    Args:
        b: Button widget
        ui_components: Dictionary komponen UI
    """
    logger = ui_components.get('logger')
    
    try:
        # Nonaktifkan tombol selama proses berjalan
        _disable_buttons(ui_components, True)
        
        # Tampilkan progress bar
        if 'progress_bar' in ui_components and 'progress_message' in ui_components:
            ui_components['progress_bar'].layout.visibility = 'visible'
            ui_components['progress_message'].layout.visibility = 'visible'
            ui_components['progress_bar'].value = 0
            ui_components['progress_message'].value = "Menyiapkan direktori lokal..."
        
        # Import module
        from smartcash.ui.setup.directory_manager import create_project_structure_async
        
        # Jalankan operasi dalam thread terpisah
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(create_project_structure_async, ui_components)
            future.result()  # Wait for completion
            
        # Reset progress setelah selesai
        _reset_progress(ui_components)
        
    except Exception as e:
        # Log error
        if logger:
            logger.error(f"❌ Error saat setup direktori: {str(e)}")
        
        # Tampilkan error di UI
        if 'status' in ui_components:
            with ui_components['status']:
                from smartcash.ui.utils.alert_utils import create_status_indicator
                display(create_status_indicator("error", f"❌ Error saat setup direktori: {str(e)}"))
    finally:
        # Re-aktifkan tombol
        _disable_buttons(ui_components, False)

def _disable_buttons(ui_components: Dict[str, Any], disabled: bool) -> None:
    """
    Nonaktifkan/aktifkan tombol-tombol.
    
    Args:
        ui_components: Dictionary komponen UI
        disabled: True untuk nonaktifkan, False untuk aktifkan
    """
    for button_name in ['drive_button', 'directory_button']:
        if button_name in ui_components:
            ui_components[button_name].disabled = disabled

def _setup_cleanup(ui_components: Dict[str, Any]) -> None:
    """Setup cleanup function."""
    def cleanup_resources():
        """Fungsi untuk membersihkan resources."""
        try:
            # Reset progress
            _reset_progress(ui_components)
            
            # Unregister observer group jika ada
            if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                try:
                    ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                except Exception:
                    pass
            
            # Reset logging
            try:
                from smartcash.ui.utils.logging_utils import reset_logging
                reset_logging()
            except ImportError:
                pass
            
            # Log cleanup
            logger = ui_components.get('logger')
            if logger:
                logger.debug("🧹 Cleanup env_config_handlers berhasil")
        except Exception as e:
            logger = ui_components.get('logger')
            if logger:
                logger.warning(f"⚠️ Error saat cleanup: {str(e)}")
    
    # Tetapkan fungsi cleanup ke ui_components
    ui_components['cleanup'] = cleanup_resources
    
    # Register cleanup dengan IPython event
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_resources)
    except (ImportError, AttributeError):
        pass