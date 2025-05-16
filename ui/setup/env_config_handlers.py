"""
File: smartcash/ui/setup/env_config_handlers.py
Deskripsi: Handler untuk UI konfigurasi environment
"""

import ipywidgets as widgets
from typing import Dict, Any, Callable, Optional
from IPython import get_ipython

from smartcash.ui.setup.environment_detector import detect_environment
from smartcash.ui.setup.env_config_initializer import _disable_ui_during_processing, _cleanup_ui
from smartcash.common.environment import get_environment_manager
from smartcash.ui.utils.ui_logger import log_to_ui
from smartcash.ui.utils.logging_utils import create_cleanup_function

def setup_env_config_handlers(ui_components: Dict[str, Any], env_manager: Any, config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup handler untuk UI konfigurasi environment
    
    Args:
        ui_components: Dictionary berisi komponen UI
        env_manager: Environment manager
        config: Konfigurasi aplikasi
    
    Returns:
        Dictionary berisi komponen UI yang telah diupdate
    """
    # Detect environment
    ui_components = detect_environment(ui_components, env_manager)
    
    # Setup handler untuk tombol (hanya drive dan directory)
    _setup_drive_button_handler(ui_components)
    _setup_directory_button_handler(ui_components)
    
    # Setup cleanup function
    cleanup_func = create_cleanup_function(ui_components)
    _register_cleanup_event(cleanup_func)
    
    # Log info
    if 'logger' in ui_components:
        ui_components['logger'].info("Environment config handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_drive_button_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol connect drive
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    def on_drive_button_click(b):
        # Nonaktifkan UI selama proses
        _disable_ui_during_processing(ui_components, True)
        
        # Update progress
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Menghubungkan ke Google Drive..."
        
        # Log info
        log_to_ui(ui_components, "Menghubungkan ke Google Drive...", "info", "🔄")
        
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Mount drive
            success, message = env_manager.mount_drive()
            
            # Update progress
            ui_components['progress_bar'].value = 10
            
            if success:
                # Log success
                log_to_ui(ui_components, message, "success", "✅")
                
                # Sembunyikan tombol
                ui_components['drive_button'].layout.display = 'none'
                
                # Detect environment lagi
                detect_environment(ui_components, env_manager)
            else:
                # Log error
                log_to_ui(ui_components, message, "error", "❌")
        except Exception as e:
            # Log error
            log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        finally:
            # Cleanup UI
            _cleanup_ui(ui_components)
    
    # Register handler
    ui_components['drive_button'].on_click(on_drive_button_click)

def _setup_directory_button_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol setup direktori
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    def on_directory_button_click(b):
        # Nonaktifkan UI selama proses
        _disable_ui_during_processing(ui_components, True)
        
        # Update progress
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Membuat struktur direktori..."
        
        # Log info
        log_to_ui(ui_components, "Membuat struktur direktori...", "info", "🔄")
        
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Setup project structure
            result = env_manager.setup_project_structure()
            
            # Update progress
            ui_components['progress_bar'].value = 10
            
            # Log success
            created = result.get('created', 0)
            existing = result.get('existing', 0)
            log_to_ui(
                ui_components,
                f"Berhasil membuat struktur direktori: {created} direktori baru, {existing} sudah ada",
                "success",
                "✅"
            )
        except Exception as e:
            # Log error
            log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        finally:
            # Cleanup UI
            _cleanup_ui(ui_components)
    
    # Register handler
    ui_components['directory_button'].on_click(on_directory_button_click)

def _setup_check_button_handler(ui_components: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol cek environment
    
    Args:
        ui_components: Dictionary berisi komponen UI
    """
    def on_check_button_click(b):
        # Nonaktifkan UI selama proses
        _disable_ui_during_processing(ui_components, True)
        
        # Update progress
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Memeriksa environment..."
        
        # Log info
        log_to_ui(ui_components, "Memeriksa environment...", "info", "🔄")
        
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Cek environment
            env_info = env_manager.check_environment()
            
            # Update progress
            ui_components['progress_bar'].value = 10
            
            # Log success
            log_to_ui(ui_components, "Environment berhasil diperiksa", "success", "✅")
            
            # Detect environment lagi
            detect_environment(ui_components, env_manager)
        except Exception as e:
            # Log error
            log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        finally:
            # Cleanup UI
            _cleanup_ui(ui_components)
    
    # Register handler
    if 'check_button' in ui_components:
        ui_components['check_button'].on_click(on_check_button_click)

def _setup_save_button_handler(ui_components: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Setup handler untuk tombol simpan konfigurasi
    
    Args:
        ui_components: Dictionary berisi komponen UI
        config: Konfigurasi aplikasi
    """
    def on_save_button_click(b):
        # Nonaktifkan UI selama proses
        _disable_ui_during_processing(ui_components, True)
        
        # Update progress
        ui_components['progress_bar'].layout.visibility = 'visible'
        ui_components['progress_message'].layout.visibility = 'visible'
        ui_components['progress_bar'].value = 0
        ui_components['progress_message'].value = "Menyimpan konfigurasi..."
        
        # Log info
        log_to_ui(ui_components, "Menyimpan konfigurasi...", "info", "🔄")
        
        try:
            # Dapatkan environment manager
            env_manager = get_environment_manager()
            
            # Simpan konfigurasi
            success, message = env_manager.save_environment_config()
            
            # Update progress
            ui_components['progress_bar'].value = 10
            
            if success:
                # Log success
                log_to_ui(ui_components, message, "success", "✅")
            else:
                # Log error
                log_to_ui(ui_components, message, "error", "❌")
        except Exception as e:
            # Log error
            log_to_ui(ui_components, f"Error: {str(e)}", "error", "❌")
        finally:
            # Cleanup UI
            _cleanup_ui(ui_components)
    
    # Register handler
    if 'save_button' in ui_components:
        ui_components['save_button'].on_click(on_save_button_click)

def _register_cleanup_event(cleanup_func: Callable) -> bool:
    """
    Register cleanup function ke IPython event
    
    Args:
        cleanup_func: Fungsi cleanup
    
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        # Dapatkan IPython instance
        ipython = get_ipython()
        
        # Register event
        if ipython:
            ipython.events.register('pre_run_cell', cleanup_func)
            return True
    except Exception:
        pass
    
    return False
