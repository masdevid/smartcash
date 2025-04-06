"""
File: smartcash/ui/setup/env_config_handlers.py
Deskripsi: Setup handler untuk UI konfigurasi environment dengan implementasi DRY
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor

def setup_env_config_handlers(ui_components: Dict[str, Any], env=None, config=None) -> Dict[str, Any]:
    """
    Setup handler untuk UI konfigurasi environment.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    # Setup environment detector
    from smartcash.ui.setup.environment_detector import detect_environment
    ui_components = detect_environment(ui_components, env)
    
    # Setup drive button handler
    _setup_drive_button_handler(ui_components)
    
    # Setup directory button handler
    _setup_directory_button_handler(ui_components)
    
    # Setup cleanup handler menggunakan utility dari common
    from smartcash.ui.utils.logging_utils import create_cleanup_function
    ui_components['cleanup'] = create_cleanup_function(ui_components)
    
    # Register cleanup dengan IPython event
    _register_cleanup_event(ui_components['cleanup'])
    
    # Log sukses setup
    if logger := ui_components.get('logger'):
        logger.info("✅ Environment config handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_drive_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol connect drive."""
    drive_button = ui_components.get('drive_button')
    if not drive_button:
        return
        
    def on_drive_button_click(b):
        """Handler untuk mount Google Drive."""
        logger = ui_components.get('logger')
        
        try:
            # Nonaktifkan tombol selama proses
            drive_button.disabled = True
            
            # Log status
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, "Menghubungkan ke Google Drive...", "info", "🔄")
            
            # Dapatkan environment manager
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            # Mount drive menggunakan fungsi dari environment_manager
            success, message = env_manager.mount_drive()
            
            # Update environment dan UI
            from smartcash.ui.setup.environment_detector import detect_environment
            detect_environment(ui_components)
            
            # Log hasil berdasarkan status
            status_type = "success" if success else "error"
            log_to_ui(ui_components, message, status_type, "✅" if success else "❌")
            
            # Sembunyikan tombol drive jika sukses
            if success:
                drive_button.layout.display = 'none'
        
        except Exception as e:
            # Tampilkan error menggunakan UI logger
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, f"Error saat menghubungkan ke Drive: {str(e)}", "error", "❌")
            if logger:
                logger.error(f"❌ Error saat menghubungkan ke Drive: {str(e)}")
        
        finally:
            # Aktifkan kembali tombol
            drive_button.disabled = False
    
    # Register handler
    drive_button.on_click(on_drive_button_click)

def _setup_directory_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol setup direktori."""
    directory_button = ui_components.get('directory_button')
    if not directory_button:
        return
        
    def on_directory_button_click(b):
        """Handler untuk setup direktori proyek."""
        logger = ui_components.get('logger')
        
        try:
            # Nonaktifkan tombol selama proses
            directory_button.disabled = True
            
            # Log status
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, "Membuat struktur direktori proyek...", "info", "🔄")
            
            # Fungsi callback untuk menjembatani progress tracking
            def update_progress(current, total, message):
                if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                    ui_components['progress_bar'].value = current
                    ui_components['progress_bar'].max = total
                    ui_components['progress_message'].value = message
                    ui_components['progress_bar'].layout.visibility = 'visible'
                    ui_components['progress_message'].layout.visibility = 'visible'
            
            # Dapatkan environment manager
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager()
            
            # Setup direktori proyek dengan progress tracking
            stats = env_manager.setup_project_structure(
                use_drive=env_manager.is_drive_mounted,
                progress_callback=update_progress
            )
            
            # Buat symlinks jika di Colab dan drive terpasang
            if env_manager.is_colab and env_manager.is_drive_mounted:
                # Import symlink helper untuk membuat symlinks dengan path yang benar
                from smartcash.ui.setup.environment_symlink_helper import create_drive_symlinks
                
                symlink_stats = create_drive_symlinks(env_manager.drive_path, ui_components)
                
                # Gabungkan statistik
                for key in symlink_stats:
                    stats[key] = stats.get(key, 0) + symlink_stats.get(key, 0)
            
            # Format pesan sukses
            message = (f"Berhasil membuat struktur direktori: "
                     f"{stats.get('created', 0)} direktori baru, "
                     f"{stats.get('existing', 0)} sudah ada")
            
            # Update UI berdasarkan hasil
            status_type = "success" if stats.get('created', 0) > 0 else "info"
            icon = "✅" if stats.get('created', 0) > 0 else "ℹ️"
            
            # Sembunyikan progress bar setelah selesai
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'hidden'
                ui_components['progress_message'].layout.visibility = 'hidden'
            
            # Log hasil
            log_to_ui(ui_components, message, status_type, icon)
            if logger:
                if status_type == "success":
                    logger.success(f"{icon} {message}")
                else:
                    logger.info(f"{icon} {message}")
        
        except Exception as e:
            # Tampilkan error
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, f"Error saat setup direktori: {str(e)}", "error", "❌")
            if logger:
                logger.error(f"❌ Error saat setup direktori: {str(e)}")
            
            # Reset progress jika tersedia
            if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                ui_components['progress_bar'].layout.visibility = 'hidden'
                ui_components['progress_message'].layout.visibility = 'hidden'
        
        finally:
            # Aktifkan kembali tombol
            directory_button.disabled = False
    
    # Register handler
    directory_button.on_click(on_directory_button_click)

def _register_cleanup_event(cleanup_func: callable) -> bool:
    """Register cleanup function ke IPython pre_run_cell event."""
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython and hasattr(ipython, 'events'):
            # Unregister existing handlers terlebih dahulu untuk mencegah duplikasi
            if hasattr(ipython.events, '_events') and 'pre_run_cell' in ipython.events._events:
                for handler in list(ipython.events._events['pre_run_cell']):
                    if handler.__qualname__.endswith('cleanup'):
                        ipython.events.unregister('pre_run_cell', handler)
            
            # Register cleanup baru
            ipython.events.register('pre_run_cell', cleanup_func)
            return True
            
    except (ImportError, AttributeError):
        pass
        
    return False