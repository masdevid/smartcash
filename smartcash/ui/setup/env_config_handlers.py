"""
File: smartcash/ui/setup/env_config_handlers.py
Deskripsi: Setup handler untuk UI konfigurasi environment dengan penanganan cleanup yang ditingkatkan
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
    
    # Setup cleanup handler
    _setup_cleanup_handler(ui_components)
    
    # Log sukses setup
    logger = ui_components.get('logger')
    if logger:
        logger.info("✅ Environment config handlers berhasil diinisialisasi")
    
    return ui_components

def _setup_drive_button_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk tombol connect drive."""
    from IPython.display import display, clear_output
    
    drive_button = ui_components.get('drive_button')
    if not drive_button:
        return
        
    def on_drive_button_click(b):
        """Handler untuk mount Google Drive."""
        logger = ui_components.get('logger')
        
        try:
            # Nonaktifkan tombol selama proses
            drive_button.disabled = True
            
            # Tampilkan status
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, "Menghubungkan ke Google Drive...", "info", "🔄")
            
            # Mount drive
            from google.colab import drive
            drive.mount('/content/drive')
            
            # Update environment
            from smartcash.ui.setup.environment_detector import detect_environment
            detect_environment(ui_components)
            
            # Log sukses
            log_to_ui(ui_components, "Berhasil terhubung ke Google Drive!", "success", "✅")
            if logger:
                logger.success("✅ Berhasil terhubung ke Google Drive")
            
            # Sembunyikan tombol drive setelah sukses
            drive_button.layout.display = 'none'
        
        except Exception as e:
            # Tampilkan error
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
            
            # Tampilkan status
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, "Membuat struktur direktori proyek...", "info", "🔄")
            
            with ThreadPoolExecutor(max_workers=1) as executor:
                # Setup direktori dalam thread terpisah
                future = executor.submit(_setup_project_directory, ui_components)
                result = future.result()
                
                # Update UI berdasarkan hasil
                status_type = "success" if result.get('status') == 'success' else "warning"
                icon = "✅" if result.get('status') == 'success' else "⚠️"
                
                log_to_ui(ui_components, result.get('message', 'Selesai setup direktori'), status_type, icon)
                if logger:
                    if status_type == "success":
                        logger.success(f"{icon} {result.get('message')}")
                    else:
                        logger.warning(f"{icon} {result.get('message')}")
        
        except Exception as e:
            # Tampilkan error
            from smartcash.ui.utils.ui_logger import log_to_ui
            log_to_ui(ui_components, f"Error saat setup direktori: {str(e)}", "error", "❌")
            if logger:
                logger.error(f"❌ Error saat setup direktori: {str(e)}")
        
        finally:
            # Aktifkan kembali tombol
            directory_button.disabled = False
    
    # Register handler
    directory_button.on_click(on_directory_button_click)

def _setup_project_directory(ui_components: Dict[str, Any]) -> Dict[str, Any]:
    """
    Setup direktori proyek.
    
    Args:
        ui_components: Dictionary komponen UI
        
    Returns:
        Dictionary berisi status dan pesan
    """
    try:
        # Dapatkan environment manager
        from smartcash.common.environment import get_environment_manager
        env_manager = get_environment_manager()
        logger = ui_components.get('logger')
        
        # Setup direktori
        stats = env_manager.setup_project_structure(
            use_drive=env_manager.is_drive_mounted,
            progress_callback=lambda current, total, message: _update_setup_progress(ui_components, current, total, message)
        )
        
        # Buat symlinks jika di Colab dan drive terpasang
        if env_manager.is_colab and env_manager.is_drive_mounted:
            symlink_stats = env_manager.create_symlinks(
                progress_callback=lambda current, total, message: _update_setup_progress(ui_components, current, total, message)
            )
            
            # Gabungkan statistik
            for key in symlink_stats:
                if key in stats:
                    stats[key] += symlink_stats[key]
                else:
                    stats[key] = symlink_stats[key]
        
        # Format pesan sukses
        message = f"Berhasil membuat struktur direktori: {stats['created']} direktori baru, {stats['existing']} sudah ada"
        if 'created' in stats and stats['created'] > 0:
            return {'status': 'success', 'message': message}
        else:
            return {'status': 'info', 'message': 'Struktur direktori sudah lengkap'}
    
    except Exception as e:
        return {'status': 'error', 'message': f"Error saat setup direktori: {str(e)}"}

def _update_setup_progress(ui_components: Dict[str, Any], current: int, total: int, message: str) -> None:
    """
    Update progress setup direktori.
    
    Args:
        ui_components: Dictionary komponen UI
        current: Nilai progress saat ini
        total: Total nilai progress
        message: Pesan progress
    """
    # Update progress bar jika tersedia
    progress_bar = ui_components.get('progress_bar')
    progress_message = ui_components.get('progress_message')
    
    if progress_bar:
        progress_bar.max = total
        progress_bar.value = current
        progress_bar.layout.visibility = 'visible'
    
    if progress_message:
        progress_message.value = message
        progress_message.layout.visibility = 'visible'
    
    # Coba update juga melalui progress tracker jika tersedia
    tracker_key = 'env_config_tracker'
    if tracker_key in ui_components:
        tracker = ui_components[tracker_key]
        tracker.update(current, message)

def _setup_cleanup_handler(ui_components: Dict[str, Any]) -> None:
    """Setup handler untuk cleanup saat cell dijalankan ulang."""
    # Import komponen untuk cleanup dengan error handling
    try:
        from smartcash.ui.utils.logging_utils import create_cleanup_function, restore_stdout, reset_logging
        
        def enhanced_cleanup():
            """Enhanced cleanup function dengan penanganan error yang lebih baik."""
            try:
                # Restore stdout ke aslinya
                restore_stdout(ui_components)
                
                # Reset semua konfigurasi logging
                reset_logging()
                
                # Unregister observer jika ada
                if 'observer_manager' in ui_components and 'observer_group' in ui_components:
                    try:
                        ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
                    except Exception:
                        pass
                
                # Reset progress
                if 'progress_bar' in ui_components and 'progress_message' in ui_components:
                    ui_components['progress_bar'].layout.visibility = 'hidden'
                    ui_components['progress_message'].layout.visibility = 'hidden'
                
                # Reset log handler
                logger = ui_components.get('logger')
                if logger and hasattr(logger, '_callbacks'):
                    logger._callbacks = []
                
                # Hapus interceptor stdout
                if 'custom_stdout' in ui_components:
                    ui_components.pop('custom_stdout', None)
                
                # Log sukses cleanup
                logger = ui_components.get('logger')
                if logger:
                    logger.debug("🧹 Cleanup env_config berhasil dijalankan")
            except Exception:
                # Ignore errors during cleanup
                pass
        
        # Tetapkan fungsi cleanup ke ui_components
        ui_components['cleanup'] = enhanced_cleanup
        
        # Register cleanup dengan IPython event
        try:
            from IPython import get_ipython
            ipython = get_ipython()
            if ipython:
                # Unregister existing handlers terlebih dahulu untuk mencegah duplikasi
                if hasattr(ipython.events, '_events'):
                    for event_type in ipython.events._events:
                        if event_type == 'pre_run_cell':
                            existing_handlers = ipython.events._events[event_type]
                            for handler in list(existing_handlers):
                                if handler.__qualname__.endswith('cleanup'):
                                    ipython.events.unregister('pre_run_cell', handler)
                
                # Register cleanup baru
                ipython.events.register('pre_run_cell', enhanced_cleanup)
        except (ImportError, AttributeError):
            pass
    
    except ImportError:
        # Fallback minimal jika komponen tidak tersedia
        def minimal_cleanup():
            # Reset stdout jika ada
            if 'original_stdout' in ui_components:
                import sys
                sys.stdout = ui_components['original_stdout']
        
        # Tetapkan fungsi cleanup ke ui_components
        ui_components['cleanup'] = minimal_cleanup