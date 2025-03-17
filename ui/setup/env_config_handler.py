"""
File: smartcash/ui/setup/env_config_handler.py
Deskripsi: Handler untuk konfigurasi environment SmartCash
"""

from typing import Dict, Any, Optional
from pathlib import Path
import ipywidgets as widgets
from IPython.display import display, HTML, clear_output

def setup_env_config_handlers(
    ui_components: Dict[str, Any], 
    env: Optional[Any] = None, 
    config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Setup handler untuk konfigurasi environment.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi environment
        
    Returns:
        Dictionary UI components yang telah ditambahkan handler
    """
    # Import dependencies
    from smartcash.ui.handlers.environment_handler import (
        detect_environment, 
        check_smartcash_dir, 
        filter_drive_tree,
        sync_configs
    )
    from smartcash.ui.handlers.config_handler import setup_config_handlers
    from smartcash.ui.handlers.observer_handler import setup_observer_handlers
    
    # Setup config handlers
    ui_components = setup_config_handlers(ui_components, config)
    
    # Setup observer handlers
    ui_components = setup_observer_handlers(ui_components)
    
    # Tambahkan fungsi-fungsi handler
    def handle_drive_connection(b):
        """Handler untuk koneksi Google Drive."""
        logger = ui_components.get('logger')
        try:
            # Pastikan environment manager tersedia
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager(logger=logger)
            
            # Mount drive
            mount_success, mount_msg = env_manager.mount_drive()
            
            if mount_success:
                # Buat symlink
                symlink_stats = env_manager.create_symlinks()
                
                # Sinkronisasi konfigurasi
                sync_configs(
                    [env_manager.get_path('configs')],
                    [env_manager.drive_path / 'configs'],
                    logger
                )
                
                # Tampilkan tree drive
                with ui_components['info_panel']:
                    clear_output(wait=True)
                    tree_html = env_manager.get_directory_tree(
                        env_manager.drive_path, 
                        max_depth=3
                    )
                    display(HTML(filter_drive_tree(tree_html)))
            else:
                # Tampilkan pesan error
                with ui_components['status']:
                    display(HTML(f"""
                    <div style='color:red; padding:10px; background:#f8d7da; border-radius:4px;'>
                        ❌ Gagal koneksi Google Drive: {mount_msg}
                    </div>
                    """))
        except Exception as e:
            if logger:
                logger.error(f"❌ Error koneksi Drive: {str(e)}")
    
    def handle_local_dir_setup(b):
        """Handler untuk setup direktori lokal."""
        logger = ui_components.get('logger')
        try:
            # Pastikan environment manager tersedia
            from smartcash.common.environment import get_environment_manager
            env_manager = get_environment_manager(logger=logger)
            
            # Setup struktur proyek
            dir_stats = env_manager.setup_project_structure()
            
            # Tampilkan tree direktori
            with ui_components['info_panel']:
                clear_output(wait=True)
                tree_html = env_manager.get_directory_tree(
                    env_manager.base_dir, 
                    max_depth=3
                )
                display(HTML(tree_html))
        except Exception as e:
            if logger:
                logger.error(f"❌ Error setup direktori: {str(e)}")
    
    # Tambahkan handler ke tombol
    ui_components['drive_button'].on_click(handle_drive_connection)
    ui_components['dir_button'].on_click(handle_local_dir_setup)
    
    # Deteksi environment
    detect_environment(ui_components, env)
    
    # Cek direktori smartcash
    check_smartcash_dir(ui_components)
    
    return ui_components