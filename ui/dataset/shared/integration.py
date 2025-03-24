"""
File: smartcash/ui/dataset/shared/integration.py
Deskripsi: Utilitas integrasi untuk menghubungkan modul preprocessing dan augmentasi dengan pendekatan DRY
"""

from typing import Dict, Any
from IPython.display import display

def apply_shared_handlers(ui_components: Dict[str, Any], env=None, config=None, module_type='preprocessing') -> Dict[str, Any]:
    """
    Aplikasikan shared handlers untuk preprocessing dan augmentation dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager
        config: Konfigurasi aplikasi
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang telah diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Import all shared handlers
    from smartcash.ui.dataset.shared.status_panel import update_status_panel
    from smartcash.ui.dataset.shared.progress_handler import setup_throttled_progress_callback
    from smartcash.ui.dataset.shared.visualization_handler import setup_shared_visualization_handlers
    from smartcash.ui.dataset.shared.cleanup_handler import setup_shared_cleanup_handler
    
    try:
        # Setup progress handler
        progress_callback, register_progress_callback, reset_progress_bar = setup_throttled_progress_callback(ui_components, logger)
        
        # Tambahkan fungsi-fungsi progress ke ui_components
        ui_components.update({
            'progress_callback': progress_callback,
            'register_progress_callback': register_progress_callback,
            'reset_progress_bar': reset_progress_bar
        })
        
        # Register progress callback ke manager jika tersedia
        manager_key = 'dataset_manager' if module_type == 'preprocessing' else 'augmentation_manager'
        if manager_key in ui_components:
            register_progress_callback(ui_components[manager_key])
        
        # Setup visualization handler
        ui_components = setup_shared_visualization_handlers(ui_components, env, config, module_type)
        
        # Setup cleanup handler
        ui_components = setup_shared_cleanup_handler(ui_components, env, config, module_type)
        
        # Setup event handler untuk observer jika tersedia
        try:
            from smartcash.ui.handlers.observer_handler import setup_observer_handlers
            ui_components = setup_observer_handlers(ui_components, f"{module_type}_observers")
            if logger: logger.debug(f"{ICONS.get('info', 'ℹ️')} Observer berhasil diinisialisasi")
        except ImportError:
            pass
        
        if logger: logger.info(f"{ICONS.get('success', '✅')} Shared handlers berhasil diaplikasikan untuk {module_type}")
    
    except Exception as e:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Error applying shared handlers: {str(e)}")
    
    return ui_components

def create_cleanup_function(ui_components: Dict[str, Any], module_type: str = 'preprocessing') -> None:
    """
    Buat fungsi cleanup untuk UI components.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Cleanup function untuk dijalankan saat cell di-reset
    def cleanup_resources():
        """Bersihkan resources yang digunakan."""
        if logger: logger.info(f"{ICONS.get('cleanup', '🧹')} Membersihkan resources {module_type}")
        
        # Unregister observer
        if 'observer_manager' in ui_components and 'observer_group' in ui_components:
            try:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            except Exception as e:
                if logger: logger.debug(f"{ICONS.get('warning', '⚠️')} Error unregister observer: {str(e)}")
        
        # Reset logging jika tersedia
        try:
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
        except ImportError:
            pass
    
    # Tambahkan cleanup function ke UI components
    ui_components['cleanup'] = cleanup_resources
    
    # Register cleanup handler ke IPython events jika belum
    try:
        from IPython import get_ipython
        if get_ipython() and 'cleanup' in ui_components and callable(ui_components['cleanup']):
            cleanup = ui_components['cleanup']
            
            # Cek apakah event handler sudah terdaftar untuk mencegah duplikasi
            handler_key = f"_{module_type}_cleanup_registered"
            if not hasattr(get_ipython().events, handler_key):
                get_ipython().events.register('pre_run_cell', cleanup)
                setattr(get_ipython().events, handler_key, True)
    except Exception:
        pass