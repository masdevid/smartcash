"""
File: smartcash/ui/dataset/shared/integration.py
Deskripsi: Utilitas integrasi bersama untuk menghubungkan berbagai komponen UI 
dengan pendekatan DRY antara preprocessing dan augmentasi
"""

from typing import Dict, Any, Optional, Callable
from IPython.display import display

def apply_shared_handlers(ui_components: Dict[str, Any], env=None, config=None, 
                         module_type: str = 'preprocessing') -> Dict[str, Any]:
    """
    Aplikasikan semua shared handlers ke UI components dengan pendekatan DRY.
    
    Args:
        ui_components: Dictionary komponen UI
        env: Environment manager (optional)
        config: Konfigurasi aplikasi (optional)
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
        
    Returns:
        Dictionary UI components yang diupdate
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    try:
        # 1. Setup Progress Handler - Essential untuk semua operasi
        ui_components = _setup_progress_handler(ui_components, logger)
        
        # 2. Setup Visualization Handler
        ui_components = _setup_visualization_handler(ui_components, env, config, module_type)
        
        # 3. Setup Cleanup Handler
        ui_components = _setup_cleanup_handler(ui_components, env, config, module_type)
        
        # 4. Setup Summary Handler
        ui_components = _setup_summary_handler(ui_components, env, config, module_type)
        
        # 5. Setup Observer untuk event handling (jika tersedia)
        ui_components = _setup_observer_handler(ui_components, module_type)
        
        if logger: logger.info(f"{ICONS.get('success', '✅')} Shared handlers berhasil diintegrasikan untuk {module_type}")
    except Exception as e:
        if logger: logger.warning(f"{ICONS.get('warning', '⚠️')} Error integrasi handlers: {str(e)}")
    
    return ui_components

def create_cleanup_function(ui_components: Dict[str, Any], module_type: str = 'preprocessing') -> None:
    """
    Buat dan register fungsi cleanup untuk resources UI.
    
    Args:
        ui_components: Dictionary komponen UI
        module_type: Tipe modul ('preprocessing' atau 'augmentation')
    """
    logger = ui_components.get('logger')
    from smartcash.ui.utils.constants import ICONS
    
    # Definisikan cleanup function
    def cleanup_resources():
        """Cleanup global resources untuk menghindari memory leaks."""
        if logger: logger.debug(f"{ICONS.get('cleanup', '🧹')} Membersihkan resources {module_type}")
        
        # 1. Unregister observer jika ada
        if 'observer_manager' in ui_components and 'observer_group' in ui_components:
            try:
                ui_components['observer_manager'].unregister_group(ui_components['observer_group'])
            except Exception:
                pass
        
        # 2. Reset progress tracking
        if 'reset_progress_bar' in ui_components and callable(ui_components['reset_progress_bar']):
            ui_components['reset_progress_bar']()
        
        # 3. Reset logging
        try:
            from smartcash.ui.utils.logging_utils import reset_logging
            reset_logging()
        except ImportError:
            pass
        
        # 4. Clean other registered resources
        if 'resources' in ui_components:
            for resource, cleanup_func in ui_components['resources']:
                try:
                    if cleanup_func and callable(cleanup_func):
                        cleanup_func(resource)
                    elif hasattr(resource, 'close') and callable(resource.close):
                        resource.close()
                except Exception:
                    pass
            
            ui_components['resources'] = []
    
    # Simpan fungsi ke ui_components
    ui_components['cleanup'] = cleanup_resources
    
    # Register sebagai event handler untuk IPython jika dalam notebook
    try:
        from IPython import get_ipython
        ipython = get_ipython()
        
        if ipython:
            # Cek apakah sudah terdaftar untuk hindari duplikasi
            handler_key = f"_{module_type}_cleanup_registered"
            if not hasattr(ipython.events, handler_key):
                ipython.events.register('pre_run_cell', cleanup_resources)
                setattr(ipython.events, handler_key, True)
                
                if logger: logger.debug(f"{ICONS.get('info', 'ℹ️')} Cleanup function berhasil didaftarkan")
    except (ImportError, AttributeError):
        # Bukan environment notebook, atau IPython tidak tersedia
        pass

# === FUNGSI HELPER INTERNAL ===

def _setup_progress_handler(ui_components: Dict[str, Any], logger) -> Dict[str, Any]:
    """Setup progress handler dengan throttling dan normalisasi."""
    try:
        from smartcash.ui.dataset.shared.progress_handler import setup_throttled_progress_callback
        
        # Buat callback dengan throttling
        progress_callback, register_progress_callback, reset_progress_bar = setup_throttled_progress_callback(ui_components, logger)
        
        # Tambahkan ke ui_components
        ui_components.update({
            'progress_callback': progress_callback,
            'register_progress_callback': register_progress_callback,
            'reset_progress_bar': reset_progress_bar
        })
        
        # Register ke manager jika tersedia
        manager_key = 'dataset_manager' if 'preprocessing_running' in ui_components else 'augmentation_manager'
        if manager_key in ui_components and ui_components[manager_key]:
            register_progress_callback(ui_components[manager_key])
    except ImportError:
        if logger: logger.debug(f"ℹ️ Progress handler tidak tersedia, menggunakan default")
    
    return ui_components

def _setup_visualization_handler(ui_components: Dict[str, Any], env, config, module_type: str) -> Dict[str, Any]:
    """Setup visualization handler untuk dataset visualization."""
    try:
        from smartcash.ui.dataset.shared.visualization_handler import setup_shared_visualization_handlers
        ui_components = setup_shared_visualization_handlers(ui_components, env, config, module_type)
    except ImportError:
        if logger: logger.debug(f"ℹ️ Visualization handler tidak tersedia, menggunakan default")
    
    return ui_components

def _setup_cleanup_handler(ui_components: Dict[str, Any], env, config, module_type: str) -> Dict[str, Any]:
    """Setup cleanup handler untuk pembersihan data."""
    try:
        from smartcash.ui.dataset.shared.cleanup_handler import setup_shared_cleanup_handler
        ui_components = setup_shared_cleanup_handler(ui_components, env, config, module_type)
    except ImportError:
        if logger: logger.debug(f"ℹ️ Cleanup handler tidak tersedia, menggunakan default")
    
    return ui_components

def _setup_summary_handler(ui_components: Dict[str, Any], env, config, module_type: str) -> Dict[str, Any]:
    """Setup summary handler untuk ringkasan hasil."""
    try:
        from smartcash.ui.dataset.shared.summary_handler import setup_shared_summary_handler
        ui_components = setup_shared_summary_handler(ui_components, env, config, module_type)
    except ImportError:
        if logger: logger.debug(f"ℹ️ Summary handler tidak tersedia, menggunakan default")
    
    return ui_components

def _setup_observer_handler(ui_components: Dict[str, Any], module_type: str) -> Dict[str, Any]:
    """Setup observer handler untuk event system."""
    logger = ui_components.get('logger')
    
    try:
        from smartcash.ui.handlers.observer_handler import setup_observer_handlers
        ui_components = setup_observer_handlers(ui_components, f"{module_type}_observers")
        if logger: logger.debug(f"ℹ️ Observer berhasil diinisialisasi")
    except ImportError:
        if logger: logger.debug(f"ℹ️ Observer tidak tersedia")
    
    return ui_components