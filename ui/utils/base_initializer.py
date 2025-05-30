"""
File: smartcash/ui/utils/base_initializer.py
Deskripsi: Base initializer untuk semua modul UI dengan pendekatan DRY
"""

from typing import Dict, Any, Tuple, Callable, List, Optional
from IPython.display import display

def initialize_module_ui(
    module_name: str,
    create_ui_func: Callable[[Any, Any], Dict[str, Any]],
    setup_config_handler_func: Optional[Callable] = None,
    setup_specific_handlers_func: Optional[Callable] = None,
    detect_state_func: Optional[Callable] = None,
    button_keys: List[str] = None,
    multi_progress_config: Optional[Dict[str, str]] = None,
    observer_group: Optional[str] = None
) -> Dict[str, Any]:
    """
    Base initializer untuk modul UI dengan pendekatan DRY.
    
    Args:
        module_name: Nama modul (contoh: 'preprocessing', 'augmentation')
        create_ui_func: Fungsi untuk membuat UI komponen
        setup_config_handler_func: Fungsi untuk setup config handler
        setup_specific_handlers_func: Fungsi untuk setup handler spesifik modul
        detect_state_func: Fungsi untuk deteksi state modul
        button_keys: List nama tombol yang perlu diattach dengan ui_components
        multi_progress_config: Konfigurasi untuk multi progress tracking
        observer_group: Nama grup observer untuk modul
        
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    ui_components = {'module_name': module_name}
    
    try:
        # Setup environment dan config (gunakan singleton)
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        env, config = setup_notebook_environment(module_name)
        
        # Buat komponen UI
        ui_components = create_ui_func(env, config)
        
        # Setup logging untuk UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, module_name)
        ui_components['logger'] = logger
        
        # Setup config handler jika ada
        if setup_config_handler_func:
            ui_components = setup_config_handler_func(ui_components, config, env)
        
        # Setup progress tracking dengan handler bersama
        from smartcash.ui.handlers.progress_handler import setup_throttled_progress_callback
        progress_callback, register_callback, reset_func = setup_throttled_progress_callback(ui_components, logger)
        ui_components.update({
            'progress_callback': progress_callback,
            'register_progress_callback': register_callback,
            'reset_progress_bar': reset_func
        })
        
        # Setup button dan cleanup handler spesifik untuk setiap modul
        if module_name == 'preprocessing':
            # Handler untuk preprocessing
            from smartcash.ui.dataset.preprocessing.handlers.button_handler import setup_preprocessing_button_handlers
            ui_components = setup_preprocessing_button_handlers(ui_components, module_name, config, env)
            
            from smartcash.ui.dataset.preprocessing.handlers.cleanup_handler import setup_cleanup_handler
            ui_components = setup_cleanup_handler(ui_components, env, config)
            
        elif module_name == 'augmentation':
            # Handler untuk augmentation
            from smartcash.ui.dataset.augmentation.handlers.button_handler import setup_augmentation_button_handlers
            ui_components = setup_augmentation_button_handlers(ui_components, module_name, config, env)
            
            from smartcash.ui.dataset.augmentation.handlers.cleanup_handler import setup_cleanup_handler
            ui_components = setup_cleanup_handler(ui_components, env, config)
            
        elif module_name == 'download':
            # Handler untuk download
            from smartcash.ui.dataset.download.handlers.setup_handlers import setup_download_handlers
            ui_components = setup_download_handlers(ui_components, env, config)
            
            from smartcash.ui.dataset.download.handlers.cleanup_handler import setup_download_cleanup_handler
            ui_components = setup_download_cleanup_handler(ui_components, module_name, config, env)
        
        # Setup handler spesifik modul jika ada
        if setup_specific_handlers_func:
            ui_components = setup_specific_handlers_func(ui_components, env, config)
        
        # Setup observer integration
        if observer_group:
            try:
                from smartcash.ui.handlers.observer_handler import setup_observer_handlers
                ui_components = setup_observer_handlers(ui_components, observer_group)
            except ImportError:
                pass
        
        # Deteksi status data jika fungsi tersedia
        if detect_state_func:
            ui_components = detect_state_func(ui_components)
        
        # Register cleanup function untuk cell execution
        from smartcash.ui.utils.logging_utils import register_cleanup_on_cell_execution
        register_cleanup_on_cell_execution(ui_components)
        
        # Attach UI components ke tombol utama untuk akses data
        if button_keys:
            for k in button_keys:
                if k in ui_components and ui_components[k] is not None:
                    setattr(ui_components[k], 'ui_components', ui_components)
        
        # Tampilkan UI
        display(ui_components['ui'])
        
    except Exception as e:
        # Fallback jika ada error
        try:
            # Coba import logger untuk logging error
            try:
                from smartcash.common.logger import get_logger
                logger = get_logger(f"{module_name}_initializer")
                # Hanya log error kritis
                pass
            except ImportError:
                # Jika tidak bisa import logger, tidak perlu print error ke console
                pass
            
            # Buat UI fallback dengan pesan error
            from smartcash.ui.utils.fallback_utils import create_fallback_ui
            ui_components = create_fallback_ui(ui_components, f"⚠️ Error saat inisialisasi {module_name}: {str(e)}", "error")
            display(ui_components['ui'])
        except Exception as inner_e:
            # Jika fallback juga error, buat UI minimal
            print(f"⚠️ Error saat membuat fallback UI: {str(inner_e)}")
            from IPython.display import HTML
            display(HTML(f"<div style='color:red; padding:10px; border:1px solid red;'>⚠️ Error saat inisialisasi {module_name}: {str(e)}</div>"))
            # Pastikan ui_components memiliki minimal UI untuk mencegah error lanjutan
            if 'ui' not in ui_components:
                ui_components['ui'] = HTML(f"<div>Error UI</div>")
    
    return ui_components
