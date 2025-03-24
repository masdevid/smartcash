"""
File: smartcash/ui/cell_template.py
Deskripsi: Template utama untuk inisialisasi sel notebook dengan integrasi standar dan tanpa duplikasi
"""

from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor

def setup_cell(
    cell_name: str,
    create_ui_func: Callable[[Any, Dict[str, Any]], Dict[str, Any]],
    setup_handlers_func: Callable[[Dict[str, Any], Any, Dict[str, Any]], Dict[str, Any]],
    init_async_func: Optional[Callable[[Dict[str, Any], Any, Dict[str, Any]], None]] = None
) -> Dict[str, Any]:
    """
    Setup sel notebook dengan standarisasi dan menghindari duplikasi.
    
    Args:
        cell_name: Nama sel/modul
        create_ui_func: Fungsi untuk membuat komponen UI
        setup_handlers_func: Fungsi untuk setup handlers
        init_async_func: Fungsi inisialisasi yang akan dijalankan secara asinkron (opsional)
        
    Returns:
        Dictionary UI components yang telah diinisialisasi
    """
    # Inisialisasi ui_components dengan nilai default
    ui_components = {'status': None, 'module_name': cell_name}
    
    try:
        # Import komponen standar dengan pendekatan konsolidasi
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        from smartcash.ui.utils.ui_logger import log_to_ui, intercept_stdout_to_ui
        
        # Setup environment dan load konfigurasi
        env, config = setup_notebook_environment(cell_name)
        
        # Buat komponen UI dengan fungsi yang diberikan
        ui_components = create_ui_func(env, config)
        ui_components['module_name'] = cell_name
        
        # Setup logging dan hubungkan ke UI
        if 'status' in ui_components:
            log_to_ui(ui_components, f"🚀 Inisialisasi {cell_name} dimulai", "info")
        
        # Tangkap stdout ke UI untuk mencegah duplikasi di konsol dan UI
        intercept_stdout_to_ui(ui_components)
        
        # Setup logger dengan arahkan ke UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger: 
            ui_components['logger'] = logger
            logger.info(f"✅ Logger {cell_name} berhasil diinisialisasi")
        
        # Jalankan inisialisasi asinkron jika disediakan
        if init_async_func:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(init_async_func, ui_components, env, config)
        
        # Setup handlers
        ui_components = setup_handlers_func(ui_components, env, config)
        
        # Log selesai inisialisasi
        if logger: 
            logger.info(f"✅ {cell_name} selesai diinisialisasi")
        
        # Setup cleanup for IPython events
        if 'cleanup' in ui_components and callable(ui_components['cleanup']):
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython:
                    ipython.events.register('pre_run_cell', ui_components['cleanup'])
            except (ImportError, AttributeError):
                pass
    
    except Exception as e:
        # Fallback sederhana jika terjadi error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui, show_status
        ui_components = create_fallback_ui(
            ui_components, 
            f"❌ Error saat inisialisasi {cell_name}: {str(e)}", 
            "error"
        )
        show_status(f"Error: {str(e)}", "error", ui_components)
        
        # Log error jika logger tersedia
        if 'logger' in ui_components:
            ui_components['logger'].error(f"❌ Error setup {cell_name}: {str(e)}")
       
    # Return UI components
    return ui_components