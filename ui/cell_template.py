"""
File: smartcash/ui/cell_template.py
Deskripsi: Template utama untuk inisialisasi sel notebook dengan integrasi standar dan tanpa duplikasi
"""

from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML as DisplayHTML

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
        
        # Buat UI dasar minimal untuk intercept stdout sebelum inisialisasi lainnya
        import ipywidgets as widgets
        if 'status' not in ui_components:
            ui_components['status'] = widgets.Output()
            
        # Setup environment dan load konfigurasi (setelah intercept stdout)
        env, config = setup_notebook_environment(cell_name)
        
        # Buat komponen UI dengan fungsi yang diberikan
        ui_components = create_ui_func(env, config)
        ui_components['module_name'] = cell_name
        
        # Pastikan stdout tetap terintercept setelah UI dibuat
        if 'custom_stdout' in ui_components and 'status' in ui_components:
            # Perbarui referensi ke output widget jika berubah
            ui_components['custom_stdout'].ui_components = ui_components
        
        # Setup logger dengan arahkan ke UI
        logger = setup_ipython_logging(ui_components, cell_name)
        if logger: 
            ui_components['logger'] = logger
            # Hanya log level info atau lebih tinggi untuk mengurangi noise
            logger.info(f"✅ {cell_name} dimulai")
        
        # Setup handlers
        ui_components = setup_handlers_func(ui_components, env, config)
        
        # Jalankan inisialisasi asinkron jika disediakan
        if init_async_func:
            with ThreadPoolExecutor(max_workers=1) as executor:
                executor.submit(init_async_func, ui_components, env, config)
        
        # Setup cleanup for IPython events - gunakan implementasi yang lebih sederhana
        if 'cleanup' in ui_components and callable(ui_components['cleanup']):
            try:
                from IPython import get_ipython
                ipython = get_ipython()
                if ipython:
                    # Jangan daftarkan cleanup dua kali
                    if hasattr(ipython.events, '_events') and 'pre_run_cell' in ipython.events._events:
                        existing_handlers = ipython.events._events['pre_run_cell']
                        for handler in list(existing_handlers):
                            if getattr(handler, '__name__', '') == ui_components['cleanup'].__name__:
                                ipython.events.unregister('pre_run_cell', handler)
                    # Daftarkan cleanup baru
                    ipython.events.register('pre_run_cell', ui_components['cleanup'])
            except (ImportError, AttributeError):
                pass
    
    except Exception as e:
        # Fallback sederhana untuk error
        error_html = f"""
        <div style="padding:10px; background-color:#f8d7da; 
                   color:#721c24; border-radius:4px; margin:5px 0;
                   border-left:4px solid #721c24;">
            <p style="margin:5px 0">❌ Error inisialisasi {cell_name}: {str(e)}</p>
        </div>
        """
        
        # Tampilkan error langsung
        display(DisplayHTML(error_html))
        
        # Set UI components sederhana untuk dikembalikan
        import ipywidgets as widgets
        fallback_output = widgets.Output()
        ui_components = {
            'module_name': cell_name,
            'ui': fallback_output,
            'status': fallback_output
        }
       
    # Return UI components
    return ui_components