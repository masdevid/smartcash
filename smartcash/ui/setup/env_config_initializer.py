"""
File: smartcash/ui/setup/env_config_initializer.py
Deskripsi: Initializer untuk modul konfigurasi environment dengan pendekatan modular dan efisien
"""

from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, HTML

def initialize_env_config_ui():
    """
    Inisialisasi UI dan handler untuk konfigurasi environment dengan pendekatan cell template.
    
    Returns:
        Dictionary komponen UI yang terinisialisasi
    """
    try:
        # Gunakan cell template standar untuk konsistensi
        from smartcash.ui.cell_template import setup_cell
        from smartcash.ui.setup.env_config_component import create_env_config_ui
        from smartcash.ui.setup.env_config_handlers import setup_env_config_handlers
        
        # Inisialisasi async opsional untuk verifikasi konfigurasi
        def init_async(ui_components, env, config):
            try:
                # Verifikasi konfigurasi default
                from smartcash.ui.setup.drive_sync_initializer import initialize_configs
                logger = ui_components.get('logger')
                
                # Jalankan sinkronisasi konfigurasi
                success, message = initialize_configs(logger)
                if logger:
                    logger.info(f"🔄 Sinkronisasi konfigurasi: {message}")
            except Exception as e:
                logger = ui_components.get('logger')
                if logger:
                    logger.warning(f"⚠️ Error saat inisialisasi konfigurasi: {str(e)}")
        
        # Gunakan cell template untuk setup standar
        ui_components = setup_cell(
            cell_name="env_config",
            create_ui_func=create_env_config_ui,
            setup_handlers_func=setup_env_config_handlers,
            init_async_func=init_async
        )
        
        # Pastikan UI ditampilkan jika ada
        if 'ui' in ui_components and ui_components['ui'] is not None:
            display(ui_components['ui'])
            
        return ui_components
        
    except Exception as e:
        # Fallback ultra-minimal jika semua gagal
        error_html = f"""
        <div style="padding:10px; background-color:#f8d7da; 
                   color:#721c24; border-radius:4px; margin:5px 0;
                   border-left:4px solid #721c24;">
            <p style="margin:5px 0">❌ Error inisialisasi environment config: {str(e)}</p>
        </div>
        """
        display(HTML(error_html))
        
        # Return minimal components
        import ipywidgets as widgets
        output = widgets.Output()
        display(output)
        return {'module_name': 'env_config', 'ui': output, 'status': output}