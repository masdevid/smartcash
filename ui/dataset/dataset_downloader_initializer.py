"""
File: smartcash/ui/dataset/dataset_downloader_initializer.py
Deskripsi: Initializer untuk modul download dataset dengan pendekatan modular
"""

from typing import Dict, Any
from IPython.display import display

def initialize_dataset_downloader() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk download dataset.
    
    Returns:
        Dictionary UI components yang terinisialisasi
    """
    try:
        # Gunakan cell template standar untuk konsistensi
        from smartcash.ui.cell_template import setup_cell
        from smartcash.ui.dataset.dataset_downloader_component import create_dataset_downloader_ui
        from smartcash.ui.dataset.dataset_downloader_handler import setup_dataset_downloader_handlers
        
        # Inisialisasi async opsional untuk verifikasi konfigurasi
        def init_async(ui_components, env, config):
            try:
                # Dapatkan info endpoint dan dataset
                from smartcash.ui.dataset.handlers.endpoint_handler import get_available_endpoints
                logger = ui_components.get('logger')
                
                # Dapatkan list endpoint yang tersedia
                endpoints = get_available_endpoints(ui_components)
                if logger and endpoints:
                    logger.info(f"🔍 {len(endpoints)} endpoint dataset terdeteksi")
            except Exception as e:
                logger = ui_components.get('logger')
                if logger:
                    logger.warning(f"⚠️ Error saat mendapatkan endpoint: {str(e)}")
        
        # Gunakan cell template untuk setup standar
        ui_components = setup_cell(
            cell_name="dataset_downloader",
            create_ui_func=create_dataset_downloader_ui,
            setup_handlers_func=setup_dataset_downloader_handlers,
            init_async_func=init_async
        )
        
        # Pastikan UI ditampilkan jika ada
        if 'ui' in ui_components and ui_components['ui'] is not None:
            display(ui_components['ui'])
            
        return ui_components
        
    except Exception as e:
        # Fallback minimal jika terjadi error
        from IPython.display import HTML
        error_html = f"""
        <div style="padding:10px; background-color:#f8d7da; 
                   color:#721c24; border-radius:4px; margin:5px 0;
                   border-left:4px solid #721c24;">
            <p style="margin:5px 0">❌ Error inisialisasi dataset downloader: {str(e)}</p>
        </div>
        """
        display(HTML(error_html))
        
        # Return minimal components
        import ipywidgets as widgets
        output = widgets.Output()
        display(output)
        return {'module_name': 'dataset_downloader', 'ui': output, 'status': output}