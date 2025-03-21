"""
File: smartcash/ui/training_config/cell_skeleton.py
Deskripsi: Template untuk cell training config dengan error handling yang tepat
"""

# Import dasar dengan error handling yang konsisten
from IPython.display import display, HTML
import sys
if '.' not in sys.path: sys.path.append('.')

def run_cell(cell_name, config_path):
    """
    Jalankan cell training config dengan error handling yang konsisten.
    
    Args:
        cell_name: Nama modul/komponen
        config_path: Path ke file konfigurasi
    """
    try:
        # Setup environment dan load config
        from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
        env, config = setup_notebook_environment(cell_name, config_path)
        
        # Setup UI components
        ui_components = setup_ui_component(env, config, cell_name)
        
        # Tambahkan logging yang terintegrasi
        from smartcash.ui.utils.logging_utils import setup_ipython_logging, alert_to_ui

        logger = setup_ipython_logging(ui_components, f"cell_{cell_name}")
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Inisialisasi komponen {cell_name}")
        
        # Setup handler secara manual dengan error handling yang lebih baik
        try:
            handler_module = f"smartcash.ui.training_config.{cell_name}_handler"
            handler_function = f"setup_{cell_name}_handlers"
            
            module = __import__(handler_module, fromlist=[handler_function])
            handler_func = getattr(module, handler_function)
            
            ui_components = handler_func(ui_components, env, config)
            
            # Cek apakah handler berhasil setup
            if logger: logger.info(f"✅ Cell {cell_name} berhasil diinisialisasi")

        except Exception as e:
            alert_to_ui(f"Error Setup Handler: {str(e)}", 'error', ui_components)
            
        # Tampilkan UI
        display_ui(ui_components)
        
        return ui_components

    except ImportError as e:
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", 'warning', ui_components)
    except Exception as e:
        from smartcash.ui.utils.fallback_utils import create_status_message
        create_status_message(f"{str(e)}", 'Error Inisialisasi', 'error')