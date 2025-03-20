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
        # Tambahkan CSS styles untuk UI yang konsisten
        from smartcash.ui.helpers.ui_helpers import inject_css_styles
        inject_css_styles()
        
        # Setup environment dan load config
        from smartcash.ui.utils.cell_utils import setup_notebook_environment, setup_ui_component, display_ui
        env, config = setup_notebook_environment(cell_name, config_path)
        
        # Setup UI components
        ui_components = setup_ui_component(env, config, cell_name)
        
        # Tambahkan logging yang terintegrasi
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
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
            if logger:
                logger.info(f"✅ Handler {cell_name} berhasil diinisialisasi")
        except Exception as e:
            if 'status' in ui_components:
                with ui_components['status']:
                    display(HTML(f"""
                    <div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'>
                        <h3>❌ Error Setup Handler</h3>
                        <p>{str(e)}</p>
                    </div>
                    """))
            else:
                display(HTML(f"""
                <div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'>
                    <h3>❌ Error Setup Handler</h3>
                    <p>{str(e)}</p>
                </div>
                """))
            
        # Tampilkan UI
        display_ui(ui_components)
        
        return ui_components

    except ImportError as e:
        display(HTML(f"""
        <div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'>
            <h3>❌ Error Import Module</h3>
            <p>{str(e)}</p>
        </div>
        """))
    except Exception as e:
        display(HTML(f"""
        <div style='padding:10px; background:#f8d7da; color:#721c24; border-radius:5px'>
            <h3>❌ Error Inisialisasi</h3>
            <p>{str(e)}</p>
        </div>
        """))
        
# Contoh penggunaan:
# ui_components = run_cell("backbone_selection", "configs/model_config.yaml")
# ui_components = run_cell("hyperparameters", "configs/training_config.yaml")
# ui_components = run_cell("training_strategy", "configs/training_config.yaml")