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
        
    Returns:
        Dict berisi komponen UI
    """
    ui_components = {'module_name': cell_name}
    
    try:
        # Setup environment dan load config
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Dapatkan environment dan config
        env = get_environment_manager()
        config_manager = get_config_manager()
        config = config_manager.load_config(config_path) if config_path else config_manager.config
        
        # Tambahkan logging yang terintegrasi
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        
        logger = setup_ipython_logging(ui_components, f"cell_{cell_name}")
        if logger:
            ui_components['logger'] = logger
            logger.info(f"🚀 Inisialisasi komponen {cell_name}")
        
        # Tampilkan header
        display(HTML(f"<h2>Konfigurasi {cell_name.replace('_', ' ').title()}</h2>"))
        
        # Setup UI components berdasarkan jenis komponen
        try:
            # Tentukan path initializer berdasarkan struktur folder baru
            if cell_name == 'backbone':
                from smartcash.ui.training_config.backbone.backbone_initializer import initialize_backbone_ui
                ui_components = initialize_backbone_ui(env, config)
            elif cell_name == 'hyperparameters':
                from smartcash.ui.training_config.hyperparameters.hyperparameters_initializer import initialize_hyperparameters_ui
                ui_components = initialize_hyperparameters_ui(env, config)
            elif cell_name == 'training_strategy':
                from smartcash.ui.training_config.training_strategy.training_strategy_initializer import initialize_training_strategy_ui
                ui_components = initialize_training_strategy_ui(env, config)
            else:
                # Fallback untuk komponen yang belum direfaktor
                handler_module = f"smartcash.ui.training_config.{cell_name}_handler"
                handler_function = f"setup_{cell_name}_handlers"
                
                module = __import__(handler_module, fromlist=[handler_function])
                handler_func = getattr(module, handler_function)
                
                ui_components = handler_func(ui_components, env, config)
            
            # Cek apakah handler berhasil setup
            if logger: logger.info(f"✅ Cell {cell_name} berhasil diinisialisasi")
            
        except Exception as e:
            # Tampilkan error
            from smartcash.ui.utils.alert_utils import create_alert
            error_msg = f"Error Setup {cell_name}: {str(e)}"
            if logger: logger.error(f"❌ {error_msg}")
            
            if 'status' in ui_components:
                with ui_components['status']:
                    display(HTML(f"<p style='color:red'>❌ {error_msg}</p>"))
            else:
                display(HTML(f"<div style='color:red; padding:10px; border:1px solid red; border-radius:5px; margin:10px 0;'>❌ {error_msg}</div>"))
        
        return ui_components

    except ImportError as e:
        error_msg = f"⚠️ Beberapa komponen tidak tersedia: {str(e)}"
        display(HTML(f"<div style='color:orange; padding:10px; border:1px solid orange; border-radius:5px; margin:10px 0;'>{error_msg}</div>"))
        return ui_components
    except Exception as e:
        error_msg = f"❌ Error inisialisasi: {str(e)}"
        display(HTML(f"<div style='color:red; padding:10px; border:1px solid red; border-radius:5px; margin:10px 0;'>{error_msg}</div>"))
        return ui_components