"""
File: smartcash/ui/training_config/backbone/backbone_initializer.py
Deskripsi: Initializer untuk UI pemilihan backbone model
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output

def initialize_backbone_ui(env: Any = None, config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Inisialisasi UI untuk pemilihan backbone model.
    
    Args:
        env: Environment manager
        config: Konfigurasi untuk model
        
    Returns:
        Dict berisi komponen UI
    """
    # Import yang diperlukan
    import ipywidgets as widgets
    from IPython.display import HTML
    
    # Buat komponen status untuk logger dan debugging
    status_output = widgets.Output()
    debug_output = widgets.Output()
    
    try:
        # Setup UI logger dengan error handling yang lebih baik
        try:
            from smartcash.ui.utils.ui_logger import create_direct_ui_logger
            ui_components_temp = {'status': status_output}
            logger = create_direct_ui_logger(ui_components_temp, 'backbone_ui')
        except Exception as logger_error:
            # Fallback ke print jika logger gagal
            print(f"⚠️ Error saat membuat logger: {str(logger_error)}")
            logger = None
        
        # Log info jika logger tersedia
        if logger:
            logger.info("🚀 Memulai inisialisasi UI backbone model")
        else:
            print("🚀 Memulai inisialisasi UI backbone model")
        
        # Import dependency dengan error handling
        try:
            from smartcash.common.environment import get_environment_manager
            # Dapatkan environment jika belum tersedia
            env = env or get_environment_manager()
        except Exception as env_error:
            with debug_output:
                print(f"⚠️ Error saat mendapatkan environment: {str(env_error)}")
            env = None
        
        # Buat komponen UI dengan penanganan error yang lebih baik
        try:
            # Import dengan error handling
            try:
                from smartcash.ui.training_config.backbone.components.backbone_components import create_backbone_ui
            except ImportError as import_error:
                with debug_output:
                    print(f"❌ Error saat mengimpor create_backbone_ui: {str(import_error)}")
                # Coba import alternatif
                try:
                    import sys
                    import os
                    # Tambahkan path ke sys.path jika perlu
                    module_path = os.path.abspath('/Users/masdevid/Projects/smartcash/smartcash/ui/training_config/backbone/components')
                    if module_path not in sys.path:
                        sys.path.append(module_path)
                    # Import langsung dari file
                    from backbone_components import create_backbone_ui
                except ImportError as alt_error:
                    with debug_output:
                        print(f"❌ Error saat mengimpor alternatif: {str(alt_error)}")
                    raise
            
            # Buat UI components
            ui_components = create_backbone_ui(config)
            ui_components['logger'] = logger
            ui_components['module_name'] = 'backbone'
            ui_components['debug_output'] = debug_output
            
            # Setup handlers untuk tombol-tombol
            try:
                from smartcash.ui.training_config.backbone.handlers.button_handlers import setup_button_handlers
                ui_components = setup_button_handlers(ui_components, config, env)
            except Exception as handler_error:
                with debug_output:
                    print(f"⚠️ Error saat setup button handlers: {str(handler_error)}")
            
            # Tampilkan UI
            clear_output(wait=True)
            if 'ui' in ui_components:
                display(ui_components['ui'])
                if logger:
                    logger.info("✅ UI backbone model berhasil diinisialisasi")
                else:
                    print("✅ UI backbone model berhasil diinisialisasi")
            else:
                # Tampilkan error jika ui tidak ada
                display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red;'><h3>❌ Error: Komponen UI tidak ditemukan</h3></div>"))
                display(debug_output)
            
            return ui_components
            
        except Exception as ui_error:
            if logger:
                logger.error(f"❌ Error saat membuat komponen UI backbone: {str(ui_error)}")
            else:
                print(f"❌ Error saat membuat komponen UI backbone: {str(ui_error)}")
            # Tampilkan debug output untuk membantu troubleshooting
            display(debug_output)
            
    except Exception as e:
        # Tampilkan pesan error
        display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red;'><h3>❌ Error umum saat inisialisasi UI backbone</h3><p>{str(e)}</p></div>"))
        display(debug_output)
        
    # Jika terjadi error, kembalikan dictionary kosong
    return {}
