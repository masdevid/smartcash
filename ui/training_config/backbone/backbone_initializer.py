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
    # Setup UI logger
    from smartcash.ui.utils.ui_logger import create_direct_ui_logger
    
    # Buat komponen status untuk logger
    import ipywidgets as widgets
    status_output = widgets.Output()
    ui_components_temp = {'status': status_output}
    
    # Buat logger
    logger = create_direct_ui_logger(ui_components_temp, 'backbone_ui')
    
    logger.info("🚀 Memulai inisialisasi UI backbone model")
    
    try:
        # Import dependency
        from smartcash.common.environment import get_environment_manager
        
        # Dapatkan environment jika belum tersedia
        env = env or get_environment_manager()
        
        # Buat komponen UI dengan penanganan error yang lebih baik
        try:
            from smartcash.ui.training_config.backbone.components.backbone_components import create_backbone_ui
            
            # Buat UI components
            ui_components = create_backbone_ui(config)
            ui_components['logger'] = logger
            ui_components['module_name'] = 'backbone'
            
            # Setup handlers untuk tombol-tombol
            from smartcash.ui.training_config.backbone.handlers.button_handlers import setup_button_handlers
            ui_components = setup_button_handlers(ui_components, config, env)
            
            # Tampilkan UI
            clear_output(wait=True)
            display(ui_components['ui'])
            
            logger.info("✅ UI backbone model berhasil diinisialisasi")
            
            return ui_components
            
        except Exception as ui_error:
            logger.error(f"❌ Error saat membuat komponen UI backbone: {str(ui_error)}")
            
    except Exception as e:
        logger.error(f"❌ Error umum saat inisialisasi UI backbone: {str(e)}")
        # Tampilkan pesan error
        from IPython.display import HTML
        display(HTML(f"<div style='color: red; padding: 10px; border: 1px solid red;'><h3>❌ Error umum saat inisialisasi UI backbone</h3><p>{str(e)}</p></div>"))
        
    # Jika terjadi error, kembalikan dictionary kosong
    return {}
