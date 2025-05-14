"""
File: smartcash/ui/training/training_initializer.py
Deskripsi: Inisialisasi UI untuk proses training model SmartCash
"""

from typing import Dict, Any, Optional
import ipywidgets as widgets
from IPython.display import display, HTML

def initialize_training_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI untuk proses training model.
    
    Returns:
        Dict berisi komponen UI untuk training
    """
    try:
        # Import komponen dan handler
        from smartcash.ui.training.components.training_components import create_training_components
        from smartcash.ui.training.handlers.setup_handler import setup_training_handlers
        from smartcash.common.config.manager import get_config_manager
        from smartcash.common.logger import get_logger
        
        # Dapatkan logger
        logger = get_logger("training_ui")
        
        # Dapatkan ConfigManager
        config_manager = get_config_manager()
        
        # Cek apakah UI components sudah terdaftar
        ui_components = config_manager.get_ui_components('training')
        
        if ui_components:
            logger.info("🔄 Menggunakan komponen UI training yang sudah ada")
        else:
            # Buat komponen UI baru
            logger.info("🆕 Membuat komponen UI training baru")
            ui_components = create_training_components()
            
            # Tambahkan logger ke komponen
            ui_components['logger'] = logger
        
        # Setup handler untuk tombol
        ui_components = setup_training_handlers(ui_components)
        
        # Tampilkan UI
        display(ui_components['main_box'])
        
        return ui_components
        
    except Exception as e:
        # Fallback jika terjadi error
        print(f"❌ Error saat inisialisasi UI training: {str(e)}")
        
        # Tampilkan pesan error
        error_box = widgets.HTML(
            value=f"""
            <div style="color:red;padding:10px;border:1px solid red;border-radius:5px">
                <h3>❌ Error saat inisialisasi UI training</h3>
                <p>{str(e)}</p>
            </div>
            """
        )
        display(error_box)
        
        return {'error': str(e), 'error_box': error_box}
