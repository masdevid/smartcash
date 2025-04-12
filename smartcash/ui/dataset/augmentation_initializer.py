"""
File: smartcash/ui/dataset/augmentation_initializer.py
Deskripsi: Initializer untuk modul augmentasi yang mengintegrasikan semua komponen dengan pendekatan DRY
"""

from typing import Dict, Any, Optional
from IPython.display import display, clear_output
from smartcash.ui.utils.constants import ICONS

def initialize_augmentation_ui() -> Dict[str, Any]:
    """
    Inisialisasi UI modul augmentasi dengan pendekatan modular DRY.
    
    Returns:
        Dictionary UI components
    """
    ui_components = {'module_name': 'augmentation'}
    
    try:
        # Setup environment dan config dengan utils standar
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        env, config = setup_notebook_environment('augmentation')
        
        # Buat komponen UI dengan template yang ada
        from smartcash.ui.dataset.augmentation_component import create_augmentation_ui
        ui_components = create_augmentation_ui(env, config)
        
        # Setup logging terintegrasi dengan UI
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        logger = setup_ipython_logging(ui_components, 'augmentation')
        ui_components['logger'] = logger
        
        # Load konfigurasi augmentasi dengan handler yang ada
        from smartcash.ui.dataset.augmentation_config_handler import load_augmentation_config, update_ui_from_config
        config = load_augmentation_config(ui_components=ui_components)
        ui_components['config'] = config
        
        # Update UI dari konfigurasi yang dimuat
        ui_components = update_ui_from_config(ui_components, config)
        
        # Setup augmentation manager dengan shared util
        from smartcash.ui.dataset.shared.setup_utils import setup_manager
        augmentation_manager = setup_manager(ui_components, config, 'augmentation')
        ui_components['augmentation_manager'] = augmentation_manager
        
        # Aplikasikan shared handlers untuk progress, visualization, cleanup, dan summary
        from smartcash.ui.dataset.shared.integration import apply_shared_handlers, create_cleanup_function
        ui_components = apply_shared_handlers(ui_components, env, config, 'augmentation')
        create_cleanup_function(ui_components, 'augmentation')
        
        # Setup handlers spesifik augmentasi
        from smartcash.ui.dataset.augmentation_handler import setup_augmentation_handlers
        ui_components = setup_augmentation_handlers(ui_components, env, config)
        
        # Deteksi state modul (apakah sudah ada hasil augmentasi)
        from smartcash.ui.dataset.shared.setup_utils import detect_module_state
        ui_components = detect_module_state(ui_components, 'augmentation')
        
        # Setup initial visibility
        if not ui_components.get('is_augmented', False):
            # Sembunyikan container yang tidak relevan jika belum diproses
            for container in ['visualization_container', 'summary_container', 'visualization_buttons']:
                if container in ui_components:
                    ui_components[container].layout.display = 'none'
            
            # Sembunyikan tombol cleanup
            if 'cleanup_button' in ui_components:
                ui_components['cleanup_button'].layout.display = 'none'
        
        # Sembunyikan tombol stop karena belum berjalan
        if 'stop_button' in ui_components:
            ui_components['stop_button'].layout.display = 'none'
        
        # Log sukses
        if logger:
            logger.info(f"{ICONS['success']} Augmentation UI berhasil diinisialisasi")
            
    except Exception as e:
        # Fallback jika terjadi error
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        error_message = f"⚠️ Error inisialisasi augmentation: {str(e)}"
        ui_components = create_fallback_ui(ui_components, error_message, "error")
    
    return ui_components