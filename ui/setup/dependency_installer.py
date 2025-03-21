"""
File: smartcash/ui/setup/dependency_installer.py
Deskripsi: Koordinator utama untuk instalasi dependency SmartCash
"""

import ipywidgets as widgets
from typing import Dict, Any, Optional

def setup_dependency_installer():
    """Koordinator utama setup dan konfigurasi environment dengan integrasi utilities"""
    try:
        # Import komponen dengan pendekatan konsolidasi
        from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui  
        from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers
        from smartcash.ui.utils.cell_utils import setup_notebook_environment
        from smartcash.ui.utils.logging_utils import setup_ipython_logging
        
        # Setup notebook environment
        env, config = setup_notebook_environment("dependency_installer")
        
        # Buat komponen UI
        ui_components = create_dependency_installer_ui(env, config)
        
        ui_components = setup_dependency_installer_handlers(ui_components, config)
        # Setup logging untuk UI
        logger = setup_ipython_logging(ui_components, "dependency_installer")
        if logger:
            ui_components['logger'] = logger
            logger.info("🚀 Cell dependency_installer diinisialisasi")


    except ImportError as e:
        # Fallback jika modules tidak tersedia
        from smartcash.ui.utils.fallback_utils import show_status
        show_status(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}", "warning", ui_components)
    
    return ui_components
