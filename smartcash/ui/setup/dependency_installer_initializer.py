"""
File: smartcash/ui/setup/dependency_installer_initializer.py
Deskripsi: Initializer untuk instalasi dependencies dengan alur otomatis 3 tahap
"""

from typing import Dict, Any, Optional

def initialize_dependency_installer() -> Dict[str, Any]:
    """
    Inisialisasi UI dan handler untuk instalasi dependencies.
    
    Returns:
        Dictionary UI components
    """
    try:
        # Gunakan template cell untuk standarisasi
        from smartcash.ui.cell_template import setup_cell
        from smartcash.ui.setup.dependency_installer_component import create_dependency_installer_ui
        from smartcash.ui.setup.dependency_installer_handler import setup_dependency_installer_handlers
        
        # Optional async init function (untuk deteksi packages)
        def init_async(ui_components, env, config):
            # Deteksi packages yang sudah terinstall dan siapkan instalasi
            try:
                from smartcash.ui.setup.dependency_installer_handler import analyze_installed_packages
                analyze_installed_packages(ui_components)
            except Exception as e:
                if 'logger' in ui_components:
                    ui_components['logger'].warning(f"⚠️ Gagal mendeteksi packages otomatis: {str(e)}")
        
        # Gunakan template standar untuk setup
        ui_components = setup_cell(
            cell_name="dependency_installer",
            create_ui_func=create_dependency_installer_ui,
            setup_handlers_func=setup_dependency_installer_handlers,
            init_async_func=init_async
        )
        
        return ui_components
        
    except Exception as e:
        # Fallback jika cell template gagal
        from smartcash.ui.utils.fallback_utils import create_fallback_ui
        ui_components = {'module_name': 'dependency_installer'}
        ui_components = create_fallback_ui(
            ui_components, 
            f"❌ Error inisialisasi dependency installer: {str(e)}", 
            "error"
        )
        return ui_components