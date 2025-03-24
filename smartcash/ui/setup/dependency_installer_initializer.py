"""
File: smartcash/ui/setup/dependency_installer_initializer.py
Deskripsi: Initializer untuk instalasi dependencies dengan alur otomatis yang lebih robust
"""

from typing import Dict, Any
from IPython.display import display

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
        
        # Optional async init function untuk deteksi packages
        def init_async(ui_components, env, config):
            # Deteksi packages yang sudah terinstall dan siapkan instalasi
            try:
                analyze_func = ui_components.get('analyze_installed_packages')
                if analyze_func and callable(analyze_func):
                    analyze_func(ui_components)
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
        
        # Tampilkan UI jika ada
        if 'ui' in ui_components and ui_components['ui'] is not None:
            display(ui_components['ui'])
            
        return ui_components
        
    except Exception as e:
        # Fallback sederhana jika terjadi error
        from IPython.display import HTML
        error_html = f"""
        <div style="padding:10px; background-color:#f8d7da; 
                   color:#721c24; border-radius:4px; margin:5px 0;
                   border-left:4px solid #721c24;">
            <p style="margin:5px 0">❌ Error inisialisasi dependency installer: {str(e)}</p>
        </div>
        """
        display(HTML(error_html))
        
        # Return minimal components
        import ipywidgets as widgets
        output = widgets.Output()
        display(output)
        return {'module_name': 'dependency_installer', 'ui': output, 'status': output}