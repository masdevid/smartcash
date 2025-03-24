"""
File: smartcash/ui/utils/cell_utils.py
Deskripsi: Utilitas untuk cell notebook dengan integrasi logging dan komponen alert yang konsisten
"""

import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Callable
import yaml
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_notebook_environment(
    cell_name: str, 
    config_path: str = "configs/colab_config.yaml",
    setup_logger: bool = False  # Parameter baru untuk opsional setup logger
) -> tuple:
    """
    Setup environment notebook dengan konfigurasi cell dan opsional setup logger.
    
    Args:
        cell_name: Nama cell untuk identifikasi dan logging
        config_path: Path relatif ke file konfigurasi
        setup_logger: Apakah setup logger di fungsi ini
        
    Returns:
        Tuple berisi (environment_manager, config_dict)
    """
    # Import logger terkait
    from smartcash.ui.utils.logging_utils import create_dummy_logger
    
    # Buat logger dummy sementara untuk log awal
    logger = create_dummy_logger()
    
    # Pastikan smartcash dalam path
    import sys
    if '.' not in sys.path:
        sys.path.append('.')
        logger.info(f"🛠️ Menambahkan direktori saat ini ke sys.path")
        
    # Import dependencies dengan fallback
    try:
        from smartcash.common.logger import get_logger
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Setup komponen env dan config tanpa logger
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        
        # Setup logger jika diminta
        if setup_logger:
            logger = get_logger(f"cell_{cell_name}")
        
        # Load konfigurasi
        try:
            from pathlib import Path
            config_path = Path(config_path)
            if config_path.exists():
                config = config_manager.load_config(str(config_path))
                if setup_logger:
                    logger.debug(f"📄 Konfigurasi dimuat dari {config_path}")
            else:
                config = config_manager.config
                if setup_logger:
                    logger.warning(f"⚠️ File konfigurasi {config_path} tidak ditemukan, menggunakan konfigurasi default")
        except Exception as e:
            if setup_logger:
                logger.error(f"❌ Error saat memuat konfigurasi: {str(e)}")
            config = {}
            
        # Berhasil, kembalikan env_manager dan config
        return env_manager, config
        
    except ImportError as e:
        print(f"⚠️ Mode fungsionalitas terbatas - {str(e)}")
        
        # Fallback implementation sederhana
        class DummyEnv:
            def __init__(self):
                self.is_colab = 'google.colab' in sys.modules
                self.base_dir = self._get_cwd()
                self.is_drive_mounted = self._check_drive_mounted()
                self.drive_path = '/content/drive/MyDrive' if self.is_drive_mounted else None
            
            def _get_cwd(self):
                import os
                return os.getcwd()
                
            def _check_drive_mounted(self):
                import os
                return os.path.exists('/content/drive/MyDrive')
                
            def get_path(self, p):
                import os
                return os.path.join(self.base_dir, p)
        
        # Fallback config loading
        config = {}
        try:
            import yaml
            from pathlib import Path
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f) or {}
                print(f"🔄 Berhasil memuat konfigurasi dari {config_path}")
        except Exception as e:
            print(f"❌ Error saat memuat konfigurasi: {str(e)}")
            
        return DummyEnv(), config

def create_default_ui_components(cell_name: str) -> Dict[str, Any]:
    """
    Buat komponen UI default untuk cell.
    
    Args:
        cell_name: Nama cell
        
    Returns:
        Dictionary berisi widget UI default
    """
    # Import komponen UI dan konstanta
    import ipywidgets as widgets
    from smartcash.ui.utils.constants import COLORS, ICONS
    
    # Format judul cell dari nama
    title = " ".join(word.capitalize() for word in cell_name.split("_"))
    
    # Buat komponen UI default dengan styling yang lebih baik
    header = widgets.HTML(
        f"""<div style="background-color: {COLORS['header_bg']}; padding: 15px; 
                      color: {COLORS['dark']}; border-radius: 5px; 
                      margin-bottom: 15px; border-left: 5px solid {COLORS['header_border']};">
            <h2 style="color: {COLORS['secondary']}; margin-top: 0;">{title}</h2>
            <p style="color: {COLORS['secondary']}; margin-bottom: 0;">Cell untuk {title}</p>
        </div>"""
    )
    
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border=f'1px solid {COLORS["border"]}',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='8px',
            overflow='auto'
        )
    )
    
    # Return dictionary
    return {
        'ui': widgets.VBox([header, status], 
            layout=widgets.Layout(width='100%', padding='10px')),
        'header': header,
        'status': status,
        'module_name': cell_name,
        'resources': []  # Untuk tracking resources yang perlu dibersihkan
    }

def setup_ui_component(
    env: Any, 
    config: Dict[str, Any], 
    component_name: str
) -> Dict[str, Any]:
    """
    Setup UI component untuk notebook cell dengan integrasi logging.
    
    Args:
        env: Environment manager
        config: Konfigurasi
        component_name: Nama komponen UI
        
    Returns:
        Dictionary berisi widget UI
    """
    # Import komponen terkait
    from IPython.display import display, HTML
    
    # Buat default UI components
    ui_components = create_default_ui_components(component_name)
    
    # Setup logger yang terintegrasi dengan UI dan redirect semua output logging
    from smartcash.ui.utils.logging_utils import setup_ipython_logging
    
    logger = setup_ipython_logging(ui_components, f"cell_{component_name}")
    if logger:
        ui_components['logger'] = logger
        logger.info(f"🚀 Inisialisasi komponen UI '{component_name}'")
    
    # Import komponen UI dengan fallback, termasuk pencarian di subdirektori
    try:
        # Import alert components untuk pesan error/warning
        try:
            from smartcash.ui.utils.alert_utils import create_status_indicator, create_info_alert
            ui_components['create_status_indicator'] = create_status_indicator
            ui_components['create_info_alert'] = create_info_alert
        except ImportError:
            pass

        # Coba import dari lokasi baru (setelah refactor)
        import importlib
        import_locations = [
            f"smartcash.ui.setup.{component_name}_component",
            f"smartcash.ui.dataset.{component_name}_component",
            f"smartcash.ui.training_config.{component_name}_component",
            f"smartcash.ui.training.{component_name}_component",
            f"smartcash.ui.evaluation.{component_name}_component",
            f"smartcash.ui.detection.{component_name}_component"
        ]
        
        # Coba impor dari semua lokasi yang mungkin
        module_loaded = False
        for module_path in import_locations:
            try:
                ui_module = importlib.import_module(module_path)
                create_func = getattr(ui_module, f"create_{component_name}_ui")
                ui_components = create_func(env, config)
                if logger:
                    logger.info(f"✅ Komponen UI '{component_name}' berhasil dimuat dari {module_path}")
                module_loaded = True
                break
            except (ImportError, AttributeError):
                continue
        
        if not module_loaded:
            # Jika tidak ditemukan
            error_msg = f"UI component '{component_name}' tidak ditemukan. Menggunakan implementasi minimal."
            if logger:
                logger.warning(f"⚠️ {error_msg}")
            else:
                with ui_components['status']:
                    # Gunakan komponen alert jika tersedia
                    if 'create_status_indicator' in ui_components:
                        display(ui_components['create_status_indicator']("warning", error_msg))
                    else:
                        from smartcash.ui.utils.constants import COLORS, ICONS
                        display(HTML(f"""
                        <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                                  color:{COLORS['alert_warning_text']}; 
                                  border-radius:4px; margin:5px 0;
                                  border-left:4px solid {COLORS['alert_warning_text']};">
                            <p style="margin:5px 0">{ICONS['warning']} {error_msg}</p>
                        </div>
                        """))
                
    except Exception as e:
        # Catch any other errors during import
        error_msg = f"Error saat import komponen '{component_name}': {str(e)}"
        if logger:
            logger.error(f"❌ {error_msg}")
        else:
            with ui_components['status']:
                # Gunakan komponen alert jika tersedia
                if 'create_status_indicator' in ui_components:
                    display(ui_components['create_status_indicator']("error", error_msg))
                else:
                    from smartcash.ui.utils.constants import COLORS, ICONS
                    display(HTML(f"""
                    <div style="padding:10px; background-color:{COLORS['alert_danger_bg']}; 
                              color:{COLORS['alert_danger_text']}; 
                              border-radius:4px; margin:5px 0;
                              border-left:4px solid {COLORS['alert_danger_text']};">
                        <p style="margin:5px 0">{ICONS['error']} {error_msg}</p>
                    </div>
                    """))
    
    # Setup handler jika tersedia
    try:
        import importlib
        handler_locations = [
            f"smartcash.ui.{component_name.split('_')[0]}.{component_name}_handler",
            f"smartcash.ui.setup.{component_name.split('_')[0]}.{component_name}_handler",
            f"smartcash.ui.training_config.{component_name.split('_')[0]}.{component_name}_handler",
            f"smartcash.ui.training.{component_name.split('_')[0]}.{component_name}_handler",
            f"smartcash.ui.evaluation.{component_name.split('_')[0]}.{component_name}_handler",
            f"smartcash.ui.detection.{component_name.split('_')[0]}.{component_name}_handler"
        ]
        
        handler_loaded = False
        for handler_path in handler_locations:
            try:
                handler_module = importlib.import_module(handler_path)
                setup_handlers_func = getattr(handler_module, f"setup_{component_name}_handlers")
                ui_components = setup_handlers_func(ui_components, env, config)
                if logger:
                    logger.info(f"✅ Handler untuk '{component_name}' berhasil dimuat dari {handler_path}")
                handler_loaded = True
                break
            except (ImportError, AttributeError):
                continue
        
        if not handler_loaded and logger:
            logger.info(f"ℹ️ Tidak ditemukan handler untuk '{component_name}', hanya menggunakan komponen UI")
    except Exception as e:
        if logger:
            logger.warning(f"⚠️ Error saat setup handler: {str(e)}")
    
    return ui_components

def display_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display UI components dalam notebook.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    # Import komponen terkait
    from IPython.display import display, HTML
    
    logger = ui_components.get('logger')
    
    if 'ui' in ui_components:
        if logger:
            logger.debug("🖥️ Menampilkan komponen UI")
        display(ui_components['ui'])
    else:
        # Fallback jika 'ui' tidak ada
        import ipywidgets as widgets
        for key, component in ui_components.items():
            if isinstance(component, (widgets.Widget, widgets.widgets.Widget)):
                if logger:
                    logger.debug(f"🖥️ Menampilkan widget '{key}'")
                display(component)
                break
        else:
            message = "Tidak ada komponen UI yang dapat ditampilkan"
            if logger:
                logger.warning(f"⚠️ {message}")
                
            # Gunakan komponen alert jika tersedia
            if 'create_status_indicator' in ui_components:
                display(ui_components['create_status_indicator']("warning", message))
            else:
                from smartcash.ui.utils.constants import COLORS, ICONS
                display(HTML(f"""
                <div style="padding:10px; background-color:{COLORS['alert_warning_bg']}; 
                          color:{COLORS['alert_warning_text']}; 
                          border-radius:4px; margin:5px 0;">
                    <p style="margin:5px 0">{ICONS['warning']} {message}</p>
                </div>
                """))

def register_cleanup_resource(ui_components: Dict[str, Any], resource: Any, cleanup_func: Optional[Callable] = None) -> None:
    """
    Registrasi resource yang perlu dibersihkan saat cleanup.
    
    Args:
        ui_components: Dictionary komponen UI
        resource: Resource yang perlu dibersihkan
        cleanup_func: Fungsi cleanup khusus (opsional)
    """
    if 'resources' not in ui_components:
        ui_components['resources'] = []
    
    ui_components['resources'].append((resource, cleanup_func))

def cleanup_resources(ui_components: Dict[str, Any]) -> None:
    """
    Cleanup resources yang digunakan oleh UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    from smartcash.ui.utils.logging_utils import setup_ipython_logging
    
    logger = setup_ipython_logging(ui_components,'cleanup_resources')
    
    # Jalankan fungsi cleanup khusus jika ada
    if 'cleanup' in ui_components and callable(ui_components['cleanup']):
        try:
            ui_components['cleanup']()
            if logger:
                logger.info("🧹 Cleanup function berhasil dijalankan")
        except Exception as e:
            if logger:
                logger.warning(f"⚠️ Error saat menjalankan cleanup function: {str(e)}")
    
    # Cleanup resources terdaftar
    if 'resources' in ui_components:
        for resource, cleanup_func in ui_components['resources']:
            try:
                if cleanup_func and callable(cleanup_func):
                    cleanup_func(resource)
                elif hasattr(resource, 'close') and callable(resource.close):
                    resource.close()
                    
                if logger:
                    logger.debug(f"🧹 Resource {type(resource).__name__} berhasil dibersihkan")
            except Exception as e:
                if logger:
                    logger.warning(f"⚠️ Error saat membersihkan resource: {str(e)}")