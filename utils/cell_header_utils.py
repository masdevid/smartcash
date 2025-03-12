"""
File: smartcash/utils/cell_header_utils.py
Author: Refactored
Deskripsi: Utility functions untuk setup Jupyter/Colab notebook cells secara konsisten
"""

import sys
import gc
import os
import atexit
import importlib
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List, Union
from IPython.display import display, HTML

def setup_notebook_environment(
    cell_name: str,
    config_path: str = "configs/base_config.yaml",
    create_dirs: Optional[List[str]] = None,
    register_cleanup: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Setup environment standar untuk cell notebook:
    - Menambahkan smartcash ke path
    - Membuat direktori yang diperlukan
    - Setting up logger
    - Loading dan sinkronisasi konfigurasi
    - Setup dan cleanup observer manager
    
    Args:
        cell_name: Nama unik untuk cell (untuk logging dan observer group)
        config_path: Path ke file konfigurasi
        create_dirs: List direktori yang perlu dibuat
        register_cleanup: Register cleanup handler otomatis
    
    Returns:
        Tuple (environment components, config)
    """
    # Pastikan smartcash ada di path
    if not any('smartcash' in p for p in sys.path):
        sys.path.append('.')
    
    # Buat direktori standar
    os.makedirs("configs", exist_ok=True)
    os.makedirs("smartcash/ui_components", exist_ok=True)
    os.makedirs("smartcash/ui_handlers", exist_ok=True)
    
    # Buat direktori tambahan jika diperlukan
    if create_dirs:
        for directory in create_dirs:
            os.makedirs(directory, exist_ok=True)
    
    # Setup komponen
    env = {}
    config = {}
    
    # Setup logging, config dan observer
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.config_manager import get_config_manager
        from smartcash.utils.observer.observer_manager import ObserverManager
        
        # Setup logger
        logger = get_logger(cell_name)
        env['logger'] = logger
        
        # Setup config manager
        config_manager = get_config_manager(logger)
        env['config_manager'] = config_manager
        
        # Coba muat config dari Google Drive jika di Colab
        if config_manager.is_colab and config_manager.drive_mounted:
            config = config_manager.load_from_drive()
        
        # Jika belum ada konfigurasi, muat dari file lokal
        if not config:
            config = config_manager.load_config(config_path, logger=logger)
        
        # Setup observer manager
        observer_manager = ObserverManager(auto_register=True)
        env['observer_manager'] = observer_manager
        
        # Buat observer group berdasarkan nama cell
        observer_group = f"{cell_name}_observers"
        
        # Bersihkan observer dari sesi sebelumnya
        observer_manager.unregister_group(observer_group)
        
        # Setup cleanup
        if register_cleanup:
            def cleanup():
                try:
                    observer_manager.unregister_group(observer_group)
                    logger.info(f"✅ Observer {observer_group} berhasil dibersihkan")
                except Exception as e:
                    logger.error(f"❌ Error saat cleanup: {str(e)}")
            
            atexit.register(cleanup)
            env['cleanup'] = cleanup
        
    except ImportError as e:
        print(f"⚠️ Beberapa komponen tidak tersedia: {str(e)}")
    
    return env, config


def setup_ui_component(
    env: Dict[str, Any], 
    config: Dict[str, Any], 
    component_name: str
) -> Dict[str, Any]:
    """
    Setup UI component dan handler-nya dengan penanganan exception.
    
    Args:
        env: Environment dictionary dari setup_notebook_environment
        config: Konfigurasi yang akan digunakan
        component_name: Nama komponen yang akan dimuat
        
    Returns:
        Dictionary UI components
    """
    # Default empty UI
    fallback_ui = {'ui': HTML("<h3>⚠️ Komponen UI tidak tersedia</h3><p>Pastikan semua modul terinstall dengan benar</p>")}
    
    try:
        # Dinamis import UI component dan handler
        ui_module_name = f"smartcash.ui_components.{component_name}"
        handler_module_name = f"smartcash.ui_handlers.{component_name}"
        
        # Import UI component module
        ui_module = importlib.import_module(ui_module_name)
        ui_create_func = getattr(ui_module, f"create_{component_name}_ui")
        
        # Import handler module
        handler_module = importlib.import_module(handler_module_name)
        handler_setup_func = getattr(handler_module, f"setup_{component_name}_handlers")
        
        # Buat dan setup UI component
        ui_components = ui_create_func()
        ui_components = handler_setup_func(ui_components, config)
        
        # Register cleanup jika belum ada
        if 'cleanup' in ui_components and callable(ui_components['cleanup']):
            if env.get('observer_manager') and hasattr(env['observer_manager'], 'unregister_group'):
                # Tambahkan ke daftar cleanup bersama observer manager
                old_cleanup = ui_components['cleanup']
                
                def combined_cleanup():
                    old_cleanup()
                    env['observer_manager'].unregister_group(f"{env['cell_name']}_observers")
                
                ui_components['cleanup'] = combined_cleanup
                atexit.register(ui_components['cleanup'])
            else:
                # Register ke atexit langsung
                atexit.register(ui_components['cleanup'])
        
        # Log success
        if env.get('logger'):
            env['logger'].info(f"✅ UI Component {component_name} berhasil disetup")
        
        return ui_components
    
    except Exception as e:
        # Log error
        if env.get('logger'):
            env['logger'].error(f"❌ Error saat setup UI component {component_name}: {str(e)}")
        else:
            print(f"❌ Error saat setup UI component {component_name}: {str(e)}")
        
        # Return fallback UI
        return fallback_ui

def display_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display UI component dengan penanganan error.
    
    Args:
        ui_components: Dictionary UI components dari setup_ui_component
    """
    try:
        if 'ui' in ui_components:
            display(ui_components['ui'])
        else:
            display(HTML("<h3>⚠️ Komponen UI tidak valid</h3><p>Tidak ditemukan key 'ui' dalam ui_components</p>"))
    except Exception as e:
        display(HTML(f"<h3>❌ Error saat menampilkan UI</h3><p>{str(e)}</p>"))


def create_minimal_cell(component_name: str, config_path: Optional[str] = None) -> str:
    """
    Buat kode minimal untuk cell notebook.
    
    Args:
        component_name: Nama komponen UI
        config_path: Path ke file konfigurasi
        
    Returns:
        String kode cell
    """
    config_import = f"config_path=\"{config_path}\"" if config_path else "config_path=None"
    
    return f"""# Cell - {component_name.capitalize()}
# Deskripsi: Komponen UI untuk {component_name}

from smartcash.utils.cell_header_utils import setup_notebook_environment, setup_ui_component, display_ui

# Setup environment
env, config = setup_notebook_environment(
    cell_name="{component_name}",
    {config_import}
)

# Setup UI component
ui_components = setup_ui_component(env, config, "{component_name}")

# Tampilkan UI
display_ui(ui_components)
"""