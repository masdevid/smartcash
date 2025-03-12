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
    config_path: Optional[str] = None,
    create_dirs: Optional[List[str]] = None,
    import_utils: bool = True,
    force_gc: bool = True
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Setup standar untuk cell notebook, dengan penanganan path, config, dan logger.
    
    Args:
        cell_name: Nama unik cell untuk logger
        config_path: Path ke file konfigurasi untuk dimuat
        create_dirs: List direktori yang perlu dibuat
        import_utils: Otomatis import utility classes
        force_gc: Force garbage collection
        
    Returns:
        Tuple (env, config) berisi environment dan konfigurasi
    """
    # Pastikan smartcash ada di path
    if not any('smartcash' in p for p in sys.path):
        sys.path.append('.')
    
    # Force garbage collection jika diperlukan
    if force_gc:
        gc.collect()
    
    # Setup environment dict
    env = {
        'cell_name': cell_name,
        'logger': None,
        'env_manager': None,
        'observer_manager': None,
        'start_time': time.time()
    }
    
    # Buat direktori yang diperlukan
    if create_dirs:
        for dir_path in create_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Import utils dan setup environment components
    try:
        if import_utils:
            from smartcash.utils.config_manager import ConfigManager, get_config_manager
            from smartcash.utils.environment_manager import EnvironmentManager
            from smartcash.utils.logger import get_logger
            from smartcash.utils.observer.observer_manager import ObserverManager
            from smartcash.utils.observer.cleanup_utils import register_notebook_cleanup
            
            # Setup logger
            env['logger'] = get_logger(cell_name)
            
            # Setup environment manager
            env['env_manager'] = EnvironmentManager(logger=env['logger'])
            
            # Setup observer manager dan cleanup otomatis
            env['observer_manager'] = ObserverManager(auto_register=True)
            env['observer_manager'].unregister_group(f"{cell_name}_observers")
            
            # Setup cleanup otomatis
            cleanup_func = register_notebook_cleanup(
                observer_managers=[env['observer_manager']],
                auto_execute=True
            )
            env['cleanup'] = cleanup_func
            
            # Load konfigurasi
            config_manager = get_config_manager(logger=env['logger'])
            config = config_manager.load_config(
                filename=config_path,
                fallback_to_pickle=True,
                logger=env['logger']
            )
            
            # Log success
            env['logger'].info(f"✅ Notebook environment berhasil disetup untuk {cell_name}")
            
            return env, config
        
        # Return minimal environment jika import_utils=False
        return env, {}
    
    except Exception as e:
        print(f"⚠️ Error saat setup environment: {str(e)}")
        return env, {}


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