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
    Setup environment dengan fallback dan error handling.
    
    Args:
        cell_name: Nama unik cell untuk logging
        config_path: Path ke file konfigurasi
        create_dirs: Direktori yang perlu dibuat
        import_utils: Import utility classes
        force_gc: Force garbage collection
    
    Returns:
        Tuple (env, config) dengan informasi environment
    """
    # Tambahkan direktori project ke path jika belum ada
    if not any('smartcash' in p for p in sys.path):
        module_path = str(Path().absolute())
        if module_path not in sys.path:
            sys.path.append(module_path)
    
    # Force garbage collection
    if force_gc:
        gc.collect()
    
    # Inisialisasi environment default
    env = {
        'cell_name': cell_name,
        'logger': None,
        'env_manager': None,
        'observer_manager': None,
        'config_manager': None,
        'status': 'initializing'
    }
    
    # Buat direktori yang diperlukan
    if create_dirs:
        for dir_path in create_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Daftar modul yang akan di-import
    modules_to_import = [
        ('config_manager', 'ConfigManager'),
        ('environment_manager', 'EnvironmentManager'),
        ('logger', 'get_logger'),
        ('observer.observer_manager', 'ObserverManager')
    ]
    
    # Import dengan error handling
    for module_name, class_name in modules_to_import:
        try:
            module = importlib.import_module(f'smartcash.utils.{module_name}')
            
            if module_name == 'config_manager':
                env['config_manager'] = getattr(module, class_name)
            elif module_name == 'environment_manager':
                env['env_manager'] = getattr(module, class_name)(logger=None)
            elif module_name == 'logger':
                env['logger'] = getattr(module, class_name)(cell_name)
            elif module_name == 'observer.observer_manager':
                env['observer_manager'] = getattr(module, class_name)(auto_register=True)
        except ImportError as e:
            if env['logger']:
                env['logger'].warning(f"⚠️ Gagal import {module_name}: {str(e)}")
    
    # Load konfigurasi dengan fallback
    config = {}
    try:
        if config_path and Path(config_path).exists():
            config = env['config_manager'].load_config(
                filename=config_path,
                fallback_to_pickle=True,
                logger=env['logger'],
                use_singleton=True
            ) if env['config_manager'] else {}
    except Exception as e:
        if env['logger']:
            env['logger'].warning(f"⚠️ Gagal load config: {str(e)}")
    
    # Update status
    env['status'] = 'ready' if all(env.values()) else 'partial'
    
    # Tambahkan info cadangan jika modul utama gagal
    if not all(env.values()):
        env.update({
            'fallback_mode': True,
            'base_dir': str(Path.cwd()),
            'python_version': sys.version
        })
    
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