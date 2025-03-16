"""
File: smartcash/ui/utils/cell_utils.py
Deskripsi: Utilitas untuk cell notebook dengan fungsi setup environment dan komponen UI
"""

import importlib
import sys
import os
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, Union
import yaml
import ipywidgets as widgets
from IPython.display import display, HTML

def setup_notebook_environment(
    cell_name: str,
    config_path: str = "configs/colab_config.yaml"
) -> Tuple[Any, Dict[str, Any]]:
    """
    Setup environment notebook dengan konfigurasi cell.
    
    Args:
        cell_name: Nama cell untuk identifikasi dan logging
        config_path: Path relatif ke file konfigurasi
        
    Returns:
        Tuple berisi (environment_manager, config_dict)
    """
    # Pastikan smartcash dalam path
    if '.' not in sys.path:
        sys.path.append('.')
        
    # Import dependencies dengan fallback
    try:
        from smartcash.utils.logger import get_logger
        from smartcash.utils.environment_manager import get_environment_manager
        from smartcash.utils.config_manager import get_config_manager
        
        # Setup komponen
        logger = get_logger(f"cell_{cell_name}")
        env_manager = get_environment_manager(logger=logger)
        config_manager = get_config_manager(logger=logger)
        
        # Load konfigurasi
        try:
            if Path(config_path).exists():
                config = config_manager.load_config(config_path)
                logger.info(f"🔄 Loaded config from {config_path}")
            else:
                config = config_manager.get_config()
                logger.warning(f"⚠️ Config file {config_path} not found, using default")
        except Exception as e:
            logger.error(f"❌ Error loading config: {e}")
            config = {}
            
        return env_manager, config
        
    except ImportError as e:
        print(f"⚠️ Limited functionality mode - {str(e)}")
        
        # Fallback implementation
        env = type('DummyEnv', (), {
            'is_colab': 'google.colab' in sys.modules,
            'base_dir': os.getcwd(),
            'get_path': lambda p: os.path.join(os.getcwd(), p)
        })
        
        # Fallback config loading
        config = {}
        try:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                print(f"🔄 Loaded config from {config_path}")
        except Exception as e:
            print(f"❌ Error loading config: {e}")
            
        return env, config

def setup_ui_component(
    env: Any, 
    config: Dict[str, Any], 
    component_name: str
) -> Dict[str, Any]:
    """
    Setup UI component untuk notebook cell.
    
    Args:
        env: Environment manager
        config: Konfigurasi
        component_name: Nama komponen UI
        
    Returns:
        Dictionary berisi widget UI
    """
    ui_components = {}
    
    # Import komponen UI dengan fallback
    try:
        # Coba import dari lokasi baru (setelah refactor)
        module_path = f"smartcash.ui.components.{component_name}"
        ui_module = importlib.import_module(module_path)
        
        # Panggil fungsi create_*_ui()
        create_func = getattr(ui_module, f"create_{component_name}_ui")
        ui_components = create_func(env, config)
        
    except (ImportError, AttributeError) as e1:
        try:
            # Fallback ke lokasi lama (sebelum refactor)
            module_path = f"smartcash.ui_components.{component_name}"
            ui_module = importlib.import_module(module_path)
            create_func = getattr(ui_module, f"create_{component_name}_ui")
            ui_components = create_func(env, config)
            
        except (ImportError, AttributeError) as e2:
            # Fallback ke implementasi minimal
            print(f"⚠️ UI component '{component_name}' not found. Using minimal implementation.")
            title = component_name.replace("_", " ").title()
            ui_components = {
                'ui': widgets.VBox([
                    widgets.HTML(f"<h2>{title}</h2>"),
                    widgets.HTML(f"<p>UI component '{component_name}' not available</p>")
                ])
            }
    
    # Setup handler jika tersedia
    try:
        handler_module_path = f"smartcash.ui.handlers.{component_name}"
        handler_module = importlib.import_module(handler_module_path)
        setup_handlers_func = getattr(handler_module, f"setup_{component_name}_handlers")
        ui_components = setup_handlers_func(ui_components, env, config)
    except (ImportError, AttributeError):
        try:
            # Fallback ke lokasi lama
            handler_module_path = f"smartcash.ui_handlers.{component_name}"
            handler_module = importlib.import_module(handler_module_path)
            setup_handlers_func = getattr(handler_module, f"setup_{component_name}_handlers")
            ui_components = setup_handlers_func(ui_components, env, config)
        except (ImportError, AttributeError):
            # No handlers available, just continue
            pass
    
    return ui_components

def display_ui(ui_components: Dict[str, Any]) -> None:
    """
    Display UI components dalam notebook.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    if 'ui' in ui_components:
        display(ui_components['ui'])
    else:
        # Fallback jika 'ui' tidak ada
        for key, component in ui_components.items():
            if isinstance(component, (widgets.Widget, widgets.widgets.Widget)):
                display(component)
                break
        else:
            display(HTML("<p>⚠️ No UI component found to display</p>"))

def cleanup_resources(ui_components: Dict[str, Any]) -> None:
    """
    Cleanup resources yang digunakan oleh UI components.
    
    Args:
        ui_components: Dictionary berisi widget UI
    """
    if 'cleanup' in ui_components and callable(ui_components['cleanup']):
        ui_components['cleanup']()
    
    # Cleanup observer jika ada
    try:
        from smartcash.utils.observer.observer_manager import ObserverManager
        observer_manager = ObserverManager.get_instance()
        if observer_manager:
            group_name = ui_components.get('observer_group', 'cell_observers')
            observer_manager.unregister_group(group_name)
    except ImportError:
        pass