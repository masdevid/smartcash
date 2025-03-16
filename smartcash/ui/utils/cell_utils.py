"""
File: smartcash/ui/utils/cell_utils.py
Author: Refactored
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
        from smartcash.common.logger import get_logger
        from smartcash.common.environment import get_environment_manager
        from smartcash.common.config import get_config_manager
        
        # Setup komponen
        logger = get_logger(f"cell_{cell_name}")
        env_manager = get_environment_manager()
        config_manager = get_config_manager()
        
        # Load konfigurasi
        try:
            if Path(config_path).exists():
                config = config_manager.load_config(config_path)
                logger.info(f"🔄 Loaded config from {config_path}")
            else:
                config = config_manager.config
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

def create_default_ui_components(cell_name: str) -> Dict[str, Any]:
    """
    Buat komponen UI default untuk cell.
    
    Args:
        cell_name: Nama cell
        
    Returns:
        Dictionary berisi widget UI default
    """
    # Format judul cell dari nama
    title = " ".join(word.capitalize() for word in cell_name.split("_"))
    
    # Buat komponen UI default
    header = widgets.HTML(
        f"""<div style="background-color: #f0f8ff; padding: 15px; color: black; 
                      border-radius: 5px; margin-bottom: 15px; border-left: 5px solid #3498db;">
            <h2 style="color: #2c3e50; margin-top: 0;">{title}</h2>
            <p style="color: #2c3e50; margin-bottom: 0;">No UI component found for this cell</p>
        </div>"""
    )
    
    status = widgets.Output(
        layout=widgets.Layout(
            width='100%',
            border='1px solid #ddd',
            min_height='100px',
            max_height='300px',
            margin='10px 0',
            padding='8px',
            overflow='auto'
        )
    )
    
    progress_bar = widgets.IntProgress(
        value=0,
        min=0,
        max=100,
        description='Progress:',
        bar_style='info',
        orientation='horizontal',
        layout=widgets.Layout(
            width='100%',
            margin='10px 0'
        )
    )
    
    # Return dictionary
    return {
        'ui': widgets.VBox([header, status, progress_bar]),
        'header': header,
        'status': status,
        'progress_bar': progress_bar,
        'module_name': cell_name
    }

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
    # Buat default UI components
    ui_components = create_default_ui_components(component_name)
    
   # Import komponen UI dengan fallback, termasuk pencarian di subdirektori
    try:
        # Coba import dari lokasi baru (setelah refactor)
        import_locations = [
            f"smartcash.ui.components.{component_name}",  # Base location
            f"smartcash.ui.components.setup.{component_name}",  # Setup subdirectory
            f"smartcash.ui.components.dataset.{component_name}",  # Dataset subdirectory
            f"smartcash.ui.components.training_config.{component_name}",  # Training config subdirectory
            f"smartcash.ui.components.training_execution.{component_name}",  # Training execution subdirectory
            f"smartcash.ui.components.model_evaluation.{component_name}",  # Model evaluation subdirectory
            f"smartcash.ui.components.detection.{component_name}"  # Detection subdirectory
        ]
        
        ui_module = None
        create_func = None
        error = None
        
        # Try each possible location
        for module_path in import_locations:
            try:
                ui_module = importlib.import_module(module_path)
                create_func = getattr(ui_module, f"create_{component_name}_ui")
                break
            except (ImportError, AttributeError) as e:
                error = e
                continue
                
        if create_func:
            ui_components = create_func(env, config)
        else:
            # Try fallback to old location
            module_path = f"smartcash.ui_components.{component_name}"
            try:
                ui_module = importlib.import_module(module_path)
                create_func = getattr(ui_module, f"create_{component_name}_ui")
                ui_components = create_func(env, config)
            except (ImportError, AttributeError):
                # Also try subdirectories in old location
                old_import_locations = [
                    f"smartcash.ui_components.setup.{component_name}",
                    f"smartcash.ui_components.dataset.{component_name}",
                    f"smartcash.ui_components.training.{component_name}"
                ]
                
                for module_path in old_import_locations:
                    try:
                        ui_module = importlib.import_module(module_path)
                        create_func = getattr(ui_module, f"create_{component_name}_ui")
                        ui_components = create_func(env, config)
                        break
                    except (ImportError, AttributeError):
                        continue
                
                # If still not found, fallback to default
                if not create_func:
                    with ui_components['status']:
                        display(HTML(f"<p style='color:#856404'>⚠️ UI component '{component_name}' not found in any directory. Using minimal implementation.</p>"))
    except Exception as e:
        # Catch any other errors during import
        with ui_components['status']:
            display(HTML(f"<p style='color:#721c24'>❌ Error importing component '{component_name}': {str(e)}</p>"))
    
    # Tambahkan progress_bar jika belum ada
    if 'progress_bar' not in ui_components:
        progress_bar = widgets.IntProgress(
            value=0,
            min=0,
            max=100,
            description='Progress:',
            bar_style='info',
            orientation='horizontal',
            layout=widgets.Layout(
                width='100%',
                margin='10px 0'
            )
        )
        ui_components['progress_bar'] = progress_bar
        
        # Tambahkan ke UI jika ui_components memiliki 'ui'
        if 'ui' in ui_components and isinstance(ui_components['ui'], widgets.Box):
            ui_children = list(ui_components['ui'].children)
            ui_children.append(progress_bar)
            ui_components['ui'].children = tuple(ui_children)
    
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
        from smartcash.components.observer import ObserverManager
        observer_manager = ObserverManager.get_instance()
        if observer_manager:
            group_name = ui_components.get('observer_group', 'cell_observers')
            observer_manager.unregister_group(group_name)
    except ImportError:
        pass